"""NetworkEngine -- the multi-queue HTTP router, Layer 1.

Transposed from `unified-library-ts/src/network/engine.ts`.

Holds a map of `queueName -> QueueState`. Each queue has its own rate limiter,
concurrency limit, retry policy and heap. Queues are created lazily on first use,
with settings supplied beforehand via `configure_queue` -- SNAPSHOT semantics, so
a queue's settings are fixed once it exists.

Knows nothing about LLMs, providers or models. The semantic layer fills in
`provider` and `model` for the hook payloads and chooses `queue_name`, whose
default formula is `"$provider/$model"`.

This is the ASYNC engine. Its synchronous sibling is `RequestExecutor`, which
runs the same retry decisions without a queue -- see that module for why a
blocking caller has nothing to queue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..bus.hook_bus import HookBus
from .queue_state import PRIORITY_INTERACTIVE, QueueState, QueueStateConfig
from .rate_limiter import RateLimiterConfig
from .request_queue import QueueConfig
from .retry import RetryConfig, merge_retry
from .types import HttpRequest, HttpResponse, SSEEvent

#: What a queue gets when nothing more specific is configured. Conservative on
#: purpose: a queue created lazily for an unknown model should not be the reason
#: an account trips a limit.
FALLBACK_LIMITS = RateLimiterConfig(rpm=30, tpm=None, rpd=None, concurrent=5)


@dataclass
class QueueSettings:
    """Per-queue configuration, looked up by name when the queue is created."""

    limits: Mapping[str, Any] | None = None
    retry: Any = None
    queue: Mapping[str, Any] | None = None


@dataclass
class NetworkEngineConfig:
    hooks: HookBus | None = None
    #: The async fetch every queue uses. Bound at queue creation.
    fetch: Callable[..., Any] | None = None
    fetch_stream: Callable[..., Any] | None = None
    queues: Mapping[str, Any] = field(default_factory=dict)
    #: Retry policy inherited by EVERY queue this engine creates.
    #:
    #: Retry is cross-cutting, not something a caller threads through each call,
    #: so it is configured once and applies everywhere. Three layers, narrowest
    #: wins: the request's, then the queue's, then this.
    retry: Any = None


class NetworkEngine:
    """`class NetworkEngine` (engine.ts:83)."""

    def __init__(self, config: NetworkEngineConfig | None = None) -> None:
        config = config or NetworkEngineConfig()
        self.hooks = config.hooks or HookBus()
        self._fetch = config.fetch
        self._fetch_stream = config.fetch_stream
        self._default_retry = config.retry
        self._settings: dict[str, QueueSettings] = {
            name: _as_settings(value) for name, value in (config.queues or {}).items()
        }
        self._queues: dict[str, QueueState] = {}

    # -- configuration -------------------------------------------------------

    def configure_queue(self, queue_name: str, settings: Any) -> None:
        """Pre-configure a queue.

        Raises if it already exists: settings are snapshotted at creation, so a
        later change would silently do nothing. `drop_queue` first to reconfigure.
        """
        if queue_name in self._queues:
            raise ValueError(
                f'NetworkEngine: queue "{queue_name}" already created -- settings are '
                f'immutable. Call drop_queue("{queue_name}") first to reconfigure.'
            )
        self._settings[queue_name] = _as_settings(settings)

    def drop_queue(self, queue_name: str) -> None:
        """Forget a queue. In-flight requests continue; the next call rebuilds it."""
        self._queues.pop(queue_name, None)

    def has_queue(self, queue_name: str) -> bool:
        return queue_name in self._queues

    def queue_names(self) -> list[str]:
        return list(self._queues)

    def get_queue_state(self, queue_name: str) -> QueueState | None:
        return self._queues.get(queue_name)

    def snapshot(self) -> list[dict[str, Any]]:
        """Numeric state of every live queue, for metrics."""
        return [queue.snapshot() for queue in self._queues.values()]

    def destroy(self) -> None:
        self._queues.clear()

    # -- submission ----------------------------------------------------------

    async def fetch(
        self, req: HttpRequest, options: Mapping[str, Any] | None = None
    ) -> HttpResponse:
        queue = self._get_or_create(_resolve_queue_name(req, options))
        options = options or {}
        return await queue.submit(
            req,
            int(options.get("priority", PRIORITY_INTERACTIVE)),
            float(options.get("estimatedTokens") or 0),
        )

    def fetch_stream(
        self, req: HttpRequest, options: Mapping[str, Any] | None = None
    ) -> AsyncIterator[SSEEvent]:
        """Returned, not awaited: the caller writes `async for` over it."""
        queue = self._get_or_create(_resolve_queue_name(req, options))
        options = options or {}
        return queue.submit_stream(
            req,
            int(options.get("priority", PRIORITY_INTERACTIVE)),
            float(options.get("estimatedTokens") or 0),
        )

    # -- internals -----------------------------------------------------------

    def _get_or_create(self, queue_name: str) -> QueueState:
        queue = self._queues.get(queue_name)
        if queue is not None:
            return queue

        settings = self._settings.get(queue_name) or QueueSettings()
        limits = _merge_limits(settings.limits)
        # Engine-wide policy first, this queue's on top. Nested groups merge, so
        # overriding one backoff knob does not discard the rest.
        retry: RetryConfig = merge_retry(self._default_retry, settings.retry)

        queue = QueueState(
            QueueStateConfig(
                queue_name=queue_name,
                fetch=_require_fetch(self._fetch, "fetch"),
                hooks=self.hooks,
                limits=limits,
                retry=retry,
                queue=_merge_queue_config(settings.queue),
            )
        )
        self._queues[queue_name] = queue
        return queue


def _require_fetch(fetch: Callable[..., Any] | None, what: str) -> Callable[..., Any]:
    if fetch is None:
        raise ValueError(f"NetworkEngine: no {what} function configured")
    return fetch


def _as_settings(value: Any) -> QueueSettings:
    if isinstance(value, QueueSettings):
        return value
    value = value or {}
    return QueueSettings(
        limits=value.get("limits"), retry=value.get("retry"), queue=value.get("queue")
    )


def _merge_limits(limits: Mapping[str, Any] | None) -> RateLimiterConfig:
    if not limits:
        return FALLBACK_LIMITS
    return RateLimiterConfig(
        rpm=limits.get("rpm", FALLBACK_LIMITS.rpm),
        tpm=limits.get("tpm", FALLBACK_LIMITS.tpm),
        rpd=limits.get("rpd", FALLBACK_LIMITS.rpd),
        concurrent=int(limits.get("concurrent", FALLBACK_LIMITS.concurrent)),
    )


def _merge_queue_config(queue: Mapping[str, Any] | None) -> QueueConfig:
    if not queue:
        return QueueConfig()
    base = QueueConfig()
    return QueueConfig(
        max_size=int(queue.get("max_size", base.max_size)),
        timeout_ms=float(queue.get("timeout_ms", base.timeout_ms)),
    )


def _resolve_queue_name(req: HttpRequest, options: Mapping[str, Any] | None) -> str:
    options = options or {}
    if options.get("queueName"):
        return str(options["queueName"])
    ctx = options.get("ctx") or {}
    if ctx.get("queueName"):
        return str(ctx["queueName"])
    return f"{req.get('provider')}/{req.get('model')}"


__all__ = [
    "FALLBACK_LIMITS",
    "NetworkEngine",
    "NetworkEngineConfig",
    "QueueSettings",
]
