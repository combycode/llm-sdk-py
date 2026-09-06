"""One request, attempted until it succeeds or the policy gives up.

This is the retry loop, and it is the half of `queue-state.ts` that a
SYNCHRONOUS caller can use. What is deliberately absent is the queue: a priority
heap, a semaphore and a concurrency limit all order work that is in flight at the
same time, and a blocking caller has exactly one request in flight. Giving the
sync path a queue would be theatre -- it would always hold one item, and the
limit would never bind.

What a single caller DOES get, and what this is:

- the error taxonomy, so `except RateLimitError` works;
- retry with exponential backoff and jitter;
- the server's `Retry-After`, honoured and capped;
- a per-attempt and a whole-sequence deadline;
- the network hooks, so cost and telemetry see the same events either way.

`AsyncQueueState` (when it lands) drives the same decisions from
`network/retry.py` around an `await`; the decisions themselves are shared, so
the two cannot disagree about whether something is retryable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from ..bus.hook_bus import HookBus
from .errors import LLMError, NetworkError, classify_error
from .errors import TimeoutError as LLMTimeoutError
from .retry import RetryConfig, calculate_backoff, should_retry
from .types import HttpRequest, HttpResponse


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _replayable(body: Any) -> bool:
    """Whether a second attempt would send the same body.

    An iterator is consumed by the first attempt, so retrying it sends an empty
    one -- the same hardening OpenAI's client shipped in 6.49. Bytes, strings and
    mappings all replay.
    """
    return not (hasattr(body, "__next__") or hasattr(body, "__anext__"))


class RequestExecutor:
    """Runs one request under a retry policy, emitting the network hooks.

    Holds no per-request state: `execute` is re-entrant, so one executor serves
    a whole engine.
    """

    def __init__(self, hooks: HookBus, queue_name: str = "") -> None:
        self._hooks = hooks
        self._queue_name = queue_name

    def execute(
        self,
        req: HttpRequest,
        send: Callable[[HttpRequest], HttpResponse],
        retry: RetryConfig,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> HttpResponse:
        """Send, and keep sending until it works or the policy says stop.

        `send` is the transport call. `sleep` is injectable so a test can assert
        the backoff schedule without waiting for it.
        """
        started = _now_ms()
        attempt = 0
        provider = req.get("provider") or ""
        model = req.get("model") or ""

        while True:
            self._emit_start(req, attempt, provider, model)
            attempt_started = _now_ms()
            try:
                response = self._attempt(req, send, retry, attempt_started)
            except LLMError as error:
                error.provider = error.provider or provider
                error.model = error.model or model
                delay = self._decide(error, attempt, retry, req, started, provider, model)
                if delay is None:
                    raise
                sleep(delay / 1000)
                attempt += 1
                continue

            self._emit_complete(req, attempt, response, attempt_started, provider, model)
            return response

    # -- one attempt ---------------------------------------------------------

    def _attempt(
        self,
        req: HttpRequest,
        send: Callable[[HttpRequest], HttpResponse],
        retry: RetryConfig,
        attempt_started: float,
    ) -> HttpResponse:
        """Send once, and turn anything that is not a 2xx into an `LLMError`.

        A transport raising is a NETWORK error rather than a crash: a DNS failure
        and a 503 are the same thing to a caller deciding whether to retry, and
        letting the raw exception through would skip the taxonomy entirely.
        """
        try:
            response = send(req)
        except LLMError:
            raise
        except Exception as exc:
            raise NetworkError(
                str(exc) or type(exc).__name__,
                provider=req.get("provider") or "",
                model=req.get("model") or "",
                retryable=True,
                raw=exc,
            ) from exc

        elapsed = _now_ms() - attempt_started
        if elapsed > retry.attempt_timeout_ms:
            raise LLMTimeoutError(
                f"attempt exceeded {retry.attempt_timeout_ms:.0f}ms",
                provider=req.get("provider") or "",
                model=req.get("model") or "",
                retryable=True,
            )

        status = int(response.get("status") or 0)
        if status >= 400:
            raise classify_error(
                req.get("provider") or "",
                status,
                response.get("body"),
                response.get("headers") or {},
                model=req.get("model") or "",
            )
        return response

    # -- the retry decision --------------------------------------------------

    def _decide(
        self,
        error: LLMError,
        attempt: int,
        retry: RetryConfig,
        req: HttpRequest,
        started: float,
        provider: str,
        model: str,
    ) -> float | None:
        """How long to wait before the next attempt, or None to give up.

        `onModelError` fires either way and carries `willRetry`, so a subscriber
        sees every failure -- not only the final one.
        """
        request_max = _request_max_retries(req)
        willing = should_retry(
            error,
            attempt,
            retry,
            elapsed_ms=_now_ms() - started,
            request_max_retries=request_max,
            replayable_body=_replayable(req.get("body")),
        )

        self._hooks.emit_sync(
            "onModelError",
            {
                "provider": provider,
                "model": model,
                "queueName": self._queue_name,
                "error": error,
                "headers": {},
                "attempt": attempt,
                "willRetry": willing,
                "trace": req.get("trace"),
            },
        )
        if not willing:
            return None

        backoff = calculate_backoff(attempt, error, retry)
        if error.kind == "rate_limit":
            self._hooks.emit_sync(
                "onRateLimitHit",
                {
                    "provider": provider,
                    "model": model,
                    "queueName": self._queue_name,
                    "retryAfterMs": error.retry_after_ms,
                    "trace": req.get("trace"),
                },
            )
        self._hooks.emit_sync(
            "onRetry",
            {
                "provider": provider,
                "model": model,
                "queueName": self._queue_name,
                "attempt": attempt + 1,
                "backoffMs": backoff,
                "reason": error.kind,
                "trace": req.get("trace"),
            },
        )
        return backoff

    # -- hooks ---------------------------------------------------------------

    def _emit_start(
        self, req: HttpRequest, attempt: int, provider: str, model: str
    ) -> None:
        self._hooks.emit_sync(
            "onRequestStart",
            {
                "provider": provider,
                "model": model,
                "queueName": self._queue_name,
                "url": req.get("url"),
                "attempt": attempt,
                "trace": req.get("trace"),
            },
        )

    def _emit_complete(
        self,
        req: HttpRequest,
        attempt: int,
        response: HttpResponse,
        attempt_started: float,
        provider: str,
        model: str,
    ) -> None:
        self._hooks.emit_sync(
            "onRequestComplete",
            {
                "provider": provider,
                "model": model,
                "queueName": self._queue_name,
                "status": response.get("status"),
                "attempt": attempt,
                "latencyMs": _now_ms() - attempt_started,
                "trace": req.get("trace"),
            },
        )


def _request_max_retries(req: Mapping[str, Any]) -> int | None:
    """The per-request `max_retries`, from either spelling.

    The wire request carries `retry` as the caller gave it: a `RetryOverride`
    dataclass or a plain mapping.
    """
    override = req.get("retry")
    if override is None:
        return None
    if isinstance(override, Mapping):
        value = override.get("max_retries", override.get("maxRetries"))
    else:
        value = getattr(override, "max_retries", None)
    return int(value) if value is not None else None


__all__ = ["RequestExecutor"]
