"""Retry policy: what to retry, how long to wait, and whose setting wins.

Transposed from `queue-state-config.ts` and the retry half of `queue-state.ts`.
Split out of both because it is the part the SYNCHRONOUS and ASYNCHRONOUS
engines share entirely: the decision to retry, the backoff arithmetic and the
precedence rules are pure, and only the waiting differs.

Config is snake_case, unlike the wire shapes -- it is written by the caller, in
Python, and the API contract's example writes it that way::

    Engine(retry={"backoff": {"initial_ms": 1}, "per_kind": {...}})

Three layers, narrowest first: `request.retry` > `queues[name].retry` >
`Engine(retry=)`. Nested groups MERGE rather than replace, so overriding one
backoff knob keeps the rest -- otherwise a one-line tweak silently resets the
whole schedule.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import LLMError


@dataclass(frozen=True)
class BackoffConfig:
    initial_ms: float = 500
    max_ms: float = 8_000
    multiplier: float = 2
    #: Fraction of the delay to randomise, each way. 0.25 spreads a 1s wait over
    #: 750-1250ms, which is what stops a fleet retrying in lockstep.
    jitter: float = 0.25


@dataclass(frozen=True)
class ErrorRetryConfig:
    """Per-kind rules. Every field is optional: an unset one defers outward."""

    retryable: bool | None = None
    max_retries: int | None = None
    fixed_backoff_ms: float | None = None


@dataclass(frozen=True)
class RetryOverride:
    """A retry policy stated as a partial override of another one.

    Every field optional, and `backoff` is a partial too -- which is the whole
    point: `Partial<RetryConfig>` in TypeScript still demanded a complete
    `backoff` object, making "override one knob, keep the rest" inexpressible
    even though the merge always supported it.

    `per_kind` is deliberately absent from the PER-REQUEST layer: one request
    cannot sensibly redefine which error classes are retryable for the queue it
    shares with everyone else.
    """

    max_retries: int | None = None
    total_timeout_ms: float | None = None
    attempt_timeout_ms: float | None = None
    max_retry_after_ms: float | None = None
    backoff: Mapping[str, Any] | None = None
    per_kind: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RetryConfig:
    """A complete policy -- what `merge_retry` produces and the engines read."""

    max_retries: int = 2
    total_timeout_ms: float = 120_000
    attempt_timeout_ms: float = 600_000
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    #: The longest `Retry-After` we are willing to honour. A server asking for
    #: longer does not get waited for: the request fails fast instead of parking,
    #: and the rate limiter is not paused for that long either. Without a cap, a
    #: single `Retry-After: 86400` silently holds a request -- and the whole
    #: limiter -- for a day.
    max_retry_after_ms: float = 120_000
    per_kind: Mapping[str, ErrorRetryConfig] = field(default_factory=dict)


#: Which kinds are worth retrying, and how hard. Rate limits get the most
#: attempts because they are the one failure that reliably clears on its own.
DEFAULT_PER_KIND: dict[str, ErrorRetryConfig] = {
    "rate_limit": ErrorRetryConfig(retryable=True, max_retries=5),
    "server_error": ErrorRetryConfig(retryable=True, max_retries=2),
    "timeout": ErrorRetryConfig(retryable=True, max_retries=2),
    "network": ErrorRetryConfig(retryable=True, max_retries=2),
    "context_overflow": ErrorRetryConfig(retryable=False),
    "auth": ErrorRetryConfig(retryable=False),
    "invalid_request": ErrorRetryConfig(retryable=False),
    "model_not_found": ErrorRetryConfig(retryable=False),
    "quota_exceeded": ErrorRetryConfig(retryable=False),
    "content_filter": ErrorRetryConfig(retryable=False),
    "unsupported": ErrorRetryConfig(retryable=False),
}

DEFAULT_RETRY = RetryConfig(per_kind=DEFAULT_PER_KIND)


def _as_dict(value: Any) -> dict[str, Any]:
    """A dataclass or a plain dict, either way -- callers write both."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {k: v for k, v in value.items() if v is not None}
    return {k: v for k, v in asdict(value).items() if v is not None}


def _per_kind(value: Any) -> dict[str, ErrorRetryConfig]:
    out: dict[str, ErrorRetryConfig] = {}
    for kind, cfg in (value or {}).items():
        out[kind] = cfg if isinstance(cfg, ErrorRetryConfig) else ErrorRetryConfig(**_as_dict(cfg))
    return out


def merge_retry(*overrides: Any, base: RetryConfig = DEFAULT_RETRY) -> RetryConfig:
    """Layer overrides onto a base, widest first.

    `backoff` and `per_kind` merge KEY BY KEY rather than replacing, at every
    layer, which is what makes a one-knob override safe.
    """
    max_retries = base.max_retries
    total_timeout_ms = base.total_timeout_ms
    attempt_timeout_ms = base.attempt_timeout_ms
    max_retry_after_ms = base.max_retry_after_ms
    backoff = asdict(base.backoff)
    per_kind = dict(base.per_kind)

    for override in overrides:
        if override is None:
            continue
        data = _as_dict(override)
        backoff.update(_as_dict(data.pop("backoff", None)))
        per_kind.update(_per_kind(data.pop("per_kind", None)))
        max_retries = int(data.get("max_retries", max_retries))
        total_timeout_ms = float(data.get("total_timeout_ms", total_timeout_ms))
        attempt_timeout_ms = float(data.get("attempt_timeout_ms", attempt_timeout_ms))
        max_retry_after_ms = float(data.get("max_retry_after_ms", max_retry_after_ms))

    return RetryConfig(
        max_retries=max_retries,
        total_timeout_ms=total_timeout_ms,
        attempt_timeout_ms=attempt_timeout_ms,
        max_retry_after_ms=max_retry_after_ms,
        backoff=BackoffConfig(**backoff),
        per_kind=per_kind,
    )


def should_retry(
    error: LLMError,
    attempt: int,
    retry: RetryConfig,
    *,
    elapsed_ms: float,
    request_max_retries: int | None = None,
    replayable_body: bool = True,
) -> bool:
    """Whether to make attempt N+1.

    Five independent reasons not to, and every one of them has cost someone a
    production incident:

    - the KIND is not retryable (retrying a 401 forever);
    - the attempts are used up;
    - the total budget is spent -- a per-attempt cap alone lets a sequence run
      for minutes;
    - the body cannot be replayed, because a streamed body is consumed by the
      first attempt and the second would send an empty one;
    - the server asked for longer than we will wait, which is a refusal rather
      than a delay: parking for hours looks identical to a hang from the
      caller's side.
    """
    kind_config = retry.per_kind.get(error.kind)
    retryable = kind_config.retryable if kind_config and kind_config.retryable is not None else error.retryable
    # Precedence: a per-request override beats the per-kind rule, which beats
    # the queue default. The override is the most specific statement of intent.
    max_retries = request_max_retries
    if max_retries is None:
        max_retries = kind_config.max_retries if kind_config else None
    if max_retries is None:
        max_retries = retry.max_retries

    retry_after_too_long = (
        error.retry_after_ms is not None and error.retry_after_ms > retry.max_retry_after_ms
    )
    return bool(
        retryable
        and attempt < max_retries
        and elapsed_ms < retry.total_timeout_ms
        and replayable_body
        and not retry_after_too_long
    )


def honored_retry_after_ms(error: LLMError, retry: RetryConfig) -> float | None:
    """The server's `Retry-After`, clamped to what we are willing to wait."""
    raw = error.retry_after_ms
    if raw is None or raw <= 0:
        return None
    return min(raw, retry.max_retry_after_ms)


def calculate_backoff(
    attempt: int,
    error: LLMError,
    retry: RetryConfig,
    *,
    rand: Any = None,
) -> float:
    """How long to wait before attempt N+1, in milliseconds.

    The server's instruction wins, then a per-kind fixed delay, then exponential
    backoff with jitter. `rand` is injectable so a test can pin the jitter
    rather than assert on a range.
    """
    honored = honored_retry_after_ms(error, retry)
    if honored is not None:
        return honored

    kind_config = retry.per_kind.get(error.kind)
    if kind_config and kind_config.fixed_backoff_ms:
        return kind_config.fixed_backoff_ms

    b = retry.backoff
    base = min(b.initial_ms * b.multiplier**attempt, b.max_ms)
    draw = (rand or random.random)()
    # `1 - (draw * jitter * 2 - jitter)` spreads the delay symmetrically about
    # `base`, by +/- jitter.
    return round(base * (1 - (draw * b.jitter * 2 - b.jitter)))


__all__ = [
    "DEFAULT_PER_KIND",
    "DEFAULT_RETRY",
    "BackoffConfig",
    "ErrorRetryConfig",
    "RetryConfig",
    "RetryOverride",
    "calculate_backoff",
    "honored_retry_after_ms",
    "merge_retry",
    "should_retry",
]
