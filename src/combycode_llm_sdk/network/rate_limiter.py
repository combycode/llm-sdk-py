"""Token buckets for RPM / TPM / RPD tracking.

Transposed from `unified-library-ts/src/network/rate-limiter.ts`.

Pure state and arithmetic -- nothing here waits, it only says HOW LONG to wait.
That is what lets the same limiter serve the async queue (which sleeps on a
loop) and any future synchronous consumer (which would sleep the thread).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

from ..util.http import parse_int_header

_MINUTE_MS = 60_000
_DAY_MS = 86_400_000


def _now_ms() -> float:
    return time.perf_counter() * 1000


class TokenBucket:
    """A bucket that refills continuously and is drained by requests."""

    def __init__(self, capacity: float, refill_interval_ms: float) -> None:
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = _now_ms()
        #: Tokens per millisecond.
        self._refill_rate = 1 / refill_interval_ms

    def try_consume(self, n: float = 1) -> bool:
        """Take `n` tokens if they are there. True when they were."""
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def wait_time_ms(self, n: float = 1) -> float:
        """How long until `n` tokens exist. 0 when they already do."""
        self._refill()
        if self._tokens >= n:
            return 0
        return math.ceil((n - self._tokens) / self._refill_rate)

    def set_remaining(self, remaining: float) -> None:
        """Force the level, from a provider's `x-ratelimit-remaining-*` header.

        The provider's count is authoritative: our own is an estimate that
        drifts, and it cannot see requests made by anything else using the key.
        """
        self._tokens = min(remaining, self._capacity)
        self._last_refill = _now_ms()

    def set_capacity(self, capacity: float) -> None:
        """Adopt a limit discovered from headers.

        The refill rate is recomputed PER MINUTE, which is the window every
        provider states these limits in.
        """
        self._capacity = capacity
        self._refill_rate = capacity / _MINUTE_MS
        self._tokens = min(self._tokens, capacity)

    def drain_until(self, reset_at_ms: float) -> None:
        """Empty the bucket and hold it empty until a moment in the future.

        Used when a 429 names a reset time: refilling through it would let the
        next request go early and earn another 429.
        """
        if reset_at_ms > _now_ms():
            self._tokens = 0
            self._last_refill = reset_at_ms

    @property
    def available(self) -> int:
        self._refill()
        return math.floor(self._tokens)

    def _refill(self) -> None:
        now = _now_ms()
        elapsed = now - self._last_refill
        # `<= 0` also covers the `drain_until` case, where `_last_refill` is set
        # into the FUTURE: no refill happens until the clock passes it.
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


@dataclass(frozen=True)
class RateLimiterConfig:
    """A queue's budget. `None` means unlimited on that axis."""

    rpm: int | None = 30
    tpm: int | None = None
    rpd: int | None = None
    concurrent: int = 5


class RateLimiter:
    """RPM + TPM + RPD together, as one answer."""

    def __init__(self, config: RateLimiterConfig) -> None:
        self._rpm = TokenBucket(config.rpm, _MINUTE_MS / config.rpm) if config.rpm else None
        self._tpm = TokenBucket(config.tpm, _MINUTE_MS / config.tpm) if config.tpm else None
        self._rpd = TokenBucket(config.rpd, _DAY_MS / config.rpd) if config.rpd else None

    def can_proceed(self, estimated_tokens: float = 0) -> bool:
        """Consume from every applicable bucket, or from none.

        Note this DOES consume when it returns True -- the check and the spend
        are one step, because two callers checking before either spends is how a
        limit is exceeded by exactly the number of callers.
        """
        if self._rpm and not self._rpm.try_consume(1):
            return False
        if self._rpd and not self._rpd.try_consume(1):
            return False
        return not (
            self._tpm and estimated_tokens > 0 and not self._tpm.try_consume(estimated_tokens)
        )

    def wait_time_ms(self, estimated_tokens: float = 0) -> float:
        """The longest wait any axis demands."""
        waits = [0.0]
        if self._rpm:
            waits.append(self._rpm.wait_time_ms(1))
        if self._rpd:
            waits.append(self._rpd.wait_time_ms(1))
        if self._tpm and estimated_tokens > 0:
            waits.append(self._tpm.wait_time_ms(estimated_tokens))
        return max(waits)

    def update_from_headers(self, headers: Mapping[str, str]) -> None:
        """Adopt what the provider says it has left.

        Ours is an estimate; theirs is the truth, and it accounts for every other
        client sharing the key.
        """
        remaining_requests = parse_int_header(headers, "x-ratelimit-remaining-requests")
        limit_requests = parse_int_header(headers, "x-ratelimit-limit-requests")
        remaining_tokens = parse_int_header(headers, "x-ratelimit-remaining-tokens")
        limit_tokens = parse_int_header(headers, "x-ratelimit-limit-tokens")

        if self._rpm:
            if limit_requests is not None:
                self._rpm.set_capacity(limit_requests)
            if remaining_requests is not None:
                self._rpm.set_remaining(remaining_requests)
        if self._tpm:
            if limit_tokens is not None:
                self._tpm.set_capacity(limit_tokens)
            if remaining_tokens is not None:
                self._tpm.set_remaining(remaining_tokens)

    def pause(self, duration_ms: float) -> None:
        """Hold everything for a while, after a 429 named a wait."""
        if self._rpm:
            self._rpm.drain_until(_now_ms() + duration_ms)

    @property
    def rpm_available(self) -> float:
        return self._rpm.available if self._rpm else math.inf

    @property
    def tpm_available(self) -> float:
        return self._tpm.available if self._tpm else math.inf


__all__ = ["RateLimiter", "RateLimiterConfig", "TokenBucket"]
