"""The queue, the limiter, and both engines.

The retry DECISIONS are tested in `test_errors_and_retry.py`; what is new here
is the machinery around them -- ordering, concurrency, lazy queue creation, and
the `Engine` facade's hook decorators.

The retry-precedence cases are `examples/features-highlight/engine_retry_policy.py`
translated: it is the reviewed contract for how the three layers combine, and it
states its own expected numbers.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from combycode_llm_sdk.helpers.engine import Engine, default_engine
from combycode_llm_sdk.network.engine import NetworkEngine, NetworkEngineConfig
from combycode_llm_sdk.network.errors import LLMError, RateLimitError
from combycode_llm_sdk.network.rate_limiter import RateLimiter, RateLimiterConfig, TokenBucket
from combycode_llm_sdk.network.request_queue import (
    QueueConfig,
    QueueFullError,
    RequestQueue,
    Semaphore,
    body_size_of,
)
from combycode_llm_sdk.network.retry import RetryOverride
from combycode_llm_sdk.transport import TransportResponse

FAST = {"initial_ms": 1, "max_ms": 2, "multiplier": 1, "jitter": 0}
RETRYABLE = {"server_error": {"retryable": True, "max_retries": 3}}


def req(url: str = "https://x/y", **over: Any) -> dict[str, Any]:
    return {"url": url, "provider": "openai", "model": "m", "body": {}, **over}


class TestTokenBucket:
    def test_it_starts_full_and_drains(self) -> None:
        bucket = TokenBucket(capacity=3, refill_interval_ms=1000)
        assert [bucket.try_consume() for _ in range(4)] == [True, True, True, False]

    def test_an_empty_bucket_reports_how_long_to_wait(self) -> None:
        bucket = TokenBucket(capacity=1, refill_interval_ms=1000)
        assert bucket.try_consume() is True
        assert 0 < bucket.wait_time_ms() <= 1000

    def test_a_provider_header_overrides_our_estimate(self) -> None:
        # Theirs is the truth: ours cannot see other clients on the same key.
        bucket = TokenBucket(capacity=100, refill_interval_ms=600)
        bucket.set_remaining(2)
        assert bucket.available == 2

    def test_draining_until_a_reset_time_stops_refilling(self) -> None:
        # Refilling through a named reset would let the next request go early
        # and earn another 429.
        bucket = TokenBucket(capacity=10, refill_interval_ms=1)
        import time as _t

        bucket.drain_until(_t.perf_counter() * 1000 + 5_000)
        assert bucket.available == 0
        assert bucket.wait_time_ms() > 0


class TestRateLimiter:
    def test_requests_per_minute_bind(self) -> None:
        limiter = RateLimiter(RateLimiterConfig(rpm=2, concurrent=5))
        assert [limiter.can_proceed() for _ in range(3)] == [True, True, False]

    def test_an_unset_axis_never_binds(self) -> None:
        limiter = RateLimiter(RateLimiterConfig(rpm=None, tpm=None, rpd=None, concurrent=1))
        assert all(limiter.can_proceed(10_000) for _ in range(50))
        assert limiter.wait_time_ms(10_000) == 0

    def test_token_budget_binds_on_estimated_tokens(self) -> None:
        limiter = RateLimiter(RateLimiterConfig(rpm=None, tpm=1000, concurrent=5))
        assert limiter.can_proceed(900) is True
        assert limiter.can_proceed(900) is False

    def test_a_pause_holds_everything(self) -> None:
        limiter = RateLimiter(RateLimiterConfig(rpm=100, concurrent=5))
        limiter.pause(5_000)
        assert limiter.can_proceed() is False


class TestRequestQueue:
    async def test_urgency_first_then_arrival(self) -> None:
        queue = RequestQueue()
        for priority, url in [(2, "low-1"), (0, "urgent"), (2, "low-2"), (1, "mid")]:
            queue.enqueue(req(url), priority, 0)
        order = []
        while (entry := queue.dequeue()) is not None:
            order.append(entry.request["url"])
        # Equal priorities keep arrival order -- otherwise a busy queue starves
        # whichever request happens to hash badly.
        assert order == ["urgent", "mid", "low-1", "low-2"]

    async def test_a_full_queue_refuses_rather_than_grows(self) -> None:
        # An unbounded queue turns a provider outage into memory exhaustion.
        queue = RequestQueue(QueueConfig(max_size=2))
        queue.enqueue(req(), 1, 0)
        queue.enqueue(req(), 1, 0)
        with pytest.raises(QueueFullError):
            queue.enqueue(req(), 1, 0)

    async def test_an_entry_past_its_deadline_is_rejected_not_run(self) -> None:
        queue = RequestQueue(QueueConfig(timeout_ms=0))
        future = queue.enqueue(req(), 1, 0)
        await asyncio.sleep(0.002)
        assert queue.dequeue() is None
        with pytest.raises(TimeoutError, match="timed out in queue"):
            await future


class TestSemaphore:
    async def test_it_reports_its_own_counts(self) -> None:
        # asyncio.Semaphore would do the waiting, but exposes no counts -- and
        # these are what the queue snapshot reports.
        sem = Semaphore(2)
        await sem.acquire()
        await sem.acquire()
        assert (sem.in_flight, sem.available, sem.waiting) == (2, 0, 0)
        waiter = asyncio.ensure_future(sem.acquire())
        await asyncio.sleep(0)
        assert sem.waiting == 1
        sem.release()
        await waiter
        assert sem.in_flight == 2


def test_body_size_is_null_safe() -> None:
    # A body-less request has no body; measuring `None` would report 4 for "null".
    assert body_size_of(None) == 0
    assert body_size_of({"a": 1}) == len('{"a":1}')
    assert body_size_of(b"1234") == 4


class TestNetworkEngine:
    async def test_it_retries_until_it_works(self) -> None:
        calls = []

        async def fetch(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            calls.append(request["url"])
            status = 500 if len(calls) < 3 else 200
            return {"status": status, "headers": {}, "body": {}}

        engine = NetworkEngine(
            NetworkEngineConfig(fetch=fetch, retry={"backoff": FAST, "per_kind": RETRYABLE})
        )
        response = await engine.fetch(req())
        assert response["status"] == 200
        assert len(calls) == 3

    async def test_concurrency_is_capped_at_the_limit(self) -> None:
        peak = live = 0

        async def slow(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.005)
            live -= 1
            return {"status": 200, "headers": {}, "body": {}}

        engine = NetworkEngine(
            NetworkEngineConfig(
                fetch=slow, queues={"q": {"limits": {"rpm": None, "concurrent": 2}}}
            )
        )
        await asyncio.gather(
            *[engine.fetch(req(), {"queueName": "q"}) for _ in range(10)]
        )
        assert peak == 2

    async def test_queues_are_created_lazily_and_named_by_provider_and_model(self) -> None:
        async def ok(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            return {"status": 200, "headers": {}, "body": {}}

        engine = NetworkEngine(NetworkEngineConfig(fetch=ok))
        assert engine.queue_names() == []
        await engine.fetch(req())
        assert engine.queue_names() == ["openai/m"]
        assert engine.has_queue("openai/m")

    async def test_settings_are_immutable_once_a_queue_exists(self) -> None:
        # They are snapshotted at creation, so a later change would silently do
        # nothing -- which is worse than refusing it.
        async def ok(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            return {"status": 200, "headers": {}, "body": {}}

        engine = NetworkEngine(NetworkEngineConfig(fetch=ok))
        await engine.fetch(req())
        with pytest.raises(ValueError, match="already created"):
            engine.configure_queue("openai/m", {"limits": {"rpm": 1}})
        engine.drop_queue("openai/m")
        engine.configure_queue("openai/m", {"limits": {"rpm": 1}})

    async def test_the_snapshot_counts_what_the_queue_has_done(self) -> None:
        async def ok(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            return {"status": 200, "headers": {}, "body": {}}

        engine = NetworkEngine(NetworkEngineConfig(fetch=ok))
        await engine.fetch(req())
        await engine.fetch(req())
        snapshot = engine.snapshot()[0]
        # Lifetime, so an idle queue still shows evidence of past activity.
        assert snapshot["processed"] == 2
        assert snapshot["queueName"] == "openai/m"

    async def test_an_unretryable_failure_reaches_the_caller_as_its_class(self) -> None:
        async def denied(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            return {"status": 401, "headers": {}, "body": {"error": {"message": "nope"}}}

        engine = NetworkEngine(NetworkEngineConfig(fetch=denied))
        with pytest.raises(LLMError) as info:
            await engine.fetch(req())
        assert info.value.kind == "auth"
        assert info.value.status == 401


class TestEngineRetryPrecedence:
    """`examples/features-highlight/engine_retry_policy.py`, with its numbers."""

    def retries_for(self, **config: Any) -> int:
        engine = Engine(
            transport=lambda request: TransportResponse(status=500, body={"error": "boom"}),
            register_as_default=False,
            **config,
        )
        seen = 0

        @engine.on_retry
        def _count(ctx: dict[str, Any]) -> None:
            nonlocal seen
            seen += 1

        with pytest.raises(LLMError):
            engine.request(url="https://x/y", body={}, provider="openai", model="gpt-5.4-nano")
        return seen

    def test_engine_wide_applies_everywhere(self) -> None:
        assert self.retries_for(retry={"backoff": FAST, "per_kind": RETRYABLE}) == 3

    def test_a_queue_override_wins_over_the_engine(self) -> None:
        assert (
            self.retries_for(
                retry={"backoff": FAST, "per_kind": RETRYABLE},
                queues={
                    "openai/gpt-5.4-nano": {
                        "retry": {
                            "per_kind": {"server_error": {"retryable": True, "max_retries": 1}}
                        }
                    }
                },
            )
            == 1
        )

    def test_one_overridden_knob_keeps_the_rest_of_the_backoff(self) -> None:
        # Without the merge this would fall back to the built-in 500ms schedule
        # and take seconds instead of milliseconds.
        assert (
            self.retries_for(
                retry={
                    "backoff": FAST,
                    "per_kind": {"server_error": {"retryable": True, "max_retries": 2}},
                },
                queues={"openai/gpt-5.4-nano": {"retry": {"backoff": {"multiplier": 1}}}},
            )
            == 2
        )

    def test_a_per_request_override_is_the_narrowest_and_wins(self) -> None:
        engine = Engine(
            transport=lambda request: TransportResponse(status=500, body={"error": "boom"}),
            register_as_default=False,
            retry={"backoff": FAST, "per_kind": RETRYABLE},
        )
        seen = 0

        @engine.on_retry
        def _count(ctx: dict[str, Any]) -> None:
            nonlocal seen
            seen += 1

        with pytest.raises(LLMError):
            engine.request(
                url="https://x/health",
                body={},
                provider="openai",
                model="gpt-5.4-nano",
                retry=RetryOverride(max_retries=0),
            )
        assert seen == 0


class TestEngineHooks:
    def make(self) -> Engine:
        return Engine(
            transport=lambda request: TransportResponse(status=200, body={"ok": True}),
            register_as_default=False,
        )

    def test_the_decorator_returns_the_function_you_wrote(self) -> None:
        # The obvious design -- returning the unsubscribe function -- rebinds the
        # name and throws the handler away.
        engine = self.make()
        seen: list[Any] = []

        @engine.on_request_complete
        def track(ctx: dict[str, Any]) -> None:
            seen.append(ctx["status"])

        engine.request(url="https://x/y", provider="openai", model="m", body={})
        assert seen == [200]
        assert track({"status": 999}) is None  # still the plain function
        assert seen == [200, 999]

    def test_unsubscribe_rides_on_the_handler(self) -> None:
        engine = self.make()
        seen: list[Any] = []

        @engine.on_request_complete
        def track(ctx: dict[str, Any]) -> None:
            seen.append(ctx["status"])

        track.unsubscribe()  # type: ignore[attr-defined]
        engine.request(url="https://x/y", provider="openai", model="m", body={})
        assert seen == []

    def test_the_dynamic_form_takes_the_snake_case_name(self) -> None:
        # The escape hatch a plugin loader needs, in the same vocabulary as
        # everything else.
        engine = self.make()
        seen: list[Any] = []
        engine.on("on_request_complete", lambda ctx: seen.append(ctx["status"]))
        engine.request(url="https://x/y", provider="openai", model="m", body={})
        assert seen == [200]

    def test_on_any_receives_every_event_as_one_value(self) -> None:
        # `event.type` is the SNAKE name -- the same name the subscription uses
        # (`engine.on_request_start`), so an event's type can be pasted straight
        # back into a subscription rather than translated.
        engine = self.make()
        kinds: list[str] = []
        engine.on_any(lambda event: kinds.append(event.type))
        engine.request(url="https://x/y", provider="openai", model="m", body={})
        assert "on_request_start" in kinds
        assert "on_request_complete" in kinds

    def test_every_hook_in_the_catalog_has_a_decorator(self) -> None:
        """The boilerplate is explicit for autocomplete; this stops it drifting."""
        import re

        from combycode_llm_sdk.bus.hook_map import HOOK_NAMES

        missing = [
            name
            for name in HOOK_NAMES
            if not hasattr(Engine, re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower())
        ]
        assert missing == []


class TestDefaultRegistration:
    def test_an_engine_registers_itself_unless_told_not_to(self) -> None:
        engine = Engine(transport=lambda r: TransportResponse(status=200), register_as_default=True)
        assert default_engine() is engine

    def test_opting_out_leaves_the_default_alone(self) -> None:
        first = Engine(transport=lambda r: TransportResponse(status=200))
        Engine(transport=lambda r: TransportResponse(status=200), register_as_default=False)
        assert default_engine() is first


class TestRateLimitPausesTheQueue:
    async def test_a_429_pauses_the_whole_queue_not_just_the_request(self) -> None:
        # A 429 is about the KEY, so letting the next queued request straight
        # through would earn another one.
        calls = []

        async def limited(request: dict[str, Any], options: Any = None) -> dict[str, Any]:
            calls.append(1)
            if len(calls) == 1:
                return {
                    "status": 429,
                    "headers": {"retry-after-ms": "5"},
                    "body": {"error": "slow down"},
                }
            return {"status": 200, "headers": {}, "body": {}}

        engine = NetworkEngine(
            NetworkEngineConfig(
                fetch=limited,
                retry={"backoff": FAST, "per_kind": {"rate_limit": {"retryable": True, "max_retries": 2}}},
            )
        )
        hits: list[Any] = []
        engine.hooks.on("onRateLimitHit", lambda ctx: hits.append(ctx["retryAfterMs"]))
        response = await engine.fetch(req())
        assert response["status"] == 200
        assert hits == [5]

    def test_a_retry_after_beyond_the_cap_is_refused_not_waited_out(self) -> None:
        # Parking for a day looks identical to a hang from the caller's side.
        engine = Engine(
            transport=lambda request: TransportResponse(
                status=429, body={"error": "slow"}, headers={"retry-after": "86400"}
            ),
            register_as_default=False,
            retry={"backoff": FAST},
        )
        retries: list[Any] = []
        engine.on_retry(lambda ctx: retries.append(ctx))
        with pytest.raises(RateLimitError) as info:
            engine.request(url="https://x/y", provider="openai", model="m", body={})
        assert retries == []
        assert info.value.retry_after == 86400.0
