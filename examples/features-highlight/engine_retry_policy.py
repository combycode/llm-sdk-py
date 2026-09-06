"""Retry is configured ONCE, on the engine.

Retry is cross-cutting -- nobody wants to thread it through every `complete()`
call -- so it lives on the engine and every request inherits it. A provider
needing different treatment is a per-queue override; a request needing to be
less patient (a health check that should fail fast) is a per-request one.

    request.retry  >  queues[name].retry  >  Engine(retry=...)

Nested groups MERGE rather than replace, so overriding one `backoff` knob keeps
the rest -- otherwise a one-line tweak would silently reset the whole schedule.

No API key and no network: a stub transport always fails and we count retries.
"""

from _check import check, report

from combycode_llm_sdk import Engine, LLMError, RetryOverride, TransportResponse

FAST = {"initial_ms": 1, "max_ms": 2, "multiplier": 1, "jitter": 0}


def always_fails(request):
    return TransportResponse(status=500, body={"error": "boom"})


def asks_for_a_day(request):
    """A rate-limited server that asks us to wait a full day."""
    return TransportResponse(
        status=429, body={"error": "slow down"}, headers={"retry-after": "86400"}
    )


def retries_for(**config) -> int:
    """How many retries one request costs under a given engine configuration."""
    engine = Engine(transport=always_fails, register_as_default=False, **config)
    seen = 0

    @engine.on_retry
    def _count(ctx) -> None:
        nonlocal seen
        seen += 1

    try:
        engine.request(
            url="https://example.invalid/v1/x",
            body={"hello": "world"},
            provider="openai",
            model="gpt-5.4-nano",
        )
    except LLMError:
        pass  # expected -- the point is how many attempts it made on the way
    return seen


# 1. Engine-wide: every queue inherits it, nothing passed per call.
engine_wide = retries_for(
    retry={"backoff": FAST, "per_kind": {"server_error": {"retryable": True, "max_retries": 3}}},
)

# 2. One provider treated differently, keyed by queue name (`provider/model`).
per_queue = retries_for(
    retry={"backoff": FAST, "per_kind": {"server_error": {"retryable": True, "max_retries": 3}}},
    queues={
        "openai/gpt-5.4-nano": {
            "retry": {"per_kind": {"server_error": {"retryable": True, "max_retries": 1}}}
        }
    },
)

# 3. Only ONE backoff knob overridden: the engine's other backoff values survive,
#    so this stays fast instead of falling back to the built-in 500ms schedule.
merged = retries_for(
    retry={"backoff": FAST, "per_kind": {"server_error": {"retryable": True, "max_retries": 2}}},
    queues={"openai/gpt-5.4-nano": {"retry": {"backoff": {"multiplier": 1}}}},
)

# 4. One request less patient than the queue it shares. The narrowest layer, and
#    the reason it exists: a health check fails fast without re-tuning the engine
#    for everybody else.
#
#    Two clocks on purpose: `attempt_timeout_ms` bounds ONE attempt,
#    `total_timeout_ms` bounds the whole sequence including the waits between.
impatient = RetryOverride(
    max_retries=0,
    attempt_timeout_ms=2_000,
    total_timeout_ms=5_000,
    max_retry_after_ms=5_000,
    backoff={"initial_ms": 50, "max_ms": 200},
)

engine = Engine(
    transport=always_fails,
    register_as_default=False,
    retry={"backoff": FAST, "per_kind": {"server_error": {"retryable": True, "max_retries": 3}}},
)
per_request = 0


@engine.on_retry
def _count_request(ctx) -> None:
    global per_request
    per_request += 1


try:
    engine.request(
        url="https://example.invalid/health",
        body={},
        provider="openai",
        model="gpt-5.4-nano",
        retry=impatient,
    )
except LLMError:
    pass

check(engine_wide == 3, f"engine-wide retry should be 3, got {engine_wide}")
check(per_queue == 1, f"per-queue override should win, got {per_queue}")
check(merged == 2, f"merged backoff should keep max_retries=2, got {merged}")
check(per_request == 0, f"per-request max_retries=0 must win, got {per_request}")

report(engine_wide=engine_wide, per_queue=per_queue, merged=merged, per_request=per_request)
