"""The error taxonomy and the retry policy -- the two pure halves of the engine.

Both are shared by the synchronous and asynchronous engines, so a bug here is a
bug in both, and neither a differential nor a live run would tell you which.
They are pure functions, which is why they get expectation tests.

The precedence cases mirror `examples/features-highlight/engine_retry_policy.py`
-- the reviewed contract for how the three layers combine.
"""

from __future__ import annotations

import time

import pytest

from combycode_llm_sdk.network.errors import (
    AuthError,
    ContextOverflowError,
    InvalidRequestError,
    LLMError,
    ModelNotFoundError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    UnsupportedError,
    classify_error,
    extract_error_message,
    parse_retry_after,
)
from combycode_llm_sdk.network.retry import (
    DEFAULT_RETRY,
    BackoffConfig,
    ErrorRetryConfig,
    RetryOverride,
    calculate_backoff,
    honored_retry_after_ms,
    merge_retry,
    should_retry,
)


class TestClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, AuthError),
            (403, AuthError),
            (429, RateLimitError),
            (402, QuotaExceededError),
            (413, QuotaExceededError),
            (500, ServerError),
            (503, ServerError),
        ],
    )
    def test_status_decides_the_kind(self, status: int, expected: type[LLMError]) -> None:
        err = classify_error("openai", status, {"error": {"message": "x"}})
        assert isinstance(err, expected)

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("This model's maximum context length is 8192 tokens", ContextOverflowError),
            ("The model `gpt-9` does not exist", ModelNotFoundError),
            ("This endpoint does not support streaming", UnsupportedError),
            ("Invalid value for 'temperature'", InvalidRequestError),
        ],
    )
    def test_a_400_is_told_apart_by_its_message(
        self, message: str, expected: type[LLMError]
    ) -> None:
        # Every one of these is a 400, so the message is the only signal there is.
        err = classify_error("openai", 400, {"error": {"message": message}})
        assert isinstance(err, expected)

    def test_only_server_errors_and_rate_limits_are_retryable(self) -> None:
        assert classify_error("openai", 500, {}).retryable is True
        assert classify_error("openai", 429, {}).retryable is True
        assert classify_error("openai", 401, {}).retryable is False
        assert classify_error("openai", 400, {}).retryable is False

    def test_provider_and_model_ride_along_for_the_caller(self) -> None:
        # The contract: `except LLMError as e: e.provider, e.model, e.status`.
        err = classify_error("anthropic", 500, {}, model="claude-haiku-4.5")
        assert (err.provider, err.model, err.status) == ("anthropic", "claude-haiku-4.5", 500)

    def test_the_kind_string_survives_alongside_the_class(self) -> None:
        # Both readings work; neither is a second source of truth.
        assert classify_error("openai", 429, {}).kind == "rate_limit"


class TestErrorMessage:
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"error": {"message": "nested"}}, "nested"),
            ({"error": "flat"}, "flat"),
            ({"message": "top level"}, "top level"),
            (None, "Unknown error"),
            ("a bare string", "a bare string"),
        ],
    )
    def test_the_message_is_found_wherever_the_provider_put_it(
        self, body: object, expected: str
    ) -> None:
        assert extract_error_message(body) == expected


class TestRetryAfter:
    def test_delay_seconds(self) -> None:
        assert parse_retry_after({"retry-after": "30"}) == 30_000

    def test_the_millisecond_header_wins_when_present(self) -> None:
        assert parse_retry_after({"retry-after-ms": "1500", "retry-after": "30"}) == 1500

    def test_an_http_date_is_parsed_too(self) -> None:
        # The date form used to return nothing, which was safe but silently
        # ignored the server's instruction.
        future = time.time() + 120
        stamp = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(future))
        parsed = parse_retry_after({"retry-after": stamp})
        assert parsed is not None
        assert 100_000 < parsed < 130_000

    @pytest.mark.parametrize("value", ["not-a-date", "", "NaN"])
    def test_an_unusable_value_is_discarded_not_propagated(self, value: str) -> None:
        # A sleep of NaN returns immediately, turning one bad header into a
        # retry storm.
        assert parse_retry_after({"retry-after": value}) is None

    def test_a_past_date_does_not_become_a_negative_wait(self) -> None:
        past = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() - 3600))
        assert parse_retry_after({"retry-after": past}) is None

    def test_retry_after_is_exposed_to_callers_in_seconds(self) -> None:
        # `time.sleep(err.retry_after)` should be right.
        err = classify_error("openai", 429, {}, {"retry-after": "30"})
        assert err.retry_after_ms == 30_000
        assert err.retry_after == 30.0


class TestPolicyMerging:
    def test_an_empty_override_is_the_default(self) -> None:
        assert merge_retry(None) == DEFAULT_RETRY

    def test_one_backoff_knob_keeps_the_rest(self) -> None:
        # The whole reason the override type exists: a one-line tweak must not
        # silently reset the schedule.
        merged = merge_retry({"backoff": {"multiplier": 1}})
        assert merged.backoff == BackoffConfig(
            initial_ms=DEFAULT_RETRY.backoff.initial_ms,
            max_ms=DEFAULT_RETRY.backoff.max_ms,
            multiplier=1,
            jitter=DEFAULT_RETRY.backoff.jitter,
        )

    def test_per_kind_merges_key_by_key(self) -> None:
        merged = merge_retry({"per_kind": {"server_error": {"max_retries": 9}}})
        assert merged.per_kind["server_error"].max_retries == 9
        # Every other kind survives untouched.
        assert merged.per_kind["auth"].retryable is False

    def test_layers_apply_widest_first(self) -> None:
        # Engine-wide, then the queue's, then the request's.
        merged = merge_retry(
            {"max_retries": 5, "backoff": {"initial_ms": 10}},
            {"max_retries": 3},
            RetryOverride(max_retries=0),
        )
        assert merged.max_retries == 0
        assert merged.backoff.initial_ms == 10

    def test_a_dataclass_override_and_a_dict_override_are_equivalent(self) -> None:
        assert merge_retry(RetryOverride(max_retries=7)) == merge_retry({"max_retries": 7})


class TestShouldRetry:
    def rate_limited(self, retry_after_ms: float | None = None) -> RateLimitError:
        return RateLimitError("slow down", retryable=True, retry_after_ms=retry_after_ms)

    def test_a_retryable_kind_within_budget_retries(self) -> None:
        assert should_retry(self.rate_limited(), 0, DEFAULT_RETRY, elapsed_ms=0) is True

    def test_an_unretryable_kind_never_retries(self) -> None:
        assert should_retry(AuthError("nope"), 0, DEFAULT_RETRY, elapsed_ms=0) is False

    def test_attempts_run_out(self) -> None:
        # rate_limit gets 5 by default.
        assert should_retry(self.rate_limited(), 5, DEFAULT_RETRY, elapsed_ms=0) is False

    def test_the_total_budget_stops_it_even_with_attempts_left(self) -> None:
        # A per-attempt cap alone lets a sequence run for minutes.
        assert (
            should_retry(self.rate_limited(), 0, DEFAULT_RETRY, elapsed_ms=999_999) is False
        )

    def test_a_body_that_cannot_be_replayed_is_not_retried(self) -> None:
        # A streamed body is consumed by the first attempt; the second would
        # send an empty one.
        assert (
            should_retry(
                self.rate_limited(), 0, DEFAULT_RETRY, elapsed_ms=0, replayable_body=False
            )
            is False
        )

    def test_a_retry_after_longer_than_we_will_wait_is_a_refusal(self) -> None:
        # Parking for a day looks identical to a hang from the caller's side.
        a_day = 86_400_000
        assert should_retry(self.rate_limited(a_day), 0, DEFAULT_RETRY, elapsed_ms=0) is False

    def test_a_per_request_max_beats_the_per_kind_rule(self) -> None:
        # The example's case 4: a health check fails fast without re-tuning the
        # engine for everybody else.
        assert (
            should_retry(
                self.rate_limited(), 0, DEFAULT_RETRY, elapsed_ms=0, request_max_retries=0
            )
            is False
        )


class TestBackoff:
    def test_the_servers_instruction_wins(self) -> None:
        err = RateLimitError("x", retryable=True, retry_after_ms=4_000)
        assert calculate_backoff(0, err, DEFAULT_RETRY) == 4_000

    def test_it_is_clamped_to_what_we_are_willing_to_wait(self) -> None:
        err = RateLimitError("x", retryable=True, retry_after_ms=86_400_000)
        assert honored_retry_after_ms(err, DEFAULT_RETRY) == DEFAULT_RETRY.max_retry_after_ms

    def test_a_fixed_per_kind_delay_beats_the_curve(self) -> None:
        retry = merge_retry({"per_kind": {"server_error": {"fixed_backoff_ms": 42}}})
        assert calculate_backoff(3, ServerError("x", retryable=True), retry) == 42

    def test_the_curve_is_exponential_and_capped(self) -> None:
        # Jitter pinned at the midpoint, so this asserts the curve rather than a
        # range.
        retry = merge_retry({"backoff": {"initial_ms": 100, "max_ms": 400, "multiplier": 2}})
        err = ServerError("x", retryable=True)
        mid = (lambda: 0.5)
        assert [calculate_backoff(n, err, retry, rand=mid) for n in range(4)] == [
            100,
            200,
            400,
            400,
        ]

    def test_jitter_spreads_the_delay_both_ways(self) -> None:
        retry = merge_retry({"backoff": {"initial_ms": 1000, "multiplier": 1, "jitter": 0.25}})
        err = ServerError("x", retryable=True)
        assert calculate_backoff(0, err, retry, rand=lambda: 0.0) == 1250
        assert calculate_backoff(0, err, retry, rand=lambda: 1.0) == 750

    def test_no_jitter_is_exact(self) -> None:
        retry = merge_retry({"backoff": {"initial_ms": 1, "max_ms": 2, "multiplier": 1, "jitter": 0}})
        assert calculate_backoff(0, ServerError("x", retryable=True), retry) == 1


def test_the_defaults_say_something_about_every_kind() -> None:
    """A kind with no rule falls through to the error's own `retryable`, which is
    a weaker statement than the taxonomy can make. Every kind should be named."""
    from combycode_llm_sdk.network.errors import ERROR_KINDS

    missing = [k for k in ERROR_KINDS if k not in DEFAULT_RETRY.per_kind]
    assert missing == []
    assert isinstance(DEFAULT_RETRY.per_kind["rate_limit"], ErrorRetryConfig)
