"""Two bugs that 828 passing tests did not see, and a live call found at once.

Both are recorded here because both were invisible to every existing test for
the same structural reason: the corpus differentials build a request dict
themselves and replay a recorded response, so nothing between `LLM.complete(...)`
and the wire was ever exercised. The first real call to Anthropic failed on both
within one request.

They are the two failure shapes this port exists to prevent -- a request that is
wrong in a way no fixture shows, and an error that reaches the caller as a clean
empty answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk import LLM
from combycode_llm_sdk.network.errors import AuthError, InvalidRequestError, LLMError
from combycode_llm_sdk.transport import TransportRequest, TransportResponse

OK_BODY = {
    "id": "msg_1",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


class Recorder:
    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self.body = body if body is not None else OK_BODY
        self.seen: list[TransportRequest] = []

    def __call__(self, request: TransportRequest) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(status=self.status, body=self.body)


def llm(transport: Any, **over: Any) -> LLM:
    return LLM(
        model="anthropic/claude-haiku-4.5", api_key="k", transport=transport, **over
    )


class TestAnUnsetOptionIsAbsentNotNull:
    """`temperature: Input should be a valid number` -- a 400 for a parameter
    the caller never set.

    TypeScript writes `maxTokens: options.maxTokens`; an unset one is
    `undefined`, which the spec treats as missing. Filling every unset option
    with Python's `None` put `"temperature": null` on the wire, and Anthropic
    rejects it.
    """

    def test_unset_options_do_not_reach_the_wire(self) -> None:
        recorder = Recorder()
        llm(recorder).complete("hi", max_tokens=32)
        body = recorder.seen[0].body
        assert body["max_tokens"] == 32
        for never_set in ("temperature", "top_p", "top_k", "stop_sequences", "tools"):
            assert never_set not in body, f"{never_set} was sent unset"

    def test_no_key_in_the_body_has_a_null_value(self) -> None:
        # The general form of the same bug: any `null` in a provider body is a
        # parameter we could not have meant to send.
        recorder = Recorder()
        llm(recorder).complete("hi", max_tokens=32)
        nulls = [k for k, v in recorder.seen[0].body.items() if v is None]
        assert nulls == []

    def test_an_option_set_to_zero_is_still_sent(self) -> None:
        # The fix must not become "drop anything falsy": 0 and "" are real
        # values a caller can mean, and temperature=0 is the common one.
        recorder = Recorder()
        llm(recorder).complete("hi", max_tokens=32, temperature=0)
        assert recorder.seen[0].body["temperature"] == 0

    def test_the_streamed_path_is_clean_too(self) -> None:
        # `stream()` normalizes through the same function, so it had the bug and
        # needs the same guard.
        recorder = Recorder(body=iter([b'data: {"type":"message_stop"}\n\n']))
        list(llm(recorder).stream("hi", max_tokens=8))
        assert [k for k, v in recorder.seen[0].body.items() if v is None] == []


class TestAnErrorIsRaisedNotSwallowed:
    """A 400 came back as `text=""`, `finish_reason="stop"`, and no exception.

    A bare `LLM` handed its transport straight to the client, so nothing checked
    the status: the caller saw a model with nothing to say rather than a request
    that was rejected. The same class of bug as a cost of $0.00 on 72k tokens.
    """

    @pytest.mark.parametrize(
        ("status", "expected"),
        [(400, InvalidRequestError), (401, AuthError), (500, LLMError)],
    )
    def test_a_failure_status_raises_its_own_class(
        self, status: int, expected: type[LLMError]
    ) -> None:
        recorder = Recorder(status=status, body={"error": {"message": "nope"}})
        with pytest.raises(expected):
            llm(recorder).complete("hi", max_tokens=8)

    def test_the_provider_message_survives_to_the_caller(self) -> None:
        recorder = Recorder(
            status=400, body={"error": {"message": "temperature: Input should be a valid number"}}
        )
        with pytest.raises(LLMError, match="valid number"):
            llm(recorder).complete("hi", max_tokens=8)

    def test_it_never_returns_an_empty_completion_instead(self) -> None:
        # The exact shape of the bug: a completion that looks finished.
        recorder = Recorder(status=400, body={"error": {"message": "bad"}})
        try:
            result = llm(recorder).complete("hi", max_tokens=8)
        except LLMError:
            return
        pytest.fail(
            f"a 400 produced a completion instead of raising: "
            f"text={result.text!r} finish_reason={result.finish_reason!r}"
        )

    def test_a_retryable_failure_is_retried_before_it_raises(self) -> None:
        # The executor is on this path for retry as well as classification, so
        # a bare LLM is not less robust than one built with an engine.
        class Flaky(Recorder):
            def __call__(self, request: TransportRequest) -> TransportResponse:
                self.seen.append(request)
                if len(self.seen) < 3:
                    return TransportResponse(status=500, body={"error": "boom"})
                return TransportResponse(status=200, body=OK_BODY)

        flaky = Flaky()
        assert llm(flaky).complete("hi", max_tokens=8).text == "OK"
        assert len(flaky.seen) == 3
