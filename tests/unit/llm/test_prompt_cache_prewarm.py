"""A response with no output is an empty answer, not a broken one.

Transposed from `unified-library-ts/tests/unit/llm/prompt-cache-prewarm.test.ts`.

``prompt_cache_options.prewarm: True`` asks OpenAI to write the prompt cache and
generate nothing -- it overrides ``generate`` to false. What comes back is
``status: "completed"`` with an **empty ``output[]``**, a shape most of the
parse path never sees: no message item, no text, no tool call.

That is the shape a finish-reason extractor gets wrong. Reporting it as a
failure, or as ``length``, would make a successful cache warm look like a broken
request, and the caller could not tell it from a real empty completion.

Measured 2026-09-30 on ``gpt-5.6-terra``, which is also where the feature itself
was confirmed: the prewarm call returned 0 output items and 0 cached tokens, and
the next call on the same 4177-token prompt read 4174 from cache. The prompt
carried a per-run nonce, so that hit can only have come from the prewarm.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

#: The body OpenAI returns for a prewarm: completed, and empty.
PREWARMED: dict[str, Any] = {
    "id": "resp_prewarm",
    "object": "response",
    "status": "completed",
    "model": "gpt-5.6-terra-2026-09-01",
    "output": [],
    "usage": {
        "input_tokens": 4177,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4174},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 4177,
    },
}


def _adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter({"apiKey": "k"})


class TestAPrewarmResponse:
    def test_it_parses_as_a_finished_empty_result(self) -> None:
        r = _adapter().parse_response(PREWARMED, 0)
        # `stop`, not `error` and not `length`: the request did exactly what
        # it was asked to do.
        assert r["finishReason"] == "stop"
        assert r["text"] == ""
        assert r["content"] == []
        assert r["toolCalls"] == []

    def test_it_reports_no_failure(self) -> None:
        # The distinguishing check. An empty `output[]` must not be mistaken
        # for a provider error, or a successful cache warm reads as broken.
        assert _adapter().parse_response(PREWARMED, 0).get("error") is None

    def test_it_still_accounts_the_tokens_it_was_billed_for(self) -> None:
        # A prewarm is not free: it pays for the input it wrote to the cache.
        usage = _adapter().parse_response(PREWARMED, 0)["usage"]
        assert usage["inputTokens"] == 4177
        assert usage["outputTokens"] == 0
        assert usage["cacheWriteTokens"] == 4174


class TestTheRequestSide:
    def body(self, provider_options: dict[str, Any]) -> dict[str, Any]:
        built = _adapter().build_request(
            {
                "model": "gpt-5.6-terra",
                "messages": [{"role": "user", "content": "long prompt"}],
                "providerOptions": provider_options,
            }
        )
        return dict(built.body)

    def test_it_forwards_prewarm_to_prompt_cache_options(self) -> None:
        got = self.body({"promptCacheOptions": {"prewarm": True, "ttl": "30m"}})
        assert got["prompt_cache_options"] == {"prewarm": True, "ttl": "30m"}

    def test_it_sends_nothing_when_the_caller_asked_for_nothing(self) -> None:
        # `prompt_cache_options` is refused outright on a pre-5.6 model, so an
        # empty object must not ride along on every request.
        assert "prompt_cache_options" not in self.body({})

    def test_it_keeps_a_field_the_typed_shape_does_not_know(self) -> None:
        got = self.body({"promptCacheOptions": {"prewarm": True, "some_future_knob": 7}})
        assert got["prompt_cache_options"] == {"prewarm": True, "some_future_knob": 7}
