"""A reasoning effort the catalog advertises has to reach the wire.

Transposed from `unified-library-ts/tests/unit/llm/xai-reasoning-effort.test.ts`.

Three faults behind one field:

1. The xAI overlay deleted `reasoning` for every model whose id did not contain
   `multi-agent`, while the catalog advertised `effortControl: True` with `xhigh`
   for grok-4.5/4.6 -- the catalog promising a control the request never carried.
2. On chat-completions the spec built `reasoning: {effort}`, the RESPONSES shape.
   OpenAI answers `400 Unknown parameter: 'reasoning'` to it, so asking for
   thinking on that surface failed every single time. xAI hid the identical bug
   behind the overlay above, which is why only one of the two ever errored.
3. `effort: "max"` -- a value in the library's own type and docs -- was passed
   through raw, and neither provider has it: both answer 400, OpenAI naming the
   value.

Measured 2026-09-30 on `/v1/responses`, reasoning tokens on a hard prompt (a
trivial one cannot separate the efforts, which is how "accepted and inert"
hides): grok-4.6 x6.8, grok-4.5 x36.6, grok-4.3 x7.4, ranges disjoint; and on
chat-completions grok-4.6 x10. The whole grok-4.20 line answers 400 "does not
support parameter reasoningEffort", which is why this is a table and not a
version comparison: 4.20 refuses the field while the numerically LOWER 4.3
honours it.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.openai.completions import OpenAIAdapter
from combycode_llm_sdk.llm.providers.openrouter.completions import OpenRouterAdapter
from combycode_llm_sdk.llm.providers.xai.completions import XAIAdapter
from combycode_llm_sdk.llm.providers.xai.reasoning import (
    xai_takes_reasoning_effort,
    xai_uses_effort_as_agent_count,
)
from combycode_llm_sdk.llm.providers.xai.responses import XAIResponsesAdapter


def body(adapter: Any, model: str, effort: str | None = "xhigh") -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "maxTokens": 4096,
    }
    if effort is not None:
        req["thinking"] = {"mode": "on", "effort": effort}
    return dict(adapter.build_request(req).body)


class TestTheCapabilityTable:
    def test_yes_for_the_models_measured_to_honour_it(self) -> None:
        for model in ("grok-4.3", "grok-4.5", "grok-4.6", "grok-4.7"):
            assert xai_takes_reasoning_effort(model) is True

    def test_no_for_the_4_20_line_which_refuses_the_parameter_by_name(self) -> None:
        # A 400 is not a failure mode worth risking on a guess, and this is the
        # case that rules out any rule shaped like a version comparison.
        for model in ("grok-4.20", "grok-4.20-non-reasoning", "grok-4.20-0309-reasoning"):
            assert xai_takes_reasoning_effort(model) is False

    def test_no_for_multi_agent_where_the_field_means_an_agent_count(self) -> None:
        assert xai_takes_reasoning_effort("grok-4.20-multi-agent") is False
        assert xai_uses_effort_as_agent_count("grok-4.20-multi-agent") is True

    def test_defaults_to_no_for_a_model_released_after_this_build(self) -> None:
        # The conservative direction: an unsent field costs the caller the control
        # they asked for, a rejected one costs them the whole request.
        assert xai_takes_reasoning_effort("grok-5") is False
        assert xai_takes_reasoning_effort("grok-build-0.1") is False

    def test_the_provider_prefix_does_not_change_the_answer(self) -> None:
        assert xai_takes_reasoning_effort("xai/grok-4.6") is True
        assert xai_takes_reasoning_effort("XAI/GROK-4.20") is False


class TestTheResponsesSurface:
    adapter = XAIResponsesAdapter({"apiKey": "k"})

    def test_it_carries_the_effort_for_a_model_that_honours_it(self) -> None:
        assert body(self.adapter, "grok-4.6")["reasoning"]["effort"] == "xhigh"

    def test_it_still_drops_it_for_the_4_20_line(self) -> None:
        assert "reasoning" not in body(self.adapter, "grok-4.20")

    def test_it_carries_it_for_multi_agent_which_wants_it_for_its_own_reason(self) -> None:
        assert body(self.adapter, "grok-4.20-multi-agent")["reasoning"]["effort"] == "xhigh"

    def test_nothing_when_the_caller_asked_for_no_thinking(self) -> None:
        assert "reasoning" not in body(self.adapter, "grok-4.6", None)


class TestChatCompletionsSendsReasoningEffortNotAnObject:
    openai = OpenAIAdapter({"apiKey": "k"})
    xai = XAIAdapter({"apiKey": "k"})
    openrouter = OpenRouterAdapter({"apiKey": "k"})

    def test_openai_sends_the_string_and_no_object(self) -> None:
        b = body(self.openai, "gpt-5.4-nano", "high")
        assert b["reasoning_effort"] == "high"
        assert "reasoning" not in b

    def test_xai_the_same_for_a_model_that_takes_it(self) -> None:
        assert body(self.xai, "grok-4.6", "high")["reasoning_effort"] == "high"

    def test_xai_nothing_for_the_4_20_line_on_this_surface_either(self) -> None:
        b = body(self.xai, "grok-4.20", "high")
        assert "reasoning_effort" not in b
        assert "reasoning" not in b

    def test_openrouter_keeps_the_object_which_is_the_shape_it_documents(self) -> None:
        # Probed 2026-09-30: OpenRouter accepts both forms and honours both, but
        # the object is the only one that can also carry its own `max_tokens` and
        # `exclude`. The 400 is an OpenAI fact, not an OpenRouter one.
        b = body(self.openrouter, "anthropic/claude-fable-5", "high")
        assert b["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in b


class TestMaxMapsToTheTopRung:
    def test_on_every_one_of_them(self) -> None:
        # `max` is not a value either provider accepts -- both answered 400, and
        # OpenAI named it. Every other provider already mapped effort through a
        # table; these two did not.
        assert body(XAIResponsesAdapter({"apiKey": "k"}), "grok-4.6", "max")["reasoning"][
            "effort"
        ] == "xhigh"
        assert body(OpenAIAdapter({"apiKey": "k"}), "gpt-5.4-nano", "max")["reasoning_effort"] == (
            "xhigh"
        )
        assert body(OpenRouterAdapter({"apiKey": "k"}), "anthropic/claude-fable-5", "max")[
            "reasoning"
        ] == {"effort": "xhigh"}
