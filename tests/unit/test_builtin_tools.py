"""The five prompt-backed tools the SDK ships.

They are data, so most of what can go wrong is a mismatch between the prompt and
the schema it asks the model to fill: an index numbered from one where the
schema says zero, a JSON Schema rendered as a Python repr, a length target that
never reaches the template. None of those raise -- they just make the model
answer the wrong question -- so they are pinned here.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.internal_tools.builtin import (
    BUILTIN_TOOLS,
    clarify_tool,
    classify_tool,
    register_builtin_tools,
    score_tool,
    structure_tool,
    summarize_tool,
)
from combycode_llm_sdk.internal_tools.types import InternalToolContext, InternalToolError
from combycode_llm_sdk.tokens import HeuristicCounter


class Client:
    """Answers with whatever it is told to, and keeps what it was asked."""

    def __init__(self, answer: str = "{}") -> None:
        self.answer = answer
        self.seen: list[Any] = []
        self.options: list[dict[str, Any]] = []

    def complete(self, input_: Any, **options: Any) -> Any:
        self.seen.append(input_)
        self.options.append(options)
        return type("R", (), {"text": self.answer})()


def _run(tool: Any, value: dict[str, Any], answer: str = "{}") -> tuple[Any, Client]:
    client = Client(answer)
    ctx = InternalToolContext(
        client=client,
        model_id="openai/gpt-5.4-nano",
        counter=HeuristicCounter(resolve_catalog("defaults")),
    )
    return tool.execute(value, ctx), client


def _prompt(client: Client) -> str:
    return json.dumps(client.seen[0], default=str)


class TestTheSetItself:
    def test_all_five_are_registered(self) -> None:
        assert [t.name for t in BUILTIN_TOOLS] == [
            "summarize",
            "classify",
            "score",
            "structure",
            "clarify",
        ]

    def test_every_id_carries_its_version(self) -> None:
        # The version is IN the id so two deployments can pin different wordings
        # of the same tool and say which one answered.
        for tool in BUILTIN_TOOLS:
            assert tool.id == f"orxa:{tool.name}@{tool.version}"

    def test_registering_puts_them_on_a_backend(self) -> None:
        registered: list[Any] = []
        register_builtin_tools(type("B", (), {"register": lambda _s, t: registered.append(t)})())
        assert len(registered) == 5

    def test_a_tool_run_without_a_client_says_who_is_at_fault(self) -> None:
        with pytest.raises(InternalToolError, match="needs an LLM client"):
            classify_tool.execute({"request": "x", "suggestions": ["a"]}, InternalToolContext())


class TestClassify:
    def test_the_options_are_numbered_from_zero(self) -> None:
        # The schema asks for a 0-based `selectedIndex`. Numbering the list from
        # one makes every answer off by one, and nothing in the response says so.
        _, client = _run(
            classify_tool,
            {"request": "reset my password", "suggestions": ["billing", "account", "shipping"]},
            '{"selectedIndex": 1, "confidence": 0.9, "reasoning": "account"}',
        )
        text = _prompt(client)
        assert "0. billing" in text
        assert "1. account" in text

    def test_it_parses_the_answer(self) -> None:
        out, _ = _run(
            classify_tool,
            {"request": "x", "suggestions": ["a", "b"]},
            '{"selectedIndex": 1, "confidence": 0.5, "reasoning": "because"}',
        )
        assert out == {"selectedIndex": 1, "confidence": 0.5, "reasoning": "because"}


class TestStructure:
    def test_the_schema_reaches_the_model_indented(self) -> None:
        # Mutation-driven, and it corrected a wrong belief: the template
        # renderer already json-encodes a non-string, so a Python repr never
        # reaches the model and asserting against one passes no matter what
        # `prepare_input` does. What it actually contributes is the indentation
        # -- and a nested schema flattened onto one line is where the "no extra
        # fields, no missing required fields" instruction starts getting ignored.
        _, client = _run(
            structure_tool,
            {
                "request": "Ada, 36",
                "schema": {"type": "object", "properties": {"age": {"type": "integer"}}},
            },
            '{"age": 36}',
        )
        rendered = client.seen[0]
        assert isinstance(rendered, str)
        schema_block = rendered.split("Target JSON Schema:", 1)[1]
        assert "\n" in schema_block.split("Return a valid")[0].strip()
        assert '"type": "object"' in rendered


class TestSummarize:
    def test_the_length_target_reaches_the_prompt(self) -> None:
        _, client = _run(
            summarize_tool,
            {"content": "long " * 50, "maxLength": 120},
            '{"summary": "s", "keyPoints": []}',
        )
        assert "under 120 characters" in _prompt(client)

    def test_tokens_take_precedence_over_characters(self) -> None:
        _, client = _run(
            summarize_tool,
            {"content": "long " * 50, "maxLength": 120, "maxTokens": 64},
            '{"summary": "s", "keyPoints": []}',
        )
        # maxLength still describes the SUMMARY's size to the model; maxTokens
        # is the response ceiling, and it is what the request carries.
        assert client.options[0].get("max_tokens") == 64

    def test_no_limit_still_asks_for_something_short(self) -> None:
        _, client = _run(
            summarize_tool, {"content": "long " * 50}, '{"summary": "s", "keyPoints": []}'
        )
        assert "concise" in _prompt(client)

    def test_a_focus_is_stated_and_its_absence_leaves_no_stray_line(self) -> None:
        _, with_focus = _run(
            summarize_tool,
            {"content": "text", "focus": "batteries"},
            '{"summary": "s", "keyPoints": []}',
        )
        assert "Focus: batteries" in _prompt(with_focus)
        _, without = _run(summarize_tool, {"content": "text"}, '{"summary": "s", "keyPoints": []}')
        assert "Focus:" not in _prompt(without)

    def test_the_budget_is_derived_from_the_requested_length(self) -> None:
        # A fixed ceiling truncates the long cases and overpays for the short
        # ones; this tool is called with limits spanning orders of magnitude.
        _, small = _run(
            summarize_tool, {"content": "t", "maxLength": 100}, '{"summary": "s", "keyPoints": []}'
        )
        _, large = _run(
            summarize_tool, {"content": "t", "maxLength": 8000}, '{"summary": "s", "keyPoints": []}'
        )
        assert large.options[0]["max_tokens"] > small.options[0]["max_tokens"] * 3

    def test_it_needs_a_counter_to_size_the_answer(self) -> None:
        # Refused rather than guessed: a silent default here would quietly
        # truncate every long summary.
        with pytest.raises(InternalToolError, match="token counter"):
            summarize_tool.execute(
                {"content": "t", "maxLength": 100},
                InternalToolContext(client=Client(), model_id="openai/gpt-5.4-nano"),
            )

    def test_it_has_a_strict_and_a_balanced_prompt(self) -> None:
        # Strict fact-preservation helps the models that over-paraphrase and
        # hurts the ones that then fail to compress at all.
        definition = summarize_tool.definition
        assert definition is not None
        assert {v.id for v in definition.variants} == {"strict", "balanced"}
        assert [v.id for v in definition.variants if v.is_default] == ["balanced"]

    def test_anthropic_and_google_get_the_strict_one(self) -> None:
        definition = summarize_tool.definition
        assert definition is not None
        strict = next(v for v in definition.variants if v.id == "strict")
        assert set(strict.supported_providers) == {"anthropic", "google"}


class TestScoreAndClarify:
    def test_score_lists_its_criteria(self) -> None:
        _, client = _run(
            score_tool,
            {"task": "t", "answer": "a", "criteria": ["accuracy", "clarity"]},
            '{"score": 90, "breakdown": {}, "feedback": "f"}',
        )
        text = _prompt(client)
        assert "- accuracy" in text and "- clarity" in text

    def test_score_falls_back_to_the_documented_criteria(self) -> None:
        # The schema's `default` fills them in, so the prompt is never empty.
        _, client = _run(
            score_tool, {"task": "t", "answer": "a"}, '{"score": 1, "breakdown": {}, "feedback": ""}'
        )
        assert "- accuracy" in _prompt(client)

    def test_score_gets_a_generous_ceiling(self) -> None:
        # A reasoning model spends output budget on thought before writing the
        # breakdown, and a truncated answer reads as a low score, not a failure.
        definition = score_tool.definition
        assert definition is not None
        assert (definition.model_preference.max_tokens or 0) >= 2000

    def test_clarify_lists_its_requirements(self) -> None:
        _, client = _run(
            clarify_tool,
            {"prompt": "p", "requirements": ["a budget", "a deadline"]},
            '{"satisfied": false, "missingRequirements": [], "clarificationQuestions": []}',
        )
        text = _prompt(client)
        assert "- a budget" in text and "- a deadline" in text
