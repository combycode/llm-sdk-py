"""Checking against the schema, and telling the MODEL what was wrong.

Transposed from `unified-library-ts/tests/unit/agent/schema-validation.test.ts`.

Two halves, both opt-in, and the reason for opt-in is the same in each: the
bundled validator reads the common JSON Schema keywords and not all of Draft
2020-12 (no `allOf`/`anyOf`, no formats). Always on, it would reject values that
are valid under a schema it cannot fully read -- and disagree with the provider
that had just enforced that schema.

**Structured output** (`structured["validate"]`). For where the provider's
enforcement is weaker than the schema: a surface with no strict mode, a model
ignoring the schema under load, a `required` treated as advisory. Failures go
through the SAME repair budget as a parse failure, because a value that parsed
and was wrong is exactly what re-prompting helps with.

**Tool arguments** (`validate_tool_arguments`). A failure is a tool RESULT
carrying the errors, not an exception: the model asked for something its own
schema forbids, which it can fix on the next step, and ending the run would
discard every step before it. The bound is `max_steps`.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.helpers.tool import Tool
from combycode_llm_sdk.llm.client_internal import parse_structured
from combycode_llm_sdk.llm.output_errors import InvalidFinalOutputError
from combycode_llm_sdk.results import Completion, Part, Usage

WEATHER: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "tempC": {"type": "number"}},
    "required": ["city", "tempC"],
}


class TestStructuredValidate:
    def test_it_is_off_by_default(self) -> None:
        # `tempC` is required and missing. Nothing complains, because the
        # provider was asked to enforce it and a second opinion from a partial
        # validator would only find disagreements.
        assert parse_structured('{"city":"Paris"}', WEATHER) == {"city": "Paris"}

    def test_it_reports_every_error_when_asked(self) -> None:
        # The message is what the model is re-prompted with, so one error at a
        # time costs a whole round trip per mistake, and the path tells it WHERE.
        with pytest.raises(InvalidFinalOutputError) as caught:
            parse_structured('{"tempC":"warm"}', WEATHER, validate=True)
        message = str(caught.value.__cause__)
        assert "$.city" in message
        assert "$.tempC" in message

    def test_it_raises_the_same_error_as_a_parse_failure(self) -> None:
        # A value that parsed and was wrong is the case re-prompting actually
        # helps with; a separate error type would have excluded it from the
        # budget.
        with pytest.raises(InvalidFinalOutputError) as caught:
            parse_structured('{"city":"Paris"}', WEATHER, validate=True)
        assert caught.value.raw_text == '{"city":"Paris"}'

    def test_it_passes_a_value_the_schema_accepts(self) -> None:
        assert parse_structured('{"city":"Paris","tempC":21}', WEATHER, validate=True) == {
            "city": "Paris",
            "tempC": 21,
        }

    def test_malformed_json_is_still_a_parse_failure(self) -> None:
        with pytest.raises(InvalidFinalOutputError):
            parse_structured("not json", WEATHER, validate=True)

    def test_no_schema_means_nothing_to_validate_against(self) -> None:
        # `validate=True` with no schema must not invent one, or every caller who
        # set the flag without a schema would get a crash instead of a parse.
        assert parse_structured('{"anything":1}', None, validate=True) == {"anything": 1}


def answer(text: str) -> Completion:
    return Completion(
        text=text,
        model="m",
        finish_reason="stop",
        usage=Usage(),
        parts=[Part(type="text", text=text)],
    )


def tool_call(name: str, arguments: dict[str, Any]) -> Completion:
    part = Part(
        type="tool_call", id="c1", name=name, arguments=arguments, raw={"type": "tool_call"}
    )
    return Completion(
        text="",
        model="m",
        finish_reason="tool_use",
        usage=Usage(),
        parts=[part],
        tool_calls=[part],
    )


class Client:
    """Asks for a tool once with the given arguments, then answers."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self._turns = [tool_call(name, arguments), answer("done")]
        self.seen = 0

    def complete(self, _messages: Any, **_options: Any) -> Completion:
        self.seen += 1
        return self._turns[min(self.seen - 1, len(self._turns) - 1)]

    def destroy(self) -> None:
        pass


def loop_for(arguments: dict[str, Any], **options: Any) -> tuple[AgentLoop, list[Any], list[Any]]:
    ran: list[Any] = []
    warnings: list[dict[str, Any]] = []
    hooks = HookBus()
    hooks.on("onWarning", lambda w: warnings.append(dict(w)))

    def body(**kw: Any) -> str:
        ran.append(kw)
        return "sunny"

    lookup = Tool(
        body,
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )
    loop = AgentLoop(
        Client("lookup", arguments), tools=[lookup], hooks=hooks, system="s", **options
    )
    return loop, ran, warnings


def tool_messages(loop: AgentLoop) -> str:
    return json.dumps(
        [e.message for e in loop.history.all() if e.message.get("role") == "tool"], default=str
    )


class TestValidateToolArguments:
    def test_it_is_off_by_default(self) -> None:
        loop, ran, _ = loop_for({"city": 42})
        loop.complete("go")
        assert ran == [{"city": 42}]

    def test_it_refuses_the_call_and_returns_the_errors(self) -> None:
        loop, ran, _ = loop_for({"city": 42}, validate_tool_arguments=True)
        result = loop.complete("go")

        assert ran == []  # not executed
        assert result.text == "done"  # and the run did NOT end
        assert "Invalid arguments" in tool_messages(loop)
        assert "city" in tool_messages(loop)

    def test_it_tells_the_model_what_to_do(self) -> None:
        # A bare validator message reads as an internal error, and models answer
        # those by apologising rather than re-calling the tool.
        loop, _, _ = loop_for({"city": 42}, validate_tool_arguments=True)
        loop.complete("go")
        assert "Call the tool again" in tool_messages(loop)

    def test_it_warns_so_a_model_that_never_gets_it_right_is_visible(self) -> None:
        loop, _, warnings = loop_for({"city": 42}, validate_tool_arguments=True)
        loop.complete("go")
        warning = next(w for w in warnings if w.get("code") == "tool_arguments_invalid")
        assert warning["details"]["toolName"] == "lookup"
        assert warning["details"]["errors"]

    def test_it_runs_the_tool_when_the_arguments_are_valid(self) -> None:
        loop, ran, warnings = loop_for({"city": "Paris"}, validate_tool_arguments=True)
        loop.complete("go")
        assert ran == [{"city": "Paris"}]
        assert not [w for w in warnings if w.get("code") == "tool_arguments_invalid"]

    def test_it_catches_a_missing_required_argument(self) -> None:
        loop, ran, _ = loop_for({}, validate_tool_arguments=True)
        loop.complete("go")
        assert ran == []

    def test_a_builtin_tool_has_no_parameters_to_check(self) -> None:
        # `{"type": "web_search"}` carries no schema at all; reading one off it
        # would refuse a tool that is perfectly well formed.
        hooks = HookBus()
        builtin = Tool(lambda **_kw: "results", {"type": "web_search"})
        loop = AgentLoop(
            Client("web_search", {}),
            tools=[builtin],
            hooks=hooks,
            system="s",
            validate_tool_arguments=True,
        )
        assert loop.complete("go").text == "done"
