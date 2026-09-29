"""Per-call guardrails: one on a call's arguments, one on its output.

Transposed from
`unified-library-ts/tests/unit/agent/tool-output-guardrails.test.ts`.

The distinction that shapes the output side: by the time it runs, the tool has
ALREADY RUN. Halting would be the wrong lever -- the output exists, and what is
left to control is what it touches. So a trip withholds the output and a
placeholder takes its place everywhere it would have been kept: the result the
model reads, the conversation, any checkpoint written from it, and the hook a
logger listens on.

Both halves fail closed. A guardrail that raises counts as having tripped -- a
checker that crashed has approved nothing, and the run that matters is the one
where it crashed on the output it would have caught. A message formatter that
raises falls back to the default sentence, never to the output it was deciding
about.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk import tool
from combycode_llm_sdk.agent.loop import TOOL_OUTPUT_WITHHELD, AgentLoop
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.results import Completion, Part, Usage

SECRET = "card 4111 1111 1111 1111"


def completion(**over: Any) -> Completion:
    fields: dict[str, Any] = {
        "text": "",
        "model": "m",
        "finish_reason": "stop",
        "usage": Usage(),
        "parts": [],
    }
    fields.update(over)
    return Completion(**fields)


def one_call(name: str = "lookup") -> Completion:
    part = Part(
        type="tool_call",
        id="c1",
        name=name,
        arguments={"who": "alex"},
        raw={"type": "tool_call"},
    )
    return completion(parts=[part], tool_calls=[part], finish_reason="tool_use")


@tool
def lookup(who: str) -> str:
    """Look somebody up."""
    return SECRET


@tool
def safe_lookup(who: str) -> str:
    """Look somebody up, harmlessly."""
    return "nothing sensitive here"


class Client:
    """Asks for `lookup` once, then answers. Records what it was shown."""

    id = "client_1"

    def __init__(self, tool_name: str = "lookup") -> None:
        self.seen: list[Any] = []
        self._tool_name = tool_name

    def complete(self, messages: Any, **options: Any) -> Completion:
        self.seen.append(messages)
        if len(self.seen) == 1:
            return one_call(self._tool_name)
        return completion(text="done", parts=[Part(type="text", text="done")])


class Guard:
    """A guardrail in the shape the loop accepts: a `name` and a `check`."""

    def __init__(self, name: str, decide: Any) -> None:
        self.name = name
        self._decide = decide

    def check(self, ctx: Any) -> Any:
        return self._decide(ctx)


def block_cards() -> Guard:
    return Guard(
        "no-card-numbers",
        lambda ctx: {"pass": False, "reason": "card number"}
        if "4111" in (ctx.result or "")
        else {"pass": True},
    )


def run(tool_fn: Any = lookup, **config: Any) -> tuple[str, list[dict[str, Any]]]:
    """One tool call. Returns what the model saw next, and the warnings."""
    hooks = HookBus()
    warnings: list[dict[str, Any]] = []
    hooks.on("onWarning", lambda ctx: warnings.append(dict(ctx)))
    client = Client(tool_fn.definition["name"] if hasattr(tool_fn, "definition") else "lookup")
    loop = AgentLoop(client, tools=[tool_fn], hooks=hooks, **config)
    loop.run("go")
    follow_up = str(client.seen[1]) if len(client.seen) > 1 else ""
    return follow_up, warnings


class TestWithNoGuardrailNothingChanges:
    def test_the_output_passes_through(self) -> None:
        follow_up, _ = run()
        assert "4111" in follow_up


class TestATrippedGuardrailWithholdsTheOutput:
    def test_it_replaces_what_the_model_is_shown(self) -> None:
        follow_up, _ = run(tool_output_guardrails=[block_cards()])
        assert "4111" not in follow_up
        assert TOOL_OUTPUT_WITHHELD in follow_up

    def test_it_says_so_once_naming_the_guardrail_and_the_tool(self) -> None:
        _, warnings = run(tool_output_guardrails=[block_cards()])
        withheld = [w for w in warnings if w.get("code") == "tool_output_withheld"]
        assert len(withheld) == 1
        assert "no-card-numbers" in withheld[0]["message"]
        assert "lookup" in withheld[0]["message"]

    def test_it_leaves_an_output_it_has_no_objection_to_alone(self) -> None:
        follow_up, _ = run(safe_lookup, tool_output_guardrails=[block_cards()])
        assert "nothing sensitive here" in follow_up


class TestTheReplacementText:
    def test_the_guardrails_own_wording_wins(self) -> None:
        guard = Guard(
            "g", lambda _ctx: {"pass": False, "reason": "card", "replaceWith": "Redacted by policy."}
        )
        follow_up, _ = run(tool_output_guardrails=[guard])
        assert "Redacted by policy." in follow_up

    def test_the_loops_configured_message_is_used(self) -> None:
        follow_up, _ = run(
            tool_output_guardrails=[block_cards()],
            tool_output_blocked_message="Ask your administrator.",
        )
        assert "Ask your administrator." in follow_up

    def test_a_formatter_may_name_the_guardrail_and_the_tool(self) -> None:
        follow_up, _ = run(
            tool_output_guardrails=[block_cards()],
            tool_output_blocked_message=lambda a: f"{a['toolName']} blocked by {a['guardrailName']}",
        )
        assert "lookup blocked by no-card-numbers" in follow_up

    def test_a_formatter_that_raises_falls_back_to_the_default(self) -> None:
        def broken(_args: Any) -> str:
            raise RuntimeError("formatter is broken")

        follow_up, _ = run(
            tool_output_guardrails=[block_cards()], tool_output_blocked_message=broken
        )
        assert TOOL_OUTPUT_WITHHELD in follow_up
        assert "4111" not in follow_up


class TestAGuardrailThatRaisesCountsAsTripped:
    def test_it_withholds_rather_than_passing_the_output_through(self) -> None:
        def explode(_ctx: Any) -> Any:
            raise RuntimeError("checker exploded")

        follow_up, warnings = run(tool_output_guardrails=[Guard("broken", explode)])
        assert "4111" not in follow_up
        assert TOOL_OUTPUT_WITHHELD in follow_up
        assert any("checker exploded" in w.get("message", "") for w in warnings)


class TestTheInputSide:
    def test_a_trip_denies_just_that_call(self) -> None:
        # A denial is a tool RESULT the model can work around, not an exception:
        # one bad call is not a reason to end a conversation.
        guard = Guard(
            "no-alex", lambda ctx: {"pass": False, "reason": "not allowed"}
            if ctx.arguments.get("who") == "alex"
            else {"pass": True},
        )
        follow_up, _ = run(tool_input_guardrails=[guard])
        assert "not allowed" in follow_up
        # And the tool never ran, so its output is nowhere.
        assert "4111" not in follow_up

    def test_a_pass_runs_the_tool_normally(self) -> None:
        guard = Guard("allow", lambda _ctx: {"pass": True})
        follow_up, _ = run(tool_input_guardrails=[guard])
        assert "4111" in follow_up

    def test_a_raising_input_guardrail_denies(self) -> None:
        def explode(_ctx: Any) -> Any:
            raise RuntimeError("input checker exploded")

        follow_up, _ = run(tool_input_guardrails=[Guard("broken", explode)])
        assert "4111" not in follow_up
        assert "input checker exploded" in follow_up
