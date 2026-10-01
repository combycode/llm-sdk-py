"""Approval that depends on what the call actually asks for.

Transposed from
`unified-library-ts/tests/unit/agent/argument-conditional-approval.test.ts`.

The policy saw a tool NAME and nothing else, so a rule about `transfer` had two
settings: ask about every transfer, or ask about none. Neither is the rule anyone
wants -- "a transfer over 1000 needs a human" -- and a gate that fires on every
call is one people learn to click through, which is worse than no gate because it
looks like one.

The arguments go to the POLICY rather than onto a `require_confirmation` callback
on each tool, which is where the upstream SDK puts it. "Over 1000 needs a human"
is a policy statement: it belongs beside "deploy needs approval", so that the
answer to *what requires approval here* is readable in one place. Scattered over
tool definitions it is only available by reading every tool.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.helpers.tool import Tool
from combycode_llm_sdk.permissions import PermissionPolicy, Rule, with_args
from combycode_llm_sdk.results import Completion, Part, Usage


def answer(text: str) -> Completion:
    return Completion(
        text=text,
        model="m",
        finish_reason="stop",
        usage=Usage(),
        parts=[Part(type="text", text=text)],
    )


def tool_call(arguments: dict[str, Any]) -> Completion:
    part = Part(
        type="tool_call",
        id="c1",
        name="transfer",
        arguments=arguments,
        raw={"type": "tool_call"},
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
    """Calls `transfer` once with the given arguments, then answers."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self, arguments: dict[str, Any]) -> None:
        self._turns = [tool_call(arguments), answer("done")]
        self.seen = 0

    def complete(self, _messages: Any, **_options: Any) -> Completion:
        self.seen += 1
        return self._turns[min(self.seen - 1, len(self._turns) - 1)]

    def destroy(self) -> None:
        pass


TRANSFER = Tool(
    lambda **_kw: "transferred",
    {"type": "function", "name": "transfer", "description": "Move money", "parameters": {}},
)


def amount_policy() -> PermissionPolicy:
    """Over 1000 needs a human; anything else goes."""
    return PermissionPolicy(
        [
            Rule(
                source="agent",
                action="execute",
                target=with_args("amount", lambda v: float(v) > 1000),
                effect="ask",
                reason="a transfer over 1000 needs a human",
            ),
            Rule(effect="allow"),
        ]
    )


def run(arguments: dict[str, Any], policy: PermissionPolicy | None = None) -> tuple[list[Any], str]:
    """Runs one call; reports whether the approver was consulted."""
    asked: list[Any] = []

    def approve(request: Any) -> dict[str, Any]:
        asked.append(request)
        return {"decision": "approve"}

    loop = AgentLoop(
        Client(arguments),
        tools=[TRANSFER],
        policy=policy if policy is not None else amount_policy(),
        approve=approve,
        system="s",
    )
    return asked, loop.complete("move it").text


def arguments_of(request: Any) -> Any:
    return request.get("arguments") if isinstance(request, dict) else request.arguments


class TestARuleThatDependsOnTheArguments:
    def test_it_asks_when_the_amount_is_over_the_line(self) -> None:
        asked, _ = run({"amount": 5000, "to": "acct-9"})
        assert len(asked) == 1
        assert arguments_of(asked[0]) == {"amount": 5000, "to": "acct-9"}

    def test_it_does_not_ask_when_it_is_under(self) -> None:
        # The whole point: a gate that fires on every call is one people click
        # through without reading.
        asked, text = run({"amount": 5, "to": "acct-9"})
        assert asked == []
        assert text == "done"

    def test_it_decides_on_the_boundary_the_rule_states(self) -> None:
        assert run({"amount": 1000})[0] == []
        assert len(run({"amount": 1001})[0]) == 1

    def test_the_whole_argument_object_reaches_the_policy(self) -> None:
        # A rule may read any argument, so all of them have to arrive.
        seen: list[Any] = []

        def spy(target: Any) -> bool:
            seen.append(dict(target))
            return False

        policy = PermissionPolicy(
            [Rule(source="agent", action="execute", target=spy, effect="ask"), Rule(effect="allow")]
        )
        run({"amount": 7, "to": "acct-1", "memo": "rent"}, policy)
        assert seen[0]["arguments"] == {"amount": 7, "to": "acct-1", "memo": "rent"}
        # And the name is still there: this adds to the target, not replaces it.
        assert seen[0]["toolName"] == "transfer"


class TestWithArgsAndTheAbsenceItExistsFor:
    # The predicate says YES to everything, deliberately. With `float(v) > 1000`
    # these two would pass whether or not the presence check exists, because the
    # predicate raises or returns false on a missing value anyway -- they would
    # assert the right behaviour and be satisfied by the wrong implementation.
    ALWAYS_YES = staticmethod(with_args("amount", lambda _v: True))

    def test_it_does_not_match_a_target_with_no_arguments(self) -> None:
        # A decision made before any call exists -- a pre-flight capability
        # check -- has none. A rule ABOUT an argument has nothing to say about
        # such a decision, and the alternative is a matcher that raises on those
        # paths and takes the run down with it.
        assert self.ALWAYS_YES({"kind": "tool", "toolName": "transfer"}) is False

    def test_it_does_not_match_when_the_argument_is_absent(self) -> None:
        assert self.ALWAYS_YES({"kind": "tool", "arguments": {"to": "acct-1"}}) is False
        # And it DOES match when the argument is there, so the guard is the only
        # thing the two assertions above are about.
        assert self.ALWAYS_YES({"kind": "tool", "arguments": {"amount": 1}}) is True

    def test_it_passes_the_value_through_untouched(self) -> None:
        # The predicate is the caller's; a helper that coerced first would decide
        # for them what "5000" means.
        seen: list[Any] = []

        def spy(value: Any) -> bool:
            seen.append(value)
            return True

        matcher = with_args("amount", spy)
        matcher({"kind": "tool", "arguments": {"amount": "5000"}})
        matcher({"kind": "tool", "arguments": {"amount": None}})
        assert seen == ["5000", None]

    def test_it_matches_a_present_but_falsy_argument(self) -> None:
        # `amount: 0` IS an argument. A `key in args` test rather than a
        # truthiness one is what makes "0 is under the line" a decision instead
        # of a silence.
        assert with_args("amount", lambda v: v == 0)({"kind": "tool", "arguments": {"amount": 0}})
