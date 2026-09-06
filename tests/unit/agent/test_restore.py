"""A dumped conversation, wired to a live client and live tools.

The interesting half is not the round trip -- it is the comparison between the
tools the saved run had and the ones this one provides. A tool the transcript
shows the model using, that is now absent, WILL be called again: the model
learns it exists from the history it is about to be shown. It then fails as an
unknown tool several turns later, nowhere near the restore that caused it.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.helpers.tool import Tool


class Client:
    """Restoring must not call the model."""

    def complete(self, **options: Any) -> Any:
        raise AssertionError("restore must not reach the provider")


def named(name: str) -> Tool:
    return Tool(
        lambda **kw: name,
        {"type": "function", "name": name, "description": name, "parameters": {}},
    )


def saved(**over: Any) -> dict[str, Any]:
    loop = AgentLoop(Client(), system="be brief", context="ctx", tools=[named("lookup")])
    loop.history.append({"role": "user", "content": "hello"})
    snapshot = loop.dump()
    snapshot.update(over)
    return snapshot


def warnings_from(loop: AgentLoop, seen: list[dict[str, Any]], code: str) -> list[str]:
    return [str(w["details"]["toolName"]) for w in seen if w.get("code") == code]


def restore(snapshot: dict[str, Any], tools: list[Tool]) -> tuple[AgentLoop, list[dict[str, Any]]]:
    from combycode_llm_sdk.bus.hook_bus import HookBus

    seen: list[dict[str, Any]] = []
    hooks = HookBus()
    hooks.on("onWarning", lambda w: seen.append(dict(w)))
    loop = AgentLoop.restore(snapshot, client=Client(), tools=tools, hooks=hooks)
    return loop, seen


class TestTheConversationComesBack:
    def test_the_prompt_and_context_are_restored(self) -> None:
        loop, _ = restore(saved(), [named("lookup")])
        assert loop.dump()["system"] == "be brief"
        assert loop.dump()["context"] == "ctx"

    def test_the_history_is_restored(self) -> None:
        loop, _ = restore(saved(), [named("lookup")])
        assert [e.message["content"] for e in loop.history.all()] == ["hello"]

    def test_a_snapshot_round_trips(self) -> None:
        first = saved()
        loop, _ = restore(first, [named("lookup")])
        again = loop.dump()
        assert again["system"] == first["system"]
        assert again["toolNames"] == first["toolNames"]
        assert len(again["history"]["entries"]) == len(first["history"]["entries"])

    def test_restoring_does_not_call_the_model(self) -> None:
        # The Client raises if it is touched; reaching the assert proves it.
        loop, _ = restore(saved(), [named("lookup")])
        assert loop is not None


class TestTheToolSetsAreCompared:
    def test_matching_tools_say_nothing(self) -> None:
        _, seen = restore(saved(), [named("lookup")])
        assert seen == []

    def test_a_tool_the_transcript_uses_but_nobody_provides_is_named(self) -> None:
        loop, seen = restore(saved(), [])
        assert warnings_from(loop, seen, "tool_removed") == ["lookup"]

    def test_a_tool_that_is_new_is_named_too(self) -> None:
        loop, seen = restore(saved(), [named("lookup"), named("fresh")])
        assert warnings_from(loop, seen, "tool_added") == ["fresh"]

    def test_both_halves_are_reported_at_once(self) -> None:
        # Swapping a tool is one edit and two surprises.
        loop, seen = restore(saved(), [named("replacement")])
        assert warnings_from(loop, seen, "tool_removed") == ["lookup"]
        assert warnings_from(loop, seen, "tool_added") == ["replacement"]

    def test_the_comparison_is_stable_in_order(self) -> None:
        # Sorted, so a log diff between two restores is about the tools and not
        # about dictionary ordering.
        loop, seen = restore(saved(), [named("b"), named("a"), named("c")])
        assert warnings_from(loop, seen, "tool_added") == ["a", "b", "c"]

    def test_removals_are_reported_in_a_stable_order_too(self) -> None:
        # Same reason as additions: a log diff between two restores should be
        # about which tools went missing, not about set iteration order.
        snapshot = saved(toolNames=["delta", "alpha", "charlie", "bravo", "echo"])
        loop, seen = restore(snapshot, [])
        assert warnings_from(loop, seen, "tool_removed") == [
            "alpha", "bravo", "charlie", "delta", "echo",
        ]

    def test_a_snapshot_with_no_tools_recorded_is_not_a_removal(self) -> None:
        loop, seen = restore(saved(toolNames=[]), [named("lookup")])
        assert warnings_from(loop, seen, "tool_removed") == []
        assert warnings_from(loop, seen, "tool_added") == ["lookup"]

    def test_a_builtin_is_compared_by_its_type(self) -> None:
        # It is keyed by type, so that is the name a snapshot carries.
        snapshot = saved(toolNames=["web_search"])
        _loop, seen = restore(snapshot, [Tool(lambda: "", {"type": "web_search"})])
        assert seen == []
