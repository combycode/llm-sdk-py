"""The policy, the approval gate and the checkpoint, as the loop uses them.

All three were ported long before the loop called them. What is tested here is
the wiring, and its one governing rule: a refusal is a tool RESULT, never an
exception. A denied call is an answer the model can work around; raising would
end a whole run because one tool was forbidden.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk import tool
from combycode_llm_sdk.agent.loop import CHECKPOINT_KEY_PREFIX, AgentLoop
from combycode_llm_sdk.approval import APPROVE, DENY, SKIP, ApprovalDecision
from combycode_llm_sdk.permissions import PermissionPolicy, Rule, any_of_kind
from combycode_llm_sdk.persistence import MemoryPersistence
from combycode_llm_sdk.results import Completion, Part, Usage


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


def answer(text: str) -> Completion:
    return completion(text=text, parts=[Part(type="text", text=text)])


def one_call() -> Completion:
    part = Part(
        type="tool_call", id="c1", name="weather", arguments={"city": "P"},
        raw={"type": "tool_call"},
    )
    return completion(parts=[part], tool_calls=[part], finish_reason="tool_use")


@tool
def weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


class Client:
    """Asks for `weather` once, then answers."""

    id = "client_1"

    def __init__(self) -> None:
        self.seen = 0

    def complete(self, messages: Any, **options: Any) -> Completion:
        self.seen += 1
        return one_call() if self.seen == 1 else answer("done")


def policy_of(effect: str) -> PermissionPolicy:
    return PermissionPolicy([Rule(effect=effect, target=any_of_kind("tool"))])


def run(**over: Any) -> tuple[AgentLoop, Any]:
    loop = AgentLoop(Client(), tools=[weather], **over)
    return loop, loop.complete("weather?")


def tool_results(loop: AgentLoop) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in loop.history.all():
        content = entry.message.get("content")
        if isinstance(content, list):
            out += [p for p in content if p.get("type") == "tool_result"]
    return out


class TestWithNoPolicy:
    def test_every_call_runs(self) -> None:
        loop, _ = run()
        assert [p["content"] for p in tool_results(loop)] == ["sunny in P"]


class TestThePolicy:
    def test_an_allowing_policy_lets_it_run(self) -> None:
        loop, _ = run(policy=policy_of("allow"))
        assert [p["content"] for p in tool_results(loop)] == ["sunny in P"]

    def test_a_denying_policy_answers_the_model_instead_of_raising(self) -> None:
        # The governing rule: the run continues, the model is told.
        loop, answer = run(policy=policy_of("deny"))
        (result,) = tool_results(loop)
        assert result["isError"] is True
        assert answer.text == "done", "the run finished rather than dying"

    def test_a_denied_call_never_reaches_the_tool(self) -> None:
        ran: list[str] = []

        @tool
        def weather(city: str) -> str:
            """Get the weather for a city."""
            ran.append(city)
            return "ran"

        loop = AgentLoop(Client(), tools=[weather], policy=policy_of("deny"))
        loop.complete("weather?")
        assert ran == []

    def test_a_denial_is_recorded_in_the_report(self) -> None:
        # A refused call is still a call that happened: it has to show up in
        # the run report, or the transcript says the tool was never asked for.
        loop, _ = run(policy=policy_of("deny"))
        report = loop.last_report
        assert report is not None
        errors = [c.error for step in report.steps for c in step.tool_calls if c.error]
        assert errors and "denied" in errors[0]


class TestTheApprovalGate:
    def test_an_approver_that_says_yes_lets_the_tool_run(self) -> None:
        loop, _ = run(policy=policy_of("ask"), approve=lambda req: ApprovalDecision(APPROVE))
        assert [p["content"] for p in tool_results(loop)] == ["sunny in P"]

    def test_an_approver_that_says_no_refuses(self) -> None:
        loop, _ = run(
            policy=policy_of("ask"),
            approve=lambda req: ApprovalDecision(DENY, note="not this one"),
        )
        (result,) = tool_results(loop)
        assert result["isError"] is True
        assert "not this one" in str(result["content"])

    def test_ask_with_no_approver_is_a_denial(self) -> None:
        # A rule that reads as a guard and silently permits everything is worse
        # than no rule at all.
        loop, _ = run(policy=policy_of("ask"))
        (result,) = tool_results(loop)
        assert result["isError"] is True

    def test_an_approver_that_raises_denies_rather_than_ending_the_run(self) -> None:
        def explode(req: Any) -> ApprovalDecision:
            raise RuntimeError("approval channel down")

        loop, answer = run(policy=policy_of("ask"), approve=explode)
        (result,) = tool_results(loop)
        assert result["isError"] is True
        assert answer.text == "done"

    def test_an_approver_that_returns_nothing_denies(self) -> None:
        # A missing `return` is the usual way this happens, and it is not a yes.
        loop, _ = run(policy=policy_of("ask"), approve=lambda req: None)
        assert tool_results(loop)[0]["isError"] is True

    def test_a_skip_is_not_an_error(self) -> None:
        loop, _ = run(policy=policy_of("ask"), approve=lambda req: ApprovalDecision(SKIP))
        (result,) = tool_results(loop)
        assert result.get("isError") is not True

    def test_an_approved_override_answers_instead_of_the_tool(self) -> None:
        # An APPROVE can carry an override; running the tool as well would be
        # the failure `result_for` exists to prevent.
        loop, _ = run(
            policy=policy_of("ask"),
            approve=lambda req: ApprovalDecision(APPROVE, override_result="the human said 12C"),
        )
        (result,) = tool_results(loop)
        assert result["content"] == "the human said 12C"

    def test_the_request_and_the_resolution_are_both_announced(self) -> None:
        seen: list[str] = []
        loop = AgentLoop(Client(), tools=[weather], policy=policy_of("ask"),
                         approve=lambda req: ApprovalDecision(APPROVE))
        loop.hooks.on("onApprovalRequested", lambda c: seen.append("requested"))
        loop.hooks.on("onApprovalResolved", lambda c: seen.append("resolved"))
        loop.complete("weather?")
        assert seen == ["requested", "resolved"]

    def test_nothing_is_left_pending_afterwards(self) -> None:
        loop, _ = run(policy=policy_of("ask"), approve=lambda req: ApprovalDecision(APPROVE))
        assert loop.pending_approvals == ()


class TestTheProviderSignature:
    """Gemini's thought signature must survive the approval detour.

    It is an opaque token that has to be echoed back VERBATIM or the next turn
    is rejected. It rides in the part's `raw`, and the approval layer works in
    `ToolCall`s -- so the conversion between them is the one place it can be
    silently dropped, and a dropped one fails on the NEXT request, not this one.
    """

    def call_with_signature(self) -> Any:
        part = Part(
            type="tool_call", id="c1", name="weather", arguments={"city": "P"},
            raw={"type": "tool_call", "signature": "sig-abc"},
        )
        return completion(parts=[part], tool_calls=[part], finish_reason="tool_use")

    def test_it_reaches_the_overridden_result(self) -> None:
        signed = self.call_with_signature()

        class Signing(Client):
            def complete(self, messages: Any, **options: Any) -> Completion:
                self.seen += 1
                return signed if self.seen == 1 else answer("done")

        loop = AgentLoop(
            Signing(), tools=[weather], policy=policy_of("ask"),
            approve=lambda req: ApprovalDecision(DENY, note="no"),
        )
        loop.complete("weather?")
        results = tool_results(loop)
        assert results and results[0]["id"] == "c1"


class TestTheCheckpoint:
    def test_the_run_is_written_before_the_approver_is_asked(self) -> None:
        # The point of a checkpoint is to survive a process that never returns
        # from that wait, so writing it afterwards would be writing it too late.
        store = MemoryPersistence()
        saw_snapshot: list[bool] = []

        def approver(req: Any) -> ApprovalDecision:
            saw_snapshot.append(store.has(CHECKPOINT_KEY_PREFIX + loop.id))
            return ApprovalDecision(APPROVE)

        loop = AgentLoop(Client(), tools=[weather], policy=policy_of("ask"),
                         approve=approver, checkpoint=store)
        loop.complete("weather?")
        assert saw_snapshot == [True]

    def test_the_checkpoint_names_the_pending_call(self) -> None:
        store = MemoryPersistence()
        captured: list[Any] = []

        def approver(req: Any) -> ApprovalDecision:
            captured.append(store.get(CHECKPOINT_KEY_PREFIX + loop.id))
            return ApprovalDecision(APPROVE)

        loop = AgentLoop(Client(), tools=[weather], policy=policy_of("ask"),
                         approve=approver, checkpoint=store)
        loop.complete("weather?")
        (snapshot,) = captured
        assert [p["toolName"] for p in snapshot["pendingToolCalls"]] == ["weather"]

    def test_no_checkpoint_is_written_when_nothing_asks(self) -> None:
        store = MemoryPersistence()
        AgentLoop(Client(), tools=[weather], checkpoint=store).complete("weather?")
        assert store.list() == []


class TestMetadataAndTheSnapshot:
    def test_metadata_is_carried_and_mutable(self) -> None:
        loop = AgentLoop(Client(), tools=[weather], metadata={"tenant": "acme"})
        loop.metadata["seen"] = 1
        assert loop.dump()["metadata"] == {"tenant": "acme", "seen": 1}

    def test_metadata_survives_a_round_trip(self) -> None:
        loop = AgentLoop(Client(), tools=[weather], metadata={"tenant": "acme"})
        back = AgentLoop.restore(loop.dump(), client=Client(), tools=[weather])
        assert back.metadata == {"tenant": "acme"}

    def test_a_run_that_was_not_suspended_carries_no_pending_key(self) -> None:
        # An empty list would read as "resumed from nothing" rather than "never
        # suspended".
        loop = AgentLoop(Client(), tools=[weather])
        assert "pendingToolCalls" not in loop.dump()

    def test_a_suspended_call_survives_a_restart(self) -> None:
        store = MemoryPersistence()
        captured: list[Any] = []

        def approver(req: Any) -> ApprovalDecision:
            captured.append(store.get(CHECKPOINT_KEY_PREFIX + loop.id))
            return ApprovalDecision(APPROVE)

        loop = AgentLoop(Client(), tools=[weather], policy=policy_of("ask"),
                         approve=approver, checkpoint=store)
        loop.complete("weather?")
        # What a process that died mid-wait would come back to.
        back = AgentLoop.restore(captured[0], client=Client(), tools=[weather])
        assert [p.tool_name for p in back.pending_approvals] == ["weather"]
        assert back.pending_approvals[0].arguments == {"city": "P"}
