"""A run suspends at an approval gate, and while the human is deciding the user
says something more -- "use staging, not prod". There was nowhere to put it.

Appending it to history by hand lands it BEFORE `_repair_unanswered_tool_calls`,
the gate every run passes through, so the model read the correction and then a
tool result, in that order -- the instruction arrives before the thing it is
correcting. And a message held only in the caller's variable is lost if the
process restarts between the gate and the resume, which is the whole reason the
gate is durable.

So staged input is admitted LAST, immediately before the next model call, and it
rides in the snapshot.

Transposed from `unified-library-ts/tests/unit/agent/add-input.test.ts`.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.helpers.tool import Tool
from combycode_llm_sdk.results import Completion, Part, Usage


def answer(text: str = "ok") -> Completion:
    return Completion(
        text=text,
        model="m",
        finish_reason="stop",
        usage=Usage(),
        parts=[Part(type="text", text=text)],
    )


def tool_call() -> Completion:
    part = Part(type="tool_call", id="c1", name="probe", arguments={}, raw={"type": "tool_call"})
    return Completion(
        text="",
        model="m",
        finish_reason="tool_use",
        usage=Usage(),
        parts=[part],
        tool_calls=[part],
    )


class Client:
    """A scripted client that keeps every message list it was sent, so a test can
    read what the model actually saw rather than what history holds after."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self, *turns: Completion) -> None:
        self.turns = list(turns) or [answer()]
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages: Any, **options: Any) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return self.turns[min(len(self.seen) - 1, len(self.turns) - 1)]

    def destroy(self) -> None:
        pass


def user_texts(client: Client, call: int = 0) -> list[str]:
    """The user-role texts the model was shown on the Nth call, in order."""
    out: list[str] = []
    for message in client.seen[call]:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.append(
                "".join(
                    str(p.get("text") or "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            )
    return out


class TestWhenTheStagedMessageReachesTheModel:
    def test_it_is_admitted_after_the_runs_own_input(self) -> None:
        # The order IS the feature: a correction read before the message it
        # corrects is not a correction.
        client = Client()
        loop = AgentLoop(client, system="s")
        loop.add_input("use staging, not prod")
        loop.complete("deploy")

        assert user_texts(client) == ["deploy", "use staging, not prod"]

    def test_insertion_order_is_kept_across_several_calls(self) -> None:
        client = Client()
        loop = AgentLoop(client, system="s")
        loop.add_input("first")
        loop.add_input("second")
        loop.complete("go")

        assert user_texts(client) == ["go", "first", "second"]

    def test_it_is_forgotten_once_admitted(self) -> None:
        client = Client(answer(), answer())
        loop = AgentLoop(client, system="s")
        loop.add_input("use staging")
        loop.complete("deploy")
        loop.complete("again")

        assert loop.pending_input == ()
        # The second call still SEES it -- it is history now -- but exactly once.
        assert user_texts(client, 1).count("use staging") == 1

    def test_it_is_admitted_when_the_resumed_run_passes_no_input(self) -> None:
        # A resume is `run(None)`; nothing may make admission depend on there
        # being a fresh message for it to follow.
        client = Client(answer(), answer())
        loop = AgentLoop(client, system="s")
        loop.complete("deploy")
        loop.add_input("use staging")
        loop.complete(None)

        assert "use staging" in user_texts(client, 1)

    def test_it_normalizes_the_input_forms_the_way_run_does(self) -> None:
        client = Client()
        loop = AgentLoop(client, system="s")
        loop.add_input([{"type": "text", "text": "as parts"}])
        loop.add_input([{"role": "user", "content": "as a message"}])
        loop.complete("go")

        assert user_texts(client) == ["go", "as parts", "as a message"]


class TestWhatItRefusesAndWhatItDiscards:
    def test_it_refuses_to_stage_while_a_run_is_in_flight(self) -> None:
        # Staging then would reach the NEXT run, so a caller who believed they
        # were adding to the running one got silence and a surprise one turn
        # later. A tool body is unambiguously inside a run.
        client = Client(tool_call(), answer())
        caught: list[Exception] = []

        def probe(**_: Any) -> str:
            assert loop.running is True
            try:
                loop.add_input("late")
            except RuntimeError as exc:
                caught.append(exc)
            return "done"

        loop = AgentLoop(
            client,
            system="s",
            tools=[
                Tool(
                    probe,
                    {"type": "function", "name": "probe", "description": "p", "parameters": {}},
                )
            ],
        )
        loop.complete("go")

        assert len(caught) == 1
        assert "in flight" in str(caught[0])
        # And it staged nothing: a rejected call must not half-apply.
        assert loop.pending_input == ()

    def test_clear_pending_input_discards_it(self) -> None:
        client = Client()
        loop = AgentLoop(client, system="s")
        loop.add_input("never mind")
        loop.clear_pending_input()
        loop.complete("go")

        assert loop.pending_input == ()
        assert user_texts(client) == ["go"]

    def test_it_reports_what_is_staged_in_order(self) -> None:
        loop = AgentLoop(Client(), system="s")
        loop.add_input("one")
        loop.add_input("two")
        staged = loop.pending_input
        assert len(staged) == 2
        assert staged[0]["content"] == [{"type": "text", "text": "one"}]


class TestSurvivingTheRestartItExistsFor:
    def test_it_rides_in_the_snapshot_and_the_restored_loop_admits_it(self) -> None:
        # The message exists NOWHERE else. A dropped approval can be asked for
        # again; a correction the user typed once is simply gone.
        loop = AgentLoop(Client(), system="s")
        loop.add_input("use staging")
        snapshot = loop.dump()
        assert len(snapshot["pendingInput"]) == 1

        fresh = Client()
        restored = AgentLoop.restore(snapshot, client=fresh, tools=[])
        assert len(restored.pending_input) == 1
        restored.complete("deploy")
        assert user_texts(fresh) == ["deploy", "use staging"]

    def test_it_is_absent_from_a_snapshot_that_staged_nothing(self) -> None:
        # Not an empty list: that reads as staging which was consumed, and the
        # sibling `pendingToolCalls` is omitted for the same reason.
        assert "pendingInput" not in AgentLoop(Client(), system="s").dump()

    def test_it_is_gone_from_a_snapshot_taken_after_admission(self) -> None:
        loop = AgentLoop(Client(), system="s")
        loop.add_input("use staging")
        loop.complete("deploy")
        assert "pendingInput" not in loop.dump()


class TestTheGuardRaises:
    def test_the_in_flight_refusal_is_a_runtime_error(self) -> None:
        # Asserted directly as well, because the tool-body test would pass
        # vacuously if `complete` swallowed the exception.
        loop = AgentLoop(Client(), system="s")
        loop._running = True
        with pytest.raises(RuntimeError, match="in flight"):
            loop.add_input("late")
