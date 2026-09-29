"""A tool call and its result travel together, or not at all.

Anthropic and OpenAI both reject a history holding a tool call nothing answered,
and equally a result that answers nothing. Either way the failure lands on the
NEXT request -- one turn away from whatever broke the pair, with an error naming
neither.

The malformed half is the other direction: a call whose arguments did not parse
used to fall back to `{}`, which is a VALID call rather than a failed one, so a
stream cut at `{"path": "/et` ran the tool with no arguments at all.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.history import ConversationHistory
from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.agent.loop_stream import (
    StepState,
    accumulate_stream_event,
    build_step_completion,
    finalize_unended_tool_calls,
)


def _call(call_id: str, name: str = "t") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "tool_call", "id": call_id, "name": name, "arguments": {}}],
    }


def _result(call_id: str) -> dict[str, Any]:
    return {"role": "tool", "content": [{"type": "tool_result", "id": call_id, "content": "ok"}]}


def _text(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


def _feed(events: list[dict[str, Any]]) -> StepState:
    state = StepState()
    for event in events:
        accumulate_stream_event(event, state)
    finalize_unended_tool_calls(state)
    return state


class TestTruncateKeepsPairsWhole:
    def test_drops_a_result_whose_call_would_be_cut(self) -> None:
        history = ConversationHistory()
        for message in (_text("user", "a"), _call("c1"), _result("c1"), _text("assistant", "done")):
            history.append(message)
        # Keeping 2 would start the history at the result -- an answer to nothing.
        history.truncate(2)
        messages = history.messages()
        assert [m["role"] for m in messages] == ["assistant"]
        assert messages[0]["content"] == "done"

    def test_keeps_a_pair_that_survives_the_cut(self) -> None:
        history = ConversationHistory()
        for message in (_text("user", "a"), _text("user", "b"), _call("c1"), _result("c1")):
            history.append(message)
        history.truncate(2)
        parts = [
            p
            for m in history.messages()
            for p in (m["content"] if isinstance(m["content"], list) else [])
        ]
        assert sum(1 for p in parts if p["type"] == "tool_call") == 1
        assert sum(1 for p in parts if p["type"] == "tool_result") == 1

    def test_leaves_a_history_without_tools_alone(self) -> None:
        history = ConversationHistory()
        for message in (_text("user", "a"), _text("assistant", "b"), _text("user", "c")):
            history.append(message)
        history.truncate(2)
        assert [m["content"] for m in history.messages()] == ["b", "c"]

    def test_reindexes_what_it_keeps(self) -> None:
        history = ConversationHistory()
        for message in (_text("user", "a"), _call("c1"), _result("c1"), _text("assistant", "d")):
            history.append(message)
        history.truncate(2)
        assert [e.index for e in history.entries] == [0]


class TestMalformedArguments:
    def test_a_truncated_call_is_marked_not_emptied(self) -> None:
        state = _feed(
            [
                {"type": "tool_call_start", "id": "c1", "name": "delete_files"},
                {"type": "tool_call_delta", "id": "c1", "arguments": '{"path": "/et'},
            ]
        )
        assert state.tool_calls[0]["malformed"] is True
        assert state.tool_calls[0]["arguments"] == {}
        assert state.tool_calls[0]["name"] == "delete_files"

    def test_a_genuine_no_argument_call_is_not_malformed(self) -> None:
        for args in ("", "   ", "{}"):
            events: list[dict[str, Any]] = [
                {"type": "tool_call_start", "id": "c1", "name": "ping"}
            ]
            if args:
                events.append({"type": "tool_call_delta", "id": "c1", "arguments": args})
            state = _feed(events)
            assert state.tool_calls[0]["arguments"] == {}
            assert "malformed" not in state.tool_calls[0]

    def test_a_scalar_parses_but_is_not_an_argument_object(self) -> None:
        for args in ('"just a string"', "42", "null", "[1,2]"):
            state = _feed(
                [
                    {"type": "tool_call_start", "id": "c1", "name": "t"},
                    {"type": "tool_call_delta", "id": "c1", "arguments": args},
                ]
            )
            assert state.tool_calls[0]["malformed"] is True
            assert state.tool_calls[0]["arguments"] == {}

    def test_well_formed_arguments_are_untouched(self) -> None:
        state = _feed(
            [
                {"type": "tool_call_start", "id": "c1", "name": "get_weather"},
                {"type": "tool_call_delta", "id": "c1", "arguments": '{"city":'},
                {"type": "tool_call_delta", "id": "c1", "arguments": '"Berlin"}'},
            ]
        )
        assert state.tool_calls[0]["arguments"] == {"city": "Berlin"}
        assert "malformed" not in state.tool_calls[0]

    def test_the_step_reports_malformed_tool_call_not_tool_use(self) -> None:
        # Only Google's API reports this reason itself, so the same truncation
        # elsewhere finished as `tool_use` and looked like a successful turn.
        state = _feed(
            [
                {"type": "tool_call_start", "id": "c1", "name": "x"},
                {"type": "tool_call_delta", "id": "c1", "arguments": '{"a":'},
            ]
        )
        assert build_step_completion(state, "m", 0.0).finish_reason == "malformed_tool_call"

    def test_a_step_whose_calls_parsed_still_reports_tool_use(self) -> None:
        state = _feed(
            [
                {"type": "tool_call_start", "id": "c1", "name": "x"},
                {"type": "tool_call_delta", "id": "c1", "arguments": '{"a":1}'},
            ]
        )
        assert build_step_completion(state, "m", 0.0).finish_reason == "tool_use"

    def test_one_bad_call_among_good_ones_still_fails_the_step(self) -> None:
        state = _feed(
            [
                {"type": "tool_call_start", "id": "good", "name": "a"},
                {"type": "tool_call_delta", "id": "good", "arguments": '{"ok":true}'},
                {"type": "tool_call_end", "id": "good"},
                {"type": "tool_call_start", "id": "bad", "name": "b"},
                {"type": "tool_call_delta", "id": "bad", "arguments": '{"cut'},
            ]
        )
        assert build_step_completion(state, "m", 0.0).finish_reason == "malformed_tool_call"
        by_id = {c["id"]: c for c in state.tool_calls}
        assert by_id["good"]["arguments"] == {"ok": True}
        assert by_id["bad"]["malformed"] is True


class _IdleClient:
    """Answers immediately, so the run ends after the repair."""

    id = "client_1"
    provider = "openai"
    model = "mock-model"

    def stream(self, messages: Any, options: Any = None) -> Any:
        self.seen = [dict(m) for m in messages]
        yield {"type": "text", "text": "ok"}
        yield {"type": "done", "finishReason": "stop"}

    def destroy(self) -> None:
        pass


def _interrupted() -> ConversationHistory:
    """The shape every interruption leaves: the call is the last thing in
    history, because execution never got far enough to append a result."""
    history = ConversationHistory()
    history.append(_text("user", "go"))
    history.append(_call("orphan-1", "delete_files"))
    return history


class TestAnInterruptedTurnIsRepaired:
    def _run(self) -> AgentLoop:
        loop = AgentLoop(_IdleClient(), history=_interrupted())
        list(loop.stream("next"))
        return loop

    def _parts(self, loop: AgentLoop) -> list[dict[str, Any]]:
        return [
            p
            for m in loop.history.messages()
            for p in (m["content"] if isinstance(m["content"], list) else [])
        ]

    def test_the_orphan_is_answered(self) -> None:
        parts = self._parts(self._run())
        calls = [p for p in parts if p["type"] == "tool_call"]
        results = [p for p in parts if p["type"] == "tool_result"]
        assert len(calls) == 1
        assert len(results) == 1
        assert results[0]["id"] == "orphan-1"

    def test_it_says_the_tool_never_ran_rather_than_inventing_success(self) -> None:
        result = next(p for p in self._parts(self._run()) if p["type"] == "tool_result")
        assert "never run" in result["content"]
        assert "delete_files" in result["content"]
        assert result["isError"] is True

    def test_the_repair_lands_before_the_new_user_message(self) -> None:
        roles = [m["role"] for m in self._run().history.messages()]
        assert roles.index("tool") < len(roles) - 1 - roles[::-1].index("user")

    def test_a_well_formed_history_is_left_alone(self) -> None:
        history = ConversationHistory()
        for message in (_text("user", "go"), _call("c1"), _result("c1")):
            history.append(message)
        before = len(history.messages())
        loop = AgentLoop(_IdleClient(), history=history)
        list(loop.stream("next"))
        # Only the new user turn and the assistant reply -- no repair entry.
        assert len(loop.history.messages()) == before + 2
