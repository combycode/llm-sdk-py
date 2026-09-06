"""Streaming an agent run.

A streamed step arrives as deltas and must end up the SAME shape a buffered one
produces -- everything downstream reads that shape and cannot tell how the step
was fetched. The failures worth pinning are the quiet ones: narration folded
into the answer, a tool call dropped because its `end` never arrived, a step
reported as finished when it was actually asking for a tool.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

sys.path.insert(0, "src")

from combycode_llm_sdk import tool
from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.agent.loop_stream import (
    StepState,
    accumulate_stream_event,
    build_step_completion,
    finalize_unended_tool_calls,
    tool_call_end_events,
    tool_call_start_events,
)
from combycode_llm_sdk.results import Usage


def _fold(events: list[dict[str, Any]]) -> tuple[StepState, list[dict[str, Any]]]:
    state = StepState()
    yielded = [e for e in (accumulate_stream_event(ev, state) for ev in events) if e]
    return state, yielded


class TestText:
    def test_text_is_accumulated_and_forwarded(self) -> None:
        state, out = _fold([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}])
        assert state.text == "hello"
        assert [e["text"] for e in out] == ["he", "llo"]

    def test_commentary_is_kept_out_of_the_answer(self) -> None:
        # Otherwise the model's thinking-aloud ends up in the transcript as if
        # it were the reply.
        state, _ = _fold(
            [
                {"type": "text", "text": "let me think ", "phase": "commentary"},
                {"type": "text", "text": "42"},
            ]
        )
        assert state.text == "42"
        assert state.commentary == "let me think "

    def test_the_phase_is_forwarded_not_just_used(self) -> None:
        # A consumer streaming these live has no other way to tell narration
        # from the answer; `final_answer_text` only cleans a finished message.
        _, out = _fold([{"type": "text", "text": "hm", "phase": "commentary"}])
        assert out[0]["phase"] == "commentary"

    def test_no_phase_means_no_phase_key(self) -> None:
        # Most providers report none, and inventing one would be a guess.
        _, out = _fold([{"type": "text", "text": "hi"}])
        assert "phase" not in out[0]

    def test_thinking_is_separate_from_both(self) -> None:
        state, out = _fold([{"type": "thinking", "text": "reasoning"}])
        assert state.thinking == "reasoning"
        assert state.text == ""
        assert out[0]["type"] == "thinking"


class TestToolCalls:
    CALL: ClassVar[list[dict[str, Any]]] = [
        {"type": "tool_call_start", "id": "c1", "name": "search"},
        {"type": "tool_call_delta", "id": "c1", "arguments": '{"q":'},
        {"type": "tool_call_delta", "id": "c1", "arguments": '"berlin"}'},
        {"type": "tool_call_end", "id": "c1"},
    ]

    def test_a_call_is_reassembled_from_its_deltas(self) -> None:
        state, out = _fold(list(self.CALL))
        assert state.tool_calls == [
            {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"q": "berlin"}}
        ]
        # Tool calls are not forwarded as they arrive: the loop announces them
        # once it knows the whole call, with its arguments.
        assert out == []

    def test_a_delta_with_no_id_still_lands(self) -> None:
        # Several providers send argument deltas carrying no id at all, and
        # dropping those produces a call with empty arguments and no error.
        state, _ = _fold(
            [
                {"type": "tool_call_start", "id": "c1", "name": "search"},
                {"type": "tool_call_delta", "arguments": '{"q":"x"}'},
                {"type": "tool_call_end", "id": "c1"},
            ]
        )
        assert state.tool_calls[0]["arguments"] == {"q": "x"}

    def test_unparseable_arguments_become_empty_rather_than_raising(self) -> None:
        # The model is already paid for. A tool handed `{}` fails with something
        # it can read and retry; a raised parse error ends the run.
        state, _ = _fold(
            [
                {"type": "tool_call_start", "id": "c1", "name": "search"},
                {"type": "tool_call_delta", "id": "c1", "arguments": "{not json"},
                {"type": "tool_call_end", "id": "c1"},
            ]
        )
        assert state.tool_calls[0]["arguments"] == {}

    def test_a_call_that_never_ends_is_still_closed(self) -> None:
        # Anthropic and OpenAI both stream calls that end only when the message
        # does. Without this the step's last call is dropped and the model is
        # answered as though it had not asked.
        state, _ = _fold(
            [
                {"type": "tool_call_start", "id": "c1", "name": "search"},
                {"type": "tool_call_delta", "id": "c1", "arguments": '{"q":"x"}'},
            ]
        )
        assert state.tool_calls == []
        finalize_unended_tool_calls(state)
        assert [c["id"] for c in state.tool_calls] == ["c1"]

    def test_finalizing_does_not_duplicate_a_call_that_did_end(self) -> None:
        state, _ = _fold(list(self.CALL))
        finalize_unended_tool_calls(state)
        assert len(state.tool_calls) == 1

    def test_provider_metadata_rides_along(self) -> None:
        state, _ = _fold(
            [
                {"type": "tool_call_start", "id": "c1", "name": "s", "_meta": {"sig": "abc"}},
                {"type": "tool_call_end", "id": "c1"},
            ]
        )
        assert state.tool_calls[0]["_meta"] == {"sig": "abc"}


class TestTheRest:
    def test_usage_and_finish_reason_are_captured(self) -> None:
        state, out = _fold(
            [
                {"type": "usage", "usage": Usage(input_tokens=3, output_tokens=4)},
                {"type": "done", "finishReason": "length"},
            ]
        )
        assert (state.usage.input_tokens, state.usage.output_tokens) == (3, 4)
        assert state.finish_reason == "length"
        assert out == []

    def test_citations_are_deduped_by_url(self) -> None:
        # Google repeats its grounding chunks across late chunks, and one page
        # cited twice is one source.
        cite = type("C", (), {"url": "https://x/y"})()
        state, _ = _fold([{"type": "citation", "citation": cite}] * 3)
        assert len(state.citations) == 1

    def test_an_unknown_event_is_ignored_rather_than_fatal(self) -> None:
        # The provider event set is open; a new kind must not end a run.
        state, out = _fold([{"type": "something_new", "payload": 1}])
        assert out == [] and state.text == ""


class TestTheStepItBuilds:
    def test_it_looks_like_a_buffered_step(self) -> None:
        state, _ = _fold(
            [
                {"type": "text", "text": "the answer"},
                {"type": "usage", "usage": Usage(input_tokens=5, output_tokens=2)},
                {"type": "done", "finishReason": "stop"},
            ]
        )
        step = build_step_completion(state, "openai/gpt-4.1", 12.5)
        assert step.text == "the answer"
        assert step.finish_reason == "stop"
        assert step.usage.output_tokens == 2
        assert step.latency_ms == 12.5
        assert [p.type for p in step.parts] == ["text"]

    def test_a_step_that_asked_for_a_tool_says_so(self) -> None:
        # Several providers report `stop` alongside a tool call; a loop reading
        # the raw value would end the run with the call unanswered.
        state, _ = _fold(
            [
                {"type": "tool_call_start", "id": "c1", "name": "search"},
                {"type": "tool_call_end", "id": "c1"},
                {"type": "done", "finishReason": "stop"},
            ]
        )
        step = build_step_completion(state, "m", 1.0)
        assert step.finish_reason == "tool_use"
        assert [p.name for p in step.tool_calls] == ["search"]

    def test_commentary_becomes_its_own_phase_tagged_part(self) -> None:
        state, _ = _fold(
            [
                {"type": "text", "text": "thinking...", "phase": "commentary"},
                {"type": "text", "text": "done"},
            ]
        )
        step = build_step_completion(state, "m", 1.0)
        assert [(p.text, p.phase) for p in step.parts] == [
            ("thinking...", "commentary"),
            ("done", "final_answer"),
        ]

    def test_without_commentary_no_phase_is_invented(self) -> None:
        state, _ = _fold([{"type": "text", "text": "done"}])
        step = build_step_completion(state, "m", 1.0)
        assert step.parts[0].phase is None


class TestTheEventsAConsumerSees:
    def test_tool_starts_carry_the_arguments(self) -> None:
        events = tool_call_start_events(2, [{"id": "c1", "name": "s", "arguments": {"q": "x"}}])
        assert events == [
            {
                "type": "tool_call_start",
                "step": 2,
                "callId": "c1",
                "toolName": "s",
                "arguments": {"q": "x"},
            }
        ]

    def test_tool_ends_carry_what_the_call_cost(self) -> None:
        report = type("R", (), {"call_id": "c1", "latency_ms": 7.5})()
        assert tool_call_end_events(2, [report]) == [
            {"type": "tool_call_end", "step": 2, "callId": "c1", "latencyMs": 7.5}
        ]

# -- the whole run, through AgentLoop.stream() -------------------------------


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


class StreamingClient:
    """A scripted client that streams, one script per step."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def __init__(self, *scripts: list[dict[str, Any]]) -> None:
        self.scripts = list(scripts)
        self.seen: list[list[dict[str, Any]]] = []

    def stream(self, messages: Any, options: Any = None) -> Any:
        self.seen.append([dict(m) for m in messages])
        script = self.scripts[min(len(self.seen) - 1, len(self.scripts) - 1)]
        yield from script

    def destroy(self) -> None:
        pass


ASKS_FOR_A_TOOL: list[dict[str, Any]] = [
    {"type": "text", "text": "let me check "},
    {"type": "tool_call_start", "id": "c1", "name": "get_weather"},
    {"type": "tool_call_delta", "id": "c1", "arguments": '{"city":"Berlin"}'},
    {"type": "tool_call_end", "id": "c1"},
    {"type": "usage", "usage": Usage(input_tokens=10, output_tokens=5)},
    {"type": "done", "finishReason": "tool_use"},
]

ANSWERS: list[dict[str, Any]] = [
    {"type": "text", "text": "It is sunny in Berlin."},
    {"type": "usage", "usage": Usage(input_tokens=20, output_tokens=6)},
    {"type": "done", "finishReason": "stop"},
]


class TestTheWholeRun:
    def _run(self) -> tuple[list[dict[str, Any]], Any]:
        client = StreamingClient(ASKS_FOR_A_TOOL, ANSWERS)
        loop = AgentLoop(client, tools=[get_weather])
        return list(loop.stream("what is the weather in Berlin?")), loop

    def test_the_run_reaches_an_answer_through_a_tool(self) -> None:
        events, _ = self._run()
        assert events[-1]["type"] == "done"
        assert events[-1]["response"].text == "It is sunny in Berlin."

    def test_the_event_sequence_brackets_each_step_and_each_tool(self) -> None:
        events, _ = self._run()
        kinds = [e["type"] for e in events]
        assert kinds[0] == "step_start"
        assert kinds.count("step_start") == 2
        assert kinds.count("step_end") == 2
        # The tool is announced once it is fully known, and closed once it ran.
        assert kinds.index("tool_call_start") > kinds.index("step_end")
        assert kinds.index("tool_call_end") > kinds.index("tool_call_start")
        assert kinds[-1] == "done"

    def test_the_text_arrives_as_deltas_before_the_answer_exists(self) -> None:
        events, _ = self._run()
        text = [e["text"] for e in events if e["type"] == "text"]
        assert text == ["let me check ", "It is sunny in Berlin."]

    def test_the_tool_actually_ran_and_its_result_went_back(self) -> None:
        _, loop = self._run()
        roles = [e.message["role"] for e in loop.history]
        assert roles == ["user", "assistant", "tool", "assistant"]
        tool_turn = loop.history.at(2)
        assert tool_turn is not None
        assert "sunny in Berlin" in str(tool_turn.message["content"])

    def test_the_second_step_was_sent_the_first_ones_result(self) -> None:
        _, loop = self._run()
        second = loop.client.seen[1]
        assert [m["role"] for m in second] == ["user", "assistant", "tool"]

    def test_done_carries_the_RUN_total_not_the_last_step(self) -> None:
        # A caller billing on the final usage after a two-step run would
        # otherwise be told about half of it.
        events, _ = self._run()
        usage = events[-1]["response"].usage
        assert usage.input_tokens == 30
        assert usage.output_tokens == 11

    def test_the_tool_call_start_carries_its_parsed_arguments(self) -> None:
        events, _ = self._run()
        start = next(e for e in events if e["type"] == "tool_call_start")
        assert start["toolName"] == "get_weather"
        assert start["arguments"] == {"city": "Berlin"}

    def test_the_step_report_records_both_steps(self) -> None:
        _, loop = self._run()
        report = loop.last_report
        assert report is not None
        assert report.step_count == 2
        assert report.tool_call_count == 1
        assert report.reason == "done"

    def test_a_tool_the_provider_never_closed_still_runs(self) -> None:
        # Mutation-driven. The helper was tested directly, but nothing proved the
        # LOOP calls it -- every script here sent a tool_call_end. Anthropic and
        # OpenAI both stream calls that end only when the message does, and
        # without the finalize step the call is dropped, the model is answered as
        # though it never asked, and the run looks like a clean one-step answer.
        never_ends: list[dict[str, Any]] = [
            {"type": "tool_call_start", "id": "c1", "name": "get_weather"},
            {"type": "tool_call_delta", "id": "c1", "arguments": '{"city":"Berlin"}'},
            {"type": "usage", "usage": Usage(input_tokens=10, output_tokens=5)},
            {"type": "done", "finishReason": "tool_use"},
        ]
        client = StreamingClient(never_ends, ANSWERS)
        loop = AgentLoop(client, tools=[get_weather])
        events = list(loop.stream("what is the weather in Berlin?"))

        assert [e["type"] for e in events].count("tool_call_start") == 1
        assert [e.message["role"] for e in loop.history] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        tool_turn = loop.history.at(2)
        assert tool_turn is not None
        assert "sunny in Berlin" in str(tool_turn.message["content"])

    def test_a_run_with_no_tools_is_one_step(self) -> None:
        client = StreamingClient(ANSWERS)
        loop = AgentLoop(client)
        kinds = [e["type"] for e in loop.stream("hi")]
        assert kinds.count("step_start") == 1
        assert "tool_call_start" not in kinds

    def test_max_steps_stops_a_model_that_keeps_calling(self) -> None:
        # The script never stops asking for the tool, so only the ceiling ends it.
        client = StreamingClient(ASKS_FOR_A_TOOL)
        loop = AgentLoop(client, tools=[get_weather], max_steps=3)
        events = list(loop.stream("go"))
        assert events[-1]["type"] == "done"
        report = loop.last_report
        assert report is not None
        assert report.reason == "max_steps"
        assert report.step_count == 3
