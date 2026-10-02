"""The `shell` builtin tool: two items for one call, and a call that is a request.

Transposed from `unified-library-ts/tests/unit/llm/shell-tool.test.ts`.

Every fixture here is a transcription of a live capture (2026-10-02), because the
three things that make this tool awkward are all things a reasonable guess gets
wrong:

- A container-run shell arrives as TWO output items. The `shell_call` completes
  carrying only the commands, and a separate `shell_call_output` follows with
  stdout/stderr, linked by `call_id`. Ending the tool call when the first item
  completes reports a command that printed nothing.
- `container_auto` is not what comes back. OpenAI rewrites it to
  `{"type": "container_reference", "container_id": ...}`.
- A LOCAL shell call is not a finished call at all: the model only asks. The turn
  then ends `finishReason: "stop"` with empty text, which is why it earns a warning
  rather than silence.

xAI is in here too: it supports the tool but names its stream events
`response.shell_call_arguments.*` and sends one JSON string, so the fixtures pin
that it still produces a start and an end from the ITEM alone.
"""

from __future__ import annotations

import json
import sys
from typing import Any, ClassVar

sys.path.insert(0, "src")

from combycode_llm_sdk.llm.providers.openai.parse_helpers import (
    shell_awaits_caller,
    shell_commands,
    shell_environment_name,
    shell_output_text,
)
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter
from combycode_llm_sdk.llm.shell_calls import merge_shell_calls, shell_awaiting_note

CONTAINER = {
    "type": "container_reference",
    "container_id": "cntr_6abf9290ddd881909b1563293d7c0d9609a6aff72865cc09",
}

CALL_ITEM: dict[str, Any] = {
    "id": "sh_0a11752d3fae1343006abf9292145887d1b008b4208991f3f2",
    "type": "shell_call",
    "status": "completed",
    "call_id": "call_NGiUM0QYjkGU1y2HiTOhaZrd",
    "action": {
        "commands": ["echo one", "ls /nonexistent"],
        "max_output_length": None,
        "timeout_ms": None,
    },
    "environment": CONTAINER,
}

STDERR_LINE = "ls: cannot access '/nonexistent': No such file or directory\n"

OUTPUT_ITEM: dict[str, Any] = {
    "id": "sho_0a11752d3fae1343006abf9292b66087d184d376be57f190f6",
    "type": "shell_call_output",
    "status": "completed",
    "call_id": "call_NGiUM0QYjkGU1y2HiTOhaZrd",
    "output": [
        {"outcome": {"type": "exit", "exit_code": 0}, "stdout": "one\n", "stderr": ""},
        {"outcome": {"type": "exit", "exit_code": 2}, "stdout": "", "stderr": STDERR_LINE},
    ],
}

COMBINED = "one\n" + STDERR_LINE


def stream_of(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Feed a whole stream through ONE parser, as a real stream does.

    `create_stream_parser()` rather than `parse_stream_event()`: the latter builds a
    fresh parser per event, so every piece of cross-event state resets. The join
    between a shell call and its output item is exactly that kind of state.
    """
    parse = OpenAIResponsesAdapter({"apiKey": "k"}).create_stream_parser()
    events: list[dict[str, Any]] = []
    for payload in payloads:
        sse = {"event": payload["type"], "data": json.dumps(payload)}
        events.extend(dict(e) for e in parse(sse))
    return events


CONTAINER_STREAM: list[dict[str, Any]] = [
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {**CALL_ITEM, "status": "in_progress", "action": {"commands": []}},
    },
    {
        "type": "response.shell_call_command.added",
        "command": "",
        "command_index": 0,
        "output_index": 0,
    },
    {
        "type": "response.shell_call_command.delta",
        "command_index": 0,
        "delta": "echo",
        "output_index": 0,
    },
    {
        "type": "response.shell_call_command.delta",
        "command_index": 0,
        "delta": " one",
        "output_index": 0,
    },
    {
        "type": "response.shell_call_command.done",
        "command": "echo one",
        "command_index": 0,
        "output_index": 0,
    },
    {"type": "response.output_item.done", "output_index": 0, "item": CALL_ITEM},
    {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {**OUTPUT_ITEM, "status": "in_progress", "output": []},
    },
    {
        "type": "response.shell_call_output_content.delta",
        "command_index": 0,
        "delta": {"stdout": "one\n"},
        "item_id": OUTPUT_ITEM["id"],
        "output_index": 1,
    },
    {
        "type": "response.shell_call_output_content.delta",
        "command_index": 1,
        "delta": {"stderr": STDERR_LINE},
        "item_id": OUTPUT_ITEM["id"],
        "output_index": 1,
    },
    {"type": "response.output_item.done", "output_index": 1, "item": OUTPUT_ITEM},
]


class TestReadingAShellCallItem:
    def test_it_joins_several_commands_into_one_script(self) -> None:
        # They run in order and read as a small script, so `code` is one block.
        assert shell_commands(CALL_ITEM) == "echo one\nls /nonexistent"

    def test_it_calls_a_missing_or_null_environment_local(self) -> None:
        # OpenAI sends `null` for a local call; xAI omits the key entirely. Both
        # mean nobody has run these commands.
        assert shell_environment_name({"type": "shell_call", "environment": None}) == "local"
        assert shell_environment_name({"type": "shell_call"}) == "local"

    def test_it_reports_the_container_the_provider_actually_chose(self) -> None:
        # `container_auto` was the REQUEST; this is what comes back.
        assert shell_environment_name(CALL_ITEM) == "container_reference"

    def test_it_knows_which_calls_are_waiting_on_the_caller(self) -> None:
        assert shell_awaits_caller(CALL_ITEM) is False
        assert shell_awaits_caller({"type": "shell_call", "environment": None}) is True
        assert shell_awaits_caller({"type": "web_search_call"}) is False

    def test_it_keeps_stdout_and_stderr_in_the_order_written(self) -> None:
        # stderr is not dropped: a failing command's only output is usually there,
        # and `output` is a caller's whole view of what happened.
        assert shell_output_text(OUTPUT_ITEM) == COMBINED

    def test_it_skips_the_empty_halves(self) -> None:
        assert shell_output_text({"output": [{"stdout": "", "stderr": ""}]}) == ""


class TestAContainerRunShellStreamed:
    def test_it_announces_the_tool_once_not_once_per_item(self) -> None:
        # The regression this pins: `shell_call_output` maps to a builtin call, so
        # the item-added effect reported `starts=2 ends=1` for a single `echo` until
        # it was told to skip the output half. Caught live, not by review.
        events = stream_of(CONTAINER_STREAM)
        assert len([e for e in events if e["type"] == "builtin_tool_start"]) == 1
        assert len([e for e in events if e["type"] == "builtin_tool_end"]) == 1

    def test_it_streams_the_command_text_as_it_is_composed(self) -> None:
        deltas = [
            e for e in stream_of(CONTAINER_STREAM)
            if e["type"] == "builtin_tool_delta" and e.get("code")
        ]
        assert [e["code"] for e in deltas] == ["echo", " one"]

    def test_it_does_not_re_emit_the_finished_command_on_done(self) -> None:
        # `.added` carries an empty command and `.done` repeats the whole one. A
        # consumer that appends what it is given would get `echo oneecho one`.
        codes = "".join(
            str(e["code"])
            for e in stream_of(CONTAINER_STREAM)
            if e["type"] == "builtin_tool_delta" and e.get("code")
        )
        assert codes == "echo one"

    def test_it_streams_stdout_and_stderr_as_output_deltas(self) -> None:
        deltas = [
            e for e in stream_of(CONTAINER_STREAM)
            if e["type"] == "builtin_tool_delta" and e.get("output")
        ]
        # `delta` is a MAPPING for these events -- the one shape in the group that
        # is not a plain string, and the reason this has its own effect.
        assert [e["output"] for e in deltas] == ["one\n", STDERR_LINE]

    def test_it_ends_only_when_the_output_item_lands_carrying_both_halves(self) -> None:
        end = next(e for e in stream_of(CONTAINER_STREAM) if e["type"] == "builtin_tool_end")
        assert end == {
            "type": "builtin_tool_end",
            "tool": "shell",
            "id": CALL_ITEM["id"],
            "code": "echo one\nls /nonexistent",
            "callId": CALL_ITEM["call_id"],
            "environment": "container_reference",
            "output": COMBINED,
        }

    def test_it_does_not_end_when_the_call_item_completes(self) -> None:
        # Ending here would report a shell command that printed nothing.
        events = stream_of(CONTAINER_STREAM[:6])
        assert [e for e in events if e["type"] == "builtin_tool_end"] == []


class TestALocalShellCallStreamed:
    LOCAL_ITEM: ClassVar[dict[str, Any]] = {
        "id": "sh_local",
        "type": "shell_call",
        "status": "completed",
        "call_id": "call_local",
        "action": {"commands": ["echo one"]},
        "environment": None,
    }

    def test_it_ends_as_soon_as_the_call_item_completes(self) -> None:
        events = stream_of(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**self.LOCAL_ITEM, "status": "in_progress"},
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": self.LOCAL_ITEM,
                },
            ]
        )
        end = next(e for e in events if e["type"] == "builtin_tool_end")
        assert end["code"] == "echo one"
        assert end["environment"] == "local"
        # No output will ever come, so claiming one would be a lie.
        assert "output" not in end


class TestXaiWhichStreamsItDifferently:
    # Measured: `response.shell_call_arguments.delta` carries
    # `{"commands":["echo one","echo two"]}` -- a JSON string, not a fragment of
    # displayable command text. Nothing is emitted for it; the item carries the
    # finished commands anyway.
    XAI_ITEM: ClassVar[dict[str, Any]] = {
        "id": "sc_2017f735-2333-9442-bd07-504c316d3040_0",
        "type": "shell_call",
        "status": "completed",
        "call_id": "call-b06e2b98-d48c-47a1-9883-c5259a45e0fa-0",
        "action": {"commands": ["echo one", "echo two"], "type": "exec"},
    }
    ARGS = '{"commands":["echo one","echo two"]}'

    def events(self) -> list[dict[str, Any]]:
        return stream_of(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        **self.XAI_ITEM,
                        "status": "in_progress",
                        "action": {"commands": [], "type": "exec"},
                    },
                },
                {
                    "type": "response.shell_call_arguments.delta",
                    "delta": self.ARGS,
                    "item_id": self.XAI_ITEM["id"],
                    "output_index": 0,
                },
                {
                    "type": "response.shell_call_arguments.done",
                    "arguments": self.ARGS,
                    "item_id": self.XAI_ITEM["id"],
                    "output_index": 0,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": self.XAI_ITEM,
                },
            ]
        )

    def test_it_still_reports_the_call_from_the_item_alone(self) -> None:
        events = self.events()
        assert len([e for e in events if e["type"] == "builtin_tool_start"]) == 1
        end = next(e for e in events if e["type"] == "builtin_tool_end")
        assert end["code"] == "echo one\necho two"
        assert end["environment"] == "local"

    def test_it_invents_no_progress_deltas_out_of_arguments_json(self) -> None:
        assert [e for e in self.events() if e["type"] == "builtin_tool_delta"] == []


class TestMergingTheTwoHalvesOfABufferedResponse:
    CALL: ClassVar[dict[str, Any]] = {
        "tool": "shell",
        "id": CALL_ITEM["id"],
        "code": "echo one",
        "callId": "call_1",
        "environment": "container_reference",
    }
    OUTPUT: ClassVar[dict[str, Any]] = {
        "tool": "shell",
        "id": OUTPUT_ITEM["id"],
        "callId": "call_1",
        "output": "one\n",
    }

    def test_it_reports_one_call_not_two_halves(self) -> None:
        merged = merge_shell_calls([self.CALL, self.OUTPUT])
        assert len(merged) == 1
        assert merged[0]["code"] == "echo one"
        assert merged[0]["output"] == "one\n"
        assert merged[0]["environment"] == "container_reference"

    def test_it_pairs_by_call_id_not_by_position(self) -> None:
        # Two interleaved shell calls: position would pair the wrong halves.
        merged = merge_shell_calls(
            [
                {"tool": "shell", "code": "first", "callId": "a"},
                {"tool": "shell", "code": "second", "callId": "b"},
                {"tool": "shell", "callId": "b", "output": "B"},
                {"tool": "shell", "callId": "a", "output": "A"},
            ]
        )
        assert [(c.get("code"), c.get("output")) for c in merged] == [
            ("first", "A"),
            ("second", "B"),
        ]

    def test_it_keeps_an_output_whose_call_never_arrived(self) -> None:
        # Dropping it would hide output that really happened.
        assert len(merge_shell_calls([self.OUTPUT])) == 1

    def test_it_leaves_other_tools_untouched(self) -> None:
        web = {"tool": "web_search", "query": "x"}
        assert len(merge_shell_calls([web, web])) == 2

    def test_the_output_half_does_not_overwrite_the_call_half(self) -> None:
        merged = merge_shell_calls([self.CALL, {**self.OUTPUT, "code": "something else"}])
        assert merged[0]["code"] == "echo one"

    def test_it_does_not_mutate_what_it_was_given(self) -> None:
        # The caller's list is theirs; merging is not a side effect on the parse.
        call = dict(self.CALL)
        merge_shell_calls([call, self.OUTPUT])
        assert "output" not in call


class TestTheWarningALocalCallEarns:
    def test_it_fires_for_a_call_the_caller_has_to_run(self) -> None:
        note = shell_awaiting_note(
            [{"tool": "shell", "environment": "local", "code": "echo one"}]
        )
        assert note is not None
        assert "waiting on you" in note
        # It must name what to run and how to answer, or it is just an alarm.
        assert "echo one" in note
        assert "callId" in note
        assert "container_auto" in note

    def test_it_stays_silent_when_the_provider_ran_the_commands(self) -> None:
        assert (
            shell_awaiting_note(
                [{"tool": "shell", "environment": "container_reference", "output": "one\n"}]
            )
            is None
        )

    def test_it_stays_silent_for_every_other_tool(self) -> None:
        assert shell_awaiting_note([{"tool": "code_interpreter", "code": "print(1)"}]) is None
        assert shell_awaiting_note(None) is None

    def test_it_counts_them_when_the_model_asked_for_several(self) -> None:
        note = shell_awaiting_note(
            [
                {"tool": "shell", "environment": "local", "code": "a"},
                {"tool": "shell", "environment": "local", "code": "b"},
            ]
        )
        assert note is not None
        assert "2 shell commands" in note
