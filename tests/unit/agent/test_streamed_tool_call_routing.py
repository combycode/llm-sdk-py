"""A streamed tool call's arguments belong to THAT call.

Transposed from
`unified-library-ts/tests/unit/agent/streamed-tool-call-routing.test.ts`.

The accumulator used to route an unmatched delta to the FIRST call in flight.
Google's stream registry emitted `id: ""` on every delta and every end (it had
`functionCall.id` in hand and simply did not pass it on), so with two function
calls in one response:

* ``read_file`` got ``{"path":"/a"}{"path":"/b"}`` -- both payloads concatenated,
  unparseable, refused as malformed. Visibly broken.
* ``delete_file`` got NOTHING. An empty args string is deliberately read as a
  genuine no-argument call, so it was not marked malformed -- it EXECUTED, with
  ``{}``.

The second line is why this matters: the model asked to delete ``/b`` and the
tool ran with no arguments at all. Exactly the failure ``_parse_accum`` exists to
prevent, arriving through a different door.

Fixed in both places: Google now carries its id on all three events, and the
accumulator no longer routes an unmatched delta to an arbitrary call.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.agent.loop_stream import (
    StepState,
    accumulate_stream_event,
    finalize_unended_tool_calls,
)


def run(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state = StepState()
    for event in events:
        accumulate_stream_event(event, state)
    finalize_unended_tool_calls(state)
    return list(state.tool_calls)


def start(call_id: str, name: str) -> dict[str, Any]:
    return {"type": "tool_call_start", "id": call_id, "name": name}


def delta(call_id: str, args: str) -> dict[str, Any]:
    return {"type": "tool_call_delta", "id": call_id, "arguments": args}


def end(call_id: str) -> dict[str, Any]:
    return {"type": "tool_call_end", "id": call_id}


class TestTwoParallelCallsKeepTheirOwnArguments:
    def test_when_every_event_carries_its_id(self) -> None:
        calls = run(
            [
                start("call_a", "read_file"),
                delta("call_a", '{"path":"/a"}'),
                end("call_a"),
                start("call_b", "delete_file"),
                delta("call_b", '{"path":"/b"}'),
                end("call_b"),
            ]
        )
        assert [(c["name"], c["arguments"]) for c in calls] == [
            ("read_file", {"path": "/a"}),
            ("delete_file", {"path": "/b"}),
        ]

    def test_and_when_the_deltas_carry_no_id_as_google_used_to_emit(self) -> None:
        # The regression case. `delete_file` must get `/b`, not an empty object.
        calls = run(
            [
                start("call_a", "read_file"),
                delta("", '{"path":"/a"}'),
                end(""),
                start("call_b", "delete_file"),
                delta("", '{"path":"/b"}'),
                end(""),
            ]
        )
        assert [(c["name"], c["arguments"]) for c in calls] == [
            ("read_file", {"path": "/a"}),
            ("delete_file", {"path": "/b"}),
        ]
        assert not any(c.get("malformed") for c in calls)

    def test_interleaved_which_is_what_a_real_parallel_stream_looks_like(self) -> None:
        calls = run(
            [
                start("call_a", "read_file"),
                start("call_b", "delete_file"),
                delta("call_a", '{"pa'),
                delta("call_b", '{"pa'),
                delta("call_a", 'th":"/a"}'),
                delta("call_b", 'th":"/b"}'),
                end("call_a"),
                end("call_b"),
            ]
        )
        assert [c["arguments"] for c in calls] == [{"path": "/a"}, {"path": "/b"}]


class TestAnEndIsHonouredOnce:
    def test_a_repeated_end_does_not_push_the_call_twice(self) -> None:
        # Pushing twice runs the tool twice -- for anything destructive that is
        # the whole cost of a duplicated event.
        calls = run(
            [
                start("call_a", "delete_file"),
                delta("call_a", '{"path":"/a"}'),
                end("call_a"),
                end("call_a"),
            ]
        )
        assert len(calls) == 1

    def test_nor_an_id_less_end_landing_on_an_already_pushed_call(self) -> None:
        calls = run([start("call_a", "delete_file"), delta("call_a", "{}"), end("call_a"), end("")])
        assert len(calls) == 1


class TestASingleCallStillWorksEveryWayItUsedTo:
    def test_with_ids_throughout(self) -> None:
        assert run([start("c", "f"), delta("c", '{"x":1}'), end("c")])[0]["arguments"] == {"x": 1}

    def test_with_no_ids_on_the_delta_or_end(self) -> None:
        assert run([start("c", "f"), delta("", '{"x":1}'), end("")])[0]["arguments"] == {"x": 1}

    def test_with_no_end_at_all_finalize_still_emits_it(self) -> None:
        assert run([start("c", "f"), delta("c", '{"x":1}')])[0]["arguments"] == {"x": 1}

    def test_a_truncated_argument_string_is_refused_not_guessed_at(self) -> None:
        call = run([start("c", "delete_files"), delta("c", '{"path": "/et'), end("c")])[0]
        assert call.get("malformed") is True
        assert call["arguments"] == {}


class TestTheGoogleStreamRegistryCarriesItsId:
    def test_on_the_delta_and_the_end_not_only_the_start(self) -> None:
        # The root cause: it had `functionCall.id` and passed "" on twice.
        from combycode_llm_sdk.llm.providers.google.stream_registry import (
            GOOGLE_STREAM_REGISTRY,
        )

        out: dict[str, Any] = {"events": []}
        ctx = type(
            "Ctx",
            (),
            {
                "req": {"out": out},
                "item": type("Item", (), {"value": {
                    "functionCall": {"id": "fc_1", "name": "read_file", "args": {"path": "/a"}}
                }})(),
            },
        )()
        GOOGLE_STREAM_REGISTRY.effects["googleStreamToolCall"](ctx)

        assert [(e["type"], e["id"]) for e in out["events"]] == [
            ("tool_call_start", "fc_1"),
            ("tool_call_delta", "fc_1"),
            ("tool_call_end", "fc_1"),
        ]
