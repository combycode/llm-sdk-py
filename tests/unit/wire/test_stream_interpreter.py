"""The streaming driver, primitive by primitive.

Transposed from `unified-library-ts/tests/unit/wire/stream-interpreter.test.ts`.

What separates it from the buffered interpreter is exactly two things, and both
are pinned here: `out` survives across events, and `events` is drained after
each one.

The mini-spec below is deliberately shaped like the hardest real case --
Anthropic's `server_tool_use`, whose input JSON arrives in fragments, is parsed
when its block closes, and is then paired with the result block that completes
it. If the driver can carry that, it can carry the rest.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk.wire.interpreter import Ctx, Registry
from combycode_llm_sdk.wire.stream_interpreter import create_stream_builder


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _raw(ctx: Ctx) -> dict[str, Any]:
    raw: dict[str, Any] = ctx.req["raw"]
    return raw


def _open_tool(ctx: Ctx) -> None:
    _out(ctx)["current"] = {"id": _raw(ctx).get("id"), "json": ""}


def _fragment(ctx: Ctx) -> None:
    """A fragment either feeds the open block or is emitted as a plain delta."""
    out = _out(ctx)
    piece = _raw(ctx).get("partial") or ""
    if out["current"]:
        out["current"]["json"] += piece
        return
    out["events"].append({"type": "tool_call_delta", "arguments": piece})


def _close_tool(ctx: Ctx) -> None:
    out = _out(ctx)
    if not out["current"]:
        return
    out["pending"][out["current"]["id"]] = json.loads(out["current"]["json"] or "{}")
    out["current"] = None


def _pair(ctx: Ctx) -> None:
    """Pair the result with the input collected earlier."""
    out = _out(ctx)
    call_id = _raw(ctx).get("for")
    out["events"].append(
        {"type": "builtin_tool_end", "id": call_id, "input": out["pending"].pop(call_id, None)}
    )


REG = Registry(
    predicates={"accumulating": lambda ctx: _out(ctx)["current"] is not None},
    effects={
        "openTool": _open_tool,
        "fragment": _fragment,
        "closeTool": _close_tool,
        "pair": _pair,
    },
)

SPEC: dict[str, Any] = {
    "id": "test/stream",
    "state": {
        "current": {"kind": "scalar", "default": None},
        "pending": {"kind": "scalar", "default": {}},
        "seen": {"kind": "array"},
    },
    "on": [
        {"when": {"eq": ["event.name", "ping"]}, "stop": True},
        {
            "match": "type",
            "cases": {
                "text": {"emit": "events", "as": {"type": "text", "text": {"$": "raw.text"}}},
                "open": {"effect": "openTool"},
                "fragment": {"effect": "fragment"},
                "close": {"effect": "closeTool"},
                "result": {"effect": "pair"},
                "counted": {"emit": "seen", "as": {"$": "@id"}},
            },
        },
    ],
}


def sse(data: Any, name: str | None = None) -> dict[str, Any]:
    payload = data if isinstance(data, str) else json.dumps(data)
    return {"data": payload, **({"event": name} if name else {})}


class TestPerEventOutput:
    def test_returns_only_what_this_event_produced(self) -> None:
        parse = create_stream_builder(SPEC, REG)
        assert parse(sse({"type": "text", "text": "a"})) == [{"type": "text", "text": "a"}]
        # Drained: the previous event's output must not come back a second time.
        assert parse(sse({"type": "text", "text": "b"})) == [{"type": "text", "text": "b"}]

    def test_returns_nothing_for_an_event_that_maps_to_nothing(self) -> None:
        parse = create_stream_builder(SPEC, REG)
        assert parse(sse({"type": "something_new_next_year"})) == []

    def test_a_stop_rule_EMITS_first_and_only_then_stops(self) -> None:
        # `stop` runs AFTER the rule body, so it means "handle this and go no
        # further" -- which is the early `return events` in OpenAI's moderation
        # and usage-only chunks. Anthropic's ping needs only the empty form, so a
        # spec with a body-less stop rule cannot tell the two orderings apart:
        # this case exists because a mutation moving `stop` before the body
        # passed the whole suite.
        spec = {
            "id": "test/stop-emits",
            "on": [
                {
                    "when": {"truthy": "raw.moderation"},
                    "stop": True,
                    "default": {"emit": "events", "as": {"type": "moderation"}},
                },
                {"match": "type", "cases": {"text": {"emit": "events", "as": {"type": "text"}}}},
            ],
        }
        parse = create_stream_builder(spec, REG)
        # The stop rule's own output is kept, and the later rule never runs.
        assert parse(sse({"moderation": {"flagged": False}, "type": "text"})) == [
            {"type": "moderation"}
        ]
        # Without the guard, the later rule is reached normally.
        assert parse(sse({"type": "text"})) == [{"type": "text"}]

    def test_stops_on_a_guarded_rule_without_falling_through(self) -> None:
        parse = create_stream_builder(SPEC, REG)
        # A ping whose payload would otherwise match `text`.
        assert parse(sse({"type": "text", "text": "nope"}, "ping")) == []


class TestStateSurvivesAcrossEvents:
    def test_carries_an_accumulator_from_one_event_to_the_next(self) -> None:
        parse = create_stream_builder(SPEC, REG)
        parse(sse({"type": "counted", "id": "a"}))
        parse(sse({"type": "counted", "id": "b"}))
        parse(sse({"type": "counted", "id": "c"}))
        # `seen` is state, not output, so it never appears in a return value.
        assert parse(sse({"type": "text", "text": "x"})) == [{"type": "text", "text": "x"}]

    def test_accumulates_fragments_then_pairs_them_with_a_later_event(self) -> None:
        # The whole reason this driver exists.
        parse = create_stream_builder(SPEC, REG)
        assert parse(sse({"type": "open", "id": "t1"})) == []
        assert parse(sse({"type": "fragment", "partial": '{"q":'})) == []
        assert parse(sse({"type": "fragment", "partial": '"hi"}'})) == []
        assert parse(sse({"type": "close"})) == []
        assert parse(sse({"type": "result", "for": "t1"})) == [
            {"type": "builtin_tool_end", "id": "t1", "input": {"q": "hi"}}
        ]

    def test_emits_a_fragment_as_a_delta_when_no_block_is_open(self) -> None:
        # Same event type, opposite behaviour, decided purely by carried state.
        parse = create_stream_builder(SPEC, REG)
        assert parse(sse({"type": "fragment", "partial": "abc"})) == [
            {"type": "tool_call_delta", "arguments": "abc"}
        ]

    def test_gives_each_stream_its_own_state(self) -> None:
        # Two conversations must not share a pending map. This is the case that
        # caught the shared-by-reference default on the TypeScript side.
        a = create_stream_builder(SPEC, REG)
        b = create_stream_builder(SPEC, REG)
        a(sse({"type": "open", "id": "t1"}))
        a(sse({"type": "fragment", "partial": '{"q":1}'}))
        a(sse({"type": "close"}))
        assert b(sse({"type": "result", "for": "t1"})) == [
            {"type": "builtin_tool_end", "id": "t1", "input": None}
        ]


class TestTheSseEnvelope:
    def test_survives_a_payload_that_is_not_json(self) -> None:
        parse = create_stream_builder(SPEC, REG)
        assert parse(sse("[DONE]")) == []

    def test_lets_a_spec_match_on_the_raw_payload_text(self) -> None:
        spec = {
            "id": "test/sentinel",
            "on": [
                {
                    "when": {"eq": ["event.data", "[DONE]"]},
                    "match": "nothing",
                    "default": {"emit": "events", "as": {"type": "done"}},
                }
            ],
        }
        parse = create_stream_builder(spec, REG)
        assert parse(sse("[DONE]")) == [{"type": "done"}]
        assert parse(sse({"type": "text"})) == []


class TestGuardRails:
    def test_refuses_a_spec_that_declares_the_reserved_accumulator(self) -> None:
        with pytest.raises(ValueError, match='"events" is reserved'):
            create_stream_builder({"id": "test/bad", "state": {"events": {"kind": "array"}}}, REG)

    def test_refuses_an_effect_name_nothing_registers(self) -> None:
        spec = {"id": "test/bad-effect", "on": [{"match": "type", "cases": {"text": {"effect": "nope"}}}]}
        with pytest.raises(ValueError, match='unknown effect "nope"'):
            create_stream_builder(spec, REG)(sse({"type": "text"}))

    def test_refuses_an_emit_into_an_accumulator_nobody_declared(self) -> None:
        spec = {"id": "test/bad-emit", "on": [{"match": "type", "cases": {"text": {"emit": "typo", "as": 1}}}]}
        with pytest.raises(ValueError, match='undeclared accumulator "typo"'):
            create_stream_builder(spec, REG)(sse({"type": "text"}))


class TestEachAndMatchList:
    """The two additions Google forced, pinned separately."""

    def test_each_applies_the_rule_to_every_element(self) -> None:
        spec = {
            "id": "test/each",
            "on": [
                {
                    "each": "raw.parts",
                    "default": {"emit": "events", "as": {"type": "text", "text": {"$": "@t"}}},
                }
            ],
        }
        parse = create_stream_builder(spec, REG)
        assert parse(sse({"parts": [{"t": "a"}, {"t": "b"}]})) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_each_over_a_missing_source_is_not_an_error(self) -> None:
        spec = {
            "id": "test/each",
            "on": [{"each": "raw.parts", "default": {"emit": "events", "as": {"type": "text"}}}],
        }
        assert create_stream_builder(spec, REG)(sse({"other": 1})) == []

    def test_a_match_list_takes_the_first_path_that_is_present(self) -> None:
        spec = {
            "id": "test/match-list",
            "on": [
                {
                    "match": ["event_type", "type"],
                    "cases": {"a": {"emit": "events", "as": {"hit": "a"}}},
                }
            ],
        }
        parse = create_stream_builder(spec, REG)
        # `event_type` wins when present...
        assert parse(sse({"event_type": "a", "type": "b"})) == [{"hit": "a"}]
        # ...and `type` is the fallback when it is not.
        assert parse(sse({"type": "a"})) == [{"hit": "a"}]
