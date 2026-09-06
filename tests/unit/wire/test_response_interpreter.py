"""The response interpreter, primitive by primitive.

Transposed from `unified-library-ts/tests/unit/wire/response-interpreter.test.ts`,
case for case. These are about the MACHINE, not about any provider: a bug here
is a bug in all seven parsers at once, so it is pinned separately from the
differential that replays real bodies.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk.wire.interpreter import Ctx, Registry
from combycode_llm_sdk.wire.response_interpreter import build_response


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _item(ctx: Ctx) -> dict[str, Any]:
    return ctx.item.value if ctx.item else {}


def _join_text(_arg: Any, ctx: Ctx) -> str:
    """The shape a real spec uses for `text`: a fold over collected content."""
    return "".join(p.get("text", "") for p in _out(ctx)["content"] if p.get("type") == "text")


def _attach_output(ctx: Ctx) -> None:
    """Attach a result block's stdout to the call it names."""
    b = _item(ctx)
    for call in _out(ctx)["toolCalls"]:
        if call.get("id") == b.get("tool_use_id") and isinstance(b.get("stdout"), str):
            call["output"] = b["stdout"]


REG = Registry(
    transforms={
        "upper": lambda arg, _ctx: str(arg).upper(),
        "parseJson": lambda arg, _ctx: json.loads(str(arg)),
        # A block yielding SEVERAL values, for `concat`.
        "twoFiles": lambda _arg, _ctx: [{"id": "a"}, {"id": "b"}],
        "joinText": _join_text,
    },
    predicates={"hasTools": lambda ctx: len(_out(ctx)["toolCalls"]) > 0},
    effects={"attachOutput": _attach_output},
)

#: An Anthropic-shaped spec, small enough to read in one screen.
SPEC: dict[str, Any] = {
    "id": "test/messages.response",
    "accumulators": {
        "content": {"kind": "array"},
        "toolCalls": {"kind": "array"},
        "files": {"kind": "array", "omitEmpty": True},
        "thinking": {"kind": "scalar", "default": None},
    },
    "fields": [
        {"to": "id", "from": "raw.id"},
        {"to": "model", "from": "raw.model"},
    ],
    "collect": [
        {
            "from": "raw.content",
            "match": "type",
            "cases": {
                "text": {"emit": "content", "as": {"type": "text", "text": {"$": "@text"}}},
                "thinking": {"emit": "thinking", "mode": "scalar", "as": {"$": "@thinking"}},
                "tool_use": {
                    "emit": ["content", "toolCalls"],
                    "as": {
                        "type": "tool_call",
                        "id": {"$": "@id"},
                        "name": {"$": "@name"},
                        "arguments": {"$": "@input"},
                    },
                },
                "file": {
                    "emit": "files",
                    "when": {"itemTruthy": "file_id"},
                    "as": {"id": {"$": "@file_id"}},
                },
            },
        }
    ],
    "derive": {"finishReason": {"$table": "finish", "$key": "raw.stop_reason", "$default": "stop"}},
    "tables": {"finish": {"max_tokens": "length", "refusal": "content_filter"}},
}


def body(content: list[Any], **extra: Any) -> dict[str, Any]:
    return {"id": "msg_1", "model": "claude-test", "stop_reason": "end_turn", "content": content, **extra}


class TestFields:
    def test_copies_scalars_from_the_raw_body(self) -> None:
        r = build_response(SPEC, body([]), REG)
        assert r["id"] == "msg_1"
        assert r["model"] == "claude-test"

    def test_merges_extra_for_what_the_spec_cannot_know(self) -> None:
        r = build_response(SPEC, body([]), REG, extra={"latencyMs": 42})
        assert r["latencyMs"] == 42


class TestCollect:
    def test_routes_each_block_by_its_discriminator(self) -> None:
        r = build_response(
            SPEC,
            body([{"type": "text", "text": "hello"}, {"type": "thinking", "thinking": "hmm"}]),
            REG,
        )
        assert r["content"] == [{"type": "text", "text": "hello"}]
        assert r["thinking"] == "hmm"

    def test_puts_the_SAME_object_in_every_accumulator_it_names(self) -> None:
        # Load-bearing: the hand-written adapters push one object into both
        # arrays, so a consumer mutating response.toolCalls[0] sees it in
        # content too. Copies would pass a deep-equality test and change that.
        r = build_response(
            SPEC, body([{"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}]), REG
        )
        assert r["toolCalls"][0] is r["content"][0]

    def test_a_scalar_emit_is_last_write_wins(self) -> None:
        r = build_response(
            SPEC,
            body([{"type": "thinking", "thinking": "first"}, {"type": "thinking", "thinking": "second"}]),
            REG,
        )
        assert r["thinking"] == "second"

    def test_honours_a_when_guard_on_top_of_the_discriminator(self) -> None:
        r = build_response(SPEC, body([{"type": "file", "file_id": "f_1"}, {"type": "file"}]), REG)
        assert r["files"] == [{"id": "f_1"}]

    def test_ignores_a_block_type_nothing_declares(self) -> None:
        # Providers add block types continuously. An unknown one must not take
        # the whole response down.
        r = build_response(
            SPEC, body([{"type": "some_future_thing", "x": 1}, {"type": "text", "text": "ok"}]), REG
        )
        assert r["content"] == [{"type": "text", "text": "ok"}]

    def test_survives_a_missing_or_non_array_source(self) -> None:
        assert build_response(SPEC, {"id": "x"}, REG)["content"] == []
        assert build_response(SPEC, body("not a list"), REG)["content"] == []  # type: ignore[arg-type]

    def test_concat_splices_an_array_in_where_push_would_nest_it(self) -> None:
        spec = {
            **SPEC,
            "collect": [
                {
                    "from": "raw.content",
                    "match": "type",
                    "cases": {
                        "file": {"emit": "files", "mode": "concat", "as": {"$call": "twoFiles"}}
                    },
                }
            ],
        }
        assert build_response(spec, body([{"type": "file"}]), REG)["files"] == [
            {"id": "a"},
            {"id": "b"},
        ]

    def test_concat_refuses_a_non_array(self) -> None:
        spec = {
            **SPEC,
            "collect": [
                {
                    "from": "raw.content",
                    "match": "type",
                    "cases": {"file": {"emit": "files", "mode": "concat", "as": {"notAn": "array"}}},
                }
            ],
        }
        with pytest.raises(TypeError, match='concat into "files"'):
            build_response(spec, body([{"type": "file"}]), REG)

    def test_an_effect_can_modify_something_already_collected(self) -> None:
        spec = {
            **SPEC,
            "collect": [
                {
                    "from": "raw.content",
                    "match": "type",
                    "cases": {
                        "tool_use": {
                            "emit": ["content", "toolCalls"],
                            "as": {"type": "tool_call", "id": {"$": "@id"}, "name": {"$": "@name"}},
                        },
                        "result": {"effect": "attachOutput"},
                    },
                }
            ],
        }
        r = build_response(
            spec,
            body(
                [
                    {"type": "tool_use", "id": "t1", "name": "f"},
                    {"type": "result", "tool_use_id": "t1", "stdout": "done"},
                ]
            ),
            REG,
        )
        assert r["toolCalls"][0]["output"] == "done"

    def test_rejects_an_effect_name_nothing_registers(self) -> None:
        spec = {
            **SPEC,
            "collect": [{"from": "raw.content", "match": "type", "cases": {"text": {"effect": "nope"}}}],
        }
        with pytest.raises(ValueError, match='unknown effect "nope"'):
            build_response(spec, body([{"type": "text", "text": "x"}]), REG)

    def test_refuses_to_emit_into_an_accumulator_nobody_declared(self) -> None:
        spec = {
            **SPEC,
            "collect": [
                {"from": "raw.content", "match": "type", "cases": {"text": {"emit": "typo", "as": {"a": 1}}}}
            ],
        }
        with pytest.raises(ValueError, match='undeclared accumulator "typo"'):
            build_response(spec, body([{"type": "text", "text": "x"}]), REG)


class TestAccumulatorShape:
    def test_keeps_an_empty_array_but_omits_one_marked_omitEmpty(self) -> None:
        r = build_response(SPEC, body([]), REG)
        assert r["content"] == []
        assert "files" not in r

    def test_gives_a_scalar_its_declared_default(self) -> None:
        assert build_response(SPEC, body([]), REG)["thinking"] is None

    def test_a_default_is_copied_not_shared_between_builds(self) -> None:
        # Python-only pin, and a real bug on the TypeScript side: the default was
        # handed out BY REFERENCE, so every build shared one object. Latent for
        # buffered specs (every declared default is null) and live the moment a
        # spec declares `{}` -- which the first streaming spec does.
        spec = {**SPEC, "accumulators": {**SPEC["accumulators"], "pending": {"kind": "scalar", "default": {}}}}
        first = build_response(spec, body([]), REG)
        first["pending"]["leaked"] = True
        second = build_response(spec, body([]), REG)
        assert second["pending"] == {}


class TestDerive:
    def test_runs_after_collect_and_can_read_what_was_collected(self) -> None:
        # The ordering is the point: `text` is a fold over `content`, which does
        # not exist until every block has been classified.
        spec = {
            **SPEC,
            "derive": {
                **SPEC["derive"],
                "text": {"$call": "joinText"},
                "hadTools": {"$when": {"pred": "hasTools"}, "$value": True},
            },
        }
        r = build_response(
            spec,
            body(
                [
                    {"type": "text", "text": "one "},
                    {"type": "tool_use", "id": "t", "name": "n", "input": {}},
                    {"type": "text", "text": "two"},
                ]
            ),
            REG,
        )
        assert r["text"] == "one two"
        assert r["hadTools"] is True

    def test_a_predicate_reading_out_is_false_when_nothing_was_collected(self) -> None:
        spec = {**SPEC, "derive": {"hadTools": {"$when": {"pred": "hasTools"}, "$value": True}}}
        assert "hadTools" not in build_response(spec, body([{"type": "text", "text": "x"}]), REG)

    def test_maps_through_a_table_with_a_default(self) -> None:
        assert build_response(SPEC, body([], stop_reason="max_tokens"), REG)["finishReason"] == "length"
        assert build_response(SPEC, body([], stop_reason="end_turn"), REG)["finishReason"] == "stop"

    def test_overrides_an_accumulator_of_the_same_name(self) -> None:
        spec = {**SPEC, "derive": {"thinking": "derived wins"}}
        r = build_response(spec, body([{"type": "thinking", "thinking": "collected"}]), REG)
        assert r["thinking"] == "derived wins"
