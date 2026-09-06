"""Translated from `unified-library-ts/tests/unit/llm/moderation.test.ts`.

PARTIAL by design, and named for what it covers. That file spans the whole
inline-moderation feature -- native passthrough + parse, emulated input/output,
the three stream strategies, early-abort, and the missing-key throw -- across
`moderation/native.ts`, `moderation/runner.ts` and `LLMClient`. Only
`moderation/native.ts` is in this port batch, so only its two describes are
translated here (moderation.test.ts:118 and :358). The rest of the file belongs
to the moderation area owner's `test_moderation.py` and is NOT duplicated here.

One transposition: the adapter case at moderation.test.ts:160 is driven through
the etalon `openai/responses` wire spec, whose `moderation` block calls the
ported `nativeModeration` transform. See the header of `test_strict_schema.py`
for why that substitution is the faithful one while the adapters are unported.
"""

from __future__ import annotations

from typing import Any

from combycode_llm_sdk.llm.moderation.native import (
    build_native_moderation,
    parse_native_moderation,
)
from combycode_llm_sdk.llm.wire_transforms import make_registry
from combycode_llm_sdk.wire.inherit import resolve_spec
from combycode_llm_sdk.wire.interpreter import build_from_spec
from combycode_llm_sdk.wire.registry import WIRE_SPECS


def raw_result(flagged: bool) -> dict[str, Any]:
    """moderation.test.ts:23-30."""
    return {
        "type": "moderation_result",
        "flagged": flagged,
        "categories": {"hate": flagged, "violence": False},
        "category_scores": {"hate": 0.97 if flagged else 0.01, "violence": 0.0},
        "category_applied_input_types": {"hate": ["text"]},
        "model": "omni-moderation-latest",
    }


class _MessageBuilderStub:
    """Stands in for the adapter message builder; see test_strict_schema.py."""

    def build_input_items(self, msg: Any, tool_names: Any, notes: Any = None) -> list[Any]:
        return [msg]


def body(request: dict[str, Any]) -> dict[str, Any]:
    return build_from_spec(
        resolve_spec("openai/responses", WIRE_SPECS),
        request,
        make_registry({"openai_responses": _MessageBuilderStub()}),
    ).body


class TestNativeModerationWireHelpers:
    """moderation.test.ts:118."""

    def test_build_defaults_the_model_and_honours_an_override(self) -> None:
        # moderation.test.ts:120-123
        assert build_native_moderation({}) == {"model": "omni-moderation-latest"}
        assert build_native_moderation({"model": "text-moderation-007"}) == {
            "model": "text-moderation-007"
        }

    def test_parses_the_responses_api_shape(self) -> None:
        # moderation.test.ts:127-130
        report = parse_native_moderation({"input": raw_result(False), "output": raw_result(True)})
        assert report is not None
        assert report["source"] == "native"
        assert report["input"]["flagged"] is False
        assert report["output"]["flagged"] is True

    def test_parses_the_chat_completions_shape(self) -> None:
        # moderation.test.ts:134-140
        wrapped = {
            "type": "moderation_results",
            "model": "omni-moderation-latest",
            "results": [raw_result(True)],
        }
        report = parse_native_moderation({"input": wrapped, "output": wrapped})
        assert report is not None
        assert report["input"]["flagged"] is True

    def test_surfaces_a_moderation_error_entry(self) -> None:
        # moderation.test.ts:144-147
        report = parse_native_moderation(
            {"input": {"type": "error", "code": "bad", "message": "moderation unavailable"}}
        )
        assert report is not None
        assert report["input"] == {"error": "moderation unavailable"}

    def test_returns_none_when_there_is_nothing_usable(self) -> None:
        # moderation.test.ts:151-152
        assert parse_native_moderation(None) is None
        assert parse_native_moderation({}) is None

    def test_an_empty_moderation_results_envelope_yields_no_entry(self) -> None:
        # moderation.test.ts:156-157
        empty = {"type": "moderation_results", "model": "m", "results": []}
        assert parse_native_moderation({"input": empty}) is None

    def test_the_openai_responses_surface_emits_body_moderation_natively(self) -> None:
        # moderation.test.ts:161-168
        base = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
        native = body({**base, "moderation": {"model": "omni-moderation-latest"}})
        assert native["moderation"] == {"model": "omni-moderation-latest"}
        emulated = body({**base, "moderation": {"mode": "emulate"}})
        assert "moderation" not in emulated


class TestParseNativeModerationUnrecognisedEntries:
    """moderation.test.ts:358."""

    def test_an_entry_the_sdk_cannot_read_yields_none_rather_than_a_hollow_report(self) -> None:
        # moderation.test.ts:362-366. A report claiming source:'native' with no
        # input/output would be indistinguishable from "checked and clean".
        # Nothing usable must mean nothing returned.
        assert parse_native_moderation({"input": {"type": "something_new"}}) is None
        assert parse_native_moderation({"input": {"type": "moderation_results"}}) is None
        assert (
            parse_native_moderation({"input": {"type": "moderation_results", "results": []}})
            is None
        )
        assert parse_native_moderation({"input": "a string"}) is None
        assert parse_native_moderation({"input": None}) is None

    def test_a_non_object_raw_moderation_field_yields_none(self) -> None:
        # moderation.test.ts:370-372
        assert parse_native_moderation(None) is None
        assert parse_native_moderation("nope") is None

    def test_one_readable_side_is_enough_the_other_is_simply_absent(self) -> None:
        # moderation.test.ts:376-382
        report = parse_native_moderation(
            {"input": raw_result(True), "output": {"type": "something_new"}}
        )
        assert report is not None
        assert report["source"] == "native"
        assert report["input"] is not None
        assert "output" not in report
