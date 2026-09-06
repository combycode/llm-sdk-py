"""Wire-interpreter primitives, driven by hand-written minimal specs.

Translated from `unified-library-ts/tests/unit/wire/interpreter-primitives.test.ts`
case for case. Its own opening comment says why it exists, and it applies here
more than anywhere:

    The differential suite proves the SHIPPED specs reproduce the shipped
    adapters. It cannot reach an interpreter feature no current spec happens to
    use -- and a port reimplementing the interpreter has to get those right too,
    or the first spec that uses one silently builds the wrong request.

So these are the interpreter's semantics stated on their own, and they are the
oracle this port answers to. Assertions are transposed, never reinterpreted.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.wire.inherit import spec_for_model
from combycode_llm_sdk.wire.interpreter import Registry, build_from_spec, resolve_variants
from combycode_llm_sdk.wire.media_specs import media_spec
from combycode_llm_sdk.wire.registry import WIRE_SPECS, get_wire_spec
from combycode_llm_sdk.wire.service_specs import service_spec


def empty_reg() -> Registry:
    return Registry(transforms={}, builders={}, predicates={}, effects={})


def base(**over: Any) -> dict[str, Any]:
    return {
        "id": "test/spec",
        "provider": "test",
        "api": "chat",
        "envelope": {"method": "POST", "path": "/v1/x"},
        **over,
    }


# ─── variants ───────────────────────────────────────────────────────────────


class TestResolveVariants:
    def test_id_match_is_applied_with_any_provider_prefix_stripped(self) -> None:
        spec = base(variants=[{"flag": "thinking", "idMatch": "^claude-opus"}])
        assert list(resolve_variants(spec, "anthropic/claude-opus-5", empty_reg())) == ["thinking"]
        assert list(resolve_variants(spec, "claude-opus-5", empty_reg())) == ["thinking"]
        assert list(resolve_variants(spec, "claude-haiku-5", empty_reg())) == []

    def test_id_match_is_case_insensitive_because_the_id_is_lowercased_first(self) -> None:
        spec = base(variants=[{"flag": "f", "idMatch": "grok"}])
        assert "f" in resolve_variants(spec, "GROK-4", empty_reg())

    def test_a_fn_variant_goes_through_the_registry_and_gets_the_ORIGINAL_id(self) -> None:
        seen: list[str] = []
        reg = empty_reg()

        def is_big(model: str, _ctx: Any = None) -> bool:
            seen.append(model)
            return "opus" in model

        reg.transforms["isBig"] = is_big
        spec = base(variants=[{"flag": "big", "fn": "isBig"}])
        assert "big" in resolve_variants(spec, "anthropic/claude-opus-5", reg)
        # Not the lowercased/stripped id -- the fn gets what the caller passed.
        assert seen == ["anthropic/claude-opus-5"]
        assert "big" not in resolve_variants(spec, "claude-haiku-5", reg)

    def test_an_unknown_fn_is_named_rather_than_silently_skipped(self) -> None:
        spec = base(variants=[{"flag": "big", "fn": "nope"}])
        with pytest.raises(Exception, match="unknown variant fn: nope"):
            resolve_variants(spec, "m", empty_reg())

    def test_unless_removes_a_flag_when_the_named_flag_also_matched(self) -> None:
        spec = base(
            variants=[
                {"flag": "legacy", "idMatch": "^gpt"},
                {"flag": "modern", "idMatch": "^gpt-5"},
                # legacy applies to every gpt EXCEPT where modern matched
                {"flag": "legacy", "unless": "modern"},
            ]
        )
        assert list(resolve_variants(spec, "gpt-4o", empty_reg())) == ["legacy"]
        assert list(resolve_variants(spec, "gpt-5.4", empty_reg())) == ["modern"]

    def test_unless_naming_a_flag_that_did_not_match_leaves_the_flag_alone(self) -> None:
        spec = base(
            variants=[
                {"flag": "a", "idMatch": "."},
                {"flag": "a", "unless": "never-set"},
            ]
        )
        assert list(resolve_variants(spec, "m", empty_reg())) == ["a"]

    def test_a_spec_with_no_variants_resolves_to_no_flags(self) -> None:
        assert list(resolve_variants(base(), "m", empty_reg())) == []


# ─── $map ───────────────────────────────────────────────────────────────────


def map_spec(drop_unmatched: bool = False) -> dict[str, Any]:
    template: dict[str, Any] = {
        "$map": "things",
        "$case": [
            {"when": {"itemEq": "a"}, "value": {"tag": "A"}},
            {"when": {"itemEq": "b"}, "value": {"tag": "B"}},
        ],
    }
    if drop_unmatched:
        template["$dropUnmatched"] = True
    return base(blocks=[{"name": "items", "to": "items", "template": template}])


class TestMapTemplates:
    def test_each_item_takes_the_first_matching_case(self) -> None:
        built = build_from_spec(map_spec(), {"things": ["a", "b"]}, empty_reg())
        assert built.body["items"] == [{"tag": "A"}, {"tag": "B"}]

    def test_an_item_matching_NO_case_throws_naming_its_index(self) -> None:
        # Silence here would drop one message out of a conversation and the
        # request would still be well-formed -- the worst possible failure mode.
        with pytest.raises(
            Exception, match=r"\$map item 1 matched no case and \$dropUnmatched is not set"
        ):
            build_from_spec(map_spec(), {"things": ["a", "zzz"]}, empty_reg())

    def test_drop_unmatched_makes_skipping_explicit_and_allowed(self) -> None:
        built = build_from_spec(map_spec(True), {"things": ["a", "zzz", "b"]}, empty_reg())
        assert built.body["items"] == [{"tag": "A"}, {"tag": "B"}]

    def test_a_missing_source_array_omits_the_field_entirely(self) -> None:
        assert "items" not in build_from_spec(map_spec(), {}, empty_reg()).body


# ─── overlay ops ────────────────────────────────────────────────────────────


def with_overlay(ops: list[Any]) -> dict[str, Any]:
    return base(
        blocks=[{"name": "seed", "to": "nested.value", "template": {"$": "seed"}}],
        overlays={"test": {"ops": ops}},
    )


class TestOverlayOps:
    def test_set_writes_an_evaluated_template_at_a_body_path(self) -> None:
        built = build_from_spec(
            with_overlay([{"op": "set", "to": "extra.flag", "value": {"$": "seed"}}]),
            {"seed": "v"},
            empty_reg(),
        )
        assert built.body["extra"] == {"flag": "v"}

    def test_set_skips_writing_when_the_template_omits(self) -> None:
        built = build_from_spec(
            with_overlay([{"op": "set", "to": "extra.flag", "value": {"$": "absent"}}]),
            {"seed": "v"},
            empty_reg(),
        )
        assert "extra" not in built.body

    def test_set_honours_its_when_guard(self) -> None:
        spec = with_overlay([{"op": "set", "to": "x", "value": 1, "when": {"eq": ["seed", "yes"]}}])
        assert build_from_spec(spec, {"seed": "yes"}, empty_reg()).body["x"] == 1
        assert "x" not in build_from_spec(spec, {"seed": "no"}, empty_reg()).body

    def test_delete_down_a_path_that_does_not_exist_is_a_no_op(self) -> None:
        # `nested` is a string here, so walking into `nested.deeper` hits a
        # non-object mid-path. Throwing would make an overlay unusable on any
        # optional field.
        spec = base(
            blocks=[{"name": "seed", "to": "nested", "template": "a plain string"}],
            overlays={"test": {"ops": [{"op": "delete", "from": "nested.deeper.leaf"}]}},
        )
        built = build_from_spec(spec, {}, empty_reg())
        assert built.body["nested"] == "a plain string"

    def test_delete_on_a_path_that_never_existed_is_also_a_no_op(self) -> None:
        spec = base(overlays={"test": {"ops": [{"op": "delete", "from": "never.here"}]}})
        build_from_spec(spec, {}, empty_reg())  # must not raise

    def test_delete_removes_a_real_nested_leaf(self) -> None:
        spec = base(
            blocks=[{"name": "seed", "to": "a.b", "template": 1}],
            overlays={"test": {"ops": [{"op": "delete", "from": "a.b"}]}},
        )
        assert build_from_spec(spec, {}, empty_reg()).body == {"a": {}}

    def test_an_unknown_call_effect_is_named_in_the_error(self) -> None:
        spec = base(overlays={"test": {"ops": [{"op": "call", "call": "nope"}]}})
        with pytest.raises(Exception, match="unknown overlay effect: nope"):
            build_from_spec(spec, {}, empty_reg())


# ─── spec lookup ────────────────────────────────────────────────────────────


class TestSpecLookup:
    def test_get_wire_spec_returns_a_registered_delta_and_names_an_unknown_id(self) -> None:
        known = next(iter(WIRE_SPECS))
        assert get_wire_spec(known)["id"] == known
        with pytest.raises(Exception, match="unknown wire spec: provider/does-not-exist"):
            get_wire_spec("provider/does-not-exist")

    def test_a_BASE_spec_is_refused_rather_than_built_from(self) -> None:
        # A base spec has no endpoint of its own; building one would produce a
        # request addressed at nothing.
        with pytest.raises(
            Exception,
            match="anthropic/files.base is a base spec and cannot build a request on its own",
        ):
            service_spec("anthropic/files.base")
        with pytest.raises(
            Exception, match="is a base spec and cannot build a request on its own"
        ):
            media_spec("openai/media.base")

    def test_spec_for_model_returns_the_pinned_spec_else_the_default(self) -> None:
        pins = {"default": "p/base", "models": {"model-a": "p/a"}}
        assert spec_for_model("model-a", pins) == "p/a"
        assert spec_for_model("model-unknown", pins) == "p/base"
