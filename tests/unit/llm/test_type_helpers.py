"""Translated from `unified-library-ts/tests/unit/llm/type-helpers.test.ts`.

Small type-level helpers that everything else is built on.

`contentParts` and `isBuiltinTool` are one-liners, but they are the branch points
for "is this a string or a part list" and "is this a function tool or a hosted
one" -- get either wrong and the request builder silently takes the other path
for every call.

All four describes run: `messages.ts` and `tools.ts` landed with the client
batch, and their two describes were carried here as skips until they did.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import pytest

from combycode_llm_sdk.llm.types.schema_utils import (
    StrictSupport,
    ensure_additional_properties,
    strict_support,
)


class TestContentParts:
    def test_wraps_a_plain_string_as_a_single_text_part(self) -> None:
        from combycode_llm_sdk.llm.types.messages import content_parts

        # type-helpers.test.ts:16
        assert content_parts("hello") == [{"type": "text", "text": "hello"}]

    def test_wraps_the_empty_string_too(self) -> None:
        from combycode_llm_sdk.llm.types.messages import content_parts

        # type-helpers.test.ts:20
        assert content_parts("") == [{"type": "text", "text": ""}]

    def test_returns_a_part_array_unchanged_by_identity(self) -> None:
        from combycode_llm_sdk.llm.types.messages import content_parts

        # type-helpers.test.ts:24-25
        parts = [{"type": "text", "text": "a"}]
        assert content_parts(parts) is parts

    def test_round_trips_with_content_text_for_the_string_form(self) -> None:
        from combycode_llm_sdk.llm.types.messages import (
            content_parts,
            content_text,
        )

        # type-helpers.test.ts:29
        assert content_text(content_parts("hello")) == "hello"


class TestIsFunctionToolAndIsBuiltinToolAreExactComplements:
    @pytest.mark.parametrize(
        ("label", "tool", "is_fn"),
        [
            ("explicit function", {"type": "function", "name": "f", "parameters": {}}, True),
            ("implicit function (no type)", {"name": "f", "parameters": {}}, True),
            ("web_search builtin", {"type": "web_search"}, False),
            ("code_interpreter builtin", {"type": "code_interpreter"}, False),
        ],
    )
    def test_complements(self, label: str, tool: dict[str, object], is_fn: bool) -> None:
        from combycode_llm_sdk.llm.types.tools import (
            is_builtin_tool,
            is_function_tool,
        )

        # type-helpers.test.ts:34-45
        assert is_function_tool(tool) is is_fn
        assert is_builtin_tool(tool) is (not is_fn)

    def test_an_empty_string_type_counts_as_a_function_tool_not_a_builtin(self) -> None:
        from combycode_llm_sdk.llm.types.tools import (
            is_builtin_tool,
            is_function_tool,
        )

        # type-helpers.test.ts:48-53. `type: ''` reaches here from loosely-typed
        # callers; treating it as a builtin would send a tool with no name to the
        # provider.
        tool = {"type": "", "name": "f", "parameters": {}}
        assert is_function_tool(tool) is True
        assert is_builtin_tool(tool) is False


class TestEnsureAdditionalProperties:
    def test_adds_additional_properties_false_to_an_object_schema(self) -> None:
        # type-helpers.test.ts:59-63
        assert ensure_additional_properties({"type": "object", "properties": {}}) == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def test_does_not_override_an_explicit_additional_properties(self) -> None:
        # type-helpers.test.ts:67-69
        out = ensure_additional_properties({"type": "object", "additionalProperties": True})
        assert out["additionalProperties"] is True

    def test_recurses_into_nested_object_properties(self) -> None:
        # type-helpers.test.ts:73-78
        out = ensure_additional_properties(
            {"type": "object", "properties": {"inner": {"type": "object", "properties": {}}}}
        )
        inner = out["properties"]["inner"]
        assert inner["additionalProperties"] is False

    def test_recurses_into_array_items(self) -> None:
        # type-helpers.test.ts:85-92. `{type:'array', items:{type:'object'}}` is the
        # shape of every "list of records" tool argument. Missing this leaves the
        # item schema without the flag and the provider rejects the whole tool.
        out = ensure_additional_properties(
            {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"id": {"type": "string"}}},
                    }
                },
            }
        )
        rows = out["properties"]["rows"]
        assert rows["items"]["additionalProperties"] is False

    def test_a_tuple_items_form_is_left_alone_rather_than_mangled(self) -> None:
        # type-helpers.test.ts:96-98
        items = [{"type": "string"}, {"type": "number"}]
        out = ensure_additional_properties({"type": "array", "items": items})
        assert out["items"] == items

    def test_does_not_mutate_the_input_schema(self) -> None:
        # type-helpers.test.ts:102-105
        schema = {"type": "object", "properties": {"a": {"type": "object"}}}
        before = copy.deepcopy(schema)
        ensure_additional_properties(schema)
        assert schema == before


class TestStrictSupportBranchSchemasThatAreAllValid:
    """type-helpers.test.ts:109."""

    OK: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }

    def test_any_of_whose_every_branch_is_strict_safe_passes(self) -> None:
        # type-helpers.test.ts:113-115
        assert strict_support(
            {"type": "object", "properties": {"x": {"anyOf": [self.OK, self.OK]}}, "required": ["x"]},
            "openai",
        ) == StrictSupport(ok=True)

    def test_one_of_and_all_of_are_walked_the_same_way(self) -> None:
        # type-helpers.test.ts:119-128
        assert (
            strict_support(
                {"type": "object", "properties": {"x": {"oneOf": [self.OK]}}, "required": ["x"]},
                "openai",
            ).ok
            is True
        )
        assert (
            strict_support(
                {"type": "object", "properties": {"x": {"allOf": [self.OK]}}, "required": ["x"]},
                "openai",
            ).ok
            is True
        )
        bad = {
            "type": "object",
            "properties": {"q": {"type": "string"}, "p": {"type": "number"}},
            "required": ["q"],
        }
        assert (
            strict_support(
                {
                    "type": "object",
                    "properties": {"x": {"allOf": [self.OK, bad]}},
                    "required": ["x"],
                },
                "openai",
            ).ok
            is False
        )

    def test_a_non_array_any_of_is_skipped_rather_than_crashing_the_walk(self) -> None:
        # type-helpers.test.ts:132-137
        assert (
            strict_support(
                {
                    "type": "object",
                    "properties": {"x": {"anyOf": "not-a-list", "type": "string"}},
                    "required": ["x"],
                },
                "openai",
            ).ok
            is True
        )
