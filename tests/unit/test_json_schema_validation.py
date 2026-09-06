"""Checking a value against a schema, and the MCP contract that needs it.

The validator is small on purpose, so the tests that matter are the edges where
Python and JSON disagree about types -- `True` is an `int` here and is not a
number there, and `3.0` is an integer to JSON Schema and not to `isinstance`.

The MCP half is a contract rather than a hint: declaring an output schema to a
provider makes the provider REQUIRE a JSON result matching it, so the schema,
the validation and returning structured data have to travel together.
"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.mcp.tools import mcp_tool_to_tool
from combycode_llm_sdk.util.json_schema import validate_json_schema

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["city"],
}


def ok(schema: dict[str, Any], value: Any) -> bool:
    return validate_json_schema(schema, value) == []


class TestTypes:
    def test_the_happy_case_passes_quietly(self) -> None:
        assert ok(SCHEMA, {"city": "Berlin", "count": 3})

    def test_a_wrong_type_says_what_it_wanted_and_what_it_got(self) -> None:
        errors = validate_json_schema({"type": "string"}, 42)
        assert errors == ["$: expected string, got number"]

    def test_a_type_mismatch_stops_the_deeper_checks(self) -> None:
        # Checking an object's properties against a string produces noise that
        # buries the one error worth reading. The value here is deliberately a
        # MAPPING against a string schema: a non-container value would skip the
        # object and array branches anyway, so it would pass this test even if
        # the early return were deleted.
        strict = {**SCHEMA, "type": "string", "additionalProperties": False}
        assert validate_json_schema(strict, {"count": "three", "extra": 1}) == [
            "$: expected string, got object"
        ]

    def test_a_type_mismatch_suppresses_the_enum_complaint_too(self) -> None:
        # `type` and `enum` together is the ordinary shape of a tool field, and
        # a value that is the wrong type is not usefully also told it is not one
        # of the permitted strings.
        schema = {"type": "string", "enum": ["a", "b"]}
        assert validate_json_schema(schema, 5) == ["$: expected string, got number"]

    def test_a_union_of_types_passes_on_any_of_them(self) -> None:
        schema = {"type": ["string", "null"]}
        assert ok(schema, "x")
        assert ok(schema, None)
        assert not ok(schema, 5)

    def test_a_bool_is_not_a_number(self) -> None:
        # `bool` is an int subclass in Python and `true` is not a number in
        # JSON Schema, so this passes by accident unless it is excluded.
        assert not ok({"type": "number"}, True)
        assert not ok({"type": "integer"}, True)
        assert ok({"type": "boolean"}, True)

    def test_a_whole_float_satisfies_integer(self) -> None:
        # JSON has one number type. A value that arrived on the wire as `3.0`
        # must not fail a field it would have passed as `3`.
        assert ok({"type": "integer"}, 3.0)
        assert not ok({"type": "integer"}, 3.5)

    def test_a_string_is_not_an_array(self) -> None:
        # Both are sequences in Python, and iterating a string as an array is
        # how a one-word answer becomes five single-letter errors.
        assert not ok({"type": "array"}, "abc")

    def test_null_is_its_own_type(self) -> None:
        assert ok({"type": "null"}, None)
        assert not ok({"type": "string"}, None)

    def test_an_unknown_type_keyword_is_not_a_failure(self) -> None:
        # The schema may be newer than this validator.
        assert ok({"type": "future-thing"}, "anything")

    def test_a_schema_with_no_type_accepts_anything(self) -> None:
        assert ok({}, {"whatever": 1})


class TestObjects:
    def test_a_missing_required_property_is_named(self) -> None:
        assert validate_json_schema(SCHEMA, {"count": 1}) == [
            "$.city: required property missing"
        ]

    def test_an_optional_property_may_be_absent(self) -> None:
        assert ok(SCHEMA, {"city": "Berlin"})

    def test_a_nested_error_carries_its_path(self) -> None:
        errors = validate_json_schema(SCHEMA, {"city": "Berlin", "count": "three"})
        assert errors == ["$.count: expected integer, got string"]

    def test_additional_properties_are_allowed_unless_refused(self) -> None:
        assert ok(SCHEMA, {"city": "Berlin", "extra": 1})
        strict = {**SCHEMA, "additionalProperties": False}
        assert validate_json_schema(strict, {"city": "Berlin", "extra": 1}) == [
            "$.extra: additional property not allowed"
        ]

    def test_deep_nesting_keeps_the_whole_path(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}},
        }
        assert validate_json_schema(schema, {"a": {"b": 1}}) == [
            "$.a.b: expected string, got number"
        ]


class TestArraysEnumsAndConst:
    def test_every_item_is_checked_and_indexed(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        assert validate_json_schema(schema, [1, "two", 3, "four"]) == [
            "$[1]: expected integer, got string",
            "$[3]: expected integer, got string",
        ]

    def test_an_empty_array_passes(self) -> None:
        assert ok({"type": "array", "items": {"type": "integer"}}, [])

    def test_enum_membership_is_structural(self) -> None:
        assert ok({"enum": ["a", {"k": 1}]}, {"k": 1})
        assert not ok({"enum": ["a", "b"]}, "c")

    def test_const_is_structural_too(self) -> None:
        assert ok({"const": {"k": [1, 2]}}, {"k": [1, 2]})
        assert not ok({"const": {"k": [1, 2]}}, {"k": [2, 1]})


class Client:
    """An MCP client that answers with whatever it was handed."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return self.result

    def call_tool_task(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {"taskId": "t-1"}

    def await_task(self, task_id: str) -> dict[str, Any]:
        return {"status": "completed"}

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        return self.result


WEATHER = {
    "name": "weather",
    "description": "the weather",
    "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
    "outputSchema": {
        "type": "object",
        "properties": {"tempC": {"type": "integer"}},
        "required": ["tempC"],
    },
}


#: The same tool, but one the server will only run as a task.
TASK_ONLY = {**WEATHER, "execution": {"taskSupport": "required"}}


def build(result: dict[str, Any], **over: Any) -> tuple[Any, Client]:
    client = Client(result)
    tool = mcp_tool_to_tool(client, WEATHER, "srv", **over)  # type: ignore[arg-type]
    return tool, client


class TestTheMcpContract:
    def test_output_validation_is_off_by_default(self) -> None:
        # Saying yes CHANGES what the tool returns, so it cannot be the default:
        # a provider told about an output schema then requires a JSON result,
        # and every existing tool answering in prose would start failing.
        tool, _ = build({"content": [{"type": "text", "text": "warm"}]})
        assert "outputSchema" not in tool.definition
        assert tool.func(city="Berlin") == "warm"

    def test_opting_in_declares_the_schema(self) -> None:
        tool, _ = build({"content": []}, validate_output=True)
        assert tool.definition["outputSchema"] == WEATHER["outputSchema"]

    def test_a_valid_structured_result_comes_back_as_json(self) -> None:
        # Not the rendered text: the provider was promised JSON matching the
        # schema, and prose is what it rejects the turn for.
        tool, _ = build(
            {"content": [{"type": "text", "text": "warm"}], "structuredContent": {"tempC": 21}},
            validate_output=True,
        )
        assert json.loads(str(tool.func(city="Berlin"))) == {"tempC": 21}

    def test_a_broken_contract_is_reported_to_the_model(self) -> None:
        # Returned rather than raised: a server that broke its own contract is a
        # disappointing answer, not a broken connection, and a model told what
        # was wrong often calls again correctly.
        tool, _ = build(
            {"content": [], "structuredContent": {"tempC": "warm"}}, validate_output=True
        )
        out = str(tool.func(city="Berlin"))
        assert out.startswith("Tool output failed schema validation")
        assert "expected integer" in out

    def test_a_missing_required_field_is_caught(self) -> None:
        tool, _ = build({"content": [], "structuredContent": {}}, validate_output=True)
        assert "required property missing" in str(tool.func(city="Berlin"))

    def test_a_call_with_no_structured_half_falls_back_to_the_text(self) -> None:
        # MCP sends structuredContent exactly when a tool publishes an
        # outputSchema, so its absence means this call had nothing structured.
        tool, _ = build(
            {"content": [{"type": "text", "text": "warm"}]}, validate_output=True
        )
        assert tool.func(city="Berlin") == "warm"

    def test_an_error_result_stays_an_error_even_when_it_validates(self) -> None:
        # isError is the tool telling the model it failed; replacing that with
        # the structured payload would hide it.
        tool, _ = build(
            {
                "content": [{"type": "text", "text": "upstream down"}],
                "structuredContent": {"tempC": 0},
                "isError": True,
            },
            validate_output=True,
        )
        assert str(tool.func(city="Berlin")).startswith("Tool error:")

    def test_a_tool_with_no_output_schema_is_unaffected_by_opting_in(self) -> None:
        client = Client({"content": [{"type": "text", "text": "hi"}]})
        bare = {"name": "echo", "description": "d", "inputSchema": {"type": "object"}}
        tool = mcp_tool_to_tool(client, bare, "srv", validate_output=True)  # type: ignore[arg-type]
        assert "outputSchema" not in tool.definition
        assert tool.func() == "hi"


class TestTheContractHoldsForTaskOnlyTools:
    """`taskSupport: "required"` changes how a tool is CALLED, not what it owes.

    A task-only tool that declares an output schema has made the same promise to
    the provider as any other, so its result goes through the same validation
    and comes back as JSON. Returning rendered prose here would be rejected by
    the provider for a reason that has nothing to do with tasks.
    """

    def build_task(self, result: dict[str, Any], **over: Any) -> tuple[Any, Client]:
        client = Client(result)
        tool = mcp_tool_to_tool(client, TASK_ONLY, "srv", **over)  # type: ignore[arg-type]
        return tool, client

    def test_a_task_result_comes_back_as_json_too(self) -> None:
        tool, _ = self.build_task(
            {"content": [{"type": "text", "text": "warm"}], "structuredContent": {"tempC": 21}},
            validate_output=True,
        )
        assert json.loads(str(tool.func(city="Berlin"))) == {"tempC": 21}

    def test_a_task_result_is_validated_too(self) -> None:
        tool, _ = self.build_task(
            {"content": [], "structuredContent": {"tempC": "warm"}}, validate_output=True
        )
        assert str(tool.func(city="Berlin")).startswith("Tool output failed schema validation")

    def test_a_task_result_is_still_prose_without_opting_in(self) -> None:
        tool, _ = self.build_task(
            {"content": [{"type": "text", "text": "warm"}], "structuredContent": {"tempC": 21}}
        )
        assert tool.func(city="Berlin") == "warm"
