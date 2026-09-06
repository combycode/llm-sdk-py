"""`@tool` -- name, description and schema, all from the function itself.

The point of the decorator is that there is ONE statement of each fact. These
tests are mostly about the derivation being faithful to the signature, because a
schema that disagrees with the body is a runtime TypeError delivered by a model
mid-conversation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from combycode_llm_sdk import tool
from combycode_llm_sdk.helpers.tool import ToolSchemaError


@dataclass
class Where:
    """Module level on purpose: `get_type_hints` resolves an annotation against
    the defining module's namespace, so a dataclass declared inside a test
    function is invisible to it -- a property of the language, not of `@tool`."""

    city: str


class TestTheThreeSources:
    def test_the_name_is_the_function_name(self) -> None:
        @tool
        def get_weather(city: str) -> str:
            """Get the weather."""
            return "sunny"

        assert get_weather.definition["name"] == "get_weather"

    def test_the_description_is_the_docstring_summary(self) -> None:
        @tool
        def f(x: str) -> str:
            """Do the thing.

            This paragraph is for whoever reads the code -- it explains a
            workaround the model cannot see and must not be billed for.
            """
            return x

        assert f.definition["description"] == "Do the thing."

    def test_a_multi_line_summary_is_folded_not_truncated(self) -> None:
        @tool
        def f(x: str) -> str:
            """Count the words in a passage of text and
            return the total.
            """
            return x

        assert f.definition["description"] == (
            "Count the words in a passage of text and return the total."
        )

    def test_no_docstring_is_refused_at_decoration(self) -> None:
        # Better here than as a tool the model is told nothing about.
        with pytest.raises(ToolSchemaError, match="no docstring"):

            @tool
            def f(x: str) -> str:
                return x


class TestTheSchema:
    def test_scalars_map_to_their_json_types(self) -> None:
        @tool
        def f(a: str, b: int, c: float, d: bool) -> str:
            """Take one of each."""
            return ""

        props = f.definition["parameters"]["properties"]
        assert [props[k]["type"] for k in "abcd"] == [
            "string",
            "integer",
            "number",
            "boolean",
        ]

    def test_bool_is_not_reported_as_an_integer(self) -> None:
        # `bool` subclasses `int`, so a chain of issubclass checks gets this
        # wrong and the model is invited to send `1` for a flag.
        @tool
        def f(flag: bool) -> str:
            """Take a flag."""
            return ""

        assert f.definition["parameters"]["properties"]["flag"]["type"] == "boolean"

    def test_a_default_makes_the_parameter_optional(self) -> None:
        @tool
        def f(city: str, unit: str = "c") -> str:
            """Take an optional unit."""
            return ""

        assert f.definition["parameters"]["required"] == ["city"]

    def test_no_parameters_omits_required_entirely(self) -> None:
        # `08_multistep_loop.py`'s `get_user_city()`. An empty `required: []` is a
        # different document from an absent one to a strict validator.
        @tool
        def get_user_city() -> str:
            """Get the user's city."""
            return "Paris"

        params = get_user_city.definition["parameters"]
        assert params["properties"] == {}
        assert "required" not in params

    def test_a_literal_becomes_an_enum(self) -> None:
        @tool
        def f(unit: Literal["c", "f"]) -> str:
            """Take a unit."""
            return ""

        assert f.definition["parameters"]["properties"]["unit"] == {
            "type": "string",
            "enum": ["c", "f"],
        }

    def test_a_list_becomes_an_array_with_items(self) -> None:
        @tool
        def f(cities: list[str]) -> str:
            """Take several cities."""
            return ""

        assert f.definition["parameters"]["properties"]["cities"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_optional_is_the_inner_type_admitting_null(self) -> None:
        @tool
        def f(note: str | None = None) -> str:
            """Take an optional note."""
            return ""

        assert f.definition["parameters"]["properties"]["note"]["type"] == [
            "string",
            "null",
        ]

    def test_a_dataclass_parameter_expands_to_an_object(self) -> None:
        @tool
        def f(where: Where) -> str:
            """Take a place."""
            return ""

        assert f.definition["parameters"]["properties"]["where"]["properties"] == {
            "city": {"type": "string"}
        }

    def test_an_unannotated_parameter_is_refused(self) -> None:
        # The alternative is an empty schema and a model guessing at the name.
        with pytest.raises(ToolSchemaError, match="no type annotation"):

            @tool
            def f(x) -> str:  # type: ignore[no-untyped-def]
                """Take a mystery."""
                return ""

    def test_an_unmappable_annotation_is_refused(self) -> None:
        with pytest.raises(ToolSchemaError, match="no JSON Schema"):

            @tool
            def f(when: complex) -> str:
                """Take a complex number."""
                return ""

    def test_varargs_name_nothing_the_model_could_fill(self) -> None:
        @tool
        def f(city: str, *rest: Any, **extra: Any) -> str:
            """Take a city."""
            return ""

        assert list(f.definition["parameters"]["properties"]) == ["city"]


class TestTheDecoratedObject:
    def test_it_is_still_callable(self) -> None:
        # A decorator that trades the function for a descriptor makes the body
        # untestable without a model in the loop.
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert add(2, 3) == 5

    def test_async_is_detected_not_declared(self) -> None:
        @tool
        async def fetch(symbol: str) -> str:
            """Look up a price."""
            return symbol

        assert fetch.is_async is True
        assert asyncio.run(fetch("ACME")) == "ACME"

    def test_a_sync_tool_is_not_marked_async(self) -> None:
        @tool
        def f(x: str) -> str:
            """Do nothing."""
            return x

        assert f.is_async is False

    def test_decorating_twice_is_the_same_tool(self) -> None:
        @tool
        def f(x: str) -> str:
            """Do nothing."""
            return x

        assert tool(f) is f
