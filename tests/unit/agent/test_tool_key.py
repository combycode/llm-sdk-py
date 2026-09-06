"""How a tool is indexed, and what a collision says.

Two kinds of tool share one registry. A function tool is keyed by name; a
builtin -- `{"type": "web_search"}` -- has no name at all, so reading
`definition["name"]` is a `KeyError` on half the registry, and the provider
matches it on its type anyway.

The collision diagnostic names the KIND as well as the key: a function tool
shadowing a builtin reads as impossible in a log, and it needs a different fix
from two functions sharing a name.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

import pytest

from combycode_llm_sdk.agent.loop import AgentLoop, ToolNameCollision
from combycode_llm_sdk.agent.tool_key import describe_tool, tool_key
from combycode_llm_sdk.helpers.tool import Tool


class Client:
    """The loop never runs here -- these are registration-only tests, and a
    client that raises proves it."""

    def complete(self, **options: Any) -> Any:
        raise AssertionError("the model must not be called in a registration test")


def function_tool(name: str, marker: str = "x") -> Tool:
    return Tool(
        lambda **kw: marker,
        {"type": "function", "name": name, "description": marker, "parameters": {}},
    )


def builtin_tool(kind: str, marker: str = "x") -> Tool:
    return Tool(lambda **kw: marker, {"type": kind})


def warnings_of(loop: AgentLoop, seen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [w for w in seen if w.get("code") == "tool_name_collision"]


def listening() -> tuple[AgentLoop, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    loop = AgentLoop(client=Client(), tools=[])
    loop.hooks.on("onWarning", lambda w: seen.append(dict(w)))
    return loop, seen


class TestTheKey:
    def test_a_function_tool_is_keyed_by_its_name(self) -> None:
        assert tool_key(function_tool("lookup")) == "lookup"

    def test_a_builtin_is_keyed_by_its_type(self) -> None:
        assert tool_key(builtin_tool("web_search")) == "web_search"

    def test_a_definition_with_no_type_is_a_function_tool(self) -> None:
        # Falsiness, not absence: a loosely-typed caller sends `{"type": ""}`,
        # and treating that as a builtin would key it under the empty string.
        assert tool_key(Tool(lambda: "", {"name": "bare"})) == "bare"
        assert tool_key(Tool(lambda: "", {"type": "", "name": "blank"})) == "blank"


class TestTheDescription:
    def test_it_names_the_kind_as_well_as_the_key(self) -> None:
        assert describe_tool(function_tool("lookup")) == "function:lookup"
        assert describe_tool(builtin_tool("web_search")) == "builtin:web_search"


class TestRegistration:
    def test_a_builtin_registers_under_its_type(self) -> None:
        # `Tool.name` raises here, which is why the registry does not use it.
        loop = AgentLoop(client=Client(), tools=[builtin_tool("web_search")])
        assert loop.tool_names() == ["web_search"]

    def test_builtins_of_different_types_do_not_collide(self) -> None:
        loop = AgentLoop(
            client=Client(),
            tools=[builtin_tool("web_search"), builtin_tool("code_interpreter")],
        )
        assert sorted(loop.tool_names()) == ["code_interpreter", "web_search"]

    def test_two_builtins_of_one_type_do_collide(self) -> None:
        loop, seen = listening()
        loop.register_tool(builtin_tool("web_search", "first"))
        loop.register_tool(builtin_tool("web_search", "second"))
        assert loop.tool_names() == ["web_search"]
        (collision,) = warnings_of(loop, seen)
        assert collision["details"]["shadowed"] == "builtin:web_search"
        assert collision["details"]["winner"] == "builtin:web_search"

    def test_a_function_tool_shadowing_a_builtin_is_described_as_exactly_that(self) -> None:
        loop, seen = listening()
        loop.register_tool(builtin_tool("web_search"))
        loop.register_tool(function_tool("web_search"))
        (collision,) = warnings_of(loop, seen)
        assert collision["details"]["shadowed"] == "builtin:web_search"
        assert collision["details"]["winner"] == "function:web_search"

    def test_the_same_tool_object_registered_twice_is_not_a_collision(self) -> None:
        # Idempotent re-registration shadows nothing, and warning about it would
        # fire on any code that re-adds a tool defensively.
        loop, seen = listening()
        one = function_tool("same")
        loop.register_tool(one)
        loop.register_tool(one)
        assert warnings_of(loop, seen) == []

    def test_two_distinct_tools_of_one_name_still_collide(self) -> None:
        loop, seen = listening()
        loop.register_tool(function_tool("dup", "first"))
        loop.register_tool(function_tool("dup", "second"))
        (collision,) = warnings_of(loop, seen)
        assert collision["details"]["key"] == "dup"
        assert "dup" in collision["message"]

    def test_the_later_registration_wins(self) -> None:
        loop, _ = listening()
        loop.register_tool(function_tool("dup", "first"))
        loop.register_tool(function_tool("dup", "second"))
        assert loop._tools["dup"].func() == "second"

    def test_the_error_policy_refuses_a_builtin_collision_too(self) -> None:
        with pytest.raises(ToolNameCollision, match="web_search"):
            AgentLoop(
                client=Client(),
                tool_name_collision="error",
                tools=[builtin_tool("web_search"), builtin_tool("web_search")],
            )

    def test_the_error_policy_still_allows_the_same_object_twice(self) -> None:
        one = function_tool("same")
        loop = AgentLoop(client=Client(), tool_name_collision="error", tools=[one])
        loop.register_tool(one)
        assert loop.tool_names() == ["same"]
