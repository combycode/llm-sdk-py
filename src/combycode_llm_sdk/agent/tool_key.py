"""How a tool is INDEXED in the loop, and how it is named when two collide.

Transposed from `unified-library-ts/src/agent/tool-key.ts`.

Two kinds of tool share one registry. A function tool is keyed by its name; a
builtin -- `{"type": "web_search"}` -- has no name at all and is keyed by its
type, because that is what the provider matches on and what a second builtin of
the same type would collide with.

Reading `definition["name"]` directly is therefore wrong for half the registry:
it is a `KeyError` on every builtin. The two functions here are the only places
that need to know the difference.
"""

from __future__ import annotations

from typing import Any

from ..llm.types.tools import is_function_tool


def tool_key(tool: Any) -> str:
    """The registry key: the function name, else the builtin type."""
    definition = tool.definition
    if is_function_tool(definition):
        return str(definition["name"])
    return str(definition.get("type") or "")


def describe_tool(tool: Any) -> str:
    """A short label for collision diagnostics.

    Names the KIND as well as the key, because a function tool shadowing a
    builtin (or the reverse) is the case that reads as impossible in a log, and
    it needs a different fix from two functions sharing a name.
    """
    definition = tool.definition
    if is_function_tool(definition):
        return f"function:{definition['name']}"
    return f"builtin:{definition.get('type') or ''}"


__all__ = ["describe_tool", "tool_key"]
