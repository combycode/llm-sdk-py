"""Universal tool schema definitions.

Transposed from `unified-library-ts/src/llm/types/tools.ts`.

Shapes, as documented in `messages.py` -- plain dicts with camelCase keys,
because the request specs address them by those names:

- `FunctionTool` `{type?:'function', name, description, parameters, strict?,
  cache?, allowedCallers?, outputSchema?}`
- `BuiltinTool`  `{type: 'image_generation'|'web_search'|'web_fetch'|
  'code_interpreter'|'file_search'|'mcp'|'programmatic_tool_calling', params?}`
- `McpToolParams` the typed shape for an `mcp` builtin's `params`, forwarded
  verbatim: `{server_label, server_url?, connector_id?, tunnel_id?,
  authorization?, headers?, require_approval?, allowed_tools?,
  server_description?, ...}`. Exactly one of `server_url`, `connector_id` or
  `tunnel_id` identifies the server, and OpenAI enforces that.
- `ToolChoice`   `'auto' | 'none' | 'required' | {name}`
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: `type JsonSchema = Record<string, unknown>` (tools.ts:58). Defined in
#: `schema_utils` and re-exported here, where the TypeScript declares it, so
#: there is one definition rather than two.
from .schema_utils import JsonSchema

#: `interface FunctionTool` (tools.ts:3).
FunctionTool = dict[str, Any]

#: `interface BuiltinTool` (tools.ts:20).
BuiltinTool = dict[str, Any]

#: `interface McpToolParams` (tools.ts:41).
McpToolParams = dict[str, Any]

#: `type Tool = FunctionTool | BuiltinTool` (tools.ts:54).
Tool = dict[str, Any]

#: `type ToolChoice` (tools.ts:56).
ToolChoice = str | dict[str, Any]


def is_function_tool(tool: Mapping[str, Any]) -> bool:
    """`!tool.type || tool.type === 'function'`.

    Falsiness, not `is None`: `{'type': ''}` reaches here from loosely-typed
    callers, and treating it as a builtin would send a tool with no name to the
    provider.
    """
    return not tool.get("type") or tool.get("type") == "function"


def is_builtin_tool(tool: Mapping[str, Any]) -> bool:
    """`!!tool.type && tool.type !== 'function'` -- the exact complement."""
    return bool(tool.get("type")) and tool.get("type") != "function"


__all__ = [
    "BuiltinTool",
    "FunctionTool",
    "JsonSchema",
    "McpToolParams",
    "Tool",
    "ToolChoice",
    "is_builtin_tool",
    "is_function_tool",
]
