"""Universal tool schema definitions.

Transposed from `unified-library-ts/src/llm/types/tools.ts`.

Shapes, as documented in `messages.py` -- plain dicts with camelCase keys,
because the request specs address them by those names:

- `FunctionTool` `{type?:'function', name, description, parameters, strict?,
  cache?, allowedCallers?, outputSchema?}`
- `BuiltinTool`  `{type: 'image_generation'|'web_search'|'web_fetch'|
  'code_interpreter'|'file_search'|'mcp'|'programmatic_tool_calling'|'shell',
  params?}`

`shell` (OpenAI + xAI Responses) is the one builtin that is only provider-RUN
when `params['environment']` is a container:

- `{'environment': {'type': 'container_auto'}}` -- OpenAI runs the commands in a
  container it provisions and streams stdout/stderr back. Measured 2026-10-02 the
  request is rewritten to `container_reference` carrying the `container_id` it
  chose.
- `{'environment': {'type': 'local'}}` or omitted -- the model only ASKS; whoever
  called has to run the commands and feed the output back. The turn ends after the
  request, so `text` is empty by design and the commands are in
  `builtinToolCalls[].code`. A `shell_awaiting_caller` warning says so, because an
  empty answer with `finishReason: 'stop'` otherwise looks like success.

xAI REQUIRES `environment` (a 422 names the missing field) and accepts only
`local`, so a shell call on xAI is always the second case.
- `McpToolParams` the typed shape for an `mcp` builtin's `params`, forwarded
  verbatim: `{server_label, server_url?, connector_id?, tunnel_id?,
  authorization?, headers?, require_approval?, allowed_tools?,
  server_description?, ...}`. Exactly one of `server_url`, `connector_id` or
  `tunnel_id` identifies the server, and OpenAI enforces that.
- `ImageGenerationToolParams` the typed shape for an `image_generation`
  builtin's `params`, forwarded verbatim and spread BESIDE `type` (xAI reads
  `action` as a sibling; nested under a `params` key it would be an unknown
  field and silently absent rather than refused):
  `{action?: 'auto'|'generate'|'edit', output_format?, quality?, size?,
  background?, ...}`. Supported on openai and xai, measured live 2026-10-01
  through this library. xAI's `action` is validated rather than inert --
  `'paint'` is a 400 naming the three values -- and `'edit'` with nothing to
  edit returns no image at all, which is a 200 with a text-only answer.
  On `mimeType`: OpenAI reports `output_format` and reports it accurately,
  while xAI reports none and returns JPEG, so the parser prefers the declared
  format, falls back to the image's own magic bytes, and only then to PNG.
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
