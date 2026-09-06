"""Tools whose implementation is a prompt and a schema.

A declaration -- instructions, a user template, the shape of the answer --
becomes an executable tool that an `InternalToolRunner` resolves by id out of a
`ToolRegistry`. The prompt stays DATA: readable on `tool.definition`, versioned
in the id, swappable per model without touching code.

Written as functions instead, every such tool would restate the same six steps
around its one paragraph of prompt, and the paragraph that mattered would be the
smallest thing in the file.

Transposed from `unified-library-ts/src/plugins/internal-tools/`. Synchronous:
these are root-level helpers like `moderate()` and `transcribe()`, and the
sync/async pair the port keeps is `LLM`/`AsyncLLM`, which is what a tool calls
through.
"""

from __future__ import annotations

from .builtin import (
    BUILTIN_TOOLS,
    clarify_tool,
    classify_tool,
    register_builtin_tools,
    score_tool,
    structure_tool,
    summarize_tool,
)
from .define import apply_schema_defaults, define_llm_tool
from .ids import (
    ParsedToolId,
    format_tool_id,
    id_without_version,
    matches_version,
    parse_tool_id,
    try_parse_tool_id,
)
from .json_format import (
    JSON_API_SYSTEM_PROMPT,
    JSON_FORMAT,
    TEXT_FORMAT,
    compose_json_system_prompt,
)
from .registry import (
    DEFAULT_MIN_SCORE,
    DEFAULT_SEARCH_LIMIT,
    LocalBackend,
    ToolRegistry,
)
from .runner import InternalToolRunner, InternalToolRunnerConfig
from .template import (
    format_bulleted_list,
    format_numbered_list,
    parse_json_with_fences,
    render_template,
)
from .types import (
    CompatFile,
    InternalTool,
    InternalToolContext,
    InternalToolError,
    JsonSchema,
    LLMToolDefinition,
    ModelFilter,
    ModelPreference,
    PromptVariant,
    ResolveMaxTokensContext,
    ToolBackend,
    ToolCompat,
    ToolFilter,
    select_variant,
)

__all__ = [
    "BUILTIN_TOOLS",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SEARCH_LIMIT",
    "JSON_API_SYSTEM_PROMPT",
    "JSON_FORMAT",
    "TEXT_FORMAT",
    "CompatFile",
    "InternalTool",
    "InternalToolContext",
    "InternalToolError",
    "InternalToolRunner",
    "InternalToolRunnerConfig",
    "JsonSchema",
    "LLMToolDefinition",
    "LocalBackend",
    "ModelFilter",
    "ModelPreference",
    "ParsedToolId",
    "PromptVariant",
    "ResolveMaxTokensContext",
    "ToolBackend",
    "ToolCompat",
    "ToolFilter",
    "ToolRegistry",
    "apply_schema_defaults",
    "clarify_tool",
    "classify_tool",
    "compose_json_system_prompt",
    "define_llm_tool",
    "format_bulleted_list",
    "format_numbered_list",
    "format_tool_id",
    "id_without_version",
    "matches_version",
    "parse_json_with_fences",
    "parse_tool_id",
    "register_builtin_tools",
    "render_template",
    "score_tool",
    "select_variant",
    "structure_tool",
    "summarize_tool",
    "try_parse_tool_id",
]
