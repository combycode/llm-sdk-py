"""Normalize a provider-native builtin-tool marker to the unified name.

Transposed from `unified-library-ts/src/llm/providers/_shared/builtin-tools.ts`.

The marker may be an output-item type, a server-tool name, or a result-block
type. Unknown values fall back to a stripped form so a new provider tool still
yields a sensible name rather than an error.
"""

from __future__ import annotations

import re

_NATIVE_TO_UNIFIED: dict[str, str] = {
    # OpenAI / xAI Responses output-item types
    "web_search_call": "web_search",
    "code_interpreter_call": "code_interpreter",
    # Anthropic server_tool_use names
    "web_search": "web_search",
    "web_fetch": "web_fetch",
    "code_execution": "code_interpreter",
    "bash_code_execution": "code_interpreter",
    # Anthropic *_tool_result block types
    "web_search_tool_result": "web_search",
    "web_fetch_tool_result": "web_fetch",
    "code_execution_tool_result": "code_interpreter",
    "bash_code_execution_tool_result": "code_interpreter",
}

_CALL_SUFFIX = re.compile(r"_call$")
_RESULT_SUFFIX = re.compile(r"_tool_result$")


def unified_builtin_tool(native: str) -> str:
    known = _NATIVE_TO_UNIFIED.get(native)
    if known is not None:
        return known
    return _RESULT_SUFFIX.sub("", _CALL_SUFFIX.sub("", native))


__all__ = ["unified_builtin_tool"]
