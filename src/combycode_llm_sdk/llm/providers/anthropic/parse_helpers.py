"""The parts of Anthropic's parse that are not shape.

Transposed from the module-level helpers in
`unified-library-ts/src/llm/providers/anthropic/messages.ts`.

They live in their own module rather than beside an adapter because both the
buffered and the streaming registry need them, and on the TypeScript side
importing them from the adapter file made the adapter and its registry import
each other. Python has no reason to repeat that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def empty_usage() -> dict[str, Any]:
    """The zeroed usage every provider falls back to.

    camelCase because this is the WIRE-level unified shape, the one the specs
    name and the one the recorded corpus froze. The public Python API converts
    it to snake_case attributes; that mapping is a separate layer.
    """
    return {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "cachedTokens": 0,
        "cacheWriteTokens": 0,
        "reasoningTokens": 0,
    }


def anthropic_usage(u: Mapping[str, Any] | None) -> dict[str, Any]:
    if not u:
        return empty_usage()
    input_tokens = u.get("input_tokens") or 0
    output_tokens = u.get("output_tokens") or 0
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
        "cachedTokens": u.get("cache_read_input_tokens") or 0,
        "cacheWriteTokens": u.get("cache_creation_input_tokens") or 0,
        "reasoningTokens": 0,
    }


def anthropic_billed_tier(raw: Any) -> dict[str, Any]:
    return {"serviceTier": raw, "pricingTier": raw} if isinstance(raw, str) and raw else {}


def files_from_code_exec_block(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Hosted code-execution output files from one content block.

    Shared by the buffered and streamed paths so both surface the exact same
    files. The current tool (code_execution_20260521) emits
    `bash_code_execution_tool_result` -> `bash_code_execution_result` ->
    `content[]` of `bash_code_execution_output`; older tool versions emit the
    `code_execution_*` equivalents. Both carry `file_id`.
    (`text_editor_code_execution_tool_result` blocks are file create/view/edit
    markers with no downloadable id, so they are not surfaced.)
    """
    if block.get("type") not in ("bash_code_execution_tool_result", "code_execution_tool_result"):
        return []
    result = block.get("content")
    if not isinstance(result, Mapping):
        return []
    if result.get("type") not in ("bash_code_execution_result", "code_execution_result"):
        return []
    inner = result.get("content")
    if not isinstance(inner, list):
        return []
    files: list[dict[str, Any]] = []
    for out in inner:
        if not isinstance(out, Mapping):
            continue
        if out.get("type") in (
            "bash_code_execution_output",
            "code_execution_output",
        ) and isinstance(out.get("file_id"), str):
            files.append({"id": out["file_id"], "source": "code_execution"})
    return files


def builtin_input_payload(tool: str, inp: Mapping[str, Any] | None) -> dict[str, Any]:
    """The code (code execution) or query (web search) a hosted tool was given.

    Shared by the buffered and streamed paths.
    """
    if not inp:
        return {}
    if tool == "code_interpreter":
        code = inp.get("code", inp.get("command"))
        return {"code": code} if isinstance(code, str) else {}
    if tool == "web_search":
        return {"query": inp["query"]} if isinstance(inp.get("query"), str) else {}
    if tool == "web_fetch":
        return {"url": inp["url"]} if isinstance(inp.get("url"), str) else {}
    return {}


def result_stdout(content: Any) -> str | None:
    """stdout from a code-execution `*_tool_result` block's content, if present."""
    if isinstance(content, Mapping):
        stdout = content.get("stdout")
        if isinstance(stdout, str):
            return stdout
    return None


__all__ = [
    "anthropic_billed_tier",
    "anthropic_usage",
    "builtin_input_payload",
    "empty_usage",
    "files_from_code_exec_block",
    "result_stdout",
]
