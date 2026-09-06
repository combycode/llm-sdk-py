"""Shared response helpers used across provider adapters.

Transposed from `unified-library-ts/src/llm/providers/_shared/response-utils.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping


def extract_finish_reason(
    has_tool_calls: bool,
    provider_reason: str | None,
    reason_map: Mapping[str, str],
) -> str:
    """Table-driven finish-reason mapper.

    Returns 'tool_use' when tool calls are present; otherwise looks up the
    provider's raw reason in `reason_map`, falling back to 'stop'.
    """
    if has_tool_calls:
        return "tool_use"
    mapped = reason_map.get(provider_reason) if provider_reason is not None else None
    return mapped if mapped is not None else "stop"


__all__ = ["extract_finish_reason"]
