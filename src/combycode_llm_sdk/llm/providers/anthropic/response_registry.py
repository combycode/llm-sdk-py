"""The named escape hatches `anthropic.messages.response.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/anthropic/response-registry.ts`.

A response spec expresses SHAPE. These are the four things that are not shape: a
fold over what was collected, a usage object with a billed tier merged in, a
finish-reason decision that depends on whether any tool was called, and a
citation walk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from .._shared.builtin_tools import unified_builtin_tool
from .._shared.citations import extract_citations
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import (
    anthropic_billed_tier,
    anthropic_usage,
    builtin_input_payload,
    files_from_code_exec_block,
    result_stdout,
)


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _block(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.item.value if ctx.item else {}
    return v if isinstance(v, Mapping) else {}


#: `stop_reason` values that are NOT a clean finish.
#:
#: `model_context_window_exceeded` (anthropic-ts 0.115) means the prompt itself
#: overflowed, a truncation like max_tokens. `refusal` is a safety decline and
#: lines up with every other provider's block signal.
_FINISH = {
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}


def _builtin_call(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    """A provider-run tool call, with the code or query it was given."""
    b = _block(ctx)
    tool = unified_builtin_tool(str(b.get("name")))
    out: dict[str, Any] = {"tool": tool}
    if isinstance(b.get("id"), str):
        out["id"] = b["id"]
    inp = b.get("input")
    out.update(builtin_input_payload(tool, inp if isinstance(inp, Mapping) else None))
    return out


def _code_exec_files(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Output files from one code-execution result block; often none."""
    return files_from_code_exec_block(_block(ctx))


def _text(_arg: Any, ctx: Ctx) -> str:
    """`text` is the concatenation of the text parts, so it cannot be computed
    until every block has been classified."""
    return "".join(
        p.get("text") or ""
        for p in _out(ctx)["content"]
        if isinstance(p, Mapping) and p.get("type") == "text"
    )


def _usage_full(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    raw = _raw(ctx)
    u = raw.get("usage")
    u = u if isinstance(u, Mapping) else None
    return {**anthropic_usage(u), **anthropic_billed_tier(u.get("service_tier") if u else None)}


def _finish(_arg: Any, ctx: Ctx) -> str:
    """Depends on the collected tool calls: a turn that called a tool finished
    for that reason whatever `stop_reason` says."""
    reason = _raw(ctx).get("stop_reason")
    return extract_finish_reason(
        len(_out(ctx)["toolCalls"]) > 0,
        reason if isinstance(reason, str) else None,
        _FINISH,
    )


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]] | None:
    """Absent, not empty, when the model cited nothing. Returning None is how a
    `$call` tells the interpreter to omit the key."""
    c = extract_citations("messages", _raw(ctx))
    return c or None


def _attach_tool_output(ctx: Ctx) -> None:
    """Attach a tool result's stdout to the call it belongs to."""
    b = _block(ctx)
    block_type = b.get("type")
    if not isinstance(block_type, str) or not block_type.endswith("_tool_result"):
        return
    output = result_stdout(b.get("content"))
    if not output:
        return
    for call in _out(ctx)["builtinToolCalls"]:
        if call.get("id") == b.get("tool_use_id"):
            call["output"] = output
            return


ANTHROPIC_RESPONSE_REGISTRY = Registry(
    transforms={
        "anthropicBuiltinCall": _builtin_call,
        "anthropicCodeExecFiles": _code_exec_files,
        "anthropicText": _text,
        "anthropicUsageFull": _usage_full,
        "anthropicFinish": _finish,
        "anthropicCitations": _citations,
    },
    effects={"anthropicAttachToolOutput": _attach_tool_output},
)

__all__ = ["ANTHROPIC_RESPONSE_REGISTRY"]
