"""The named escape hatches `anthropic.messages.stream.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/anthropic/stream-registry.ts`.

Anthropic is the hardest of the five stream parsers, and all of its difficulty
is in one mechanism: a `server_tool_use` block announces a hosted tool call, its
input JSON arrives in fragments across later events, the block closes, and only
THEN can the input be parsed and filed against the id that the `*_tool_result`
block will later ask for. Three events apart, and none of them can be understood
alone.

That is what the effects below are for. Everything that is a straight mapping --
text and thinking deltas, the finish reason, usage -- stays in the spec.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from .._shared.builtin_tools import unified_builtin_tool
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import (
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


def _delta(ctx: Ctx) -> Mapping[str, Any]:
    d = _raw(ctx).get("delta")
    return d if isinstance(d, Mapping) else {}


_FINISH = {
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}


def _citation(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """The citation the ANSWER makes, which is not the same as the search results
    in the `web_search_tool_result` block: the model retrieves several pages and
    cites some of them. None when it carries no url, which omits the emit.
    """
    cite = _delta(ctx).get("citation")
    cite = cite if isinstance(cite, Mapping) else {}
    url = cite.get("url")
    if not isinstance(url, str) or not url:
        return None
    citation: dict[str, Any] = {"url": url}
    if cite.get("title"):
        citation["title"] = cite["title"]
    if cite.get("cited_text"):
        citation["text"] = cite["cited_text"]
    return {"type": "citation", "citation": citation}


def _message_delta(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Carries the running usage and, at the end, the stop reason. Either, both
    or neither may be present."""
    raw = _raw(ctx)
    events: list[dict[str, Any]] = []
    usage = raw.get("usage")
    if isinstance(usage, Mapping):
        events.append({"type": "usage", "usage": anthropic_usage(usage)})
    sr = _delta(ctx).get("stop_reason")
    if isinstance(sr, str) and sr:
        events.append(
            {
                "type": "done",
                "finishReason": extract_finish_reason(sr == "tool_use", sr, _FINISH),
            }
        )
    return events


def _message_start(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """The opening frame carries the prompt-side usage."""
    msg = _raw(ctx).get("message")
    usage = msg.get("usage") if isinstance(msg, Mapping) else None
    if not isinstance(usage, Mapping):
        return []
    return [{"type": "usage", "usage": anthropic_usage(usage)}]


def _json_delta(ctx: Ctx) -> None:
    """An input fragment either feeds the open hosted-tool block or, when none is
    open, IS a plain function tool call's arguments. Same event type, opposite
    meaning, decided entirely by carried state."""
    out = _out(ctx)
    delta = _delta(ctx)
    if out["current"]:
        out["current"]["json"] += delta.get("partial_json") or ""
        return
    out["events"].append(
        {"type": "tool_call_delta", "id": "", "arguments": delta.get("partial_json")}
    )


def _block_start(ctx: Ctx) -> None:
    out = _out(ctx)
    block = _raw(ctx).get("content_block")
    block = block if isinstance(block, Mapping) else {}
    block_type = block.get("type")

    # A regular function tool. Returns immediately in the hand-written parser, so
    # no file extraction runs for it.
    if block_type == "tool_use":
        out["current"] = None
        out["events"].append(
            {"type": "tool_call_start", "id": block.get("id"), "name": block.get("name")}
        )
        return

    if block_type == "server_tool_use":
        tool = unified_builtin_tool(str(block.get("name")))
        out["current"] = {"id": block.get("id") or "", "tool": tool, "json": ""}
        started: dict[str, Any] = {"type": "builtin_tool_start", "tool": tool}
        if isinstance(block.get("id"), str):
            started["id"] = block["id"]
        out["events"].append(started)
    elif isinstance(block_type, str) and block_type.endswith("_tool_result"):
        tool = unified_builtin_tool(block_type)
        call_id = block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else None
        inp = out["pending"].pop(call_id, None) if call_id else None
        output = result_stdout(block.get("content"))
        ended: dict[str, Any] = {"type": "builtin_tool_end", "tool": tool}
        if call_id:
            ended["id"] = call_id
        if inp:
            for key in ("code", "query", "url"):
                if inp.get(key):
                    ended[key] = inp[key]
        if output:
            ended["output"] = output
        out["events"].append(ended)

    # Server-computed code-execution results arrive complete here rather than
    # token-streamed, so their output files are surfaced at the same moment.
    for f in files_from_code_exec_block(block):
        out["events"].append({"type": "file", "file": f})


def _block_stop(ctx: Ctx) -> None:
    """Close an open hosted-tool block: parse what accumulated and file it by id
    for the result block that will ask for it."""
    out = _out(ctx)
    cur = out["current"]
    if not cur:
        return
    try:
        parsed = json.loads(cur["json"] or "{}")
    except (ValueError, TypeError):
        parsed = {}  # partial/invalid JSON -> no payload
    out["pending"][cur["id"]] = builtin_input_payload(
        cur["tool"], parsed if isinstance(parsed, Mapping) else {}
    )
    out["current"] = None


ANTHROPIC_STREAM_REGISTRY = Registry(
    transforms={
        "anthropicStreamCitation": _citation,
        "anthropicStreamMessageDelta": _message_delta,
        "anthropicStreamMessageStart": _message_start,
    },
    effects={
        "anthropicStreamJsonDelta": _json_delta,
        "anthropicStreamBlockStart": _block_start,
        "anthropicStreamBlockStop": _block_stop,
    },
)

__all__ = ["ANTHROPIC_STREAM_REGISTRY"]
