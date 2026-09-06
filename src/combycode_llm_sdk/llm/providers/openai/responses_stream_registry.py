"""The named escape hatches `openai.responses.stream.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/openai/responses-stream-registry.ts`.

The Responses stream is a clean discriminated union of event types, so most of
it is `cases`. Only one thing spans events: `phase` is announced once on
`response.output_item.added` but belongs on every text delta of that item, and
the deltas carry only `item_id`.

`oaiRespStreamFiles` is a rule of its own rather than part of the item-done
effect on purpose: xAI overrides file extraction, and on the buffered side
calling the module function directly is exactly how its override got silently
dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ...moderation.native import parse_native_moderation
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import (
    builtin_call_from_responses_item,
    files_from_responses_output_item,
    openai_responses_usage,
)


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _item(ctx: Ctx) -> Mapping[str, Any]:
    v = _raw(ctx).get("item")
    return v if isinstance(v, Mapping) else {}


def _end_payload(call: Mapping[str, Any]) -> dict[str, Any]:
    """The code, query or url a hosted tool ran, carried on its end event."""
    out: dict[str, Any] = {}
    for key in ("id", "code", "output", "query", "url"):
        if call.get(key):
            out[key] = call[key]
    return out


def _citation(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    note = _raw(ctx).get("annotation")
    note = note if isinstance(note, Mapping) else {}
    if note.get("type") != "url_citation" or not note.get("url"):
        return None
    citation: dict[str, Any] = {"url": note["url"]}
    if note.get("title"):
        citation["title"] = note["title"]
    return {"type": "citation", "citation": citation}


def _text_delta(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    """`item_id` says WHICH output item this delta belongs to -- a turn can
    interleave deltas from several items, so it is passed through for consumers
    that reassemble per item instead of concatenating."""
    raw = _raw(ctx)
    item_id = raw.get("item_id") if isinstance(raw.get("item_id"), str) else None
    phase = _out(ctx)["phaseByItem"].get(item_id) if item_id else None
    out: dict[str, Any] = {"type": "text", "text": raw.get("delta")}
    if item_id:
        out["itemId"] = item_id
    if phase is not None:
        out["phase"] = phase
    return out


def _files(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Returns STREAM EVENTS, not bare files: the unified event wraps the file as
    `{type: 'file', file}`, and returning the payload unwrapped spliced raw
    FileOutputs into the event list."""
    return [{"type": "file", "file": f} for f in files_from_responses_output_item(_item(ctx))]


def _completed(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """The terminal frame: moderation first, then usage, then done."""
    raw = _raw(ctx)
    response = raw.get("response")
    response = response if isinstance(response, Mapping) else raw
    events: list[dict[str, Any]] = []
    moderation = parse_native_moderation(response.get("moderation"))
    if moderation and moderation.get("input"):
        events.append(
            {
                "type": "moderation",
                "phase": "input",
                "result": moderation["input"],
                "source": "native",
            }
        )
    if moderation and moderation.get("output"):
        events.append(
            {
                "type": "moderation",
                "phase": "output",
                "result": moderation["output"],
                "source": "native",
            }
        )
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        events.append({"type": "usage", "usage": openai_responses_usage(usage)})
    status = response.get("status")
    events.append(
        {
            "type": "done",
            "finishReason": extract_finish_reason(
                False, status if isinstance(status, str) else None, {"incomplete": "length"}
            ),
        }
    )
    return events


def _item_added(ctx: Ctx) -> None:
    out = _out(ctx)
    raw = _raw(ctx)
    item = _item(ctx)

    # Remember this item's phase: its text deltas carry only `item_id`.
    if item.get("type") == "message" and isinstance(item.get("phase"), str):
        item_id = raw.get("item_id") if isinstance(raw.get("item_id"), str) else item.get("id")
        if item_id:
            out["phaseByItem"][item_id] = item["phase"]
    if item.get("type") == "function_call":
        out["events"].append(
            {
                "type": "tool_call_start",
                "id": item.get("call_id") or "",
                "name": item.get("name") or "",
            }
        )
    if item.get("type") == "image_generation_call":
        out["events"].append(
            {"type": "media_start", "mediaType": "image", "mimeType": "image/png"}
        )
    builtin = builtin_call_from_responses_item(item)
    if builtin:
        started: dict[str, Any] = {"type": "builtin_tool_start", "tool": builtin["tool"]}
        if builtin.get("id"):
            started["id"] = builtin["id"]
        out["events"].append(started)


def _item_done(ctx: Ctx) -> None:
    out = _out(ctx)
    raw = _raw(ctx)
    item = _item(ctx)

    builtin = builtin_call_from_responses_item(item)
    if builtin:
        out["events"].append(
            {"type": "builtin_tool_end", "tool": builtin["tool"], **_end_payload(builtin)}
        )
    if item.get("type") == "function_call":
        out["events"].append({"type": "tool_call_end", "id": item.get("call_id") or ""})
    if item.get("type") == "image_generation_call":
        out["events"].append({"type": "media_end"})
    if item.get("type") == "reasoning":
        summary = item.get("summary")
        rows = [s for s in summary if isinstance(s, Mapping)] if isinstance(summary, list) else []
        text = "\n".join(s.get("text") or "" for s in rows if s.get("type") == "summary_text")
        item_id = item.get("id") if isinstance(item.get("id"), str) else raw.get("item_id")
        if text:
            event: dict[str, Any] = {"type": "thinking", "text": text}
            if item_id:
                event["itemId"] = item_id
            out["events"].append(event)


OPENAI_RESPONSES_STREAM_REGISTRY = Registry(
    transforms={
        "oaiRespStreamCitation": _citation,
        "oaiRespStreamTextDelta": _text_delta,
        "oaiRespStreamFiles": _files,
        "oaiRespStreamCompleted": _completed,
    },
    effects={
        "oaiRespStreamItemAdded": _item_added,
        "oaiRespStreamItemDone": _item_done,
    },
)

__all__ = ["OPENAI_RESPONSES_STREAM_REGISTRY"]
