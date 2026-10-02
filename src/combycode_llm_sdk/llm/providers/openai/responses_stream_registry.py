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
from ...cache_diagnostics import openai_cache_diagnostics
from ...moderation.native import parse_native_moderation
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import (
    builtin_call_from_responses_item,
    files_from_responses_output_item,
    openai_responses_usage,
    shell_awaits_caller,
    shell_commands,
    shell_environment_name,
    shell_output_text,
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
    """The code, query or url a hosted tool ran, carried on its end event.

    `callId` / `environment` are shell-only and ride along for parity: the streamed
    `builtinToolCalls` is assembled from these events, so leaving them off would
    mean the same turn told you where its commands ran only if you did not stream
    it.
    """
    out: dict[str, Any] = {}
    for key in ("id", "code", "output", "query", "url", "callId", "environment"):
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
    diagnostics = openai_cache_diagnostics(response.get("prompt_cache_diagnostics"))
    if diagnostics is not None:
        events.append({"type": "cache_diagnostics", "diagnostics": diagnostics})
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
    # `shell_call_output` is the SECOND item of one shell invocation, so it maps to
    # a builtin call but must not announce a second start -- measured: doing so
    # reported `starts=2 ends=1` for a single `echo`.
    builtin = (
        None
        if item.get("type") == "shell_call_output"
        else builtin_call_from_responses_item(item)
    )
    if builtin:
        started: dict[str, Any] = {"type": "builtin_tool_start", "tool": builtin["tool"]}
        if builtin.get("id"):
            started["id"] = builtin["id"]
        out["events"].append(started)


def _item_done(ctx: Ctx) -> None:
    out = _out(ctx)
    raw = _raw(ctx)
    item = _item(ctx)

    # A shell call is the one builtin whose result may arrive in a LATER item, so it
    # cannot simply end here. Three cases, all measured 2026-10-02:
    #   local      -> no output item will ever come; end now, with the commands.
    #   container  -> hold the call; the `shell_call_output` item below ends it.
    #   the output -> end the held call, now carrying stdout/stderr.
    if item.get("type") == "shell_call" and not shell_awaits_caller(item):
        held: dict[str, Any] = {
            "code": shell_commands(item),
            "environment": shell_environment_name(item),
        }
        if isinstance(item.get("id"), str):
            held["id"] = item["id"]
        if isinstance(item.get("call_id"), str):
            held["callId"] = item["call_id"]
        out["openShell"] = held
    elif item.get("type") == "shell_call_output":
        waiting = out.get("openShell") or {}
        out["openShell"] = None
        output = shell_output_text(item)
        ended: dict[str, Any] = {"type": "builtin_tool_end", "tool": "shell"}
        for key in ("id", "code", "callId", "environment"):
            if waiting.get(key):
                ended[key] = waiting[key]
        if output:
            ended["output"] = output
        out["events"].append(ended)
    else:
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


def _shell_command_delta(ctx: Ctx) -> None:
    """The command text OpenAI streams while the model composes it.

    Per `command_index`, because one shell call can ask for several commands; the
    index is not forwarded because `code` is a fragment to append and the commands
    read as one script, exactly as `builtin_tool_end.code` joins them. `.added`
    carries `command: ""` and `.done` repeats the finished command, so only `.delta`
    becomes an event -- forwarding `.done` as well would duplicate every command in
    a consumer that appends what it is given.
    """
    delta = _raw(ctx).get("delta")
    if isinstance(delta, str) and delta:
        _out(ctx)["events"].append(
            {"type": "builtin_tool_delta", "tool": "shell", "code": delta}
        )


def _shell_output_delta(ctx: Ctx) -> None:
    """stdout/stderr as the provider's container produces it.

    `delta` is a MAPPING here (`{"stdout": ...}` or `{"stderr": ...}`), not a string
    -- measured, and the one shape in this group that is not a plain delta. Both
    streams become `output` rather than being split: they interleave in the order the
    command wrote them, which is the order a reader needs.
    """
    delta = _raw(ctx).get("delta")
    if not isinstance(delta, Mapping):
        return
    text = "".join(
        value
        for key in ("stdout", "stderr")
        for value in [delta.get(key)]
        if isinstance(value, str) and value
    )
    if text:
        _out(ctx)["events"].append(
            {"type": "builtin_tool_delta", "tool": "shell", "output": text}
        )


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
        "oaiRespStreamShellCommandDelta": _shell_command_delta,
        "oaiRespStreamShellOutputDelta": _shell_output_delta,
    },
)

__all__ = ["OPENAI_RESPONSES_STREAM_REGISTRY"]
