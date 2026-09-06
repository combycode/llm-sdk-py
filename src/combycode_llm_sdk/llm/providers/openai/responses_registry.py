"""The named escape hatches `openai.responses.response.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/openai/responses-registry.ts`.

Responses is the richest of the seven parsers: one `output[]` array carrying
messages, reasoning, function calls, programs, program output and generated
images, plus two things that apply to EVERY item whatever its type (output files
and hosted builtin-tool calls).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ...moderation.native import parse_native_moderation
from .._shared.citations import extract_citations
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import (
    builtin_call_from_responses_item,
    files_from_responses_output_item,
    from_wire_caller,
    openai_responses_usage,
)
from .tiers import openai_billed_tier


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _item(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.item.value if ctx.item else {}
    return v if isinstance(v, Mapping) else {}


def _rows(v: Any) -> list[Mapping[str, Any]]:
    return [x for x in v if isinstance(x, Mapping)] if isinstance(v, list) else []


def _text_from_messages(raw: Mapping[str, Any]) -> str:
    """The text a message item contributes, concatenated in output order.

    Computed from `raw` rather than from what has been collected, so it does not
    depend on which phase asks.
    """
    text = ""
    for item in _rows(raw.get("output")):
        if item.get("type") != "message":
            continue
        for c in _rows(item.get("content")):
            if c.get("type") == "output_text":
                text += c.get("text") or ""
    return text


_FINISH = {
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
    "queued": "pending",
    "in_progress": "pending",
}


def _files(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Applies to every item: container-file annotations and code-interpreter
    images, whatever the item's type."""
    return files_from_responses_output_item(_item(ctx))


def _builtin_call(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """Also every item: the hosted-tool trail. None when this item is not one."""
    return builtin_call_from_responses_item(_item(ctx))


def _message_parts(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """A message contributes one text part per `output_text`, carrying `phase`
    where the model sets it (codex-family narration vs the answer).

    `phase` normally lives on the MESSAGE item -- that is what the wire sends
    and what the TypeScript groups by. A content part carrying its own is
    strictly more specific, so it wins where present: a message whose parts
    disagree about phase can only be described by reading each one, and
    ignoring that would fold a narration and its answer into a single phase.
    """
    item = _item(ctx)
    fallback = item.get("phase") if isinstance(item.get("phase"), str) else None
    parts: list[dict[str, Any]] = []
    for c in _rows(item.get("content")):
        if c.get("type") != "output_text":
            continue
        own = c.get("phase") if isinstance(c.get("phase"), str) else None
        part: dict[str, Any] = {"type": "text", "text": c.get("text")}
        phase = own or fallback
        if phase is not None:
            part["phase"] = phase
        parts.append(part)
    return parts


def _thinking(_arg: Any, ctx: Ctx) -> str | None:
    """The reasoning summary, when the model returned one. None leaves the
    previous value alone, matching `if (summaryText) thinking = summaryText`."""
    summary = _rows(_item(ctx).get("summary"))
    text = "\n".join(s.get("text") or "" for s in summary if s.get("type") == "summary_text")
    return text or None


def _self(_arg: Any, ctx: Ctx) -> Mapping[str, Any]:
    """Kept so a `program` item can carry the reasoning that preceded it."""
    return _item(ctx)


def _tool_call(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    item = _item(ctx)
    caller = from_wire_caller(item.get("caller"))
    args = item.get("arguments")
    out: dict[str, Any] = {
        "type": "tool_call",
        "id": item.get("call_id") if item.get("call_id") is not None else item.get("id"),
        "name": item.get("name"),
        "arguments": json.loads(args) if isinstance(args, str) else (args or {}),
    }
    if caller:
        out["caller"] = caller
    return out


def _program(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    item = _item(ctx)
    bound = _out(ctx)["reasoningItems"]
    meta: dict[str, Any] = {}
    if isinstance(item.get("id"), str):
        meta["itemId"] = item["id"]
    if bound:
        meta["boundItems"] = list(bound)
    return {
        "type": "program_call",
        "id": item.get("call_id") if item.get("call_id") is not None else item.get("id"),
        "code": item.get("code") or "",
        "fingerprint": item.get("fingerprint") or "",
        "_meta": meta,
    }


def _program_result(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    item = _item(ctx)
    out: dict[str, Any] = {
        "type": "program_result",
        "id": item.get("call_id") if item.get("call_id") is not None else item.get("id"),
        "result": item.get("result") or "",
    }
    if isinstance(item.get("status"), str):
        out["status"] = item["status"]
    # Required on the way back in, unlike every other item we echo.
    if isinstance(item.get("id"), str):
        out["_meta"] = {"itemId": item["id"]}
    return out


def _image(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    item = _item(ctx)
    data = item.get("result")
    if not data:
        return None
    fmt = item.get("output_format")
    mime = "image/jpeg" if fmt == "jpeg" else "image/webp" if fmt == "webp" else "image/png"
    return {
        "type": "image_output",
        "mediaId": "",
        "mimeType": mime,
        "revisedPrompt": item.get("revised_prompt"),
        "_data": data,
    }


def _fallback_text_part(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """The `output_text` convenience field, used only when no message produced
    text AND nothing else landed in content."""
    raw = _raw(ctx)
    if _text_from_messages(raw):
        return None
    ot = raw.get("output_text")
    if not isinstance(ot, str) or not ot:
        return None
    if len(_out(ctx)["content"]) != 0:
        return None
    return {"type": "text", "text": ot}


def _text(_arg: Any, ctx: Ctx) -> str:
    raw = _raw(ctx)
    text = _text_from_messages(raw)
    if text:
        return text
    ot = raw.get("output_text")
    return ot if isinstance(ot, str) else ""


def _usage(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    raw = _raw(ctx)
    u = raw.get("usage")
    return {
        **openai_responses_usage(u if isinstance(u, Mapping) else None),
        **openai_billed_tier(raw.get("service_tier")),
    }


def _finish(_arg: Any, ctx: Ctx) -> str:
    """`incomplete` carries a sub-reason: a content_filter block must not be
    reported as a length truncation."""
    raw = _raw(ctx)
    details = raw.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, Mapping) else None
    if reason == "content_filter":
        return "content_filter"
    status = raw.get("status")
    return extract_finish_reason(
        len(_out(ctx)["toolCalls"]) > 0,
        status if isinstance(status, str) else None,
        _FINISH,
    )


def _error(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """A Responses call can fail INSIDE a 200, so there is no exception to catch
    and this is the only signal the caller gets."""
    e = _raw(ctx).get("error")
    if not isinstance(e, Mapping):
        return None
    if e.get("code") is None and e.get("message") is None:
        return None
    out: dict[str, Any] = {}
    if isinstance(e.get("code"), str):
        out["code"] = e["code"]
    if isinstance(e.get("message"), str):
        out["message"] = e["message"]
    return out or None


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]] | None:
    c = extract_citations("responses", _raw(ctx))
    return c or None


def _moderation(_arg: Any, ctx: Ctx) -> Any:
    return parse_native_moderation(_raw(ctx).get("moderation"))


OPENAI_RESPONSES_REGISTRY = Registry(
    transforms={
        "oaiRespFiles": _files,
        "oaiRespBuiltinCall": _builtin_call,
        "oaiRespMessageParts": _message_parts,
        "oaiRespThinking": _thinking,
        "oaiRespSelf": _self,
        "oaiRespToolCall": _tool_call,
        "oaiRespProgram": _program,
        "oaiRespProgramResult": _program_result,
        "oaiRespImage": _image,
        "oaiRespFallbackTextPart": _fallback_text_part,
        "oaiRespText": _text,
        "oaiRespUsage": _usage,
        "oaiRespFinish": _finish,
        "oaiRespError": _error,
        "oaiRespCitations": _citations,
        "oaiRespModeration": _moderation,
    },
)

__all__ = ["OPENAI_RESPONSES_REGISTRY"]
