"""The named escape hatches `google.interactions.response.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/google/interactions-registry.ts`.

Interactions differs from generateContent in one structural way: the items are
not a field of the response, they are the FLATTENING of `step_list`. A
`model_output` step carries a `content[]` of typed parts; a `thought` step has no
content and stands for itself. So the spec flattens first, into an internal
accumulator, and collects over that.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from .._shared.citations import extract_citations
from .._shared.constants import AUDIO_PCM16_SAMPLE_RATE_HZ
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import google_interactions_usage


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _item(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.item.value if ctx.item else {}
    return v if isinstance(v, Mapping) else {}


#: Default mime per media kind, used when the item declares none.
_DEFAULT_MIME = {"image": "image/png", "audio": "audio/pcm", "video": "video/mp4"}
_PART_TYPE = {"image": "image_output", "audio": "audio_output", "video": "video_output"}

_FINISH = {
    "failed": "error",
    # `queued` joined InteractionStatus in google 2.13: accepted but not yet run,
    # so it carries no completion and 'stop' would claim a finish that never was.
    "queued": "pending",
    "in_progress": "pending",
}


def _flatten(_arg: Any, ctx: Ctx) -> list[Mapping[str, Any]]:
    """`steps` (or the legacy `outputs`) flattened into typed items."""
    raw = _raw(ctx)
    steps = raw.get("steps")
    if not isinstance(steps, list):
        steps = raw.get("outputs")
    items: list[Mapping[str, Any]] = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, Mapping):
            continue
        content = step.get("content")
        if isinstance(content, list):
            items.extend(c for c in content if isinstance(c, Mapping))
        else:
            items.append(step)
    return items


def _text_part(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    return {"type": "text", "text": _item(ctx).get("text")}


def _tool_call(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    item = _item(ctx)
    return {
        "type": "tool_call",
        "id": item.get("id") if item.get("id") is not None else str(uuid.uuid4()),
        "name": item.get("name"),
        "arguments": item.get("arguments") or {},
    }


def _media_part(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    item = _item(ctx)
    kind = str(item.get("type"))
    mime = item.get("mime_type")
    if not isinstance(mime, str) or not mime:
        mime = item.get("mimeType")
    if not isinstance(mime, str) or not mime:
        mime = _DEFAULT_MIME.get(kind, "")
    part: dict[str, Any] = {
        "type": _PART_TYPE.get(kind),
        "mediaId": "",
        "mimeType": mime,
        "_data": item.get("data") or "",
    }
    if kind == "audio":
        part["sampleRate"] = AUDIO_PCM16_SAMPLE_RATE_HZ
    return part


def _id(_arg: Any, ctx: Ctx) -> str:
    """The response's identity -- and, on this API, the server-side handle.

    `name` is where Interactions puts it (`interactions/int_123`); there is no
    `id` field. The TypeScript reads `id` and falls through to a minted uuid,
    which means the handle a caller passes back as `state=` is one the provider
    has never seen. It still LOOKS like it works -- the request is well-formed
    and the reply is fine -- so the failure is that the conversation silently
    starts over on every turn, which is the exact saving the API exists for.

    The uuid stays as the last resort: it is a local identity for a response
    that carried none, which is worth having for message origin even though it
    is worthless as a handle.
    """
    raw = _raw(ctx)
    for key in ("id", "name"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return str(uuid.uuid4())


def _text(_arg: Any, ctx: Ctx) -> str:
    return "".join(
        p.get("text") or ""
        for p in _out(ctx)["content"]
        if isinstance(p, Mapping) and p.get("type") == "text"
    )


def _usage(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    u = _raw(ctx).get("usage")
    return google_interactions_usage(u if isinstance(u, Mapping) else None)


def _finish(_arg: Any, ctx: Ctx) -> str:
    status = _raw(ctx).get("status")
    return extract_finish_reason(
        len(_out(ctx)["toolCalls"]) > 0,
        status if isinstance(status, str) else None,
        _FINISH,
    )


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]] | None:
    c = extract_citations("interactions", _raw(ctx))
    return c or None


GOOGLE_INTERACTIONS_REGISTRY = Registry(
    transforms={
        "gaFlatten": _flatten,
        "gaTextPart": _text_part,
        "gaToolCall": _tool_call,
        "gaMediaPart": _media_part,
        "gaId": _id,
        "gaText": _text,
        "gaUsage": _usage,
        "gaFinish": _finish,
        "gaCitations": _citations,
    },
)

__all__ = ["GOOGLE_INTERACTIONS_REGISTRY"]
