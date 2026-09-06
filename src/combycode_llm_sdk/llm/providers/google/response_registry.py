"""The named escape hatches `google.generate.response.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/google/response-registry.ts`.

Google's `parts[]` has no type tag: a part IS a text part because it has a
`text` key, and a tool call because it has `functionCall`. So every rule here
returns None when the part is not its kind, and the spec lists them all against
the same array -- which is exactly what the hand-written loop does with a run of
independent `if`s.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from .._shared.citations import extract_citations
from .._shared.constants import AUDIO_PCM16_SAMPLE_RATE_HZ
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import google_usage


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _part(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.item.value if ctx.item else {}
    return v if isinstance(v, Mapping) else {}


def _candidate(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    cands = raw.get("candidates")
    if isinstance(cands, list) and cands and isinstance(cands[0], Mapping):
        return cands[0]
    return {}


def _parts_of(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    content = _candidate(raw).get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    return [p for p in parts if isinstance(p, Mapping)] if isinstance(parts, list) else []


def _has_code_exec(raw: Mapping[str, Any]) -> bool:
    """When the turn ran hosted code execution its inlineData blobs are ARTIFACTS
    (a generated chart), not conversational media. That is a property of the whole
    parts array, so it cannot be decided from one part."""
    return any(p.get("executableCode") or p.get("codeExecutionResult") for p in _parts_of(raw))


_FINISH = {
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    # Verified in google-ts src/types.ts:510. Unmapped it fell through to `stop`,
    # so a turn that failed to produce a usable tool call looked like a clean
    # finish with no content.
    "MALFORMED_FUNCTION_CALL": "malformed_tool_call",
}


def _text_part(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """`text is not None`, not truthiness: an empty string is still a text part."""
    p = _part(ctx)
    if p.get("text") is None or p.get("thought"):
        return None
    return {"type": "text", "text": p.get("text")}


def _thinking(_arg: Any, ctx: Ctx) -> str | None:
    """A part flagged `thought` carries the reasoning rather than the answer."""
    p = _part(ctx)
    return p.get("text") if p.get("thought") and p.get("text") else None


def _inline_file(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    p = _part(ctx)
    inline = p.get("inlineData")
    if not isinstance(inline, Mapping) or not _has_code_exec(_raw(ctx)):
        return None
    return {
        "data": inline.get("data"),
        "mimeType": inline.get("mimeType"),
        "source": "code_execution",
    }


def _inline_media(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    p = _part(ctx)
    inline = p.get("inlineData")
    if not isinstance(inline, Mapping) or _has_code_exec(_raw(ctx)):
        return None
    mime = inline.get("mimeType") or ""
    base = {"mediaId": "", "mimeType": mime, "_data": inline.get("data")}
    if mime.startswith("image/"):
        return {"type": "image_output", **base}
    if mime.startswith("audio/"):
        return {"type": "audio_output", **base, "sampleRate": AUDIO_PCM16_SAMPLE_RATE_HZ}
    if mime.startswith("video/"):
        return {"type": "video_output", **base}
    return None


def _tool_call(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    p = _part(ctx)
    fc = p.get("functionCall")
    if not isinstance(fc, Mapping):
        return None
    out: dict[str, Any] = {
        "type": "tool_call",
        "id": fc.get("id") if fc.get("id") is not None else str(uuid.uuid4()),
        "name": fc.get("name"),
        "arguments": fc.get("args") or {},
    }
    if p.get("thoughtSignature"):
        out["_meta"] = {"thoughtSignature": p["thoughtSignature"]}
    return out


def _builtin_calls(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Hosted tools, all decided from the whole parts array and the candidate
    metadata rather than from any single part: Google issues no call ids."""
    raw = _raw(ctx)
    parts = _parts_of(raw)
    cand = _candidate(raw)
    calls: list[dict[str, Any]] = []

    codes = [
        p["executableCode"]["code"]
        for p in parts
        if isinstance(p.get("executableCode"), Mapping) and p["executableCode"].get("code")
    ]
    outputs = [
        str(p["codeExecutionResult"].get("output") or "")
        for p in parts
        if isinstance(p.get("codeExecutionResult"), Mapping)
    ]
    if codes:
        for i, code in enumerate(codes):
            call: dict[str, Any] = {"tool": "code_interpreter", "code": code}
            if i < len(outputs) and outputs[i]:
                call["output"] = outputs[i]
            calls.append(call)
    elif _has_code_exec(raw):
        call = {"tool": "code_interpreter"}
        if outputs and outputs[0]:
            call["output"] = outputs[0]
        calls.append(call)

    grounding = cand.get("groundingMetadata")
    if grounding:
        queries = grounding.get("webSearchQueries") if isinstance(grounding, Mapping) else None
        q = queries[0] if isinstance(queries, list) and queries else None
        call = {"tool": "web_search"}
        if isinstance(q, str):
            call["query"] = q
        calls.append(call)

    url_ctx = cand.get("urlContextMetadata")
    url_meta = url_ctx.get("urlMetadata") if isinstance(url_ctx, Mapping) else None
    if isinstance(url_meta, list):
        for m in url_meta:
            if not isinstance(m, Mapping):
                continue
            call = {"tool": "web_fetch"}
            if isinstance(m.get("retrievedUrl"), str):
                call["url"] = m["retrievedUrl"]
            calls.append(call)
    return calls


def _id(_arg: Any, ctx: Ctx) -> str:
    """generateContent DOES return an id -- `responseId`. The fallback stays for
    older payloads, but minting one unconditionally made the parse
    non-deterministic: the same bytes produced a different id every time."""
    rid = _raw(ctx).get("responseId")
    return rid if isinstance(rid, str) else str(uuid.uuid4())


def _text(_arg: Any, ctx: Ctx) -> str:
    return "".join(
        p.get("text") or ""
        for p in _out(ctx)["content"]
        if isinstance(p, Mapping) and p.get("type") == "text"
    )


def _usage(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    u = _raw(ctx).get("usageMetadata")
    return google_usage(u if isinstance(u, Mapping) else None)


def _finish(_arg: Any, ctx: Ctx) -> str:
    reason = _candidate(_raw(ctx)).get("finishReason")
    return extract_finish_reason(
        len(_out(ctx)["toolCalls"]) > 0,
        reason if isinstance(reason, str) else None,
        _FINISH,
    )


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]] | None:
    c = extract_citations("generate", _raw(ctx))
    return c or None


GOOGLE_RESPONSE_REGISTRY = Registry(
    transforms={
        "googleTextPart": _text_part,
        "googleThinking": _thinking,
        "googleInlineFile": _inline_file,
        "googleInlineMedia": _inline_media,
        "googleToolCall": _tool_call,
        "googleBuiltinCalls": _builtin_calls,
        "googleId": _id,
        "googleText": _text,
        "googleUsageFull": _usage,
        "googleFinish": _finish,
        "googleCitations": _citations,
    },
)

__all__ = ["GOOGLE_RESPONSE_REGISTRY"]
