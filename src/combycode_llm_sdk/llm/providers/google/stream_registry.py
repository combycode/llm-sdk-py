"""The named escape hatches `google.generate.stream.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/google/stream-registry.ts`.

Google's parts carry no type tag, so each rule guards itself by returning None
when the part is not its kind -- the same shape as the buffered google registry,
and the same shape as the hand-written run of `if`s.

Three things span events, and all three are why this provider needs state at
all: a code-execution marker may share a chunk with its output file OR precede
it, so the flag is latched before the parts are walked; the code from an
`executableCode` part belongs on the `builtin_tool_end` of a LATER part; and
grounding metadata has no per-call markers, so the start/end pair is emitted once
per stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry, js_json
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


def _candidate(ctx: Ctx) -> Mapping[str, Any]:
    cands = _raw(ctx).get("candidates")
    if isinstance(cands, list) and cands and isinstance(cands[0], Mapping):
        return cands[0]
    return {}


def _parts_of(ctx: Ctx) -> list[Mapping[str, Any]]:
    content = _candidate(ctx).get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    return [p for p in parts if isinstance(p, Mapping)] if isinstance(parts, list) else []


def _usage(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Empty unless this chunk carries usage, so the same rule serves the
    candidate-less chunk and the trailing usage on a normal one."""
    u = _raw(ctx).get("usageMetadata")
    if not isinstance(u, Mapping):
        return []
    return [{"type": "usage", "usage": google_usage(u)}]


def _text(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """`text is not None`, not truthiness: an empty string is still text."""
    p = _part(ctx)
    if p.get("text") is None or p.get("thought"):
        return None
    return {"type": "text", "text": p.get("text")}


def _thinking(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    p = _part(ctx)
    if not (p.get("thought") and p.get("text")):
        return None
    return {"type": "thinking", "text": p["text"]}


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Grounding chunks arrive on ONE late chunk rather than spread across the
    stream -- the first `groundingMetadata` seen is usually `{}` and the
    populated one comes near the end. So this reads whichever chunk actually has
    them instead of latching on first sight the way the pair below does."""
    grounding = _candidate(ctx).get("groundingMetadata")
    chunks = grounding.get("groundingChunks") if isinstance(grounding, Mapping) else None
    out: list[dict[str, Any]] = []
    for chunk in chunks if isinstance(chunks, list) else []:
        if not isinstance(chunk, Mapping):
            continue
        web = chunk.get("web")
        web = web if isinstance(web, Mapping) else {}
        if web.get("uri"):
            citation: dict[str, Any] = {"url": web["uri"]}
            if web.get("title"):
                citation["title"] = web["title"]
            out.append({"type": "citation", "citation": citation})
    return out


def _done(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    fr = _candidate(ctx).get("finishReason")
    if not isinstance(fr, str) or not fr:
        return None
    return {
        "type": "done",
        "finishReason": extract_finish_reason(False, fr, {"MAX_TOKENS": "length"}),
    }


def _latch_code_exec(ctx: Ctx) -> None:
    """Latch the flag from ALL parts before any is handled: the marker may share
    a chunk with its output file or precede it, and it decides whether inlineData
    is an artifact or conversational media."""
    out = _out(ctx)
    for p in _parts_of(ctx):
        if p.get("executableCode") or p.get("codeExecutionResult"):
            out["codeExec"] = True


def _code_exec(ctx: Ctx) -> None:
    """The code to run, then its result. The code is carried on the END event,
    which belongs to a different part than the one that announced it."""
    out = _out(ctx)
    p = _part(ctx)
    ec = p.get("executableCode")
    if isinstance(ec, Mapping):
        code = ec.get("code")
        out["pendingCode"] = code if isinstance(code, str) else None
        out["events"].append({"type": "builtin_tool_start", "tool": "code_interpreter"})
    cer = p.get("codeExecutionResult")
    if isinstance(cer, Mapping):
        output = cer.get("output")
        ended: dict[str, Any] = {"type": "builtin_tool_end", "tool": "code_interpreter"}
        if out["pendingCode"]:
            ended["code"] = out["pendingCode"]
        if isinstance(output, str) and output:
            ended["output"] = output
        out["events"].append(ended)
        out["pendingCode"] = None


def _inline(ctx: Ctx) -> None:
    """An artifact when the turn ran code, three media events otherwise."""
    out = _out(ctx)
    p = _part(ctx)
    inline = p.get("inlineData")
    if not isinstance(inline, Mapping):
        return
    mime = inline.get("mimeType") or ""
    if out["codeExec"]:
        out["events"].append(
            {
                "type": "file",
                "file": {
                    "data": inline.get("data"),
                    "mimeType": mime,
                    "source": "code_execution",
                },
            }
        )
        return
    media_type = (
        "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else "video"
    )
    out["events"].append({"type": "media_start", "mediaType": media_type, "mimeType": mime})
    out["events"].append({"type": "media_chunk", "data": inline.get("data")})
    out["events"].append({"type": "media_end"})


def _tool_call(ctx: Ctx) -> None:
    """Google streams a function call whole rather than in fragments, so the
    start/delta/end triple is emitted from one part."""
    out = _out(ctx)
    p = _part(ctx)
    fc = p.get("functionCall")
    if not isinstance(fc, Mapping):
        return
    started: dict[str, Any] = {
        "type": "tool_call_start",
        "id": fc.get("id") or "",
        "name": fc.get("name"),
    }
    if p.get("thoughtSignature"):
        started["_meta"] = {"thoughtSignature": p["thoughtSignature"]}
    out["events"].append(started)
    if fc.get("args"):
        # js_json, not json.dumps: JSON.stringify puts no space after `:` or `,`,
        # and this string IS the emitted `arguments`.
        out["events"].append(
            {"type": "tool_call_delta", "id": "", "arguments": js_json(fc["args"])}
        )
    out["events"].append({"type": "tool_call_end", "id": ""})


def _web_search(ctx: Ctx) -> None:
    """Web search has no per-call markers: one pair, the first time grounding
    metadata appears anywhere in the stream."""
    out = _out(ctx)
    grounding = _candidate(ctx).get("groundingMetadata")
    if not grounding or out["webSearchEmitted"]:
        return
    out["webSearchEmitted"] = True
    queries = grounding.get("webSearchQueries") if isinstance(grounding, Mapping) else None
    q = queries[0] if isinstance(queries, list) and queries else None
    out["events"].append({"type": "builtin_tool_start", "tool": "web_search"})
    ended: dict[str, Any] = {"type": "builtin_tool_end", "tool": "web_search"}
    if isinstance(q, str):
        ended["query"] = q
    out["events"].append(ended)


def _url_fetch(ctx: Ctx) -> None:
    """Same for urlContext: one pair per retrieved URL, once."""
    out = _out(ctx)
    url_ctx = _candidate(ctx).get("urlContextMetadata")
    meta = url_ctx.get("urlMetadata") if isinstance(url_ctx, Mapping) else None
    if not isinstance(meta, list) or out["urlFetchEmitted"]:
        return
    out["urlFetchEmitted"] = True
    for m in meta:
        if not isinstance(m, Mapping):
            continue
        out["events"].append({"type": "builtin_tool_start", "tool": "web_fetch"})
        ended: dict[str, Any] = {"type": "builtin_tool_end", "tool": "web_fetch"}
        if isinstance(m.get("retrievedUrl"), str):
            ended["url"] = m["retrievedUrl"]
        out["events"].append(ended)


GOOGLE_STREAM_REGISTRY = Registry(
    transforms={
        "googleStreamUsage": _usage,
        "googleStreamText": _text,
        "googleStreamThinking": _thinking,
        "googleStreamCitations": _citations,
        "googleStreamDone": _done,
    },
    effects={
        "googleStreamLatchCodeExec": _latch_code_exec,
        "googleStreamCodeExec": _code_exec,
        "googleStreamInline": _inline,
        "googleStreamToolCall": _tool_call,
        "googleStreamWebSearch": _web_search,
        "googleStreamUrlFetch": _url_fetch,
    },
)

__all__ = ["GOOGLE_STREAM_REGISTRY"]
