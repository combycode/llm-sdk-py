"""The named escape hatches `openai.completions.stream.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/openai/stream-registry.ts`.

Chat Completions streams are not a discriminated union of event types the way
Anthropic's are: every chunk has the same shape and the meaning is in which
fields of `choices[0].delta` are populated. So the spec is one ordered list of
self-guarding rules, which is exactly what the hand-written parser is.

Two things carry across events and are therefore effects: tool-call fragments
correlate by `index` because only the first fragment carries an id, and audio
needs to know whether its media block is already open.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Ctx, Registry
from ...moderation.native import parse_native_moderation
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import openai_usage


def _raw(ctx: Ctx) -> Mapping[str, Any]:
    v = ctx.req.get("raw")
    return v if isinstance(v, Mapping) else {}


def _out(ctx: Ctx) -> dict[str, Any]:
    out: dict[str, Any] = ctx.req["out"]
    return out


def _choice(ctx: Ctx) -> Mapping[str, Any]:
    choices = _raw(ctx).get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return choices[0]
    return {}


def _delta(ctx: Ctx) -> Mapping[str, Any]:
    d = _choice(ctx).get("delta")
    return d if isinstance(d, Mapping) else {}


_FINISH = {
    "tool_calls": "tool_use",
    "length": "length",
    "content_filter": "content_filter",
}


def _moderation(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Native moderation arrives on its own chunk, with no choices."""
    report = parse_native_moderation(_raw(ctx).get("moderation"))
    out: list[dict[str, Any]] = []
    if report and report.get("input"):
        out.append(
            {"type": "moderation", "phase": "input", "result": report["input"], "source": "native"}
        )
    if report and report.get("output"):
        out.append(
            {"type": "moderation", "phase": "output", "result": report["output"], "source": "native"}
        )
    return out


def _has_moderation(ctx: Ctx) -> bool:
    """The early return only fires when moderation actually produced entries."""
    report = parse_native_moderation(_raw(ctx).get("moderation"))
    return bool(report and (report.get("input") or report.get("output")))


def _usage(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """Empty unless this chunk carries usage, which lets the same rule serve the
    usage-only chunk and the trailing usage on a normal one."""
    u = _raw(ctx).get("usage")
    if not isinstance(u, Mapping):
        return []
    return [{"type": "usage", "usage": openai_usage(u)}]


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]]:
    """OpenRouter `:online` sends annotations on their own chunks, one per chunk,
    BEFORE the text that cites them. `url_citation.content` is the whole scraped
    page and is deliberately not mapped to `text`, which elsewhere means the
    short passage the source supports."""
    out: list[dict[str, Any]] = []
    notes = _delta(ctx).get("annotations")
    for note in notes if isinstance(notes, list) else []:
        if not isinstance(note, Mapping):
            continue
        detail = note.get("url_citation")
        detail = detail if isinstance(detail, Mapping) else note
        if note.get("type") == "url_citation" and detail.get("url"):
            citation: dict[str, Any] = {"url": detail["url"]}
            if detail.get("title"):
                citation["title"] = detail["title"]
            out.append({"type": "citation", "citation": citation})
    return out


def _done(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    fr = _choice(ctx).get("finish_reason")
    if not isinstance(fr, str) or not fr:
        return None
    return {"type": "done", "finishReason": extract_finish_reason(False, fr, _FINISH)}


def _tool_calls(ctx: Ctx) -> None:
    """Correlate streamed tool-call fragments by `index`: the wire id, when
    present, only arrives on the first delta and the argument fragments omit it.
    Some OpenAI-compatible backends omit it entirely, so a stable `call_<uuid>`
    is synthesised ONCE per index -- assigned on first sighting and never
    changed, or parallel id-less calls collide into one."""
    out = _out(ctx)
    calls = _delta(ctx).get("tool_calls")
    for tc in calls if isinstance(calls, list) else []:
        if not isinstance(tc, Mapping):
            continue
        index = str(tc.get("index") if tc.get("index") is not None else 0)
        call_id = out["toolIdByIndex"].get(index)
        if call_id is None:
            call_id = tc.get("id") or f"call_{uuid.uuid4()}"
            out["toolIdByIndex"][index] = call_id
        fn = tc.get("function")
        fn = fn if isinstance(fn, Mapping) else {}
        if fn.get("name"):
            out["events"].append(
                {"type": "tool_call_start", "id": call_id, "name": fn["name"]}
            )
        if fn.get("arguments"):
            out["events"].append(
                {"type": "tool_call_delta", "id": call_id, "arguments": fn["arguments"]}
            )


def _audio(ctx: Ctx) -> None:
    """gpt-audio streams its reply as `delta.audio`: the transcript once up
    front, the bytes in fragments, and a final `expires_at`-only delta that
    closes it. These chunks never carry a finish_reason, so that closing delta is
    the ONLY terminal signal the stream gives."""
    out = _out(ctx)
    audio = _delta(ctx).get("audio")
    if not isinstance(audio, Mapping):
        return
    state = out["audio"]
    if audio.get("id") and not state.get("id"):
        state["id"] = audio["id"]
    if audio.get("transcript"):
        out["events"].append({"type": "text", "text": audio["transcript"]})
    if audio.get("data"):
        if not state.get("open"):
            state["open"] = True
            # Always pcm16 -- the API refuses any other format when stream=true --
            # and raw PCM has no magic bytes, so the mime is stated not sniffed.
            out["events"].append(
                {"type": "media_start", "mediaType": "audio", "mimeType": "audio/pcm"}
            )
        out["events"].append({"type": "media_chunk", "data": audio["data"]})
    if audio.get("expires_at") is not None and state.get("open"):
        state["open"] = False
        ended: dict[str, Any] = {"type": "media_end"}
        if state.get("id"):
            ended["mediaId"] = state["id"]
        out["events"].append(ended)
        out["events"].append({"type": "done", "finishReason": "stop"})


OPENAI_STREAM_REGISTRY = Registry(
    transforms={
        "openaiStreamModeration": _moderation,
        "openaiStreamUsage": _usage,
        "openaiStreamCitations": _citations,
        "openaiStreamDone": _done,
    },
    predicates={"openaiStreamHasModeration": _has_moderation},
    effects={
        "openaiStreamToolCalls": _tool_calls,
        "openaiStreamAudio": _audio,
    },
)

__all__ = ["OPENAI_STREAM_REGISTRY"]
