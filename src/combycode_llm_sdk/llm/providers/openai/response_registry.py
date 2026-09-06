"""The named escape hatches `openai.completions.response.json` calls.

Transposed from
`unified-library-ts/src/llm/providers/openai/response-registry.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....util.audio_mime import sniff_audio_mime
from ....util.base64 import base64_to_bytes
from ....wire.interpreter import Ctx, Registry
from ...moderation.native import parse_native_moderation
from .._shared.citations import extract_citations
from .._shared.response_utils import extract_finish_reason
from .parse_helpers import openai_usage
from .tiers import openai_billed_tier


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


def _message(ctx: Ctx) -> Mapping[str, Any]:
    m = _choice(ctx).get("message")
    return m if isinstance(m, Mapping) else {}


def _audio(ctx: Ctx) -> Mapping[str, Any] | None:
    a = _message(ctx).get("audio")
    return a if isinstance(a, Mapping) else None


def _text_of(ctx: Ctx) -> str:
    """The assistant's words: `message.content`, or the transcript when the reply
    was spoken (gpt-audio leaves content null)."""
    content = _message(ctx).get("content")
    if isinstance(content, str) and content:
        return content
    audio = _audio(ctx)
    transcript = audio.get("transcript") if audio else None
    return transcript if isinstance(transcript, str) else ""


_FINISH = {
    "tool_calls": "tool_use",
    "length": "length",
    "content_filter": "content_filter",
}


def _text_part(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """Omitted entirely when there are no words, so an empty reply yields no
    empty text part -- which is what the adapter's `if (text)` does."""
    text = _text_of(ctx)
    return {"type": "text", "text": text} if text else None


def _audio_part(_arg: Any, ctx: Ctx) -> dict[str, Any] | None:
    """The spoken bytes. The response carries no `format`, so the container is
    read from the bytes; the template is the last resort."""
    audio = _audio(ctx)
    data = audio.get("data") if audio else None
    if not audio or not isinstance(data, str) or not data:
        return None
    sniffed = sniff_audio_mime(base64_to_bytes(data[:16]))
    fmt = audio.get("format")
    return {
        "type": "audio_output",
        "mediaId": audio.get("id") or "",
        "mimeType": sniffed or f"audio/{fmt if isinstance(fmt, str) else 'wav'}",
        "_data": data,
    }


def _tool_call(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    """Arguments arrive as a JSON STRING and are parsed, which is the one thing
    in this file that can raise on a well-formed provider response."""
    import json

    tc = ctx.item.value if ctx.item else {}
    tc = tc if isinstance(tc, Mapping) else {}
    fn = tc.get("function")
    fn = fn if isinstance(fn, Mapping) else {}
    return {
        "type": "tool_call",
        "id": tc.get("id"),
        "name": fn.get("name"),
        "arguments": json.loads(fn.get("arguments") or "{}"),
    }


def _usage_full(_arg: Any, ctx: Ctx) -> dict[str, Any]:
    raw = _raw(ctx)
    u = raw.get("usage")
    return {
        **openai_usage(u if isinstance(u, Mapping) else None),
        **openai_billed_tier(raw.get("service_tier")),
    }


def _finish(_arg: Any, ctx: Ctx) -> str:
    reason = _choice(ctx).get("finish_reason")
    return extract_finish_reason(
        len(_out(ctx)["toolCalls"]) > 0,
        reason if isinstance(reason, str) else None,
        _FINISH,
    )


def _citations(_arg: Any, ctx: Ctx) -> list[dict[str, Any]] | None:
    c = extract_citations("completions", _raw(ctx))
    return c or None


def _thinking(_arg: Any, ctx: Ctx) -> Any:
    """Chat Completions hides reasoning text; some OpenAI-compatible providers
    (DeepSeek, xAI) return it as `reasoning_content`. Null, never absent -- and
    the accumulator's own default carries that, because a transform returning
    None means OMIT here."""
    v = _message(ctx).get("reasoning_content")
    return v if isinstance(v, str) else None


def _moderation(_arg: Any, ctx: Ctx) -> Any:
    """Absent unless moderation was requested."""
    return parse_native_moderation(_raw(ctx).get("moderation"))


OPENAI_RESPONSE_REGISTRY = Registry(
    transforms={
        "openaiTextPart": _text_part,
        "openaiText": lambda _arg, ctx: _text_of(ctx),
        "openaiAudioPart": _audio_part,
        "openaiToolCall": _tool_call,
        "openaiUsageFull": _usage_full,
        "openaiFinish": _finish,
        "openaiCitations": _citations,
        "openaiThinking": _thinking,
        "openaiModeration": _moderation,
    },
)

__all__ = ["OPENAI_RESPONSE_REGISTRY"]
