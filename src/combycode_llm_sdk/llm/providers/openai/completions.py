"""OpenAI provider adapter (Chat Completions API).

Transposed from `unified-library-ts/src/llm/providers/openai/completions.ts`.

The class is written to be SUBCLASSED: OpenRouter and xAI are Chat Completions
with an overlay, and they override `wire_flavor`, `response_spec_id`,
`response_registry`, `stream_spec_id` and `stream_registry` -- never the request
or message assembly. Keeping those five as methods rather than inlining them is
what makes those subclasses three lines each.

`openai_usage` lives in `parse_helpers.py` in this port, because the response
registry needed it before this file existed; it is re-exported here, where the
TypeScript declares it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ....wire.chat_specs import chat_spec
from ....wire.interpreter import Registry, build_from_spec, js_json
from ....wire.response_interpreter import build_response
from ....wire.response_specs import get_response_spec
from ....wire.stream_interpreter import create_stream_builder
from ....wire.stream_specs import get_stream_spec
from ...types.messages import ContentPart
from ...types.provider import ProviderHttpRequest
from ...wire_transforms import make_registry
from .._shared.dropped import NoteSink, note_replaced
from .parse_helpers import openai_usage
from .response_registry import OPENAI_RESPONSE_REGISTRY
from .stream_registry import OPENAI_STREAM_REGISTRY

#: `interface OpenAIAdapterConfig` (completions.ts:20) -- `{apiKey, baseURL?}`.
OpenAIAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.openai.com"


def audio_format(mime_type: str) -> str:
    """OpenAI `input_audio` accepts only 'wav' or 'mp3'."""
    return "mp3" if "mpeg" in mime_type or "mp3" in mime_type else "wav"


def doc_filename_for_mime(mime_type: str) -> str:
    """A filename with extension for an inline chat `file` part (the API requires one)."""
    if mime_type == "application/pdf":
        return "file.pdf"
    if mime_type == "text/plain":
        return "file.txt"
    return "file.bin"


class OpenAIAdapter:
    """`class OpenAIAdapter implements ProviderAdapter` (completions.ts:70)."""

    name = "openai"

    #: Which flavor overlay patches the shared spec. Subclasses for
    #: OpenAI-compatible backends override this and nothing else.
    wire_flavor = "openai"

    def __init__(self, config: OpenAIAdapterConfig) -> None:
        self.api_key: str = config["apiKey"]
        self._base_url: str | None = config.get("baseURL")
        # Named code the spec cannot express as data -- message/input assembly.
        # Carries `self`, so a subclass drives the same rules with its own
        # overrides.
        self.wire_registry: Registry = make_registry({"openai_completions": self})

    def auth_headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/v1/chat/completions"

    def build_request(self, req: Mapping[str, Any]) -> ProviderHttpRequest:
        return build_from_spec(
            chat_spec("openai/chat-completions"), req, self.wire_registry, self.wire_flavor
        )

    def build_messages(
        self, msg: Mapping[str, Any], notes: NoteSink = None
    ) -> list[dict[str, Any]]:
        """One universal message can become SEVERAL chat-completions messages.

        Parallel tool calls are the case that matters: the loop answers a round
        of calls with ONE tool message carrying a `tool_result` part per call,
        but this API wants a separate `{'role': 'tool'}` message per
        `tool_call_id`. Emitting only the first left the rest unanswered and the
        provider rejected the whole request with "No tool output found for
        function call <id>" -- so parallel tools were broken on every
        chat-completions backend.

        Reached through the wire registry while building the request.
        """
        if msg.get("role") == "tool":
            parts = _as_parts(msg.get("content"))
            results = [p for p in parts if p.get("type") == "tool_result"]
            if results:
                return [
                    {
                        "role": "tool",
                        "tool_call_id": result.get("id"),
                        "content": result["content"]
                        if isinstance(result.get("content"), str)
                        else js_json(result.get("content")),
                    }
                    for result in results
                ]
        return [self._build_message(msg, notes)]

    def _build_message(self, msg: Mapping[str, Any], notes: NoteSink = None) -> dict[str, Any]:
        if msg.get("role") == "assistant":
            parts = _as_parts(msg.get("content"))
            tool_calls = [p for p in parts if p.get("type") == "tool_call"]
            if tool_calls:
                text = "".join(p.get("text") or "" for p in parts if p.get("type") == "text")
                return {
                    "role": "assistant",
                    # `|| null`, not `|| ''`: this API rejects an assistant turn
                    # whose content is an empty string but accepts a null one.
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": js_json(tc.get("arguments")),
                            },
                        }
                        for tc in tool_calls
                    ],
                }

        content = msg.get("content")
        if isinstance(content, str):
            return {"role": msg.get("role"), "content": content}

        built = [self._build_content_part(p, notes) for p in content or []]

        # gpt-audio requires the text instruction to PRECEDE the input_audio part
        # (audio-first yields "please play the audio"). Keep audio parts last.
        if any(p.get("type") == "input_audio" for p in built):
            built = [p for p in built if p.get("type") != "input_audio"] + [
                p for p in built if p.get("type") == "input_audio"
            ]

        return {"role": msg.get("role"), "content": built}

    def _build_content_part(self, part: ContentPart, notes: NoteSink = None) -> dict[str, Any]:
        kind = part.get("type")
        if kind == "text":
            return {"type": "text", "text": part.get("text")}
        if kind == "image":
            source: Mapping[str, Any] = part.get("source") or {}
            if source.get("type") == "base64":
                url = f"data:{source.get('mimeType')};base64,{source.get('data')}"
            elif source.get("type") == "url":
                url = source.get("url") or ""
            else:
                url = ""
            return {
                "type": "image_url",
                "image_url": {"url": url, "detail": part.get("detail") or "auto"},
            }
        if kind == "audio":
            source = part.get("source") or {}
            if source.get("type") == "base64":
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": source.get("data"),
                        "format": audio_format(source.get("mimeType") or ""),
                    },
                }
            note_replaced(notes, "openai", f"audio ({source.get('type')} source)")
            return {"type": "text", "text": "[unsupported audio source]"}
        if kind == "document":
            # OpenAI-compatible chat file input (pdf/text). OpenRouter relies on this.
            source = part.get("source") or {}
            if source.get("type") == "provider_ref":
                return {"type": "file", "file": {"file_id": source.get("refId")}}
            if source.get("type") == "base64":
                mime = source.get("mimeType") or ""
                return {
                    "type": "file",
                    "file": {
                        "filename": doc_filename_for_mime(mime),
                        "file_data": f"data:{mime};base64,{source.get('data')}",
                    },
                }
            note_replaced(notes, "openai", f"document ({source.get('type')} source)")
            return {"type": "text", "text": "[unsupported document source]"}
        note_replaced(notes, "openai", kind)
        return {"type": "text", "text": f"[unsupported: {kind}]"}

    def enable_streaming(
        self, provider_req: ProviderHttpRequest, _req: Mapping[str, Any]
    ) -> None:
        provider_req.body["stream"] = True
        provider_req.body["stream_options"] = {"include_usage": True}

    def response_spec_id(self) -> str:
        """The response spec this adapter parses with.

        Overridden by OpenRouter, which is Chat Completions plus one rule.
        """
        return "openai/completions.response"

    def response_registry(self) -> Registry:
        return OPENAI_RESPONSE_REGISTRY

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Spec-driven; see `wire/specs/responses/openai.completions.json`."""
        return build_response(
            get_response_spec(self.response_spec_id()),
            raw,
            self.response_registry(),
            extra={"latencyMs": latency_ms, "raw": raw},
        )

    def stream_spec_id(self) -> str:
        """Overridden by OpenRouter, which appends its `:online` web-search pair."""
        return "openai/completions.stream"

    def stream_registry(self) -> Registry:
        return OPENAI_STREAM_REGISTRY

    def parse_stream_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Stateless entry, as the ProviderAdapter protocol requires."""
        return self.create_stream_parser()(event)

    def create_stream_parser(self) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
        """Stateful, and the one callers should use.

        Tool-call fragments correlate by index because only the first carries an
        id, and some OpenAI-compatible backends omit ids entirely -- so a stable
        `call_<uuid>` is synthesised once per index and held for the stream.
        """
        return create_stream_builder(
            get_stream_spec(self.stream_spec_id()), self.stream_registry()
        )


def _as_parts(content: Any) -> list[dict[str, Any]]:
    """`typeof content === 'string' ? [{type:'text', text: content}] : content`."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


__all__ = [
    "OpenAIAdapter",
    "OpenAIAdapterConfig",
    "audio_format",
    "doc_filename_for_mime",
    "openai_usage",
]
