"""Anthropic provider adapter (Messages API).

Transposed from `unified-library-ts/src/llm/providers/anthropic/messages.ts`.

Both halves are spec-driven: the request comes from a chain node under
`wire/specs/anthropic-chain/`, the response from
`wire/specs/responses/anthropic.messages.json`. What is left in this file is the
part no spec describes -- auth, the endpoint, and the message/content assembly
the request spec reaches through `anthropicMessages`.

The parse helpers (`anthropic_usage`, `anthropic_billed_tier`,
`files_from_code_exec_block`, `builtin_input_payload`, `result_stdout`) live in
`parse_helpers.py` in this port, because the response registry needed them before
this file existed. They are re-exported here, where the TypeScript declares them,
so an importer following the TypeScript's layout finds them -- one definition,
two names for it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ....runtime import is_browser
from ....util.base64 import base64_to_utf8
from ....wire.chat_specs import chat_spec, is_chat_spec
from ....wire.interpreter import Registry, build_from_spec, js_json
from ....wire.pins import ANTHROPIC_MESSAGE_PINS, pin_for
from ....wire.response_interpreter import build_response
from ....wire.response_specs import get_response_spec
from ....wire.stream_interpreter import create_stream_builder
from ....wire.stream_specs import get_stream_spec
from ...types.messages import ContentPart
from ...types.provider import ProviderHttpRequest
from ...wire_transforms import make_registry
from .._shared.dropped import NoteSink, note_replaced
from .constants import ANTHROPIC_API_VERSION
from .parse_helpers import (
    anthropic_billed_tier,
    anthropic_usage,
    builtin_input_payload,
    files_from_code_exec_block,
    result_stdout,
)
from .response_registry import ANTHROPIC_RESPONSE_REGISTRY
from .stream_registry import ANTHROPIC_STREAM_REGISTRY

#: `interface AnthropicAdapterConfig` (messages.ts:35) -- `{apiKey, baseURL?}`.
AnthropicAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.anthropic.com"


class AnthropicAdapter:
    """`class AnthropicAdapter implements ProviderAdapter` (messages.ts:121)."""

    name = "anthropic"

    def __init__(self, config: AnthropicAdapterConfig) -> None:
        self.api_key: str = config["apiKey"]
        self._base_url: str | None = config.get("baseURL")
        # Named code the spec cannot express as data -- message and content
        # assembly. Built once, carrying only this adapter, since only Anthropic
        # rules run.
        self.wire_registry: Registry = make_registry({"anthropic": self})

    def auth_headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        # Anthropic's CORS preflight rejects browser-origin requests unless this
        # opt-in header is present. Send it only in the browser (BYOK direct
        # calls); harmless to omit elsewhere. See `runtime.is_browser`.
        if is_browser():
            headers["anthropic-dangerous-direct-browser-access"] = "true"
        return headers

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/v1/messages"

    def _spec_id_for(self, req: Mapping[str, Any]) -> str:
        """The spec that builds this model's request.

        The catalog pin decides when there is one. Without it -- an engine
        running with no catalog, or a model released after this build -- the band
        comes from the pin TABLE, which is data (`wire/pins/`) rather than version
        arithmetic in code, so every port derives the same node from the same
        file instead of each re-implementing it.
        """
        wire_spec = req.get("wireSpec")
        if is_chat_spec(wire_spec) and str(wire_spec).startswith("anthropic/"):
            return str(wire_spec)
        return pin_for(req.get("model") or "", ANTHROPIC_MESSAGE_PINS)

    def build_request(self, req: Mapping[str, Any]) -> ProviderHttpRequest:
        return build_from_spec(chat_spec(self._spec_id_for(req)), req, self.wire_registry)

    def enable_streaming(
        self, provider_req: ProviderHttpRequest, _req: Mapping[str, Any]
    ) -> None:
        provider_req.body["stream"] = True

    def build_message(
        self,
        msg: Mapping[str, Any],
        _req: Mapping[str, Any],
        force_cache: bool = False,
        notes: NoteSink = None,
    ) -> dict[str, Any]:
        """Reached through the wire registry while building this adapter's request.

        `role='tool'` becomes `user`: Anthropic has no tool role, and a tool
        result is a user-turn content block.
        """
        role = "user" if msg.get("role") == "tool" else msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
        else:
            parts = [self._build_content_part(p, notes) for p in content or []]

        # The cache breakpoint goes on the LAST part, which is where the prefix
        # ends -- putting it on the first would cache nothing. `if parts` guards
        # an empty content list, which indexes out of range where JavaScript
        # would simply hand back `undefined`.
        if (msg.get("cache") or force_cache) and parts:
            parts[-1]["cache_control"] = {"type": "ephemeral"}

        return {"role": role, "content": parts}

    def _build_content_part(self, part: ContentPart, notes: NoteSink = None) -> dict[str, Any]:
        kind = part.get("type")
        if kind == "text":
            return {"type": "text", "text": part.get("text")}
        if kind == "image":
            return self._build_image(part.get("source") or {})
        if kind == "document":
            return self._build_document(part)
        if kind == "tool_call":
            return {
                "type": "tool_use",
                "id": part.get("id"),
                "name": part.get("name"),
                "input": part.get("arguments"),
            }
        if kind == "tool_result":
            content = part.get("content")
            return {
                "type": "tool_result",
                "tool_use_id": part.get("id"),
                "content": content if isinstance(content, str) else js_json(content),
            }
        # Anthropic has no block for audio or video, so the part becomes a
        # placeholder -- and the caller is told, because otherwise the model's
        # "I cannot hear audio" reads as the model's failing rather than ours.
        note_replaced(notes, "anthropic", kind)
        return {"type": "text", "text": f"[unsupported: {kind}]"}

    @staticmethod
    def _build_image(source: Mapping[str, Any]) -> dict[str, Any]:
        kind = source.get("type")
        if kind == "base64":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": source.get("mimeType"),
                    "data": source.get("data"),
                },
            }
        if kind == "url":
            return {"type": "image", "source": {"type": "url", "url": source.get("url")}}
        if kind == "provider_ref":
            return {"type": "image", "source": {"type": "file", "file_id": source.get("refId")}}
        if kind == "file":
            return {"type": "image", "source": {"type": "file", "file_id": source.get("fileId")}}
        return {"type": "image", "source": {}}

    @staticmethod
    def _build_document(part: ContentPart) -> dict[str, Any]:
        source: Mapping[str, Any] = part.get("source") or {}
        block: dict[str, Any] = {"type": "document"}
        kind = source.get("type")
        if kind == "base64":
            # Anthropic plain-text documents use a `text` source (the raw text);
            # base64 sources are only for binary docs like application/pdf.
            if source.get("mimeType") == "text/plain":
                block["source"] = {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": base64_to_utf8(source.get("data") or ""),
                }
            else:
                block["source"] = {
                    "type": "base64",
                    "media_type": source.get("mimeType"),
                    "data": source.get("data"),
                }
        elif kind == "url":
            block["source"] = {"type": "url", "url": source.get("url")}
        elif kind == "provider_ref":
            block["source"] = {"type": "file", "file_id": source.get("refId")}
        elif kind == "file":
            block["source"] = {"type": "file", "file_id": source.get("fileId")}
        if part.get("citations"):
            block["citations"] = {"enabled": True}
        return block

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Spec-driven. The block-by-block walk this replaced is in
        `wire/specs/responses/anthropic.messages.json`, and the differential over
        every recorded Anthropic body asserts the two produce the same object.
        """
        return build_response(
            get_response_spec("anthropic/messages.response"),
            raw,
            ANTHROPIC_RESPONSE_REGISTRY,
            extra={"latencyMs": latency_ms, "raw": raw},
        )

    def parse_stream_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Stateless entry, as the ProviderAdapter protocol requires: a fresh
        spec run per event, so nothing correlates across events."""
        parse = create_stream_builder(
            get_stream_spec("anthropic/messages.stream"), ANTHROPIC_STREAM_REGISTRY
        )
        return parse(event)

    def create_stream_parser(self) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
        """Stateful, and the one callers should use.

        Anthropic streams a `server_tool_use` input via `input_json_delta` and
        returns the result in a separate `*_tool_result` block, so the input has
        to be carried between them; the spec's `state` is where that lives.
        """
        return create_stream_builder(
            get_stream_spec("anthropic/messages.stream"), ANTHROPIC_STREAM_REGISTRY
        )


__all__ = [
    "AnthropicAdapter",
    "AnthropicAdapterConfig",
    "anthropic_billed_tier",
    "anthropic_usage",
    "builtin_input_payload",
    "files_from_code_exec_block",
    "result_stdout",
]
