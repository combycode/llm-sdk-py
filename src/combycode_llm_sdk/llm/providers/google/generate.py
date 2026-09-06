"""Google Gemini provider adapter (generateContent API).

Transposed from `unified-library-ts/src/llm/providers/google/generate.ts`.

`google_usage` lives in `parse_helpers.py` in this port, where the response
registry needed it first; it is re-exported here, where the TypeScript declares
it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ....wire.chat_specs import chat_spec, is_chat_spec
from ....wire.interpreter import Registry, build_from_spec
from ....wire.pins import GOOGLE_GENERATE_PINS, pin_for
from ....wire.response_interpreter import build_response
from ....wire.response_specs import get_response_spec
from ....wire.stream_interpreter import create_stream_builder
from ....wire.stream_specs import get_stream_spec
from ...types.provider import ProviderHttpRequest
from ...wire_transforms import make_registry
from .._shared.dropped import NoteSink, note_dropped
from .parse_helpers import google_usage
from .response_registry import GOOGLE_RESPONSE_REGISTRY
from .stream_registry import GOOGLE_STREAM_REGISTRY

#: `interface GoogleAdapterConfig` (generate.ts:22) -- `{apiKey, baseURL?}`.
GoogleAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GoogleAdapter:
    """`class GoogleAdapter implements ProviderAdapter` (generate.ts:45)."""

    name = "google"

    def __init__(self, config: GoogleAdapterConfig) -> None:
        self.api_key: str = config["apiKey"]
        self._base_url: str | None = config.get("baseURL")
        # Named code the spec cannot express as data -- content assembly.
        self.wire_registry: Registry = make_registry({"google": self})
        # Tool call id -> function name. Google needs the NAME in a
        # `functionResponse`, and only the call carries it, so it is remembered
        # across messages of one request.
        self.tool_call_names: dict[str, str] = {}

    def auth_headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "content-type": "application/json",
        }

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return ""  # set dynamically per request (the model is in the URL)

    def _spec_id_for(self, req: Mapping[str, Any]) -> str:
        """The spec that builds this model's request.

        Two nodes, keyed on the one thing that differs on the wire: 2.5 takes a
        token `thinkingBudget` and 400s on `thinkingLevel`, 3.x takes the level.
        Catalog pin first, then the pin TABLE -- data rather than a regex in
        code, so every port reads the same rule.
        """
        wire_spec = req.get("wireSpec")
        if is_chat_spec(wire_spec) and str(wire_spec).startswith("google/generate"):
            return str(wire_spec)
        return pin_for(req.get("model") or "", GOOGLE_GENERATE_PINS)

    def build_request(self, req: Mapping[str, Any]) -> ProviderHttpRequest:
        return build_from_spec(chat_spec(self._spec_id_for(req)), req, self.wire_registry)

    def enable_streaming(
        self, provider_req: ProviderHttpRequest, req: Mapping[str, Any]
    ) -> None:
        """Google streams from a DIFFERENT URL, not a body flag."""
        model = req.get("model") or ""
        if not model.startswith("models/"):
            model = f"models/{model}"
        provider_req.path = f"/v1beta/{model}:streamGenerateContent?alt=sse"

    def build_content(
        self, msg: Mapping[str, Any], notes: NoteSink = None
    ) -> dict[str, Any]:
        """Reached through the wire registry while building this adapter's request.

        Google has two roles only: `model` and `user`. Everything that is not an
        assistant turn -- including a tool result -- is a user turn.
        """
        role = "model" if msg.get("role") == "assistant" else "user"
        parts: list[Any] = []

        content = msg.get("content")
        if isinstance(content, str):
            parts.append({"text": content})
            return {"role": role, "parts": parts}

        for p in content or []:
            kind = p.get("type")
            before = len(parts)
            if kind == "text":
                parts.append({"text": p.get("text")})
            elif kind in ("image", "audio", "video", "document"):
                self._append_media(parts, p.get("source") or {})
            elif kind == "tool_call":
                self.tool_call_names[p["id"]] = p["name"]
                fc_part: dict[str, Any] = {
                    "functionCall": {
                        "name": p.get("name"),
                        "args": p.get("arguments"),
                        "id": p.get("id"),
                    }
                }
                meta: Mapping[str, Any] = p.get("_meta") or {}
                if meta.get("thoughtSignature"):
                    fc_part["thoughtSignature"] = meta["thoughtSignature"]
                parts.append(fc_part)
            elif kind == "tool_result":
                result = p.get("content")
                parts.append(
                    {
                        "functionResponse": {
                            # `?? ''` -- a result with no matching call still has
                            # to carry the key, and an absent name is rejected.
                            "name": self.tool_call_names.get(p["id"]) or "",
                            "id": p.get("id"),
                            "response": {"result": result}
                            if isinstance(result, str)
                            else result,
                        }
                    }
                )
            if len(parts) == before:
                # Google carries every media kind, so reaching here means the
                # SOURCE had no form -- a `path` that never resolved, say.
                note_dropped(notes, "google", kind)

        return {"role": role, "parts": parts}

    @staticmethod
    def _append_media(parts: list[Any], source: Mapping[str, Any]) -> None:
        kind = source.get("type")
        if kind == "base64":
            parts.append(
                {"inlineData": {"mimeType": source.get("mimeType"), "data": source.get("data")}}
            )
        elif kind == "url":
            parts.append(
                {
                    "fileData": {
                        "fileUri": source.get("url"),
                        "mimeType": "application/octet-stream",
                    }
                }
            )
        elif kind == "provider_ref":
            parts.append(
                {"fileData": {"fileUri": source.get("refId"), "mimeType": source.get("mimeType")}}
            )
        elif kind == "file":
            parts.append({"fileData": {"fileUri": source.get("fileId")}})

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Spec-driven; see `wire/specs/responses/google.generate.json`."""
        return build_response(
            get_response_spec("google/generate.response"),
            raw,
            GOOGLE_RESPONSE_REGISTRY,
            extra={"latencyMs": latency_ms, "raw": raw},
        )

    def parse_stream_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Stateless entry, as the ProviderAdapter protocol requires."""
        return self.create_stream_parser()(event)

    def create_stream_parser(self) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
        """Stateful, and the one callers should use.

        The code-execution flag latches across chunks and decides whether
        `inlineData` is an artifact or conversational media.
        """
        return create_stream_builder(
            get_stream_spec("google/generate.stream"), GOOGLE_STREAM_REGISTRY
        )


__all__ = ["GoogleAdapter", "GoogleAdapterConfig", "google_usage"]
