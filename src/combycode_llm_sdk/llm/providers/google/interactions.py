"""Google Interactions API adapter.

Transposed from `unified-library-ts/src/llm/providers/google/interactions.ts`.

Endpoint: `POST /v1beta/interactions`. The modern API: `input`,
`system_instruction`, `outputs` (plural), `function_result`, and
`previous_interaction_id` for stateful continuation with 72h retention.

`google_interactions_usage` lives in `parse_helpers.py` in this port, where the
response registry needed it first; it is re-exported here.
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
from ...types.messages import Message
from ...types.provider import ProviderHttpRequest
from ...wire_transforms import make_registry
from .._shared.dropped import NoteSink, note_dropped
from .interactions_registry import GOOGLE_INTERACTIONS_REGISTRY
from .interactions_stream_registry import GOOGLE_INTERACTIONS_STREAM_REGISTRY
from .parse_helpers import google_interactions_usage

#: `interface GoogleInteractionsAdapterConfig` (interactions.ts:23).
GoogleInteractionsAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GoogleInteractionsAdapter:
    """`class GoogleInteractionsAdapter implements ProviderAdapter` (interactions.ts:44)."""

    name = "google"

    def __init__(self, config: GoogleInteractionsAdapterConfig) -> None:
        self.api_key: str = config["apiKey"]
        self._base_url: str | None = config.get("baseURL")
        # Named code the spec cannot express as data -- input-item assembly.
        self.wire_registry: Registry = make_registry({"google_interactions": self})
        #: Tool call id -> name, for `function_result`. Populated from both
        #: directions: by `build_input_items` when a call is sent, and by
        #: `parse_response` when one comes back.
        self.tool_call_names: dict[str, str] = {}

    def auth_headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self.api_key,
            "content-type": "application/json",
        }

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/v1beta/interactions"

    def build_request(self, req: Mapping[str, Any]) -> ProviderHttpRequest:
        # One shape, no chain: the Interactions wire does not vary by model
        # version, so there is nothing for a pin to choose between.
        return build_from_spec(chat_spec("google/interactions"), req, self.wire_registry)

    def build_input_items(self, msg: Message, notes: NoteSink = None) -> list[Any]:
        """`step_list` input items (post May-2026).

        User turns become `{'type': 'user_input'}`, assistant turns
        `{'type': 'model_output'}`, tool results `{'type': 'function_result'}`.

        Reached through the wire registry while building the request.
        """
        items: list[Any] = []
        role = msg.get("role")

        if role in ("user", "system"):
            content = msg.get("content")
            if isinstance(content, str):
                items.append(
                    {"type": "user_input", "content": [{"type": "text", "text": content}]}
                )
            else:
                parts = self._user_parts(content or [], notes)
                if parts:
                    items.append({"type": "user_input", "content": parts})

        elif role == "assistant":
            content_items: list[Any] = []
            for p in _as_parts(msg.get("content")):
                # `p.text` truthiness, not presence: an empty assistant text part
                # would otherwise become an empty content item the API rejects.
                if p.get("type") == "text" and p.get("text"):
                    content_items.append({"type": "text", "text": p["text"]})
                if p.get("type") == "tool_call":
                    self.tool_call_names[p["id"]] = p["name"]
                    content_items.append(
                        {
                            "type": "function_call",
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "arguments": p.get("arguments"),
                        }
                    )
            if content_items:
                items.append({"type": "model_output", "content": content_items})

        elif role == "tool":
            for p in _as_parts(msg.get("content")):
                if p.get("type") != "tool_result":
                    continue
                result = p.get("content")
                items.append(
                    {
                        "type": "function_result",
                        "name": self.tool_call_names.get(p["id"]) or "",
                        "call_id": p.get("id"),
                        "result": result if isinstance(result, str) else js_json(result),
                    }
                )

        return items

    @staticmethod
    def _user_parts(content: Any, notes: NoteSink = None) -> list[Any]:
        parts: list[Any] = []
        for p in content:
            kind = p.get("type")
            source: Mapping[str, Any] = p.get("source") or {}
            if kind == "text":
                parts.append({"type": "text", "text": p.get("text")})
            elif kind == "image":
                if source.get("type") == "base64":
                    parts.append(
                        {
                            "type": "image",
                            "mime_type": source.get("mimeType"),
                            "data": source.get("data"),
                        }
                    )
                elif source.get("type") == "url":
                    parts.append({"type": "image", "uri": source.get("url")})
            # Audio takes base64 only and video takes a uri only -- the
            # Interactions API has no other form for either, so an unsupported
            # source contributes no part rather than an empty one.
            elif kind == "audio" and source.get("type") == "base64":
                parts.append(
                    {
                        "type": "audio",
                        "mime_type": source.get("mimeType"),
                        "data": source.get("data"),
                    }
                )
            elif kind == "video" and source.get("type") == "url":
                parts.append({"type": "video", "uri": source.get("url")})
            else:
                # Covers both an unknown kind and a known one arriving in a form
                # this API has no field for -- audio by url, video by base64.
                note_dropped(notes, "google", kind)
        return parts

    def enable_streaming(
        self, provider_req: ProviderHttpRequest, _req: Mapping[str, Any] | None = None
    ) -> None:
        provider_req.body["stream"] = True

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Spec-driven; see `wire/specs/responses/google.interactions.json`."""
        result = build_response(
            get_response_spec("google/interactions.response"),
            raw,
            GOOGLE_INTERACTIONS_REGISTRY,
            extra={"latencyMs": latency_ms, "raw": raw},
        )

        # The one thing the spec cannot own. `build_request` names a tool RESULT
        # by looking its call id up here, so a parse that does not record the
        # names sends the next request with an empty `name` -- silently, and only
        # on the turn AFTER the tool call. Re-fed from the built result, which
        # carries the same ids the hand-written loop used to set one at a time.
        for tc in result.get("toolCalls") or []:
            self.tool_call_names[tc.get("id")] = tc.get("name")
        return result

    def parse_stream_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Stateless entry, as the ProviderAdapter protocol requires.

        The 2.10 wire is a step machine (verified live): `step.start` opens a
        typed step (`model_output`, `function_call`, `thought`...), `step.delta`
        streams its payload (`{'type': 'text'}`, `{'type': 'arguments_delta'}`,
        `{'type': 'thought_summary'}`, internal `thought_signature`),
        `step.stop` closes it, and `interaction.completed` /
        `interaction.failed` finish the turn (usage under `interaction.usage`).
        """
        return self.create_stream_parser()(event)

    def create_stream_parser(self) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
        """Stateful, and the one callers should use.

        A function call's `arguments_delta` carries no id, so the open call's id
        is carried between its fragments and its close.
        """
        return create_stream_builder(
            get_stream_spec("google/interactions.stream"),
            GOOGLE_INTERACTIONS_STREAM_REGISTRY,
        )


def _as_parts(content: Any) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


__all__ = [
    "GoogleInteractionsAdapter",
    "GoogleInteractionsAdapterConfig",
    "google_interactions_usage",
]
