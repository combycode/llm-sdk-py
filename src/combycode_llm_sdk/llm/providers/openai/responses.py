"""OpenAI Responses API adapter.

Transposed from `unified-library-ts/src/llm/providers/openai/responses.ts`.

Endpoint: `POST /v1/responses`. The modern API: `input` (not `messages`),
`instructions` (not a system role), output items (not choices),
`function_call` / `function_call_output` for tools.

`build_input_items` is the whole of what no spec describes, and most of it is
ordering: an assistant turn emits its text, then its program, then its tool
calls, then its program result, and each of those four is a separate pass over
the same parts rather than one pass with four branches -- because the API reads
the ITEM ORDER, not the part order, and a program item must precede the calls it
made.

The parse helpers (`openai_responses_usage`, `from_wire_caller`,
`builtin_call_from_responses_item`, `files_from_responses_output_item`) live in
`parse_helpers.py` in this port and are re-exported here, where the TypeScript
declares them.
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
from ...types.messages import Message, ToolCaller
from ...types.provider import ProviderHttpRequest
from ...wire_transforms import make_registry
from .._shared.dropped import NoteSink, note_dropped
from .parse_helpers import (
    builtin_call_from_responses_item,
    files_from_responses_output_item,
    from_wire_caller,
    openai_responses_usage,
)
from .responses_registry import OPENAI_RESPONSES_REGISTRY
from .responses_stream_registry import OPENAI_RESPONSES_STREAM_REGISTRY

#: `interface OpenAIResponsesAdapterConfig` (responses.ts:30) -- `{apiKey, baseURL?}`.
OpenAIResponsesAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.openai.com"


def to_wire_caller(caller: ToolCaller) -> dict[str, Any]:
    """`caller` on the wire uses `caller_id`; our facade uses `callerId`.

    The type value is passed through unchanged -- it is an open union on our side
    (R1), and the API rejects a value it does not know (`'teleport'` -> 400
    naming `input[n].caller.type`), so an unknown value fails loudly at the
    provider rather than being dropped here.
    """
    out: dict[str, Any] = {"type": caller.get("type")}
    if caller.get("callerId") is not None:
        out["caller_id"] = caller["callerId"]
    return out


def filename_for_mime(mime_type: str) -> str:
    """A filename (with extension) for an inline `input_file` -- required by the API."""
    if mime_type == "application/pdf":
        return "file.pdf"
    if mime_type == "text/plain":
        return "file.txt"
    if mime_type.startswith("image/"):
        return f"file.{mime_type[len('image/') :]}"
    return "file.bin"


class OpenAIResponsesAdapter:
    """`class OpenAIResponsesAdapter implements ProviderAdapter` (responses.ts:222)."""

    name = "openai"

    #: Which flavor overlay patches the shared spec. Subclasses for
    #: OpenAI-compatible backends override this and nothing else.
    wire_flavor = "openai"

    def __init__(self, config: OpenAIResponsesAdapterConfig) -> None:
        self.api_key: str = config["apiKey"]
        self._base_url: str | None = config.get("baseURL")
        # Named code the spec cannot express as data -- message/input assembly.
        # Carries `self`, so a subclass drives the same rules with its own
        # overrides.
        self.wire_registry: Registry = make_registry({"openai_responses": self})

    def auth_headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/v1/responses"

    def build_request(self, req: Mapping[str, Any]) -> ProviderHttpRequest:
        return build_from_spec(
            chat_spec("openai/responses"), req, self.wire_registry, self.wire_flavor
        )

    def build_input_items(
        self,
        msg: Message,
        tool_names: dict[str, str] | None = None,
        notes: NoteSink = None,
    ) -> list[Any]:
        """Convert a universal Message to Responses API input items.

        `tool_names` is threaded ACROSS messages so a tool result can name its
        originating call -- it is a caller-owned map, not per-message state, and
        a fresh dict per call would lose every name.

        Reached through the wire registry while building the request.
        """
        names = {} if tool_names is None else tool_names
        items: list[Any] = []
        role = msg.get("role")

        if role in ("user", "system"):
            self._append_user_items(items, msg, role, notes)
        elif role == "assistant":
            self._append_assistant_items(items, msg, names)
        elif role == "tool":
            self._append_tool_items(items, msg, names)

        return items

    def _append_user_items(
        self, items: list[Any], msg: Message, role: Any, notes: NoteSink = None
    ) -> None:
        content = msg.get("content")
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            return
        parts: list[Any] = []
        for p in content or []:
            kind = p.get("type")
            source: Mapping[str, Any] = p.get("source") or {}
            # Measured against the list, not guessed from the branch taken: a
            # kind can be known and still produce nothing when its SOURCE has no
            # form here (audio, or an image by a route this API does not take).
            before = len(parts)
            if kind == "text":
                parts.append({"type": "input_text", "text": p.get("text")})
            elif kind == "image":
                if source.get("type") == "base64":
                    parts.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{source.get('mimeType')};base64,"
                            f"{source.get('data')}",
                        }
                    )
                elif source.get("type") == "url":
                    parts.append({"type": "input_image", "image_url": source.get("url")})
                elif source.get("type") == "provider_ref":
                    parts.append({"type": "input_file", "file_id": source.get("refId")})
            elif kind == "document":
                if source.get("type") == "provider_ref":
                    parts.append({"type": "input_file", "file_id": source.get("refId")})
                elif source.get("type") == "base64":
                    mime = source.get("mimeType") or ""
                    parts.append(
                        {
                            "type": "input_file",
                            # Inline file_data REQUIRES a filename (with the right
                            # extension) or the Responses API rejects the request.
                            "filename": filename_for_mime(mime),
                            "file_data": f"data:{mime};base64,{source.get('data')}",
                        }
                    )
                elif source.get("type") == "url":
                    parts.append({"type": "input_file", "url": source.get("url")})
            if len(parts) == before:
                note_dropped(notes, "openai", kind)
        # An item with empty content is rejected, so a message whose every part
        # was an unsupported kind contributes nothing rather than an empty item.
        # Which is exactly why the note matters: without it the whole message
        # would vanish from the request in silence.
        if parts:
            items.append({"role": role, "content": parts})

    def _append_assistant_items(
        self, items: list[Any], msg: Message, names: dict[str, str]
    ) -> None:
        parts = _as_parts(msg.get("content"))

        # Text content as message output items. `phase` lives on the MESSAGE, so
        # consecutive text parts sharing a phase become one item and a change of
        # phase starts a new one -- grouping by phase globally would reorder
        # commentary against the answer it precedes.
        run: list[Mapping[str, Any]] = []

        def flush() -> None:
            if not run:
                return
            item: dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": p.get("text")} for p in run],
            }
            if "phase" in run[0]:
                item["phase"] = run[0]["phase"]
            items.append(item)
            run.clear()

        for p in [p for p in parts if p.get("type") == "text"]:
            if run and run[0].get("phase") != p.get("phase"):
                flush()
            run.append(p)
        flush()

        # Programmatic tool calling: the program the model wrote, plus whatever
        # provider items it is bound to. OpenAI rejects a `program` item whose
        # `reasoning` item is missing ("provided without its required 'reasoning'
        # item"), and DROPPING the program instead is worse than an error -- the
        # model silently re-emits it and runs the whole thing again. Both
        # verified on the wire 2026-08-09.
        for p in parts:
            if p.get("type") != "program_call":
                continue
            meta: Mapping[str, Any] = p.get("_meta") or {}
            items.extend(meta.get("boundItems") or [])
            program: dict[str, Any] = {"type": "program"}
            if meta.get("itemId"):
                program["id"] = meta["itemId"]
            program["call_id"] = p.get("id")
            program["code"] = p.get("code")
            program["fingerprint"] = p.get("fingerprint")
            items.append(program)

        # Tool calls as function_call items.
        for p in parts:
            if p.get("type") != "tool_call":
                continue
            names[p["id"]] = p["name"]
            call: dict[str, Any] = {
                "type": "function_call",
                "id": f"fc_{p.get('id')}",
                "call_id": p.get("id"),
                "name": p.get("name"),
                "arguments": js_json(p.get("arguments")),
            }
            if p.get("caller"):
                call["caller"] = to_wire_caller(p["caller"])
            items.append(call)

        # The program's own return value, once it finished.
        for p in parts:
            if p.get("type") != "program_result":
                continue
            # `id` is REQUIRED here -- unlike function_call_output, which needs
            # none. A completed programmatic run replayed as history 400s without
            # it ("Missing required parameter: 'input[n].id'"), so a follow-up
            # question fails on a conversation that succeeded a moment earlier.
            meta = p.get("_meta") or {}
            out: dict[str, Any] = {"type": "program_output"}
            if meta.get("itemId"):
                out["id"] = meta["itemId"]
            out["call_id"] = p.get("id")
            out["result"] = p.get("result")
            if "status" in p:
                out["status"] = p["status"]
            items.append(out)

    def _append_tool_items(self, items: list[Any], msg: Message, names: dict[str, str]) -> None:
        for p in _as_parts(msg.get("content")):
            if p.get("type") != "tool_result":
                continue
            # `name`/`namespace` identify the tool that produced the output
            # (openai-ts 7.x). Probe-verified 2026-08-06: accepted, and a
            # non-string `namespace` is rejected, so the fields are validated
            # rather than ignored. The name comes from the matching call -- we
            # never invent one, so a result with no matching call omits it.
            name = names.get(p["id"])
            content = p.get("content")
            item: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": p.get("id"),
                "output": content if isinstance(content, str) else js_json(content),
            }
            if name is not None:
                item["name"] = name
            if "namespace" in p:
                item["namespace"] = p["namespace"]
            if p.get("caller"):
                item["caller"] = to_wire_caller(p["caller"])
            items.append(item)

    def enable_streaming(
        self, provider_req: ProviderHttpRequest, _req: Mapping[str, Any] | None = None
    ) -> None:
        provider_req.body["stream"] = True

    def response_spec_id(self) -> str:
        """Overridden by xAI, whose Responses API is this one."""
        return "openai/responses.response"

    def response_registry(self) -> Registry:
        """Overridden alongside the spec id: xAI extends file extraction, so its
        registry supplies a different `oaiRespFiles` -- the parse-side twin of its
        `files_from_output_item` override."""
        return OPENAI_RESPONSES_REGISTRY

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Spec-driven; see `wire/specs/responses/openai.responses.json`."""
        return build_response(
            get_response_spec(self.response_spec_id()),
            raw,
            self.response_registry(),
            extra={"latencyMs": latency_ms, "raw": raw},
        )

    def stream_spec_id(self) -> str:
        """Overridden by xAI, which extends file extraction."""
        return "openai/responses.stream"

    def stream_registry(self) -> Registry:
        return OPENAI_RESPONSES_STREAM_REGISTRY

    def parse_stream_event(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Stateless entry, as the ProviderAdapter protocol requires."""
        return self.create_stream_parser()(event)

    def create_stream_parser(self) -> Callable[[Mapping[str, Any]], list[dict[str, Any]]]:
        """Stateful, and the one callers should use.

        `phase` is announced once when an item is added but belongs on every text
        delta of that item, and the spec's `state` is where that is remembered.
        """
        return create_stream_builder(
            get_stream_spec(self.stream_spec_id()), self.stream_registry()
        )

    def files_from_output_item(self, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Hosted code-execution output files from one output item.

        Overridable so Responses-compatible providers with a different file shape
        (e.g. xAI, which embeds files in the code-interpreter `logs` payload) can
        extend it.
        """
        return files_from_responses_output_item(item)

    def parse_usage(self, u: Mapping[str, Any] | None) -> dict[str, Any]:
        return openai_responses_usage(u)


def _as_parts(content: Any) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


__all__ = [
    "OpenAIResponsesAdapter",
    "OpenAIResponsesAdapterConfig",
    "builtin_call_from_responses_item",
    "filename_for_mime",
    "files_from_responses_output_item",
    "from_wire_caller",
    "openai_responses_usage",
    "to_wire_caller",
]
