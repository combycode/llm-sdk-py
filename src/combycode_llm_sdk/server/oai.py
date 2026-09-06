"""OpenAI's Chat Completions shapes, in and out.

Every function here is pure. That is what lets the awkward parts -- what counts
as a valid request, which message is "the" user message, how usage is reported
when a provider does not report it -- be tested as values rather than through a
server.

The scope is the subset real clients actually send: the official SDKs, LM
Studio, Open WebUI. Fields nobody sends are omitted rather than half-supported.

Transposed from `unified-library-ts/src/server/oai-adapter.ts` and
`oai-types.ts`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

#: Roughly four characters to a token. Only used when a provider reported no
#: usage at all -- see `build_chat_response` for why a zero is not acceptable.
CHARS_PER_TOKEN = 4


class InvalidRequest(ValueError):
    """The request is not a Chat Completions request."""


def oai_content_to_text(content: Any) -> str:
    """A message's content as text, whichever of the two shapes it arrived in."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(_part_to_text(p) for p in content if isinstance(p, Mapping))
    return ""


def _part_to_text(part: Mapping[str, Any]) -> str:
    kind = part.get("type")
    if kind == "text":
        return str(part.get("text") or "")
    if kind == "image_url":
        url = str((part.get("image_url") or {}).get("url") or "")
        short = f"{url[:60]}..." if len(url) > 80 else url
        return f"[image: {short}]"
    return ""


def extract_last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The message being answered.

    The LAST user message, not the first: a client replaying a conversation
    sends the whole thing every time, and answering the first would answer the
    opening question forever.
    """
    for message in reversed(list(messages)):
        if message.get("role") == "user":
            return oai_content_to_text(message.get("content"))
    raise InvalidRequest('request.messages has no "user" entry')


def extract_system_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Every system message, joined. Clients do send more than one."""
    texts = [
        oai_content_to_text(m.get("content")) for m in messages if m.get("role") == "system"
    ]
    return "\n\n".join(t for t in texts if t)


def estimate_tokens(text: str) -> int:
    """A count for a provider that reported none.

    An estimate, and it must never be zero for non-empty text: a client reading
    `total_tokens: 0` concludes the request was free, and a billing dashboard
    built on that is wrong in the direction nobody checks.
    """
    return -(-len(text) // CHARS_PER_TOKEN) if text else 0


def validate_chat_request(body: Any) -> dict[str, Any]:
    """Refuse anything that is not a Chat Completions request, saying why.

    Checked here rather than left to fail later, because the failure otherwise
    surfaces as a provider error about a request the client never made.
    """
    if not isinstance(body, Mapping):
        raise InvalidRequest("body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise InvalidRequest("`model` must be a non-empty string")
    messages = body.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise InvalidRequest("`messages` must be a non-empty array")
    for message in messages:
        if not isinstance(message, Mapping):
            raise InvalidRequest("each message must be an object")
        if not isinstance(message.get("role"), str):
            raise InvalidRequest("each message must have a `role`")
    return dict(body)


def build_chat_response(
    *,
    model: str,
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
    response_id: str | None = None,
) -> dict[str, Any]:
    """The answer, in the shape an OpenAI client already knows how to read."""
    return {
        "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_stream_chunk(
    *,
    chunk_id: str,
    model: str,
    delta: Mapping[str, Any] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """One `chat.completion.chunk`, as a streaming client expects it."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": dict(delta or {}), "finish_reason": finish_reason}
        ],
    }


def format_sse_frame(data: Any) -> str:
    """One server-sent-event frame. The blank line is the delimiter, not decoration."""
    return f"data: {json.dumps(data)}\n\n"


#: What tells a client the stream is over. Not JSON, by OpenAI's own design.
SSE_TERMINATOR = "data: [DONE]\n\n"

#: How much text each `chat.completion.chunk` carries. Small enough that a
#: reader sees progress, large enough that a paragraph is not a hundred frames.
DEFAULT_STREAM_CHUNK_CHARS = 40


def split_for_stream(text: str, size: int = DEFAULT_STREAM_CHUNK_CHARS) -> list[str]:
    """One answer, cut into the pieces a stream delivers it in.

    A fixed width rather than a token or word boundary, and deliberately: this
    is a presentation detail, the client reassembles by concatenation, and
    inventing linguistic boundaries would mean a chunker whose bugs land in
    somebody's UI. `size` is floored at 1 so a caller cannot ask for an empty
    piece and get an endless stream of them.
    """
    if not text:
        return []
    step = max(1, size)
    return [text[i : i + step] for i in range(0, len(text), step)]


def stream_frames(
    *,
    chunk_id: str,
    model: str,
    text: str,
    finish_reason: str = "stop",
    chunk_chars: int = DEFAULT_STREAM_CHUNK_CHARS,
) -> Iterator[str]:
    """The whole SSE body of one answer, frame by frame.

    The shape a Chat Completions client expects, in order: a first frame that
    carries the ROLE and no content, then one frame per piece of text, then a
    frame carrying `finish_reason` and no content, then `[DONE]`.

    The role frame is not decoration -- a client that builds a message from the
    deltas has nothing to attribute the text to without it, and several client
    SDKs drop the whole choice.

    **This is not incremental generation.** The answer is complete before the
    first frame is built; what streams is the delivery. A caller who needs the
    model's own token timing wants `LLM.stream()`, not a server shim.
    """
    yield format_sse_frame(
        build_stream_chunk(chunk_id=chunk_id, model=model, delta={"role": "assistant"})
    )
    for piece in split_for_stream(text, chunk_chars):
        yield format_sse_frame(
            build_stream_chunk(chunk_id=chunk_id, model=model, delta={"content": piece})
        )
    yield format_sse_frame(
        build_stream_chunk(chunk_id=chunk_id, model=model, delta={}, finish_reason=finish_reason)
    )
    yield SSE_TERMINATOR


def build_models_list(ids: Sequence[str]) -> list[dict[str, Any]]:
    """Plain model ids as OpenAI's `/v1/models` rows.

    Not the same job as `Router.listing()`, which describes the models this
    server actually routes and carries their capabilities. This is for a caller
    assembling its own surface from a bare list of names, and answers the
    minimum OpenAI clients require.

    One `created` for the whole list rather than one per row: they were not
    created at measurably different times, and a differing timestamp reads as
    meaning something.
    """
    created = int(time.time())
    return [
        {"id": model_id, "object": "model", "created": created, "owned_by": "orxa"}
        for model_id in ids
    ]


def build_error_body(message: str, error_type: str = "invalid_request_error",
                     code: str | None = None) -> dict[str, Any]:
    """An error a client's own SDK will parse and raise properly."""
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code:
        error["code"] = code
    return {"error": error}


def external_tool_declarations(tools: Any) -> list[dict[str, Any]]:
    """The client's tool definitions, as declarations only.

    Declared to the model, never executed here: the bodies live in the client's
    process. A model that calls one gets an error telling it to answer in text,
    which is a far better outcome than the server inventing a result.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return out
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping) or not function.get("name"):
            continue
        out.append(
            {
                "name": str(function["name"]),
                "description": str(function.get("description") or ""),
                "parameters": dict(function.get("parameters") or {}),
            }
        )
    return out


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_STREAM_CHUNK_CHARS",
    "SSE_TERMINATOR",
    "InvalidRequest",
    "build_chat_response",
    "build_error_body",
    "build_stream_chunk",
    "estimate_tokens",
    "external_tool_declarations",
    "extract_last_user_text",
    "extract_system_text",
    "format_sse_frame",
    "oai_content_to_text",
    "split_for_stream",
    "stream_frames",
    "validate_chat_request",
]
