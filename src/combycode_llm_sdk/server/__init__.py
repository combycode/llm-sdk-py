"""This library behind an OpenAI-shaped HTTP API.

Register a model, point an existing OpenAI client at the port, and the request
reaches whichever provider that model was registered with. `create_server` in
`helpers/server.py` is the one-call form.

The split that matters: `OaiServer.handle` is a pure function from a parsed
request to a response, and `http_shell.py` is the only file that knows about
sockets. Auth, routing and error mapping are therefore things a test calls,
not things a test connects to.

Transposed from `unified-library-ts/src/server/`.

`stream: true` is answered as a real SSE stream: the frames are written and
flushed one at a time, so a reader sees the first before the last exists. What
streams is the DELIVERY -- the answer is complete before the first frame is
built, which is the design the TypeScript's `streamChunkChars` names and has
never wired up. A caller who needs the model's own token timing wants
`LLM.stream()`.
"""

from __future__ import annotations

from .app import PUBLIC_ROUTES, OaiServer
from .auth import AuthError, AuthPlugin, AuthVerifyResult, BearerKeyAuth
from .dispatch import DispatchResult, dispatch, merge_tools
from .http_shell import make_http_server, serve, write_stream, wsgi_app
from .oai import (
    DEFAULT_STREAM_CHUNK_CHARS,
    SSE_TERMINATOR,
    InvalidRequest,
    build_chat_response,
    build_error_body,
    build_stream_chunk,
    estimate_tokens,
    extract_last_user_text,
    extract_system_text,
    format_sse_frame,
    oai_content_to_text,
    split_for_stream,
    stream_frames,
    validate_chat_request,
)
from .response_store import (
    ResponseEntry,
    ResponseStore,
    ResponseTarget,
    has_fresh_provider_state,
    new_response_id,
)
from .router import (
    ModelCapabilities,
    ModelNotRegistered,
    ModelRouter,
    ResolvedTarget,
    ServerEntry,
)
from .types import HttpRequest, HttpResponse, cors_headers, json_response

__all__ = [
    "DEFAULT_STREAM_CHUNK_CHARS",
    "PUBLIC_ROUTES",
    "SSE_TERMINATOR",
    "AuthError",
    "AuthPlugin",
    "AuthVerifyResult",
    "BearerKeyAuth",
    "DispatchResult",
    "HttpRequest",
    "HttpResponse",
    "InvalidRequest",
    "ModelCapabilities",
    "ModelNotRegistered",
    "ModelRouter",
    "OaiServer",
    "ResolvedTarget",
    "ResponseEntry",
    "ResponseStore",
    "ResponseTarget",
    "ServerEntry",
    "build_chat_response",
    "build_error_body",
    "build_stream_chunk",
    "cors_headers",
    "dispatch",
    "estimate_tokens",
    "extract_last_user_text",
    "extract_system_text",
    "format_sse_frame",
    "has_fresh_provider_state",
    "json_response",
    "make_http_server",
    "merge_tools",
    "new_response_id",
    "oai_content_to_text",
    "serve",
    "split_for_stream",
    "stream_frames",
    "validate_chat_request",
    "write_stream",
    "wsgi_app",
]
