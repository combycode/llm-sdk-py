"""This library behind an OpenAI-shaped API.

`handle()` is a pure function from a parsed request to a response. No socket, no
framework, no globals -- which is what makes auth, routing and error mapping
things a test can call rather than things a test has to connect to.

One divergence from the TypeScript, and the reviewed example asks for it:
**`/health` is answered before auth.** The TypeScript runs the verifier first for
every route, so attaching auth makes the liveness probe answer 401 -- and every
orchestrator reads that as a dead process and restarts it. A liveness probe
carries no credential by design.

Transposed from `unified-library-ts/src/server/server.ts`.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from ..bus.hook_bus import HookBus
from .auth import AuthError, AuthPlugin
from .dispatch import dispatch
from .oai import (
    DEFAULT_STREAM_CHUNK_CHARS,
    InvalidRequest,
    build_error_body,
    estimate_tokens,
    extract_last_user_text,
    extract_system_text,
    stream_frames,
    validate_chat_request,
)
from .response_store import ResponseStore
from .router import ModelNotRegistered, ModelRouter, ServerEntry
from .types import (
    HttpRequest,
    HttpResponse,
    cors_headers,
    json_response,
    sse_response,
)

#: Routes that answer before any credential is looked at.
PUBLIC_ROUTES = frozenset({"/health"})


class OaiServer:
    """An OpenAI-compatible front end for whatever you registered behind it."""

    def __init__(
        self,
        *,
        entries: Sequence[ServerEntry] = (),
        auth: AuthPlugin | None = None,
        hooks: HookBus | None = None,
        agent_loader: Any = None,
        conversation_loader: Any = None,
        stream_chunk_chars: int = DEFAULT_STREAM_CHUNK_CHARS,
        response_store: ResponseStore | None = None,
    ) -> None:
        self.id = f"oai-server-{uuid.uuid4().hex[:6]}"
        self.hooks = hooks or HookBus()
        self.auth = auth
        self.agent_loader = agent_loader
        self.conversation_loader = conversation_loader
        #: How much text each streamed frame carries. A knob rather than a
        #: constant because what reads well depends on the client: a terminal
        #: wants small pieces, a browser buffering into a paragraph does not
        #: care.
        self.stream_chunk_chars = stream_chunk_chars
        #: Conversations a client can ask to continue. Absent by default: a
        #: server that remembers by accident is a privacy question nobody
        #: asked to answer.
        self.response_store = response_store
        self._router = ModelRouter(entries)

    # -- registration --------------------------------------------------------

    @property
    def router(self) -> ModelRouter:
        return self._router

    def register(self, entry: ServerEntry) -> None:
        self._router.register(entry)

    def unregister(self, model: str) -> bool:
        return self._router.unregister(model)

    def models(self) -> list[str]:
        return self._router.models()

    # -- the handler ---------------------------------------------------------

    def handle(self, request: HttpRequest) -> HttpResponse:
        """Answer one request. Pure: same request in, same response out."""
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        user_id: str | None = None
        model: str | None = None

        try:
            if request.method == "OPTIONS":
                return self._finish(
                    HttpResponse(status=204, headers=cors_headers()),
                    request_id, started, user_id, model,
                )

            if self._is_public(request):
                return self._finish(
                    self._public_route(request), request_id, started, user_id, model
                )

            if self.auth is not None:
                try:
                    user_id = self.auth.verify(request.headers).user_id
                except (AuthError, Exception) as exc:  # noqa: BLE001 -- a verifier is
                    # caller code; whatever it raises means "not authenticated",
                    # and the alternative is a 500 that reads as our fault.
                    self.hooks.emit_sync(
                        "onAuthFail",
                        {"serverId": self.id, "requestId": request_id, "reason": str(exc)},
                    )
                    return self._finish(
                        json_response(
                            build_error_body(str(exc), "authentication_error"), 401
                        ),
                        request_id, started, user_id, model,
                    )

            self.hooks.emit_sync(
                "onServerRequest",
                {
                    "serverId": self.id,
                    "requestId": request_id,
                    "method": request.method,
                    "path": request.path,
                    "userId": user_id,
                    "model": model,
                },
            )

            if request.method == "GET" and request.path == "/v1/models":
                response = json_response({"object": "list", "data": self._router.listing()})
            elif request.method == "POST" and request.path == "/v1/chat/completions":
                try:
                    parsed = validate_chat_request(request.body)
                except InvalidRequest as exc:
                    response = json_response(
                        build_error_body(str(exc), "invalid_request_error"), 400
                    )
                else:
                    model = str(parsed["model"])
                    response = self._chat_completions(parsed, user_id)
            else:
                response = json_response(
                    build_error_body(
                        f"unknown route: {request.method} {request.path}", "not_found"
                    ),
                    404,
                )
        except Exception as exc:  # noqa: BLE001 -- a handler that raises through
            # its own socket shell takes the process with it; every failure has
            # to become a response.
            response = json_response(build_error_body(str(exc), "server_error"), 500)

        return self._finish(response, request_id, started, user_id, model)

    # -- routes --------------------------------------------------------------

    def _is_public(self, request: HttpRequest) -> bool:
        return request.method == "GET" and request.path in PUBLIC_ROUTES

    def _public_route(self, request: HttpRequest) -> HttpResponse:
        return json_response({"status": "ok", "models": len(self._router)})

    def _chat_completions(self, parsed: Mapping[str, Any], user_id: str | None) -> HttpResponse:
        try:
            target = self._router.resolve(str(parsed["model"]))
        except ModelNotRegistered as exc:
            return json_response(build_error_body(str(exc), "model_not_found"), 404)

        messages = list(parsed["messages"])
        try:
            user_text = extract_last_user_text(messages)
        except InvalidRequest as exc:
            return json_response(build_error_body(str(exc), "invalid_request_error"), 400)
        system_prompt = extract_system_text(messages)

        conversation_id = str(parsed.get("user") or user_id or f"default:{target.model}")
        loop = None
        if self.agent_loader is not None:
            loop = self.agent_loader(
                user_id=user_id, model=target.model, conversation_id=conversation_id
            )

        result = dispatch(
            target=target,
            user_text=user_text,
            system_prompt=system_prompt or None,
            external_tools=parsed.get("tools"),
            max_tokens=parsed.get("max_tokens"),
            temperature=parsed.get("temperature"),
            hooks=self.hooks,
            agent=loop,
        )

        if parsed.get("stream"):
            # The answer is complete by now; what streams is the delivery. Said
            # plainly rather than implied, because a caller who needs the
            # model's own token timing wants `LLM.stream()` and not this.
            return sse_response(
                stream_frames(
                    chunk_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    model=target.model,
                    text=result.text,
                    finish_reason=result.finish_reason or "stop",
                    chunk_chars=self.stream_chunk_chars,
                )
            )

        # A provider that reported nothing gets an estimate rather than a zero:
        # `total_tokens: 0` reads as "this was free", which is the one wrong
        # answer nobody double-checks.
        prompt_tokens = result.input_tokens or estimate_tokens(user_text + (system_prompt or ""))
        completion_tokens = result.output_tokens or estimate_tokens(result.text)

        from .oai import build_chat_response

        return json_response(
            build_chat_response(
                # The REGISTERED id, not the provider's own. That is what lets a
                # client switch providers without changing anything it sends.
                model=target.model,
                text=result.text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=result.finish_reason or "stop",
            )
        )

    # -- instrumentation -----------------------------------------------------

    def _finish(
        self,
        response: HttpResponse,
        request_id: str,
        started: float,
        user_id: str | None,
        model: str | None,
    ) -> HttpResponse:
        self.hooks.emit_sync(
            "onServerResponse",
            {
                "serverId": self.id,
                "requestId": request_id,
                "status": response.status,
                "latencyMs": (time.perf_counter() - started) * 1000,
                "userId": user_id,
                "model": model,
            },
        )
        return response

    def __repr__(self) -> str:
        return f"<OaiServer {self.id} models={self._router.models()}>"


__all__ = ["PUBLIC_ROUTES", "OaiServer"]
