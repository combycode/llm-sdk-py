"""Carrying JSON-RPC to a server, and routing what it sends back.

The connection is BIDIRECTIONAL, which is the part that shapes this file. A
server does not only answer: it pushes notifications (logging, list-changed,
progress) and it makes REQUESTS of its own (sampling, elicitation, roots). So a
transport cannot be a request/response function; it has to own correlation, and
it has to have somewhere to send an inbound message that nobody asked for.

`BaseJsonRpcTransport` holds everything that is the same over any wire --
the pending map, the id counter, the routing, the timeouts -- so a concrete
transport supplies only `_send_message` plus its own lifecycle and parsing.

Threads, not a loop: `stdio.py` reads its child on a reader thread, so a pending
request waits on an `Event` rather than a future. This is the sync core; an
async twin would share this file's shape and not its mechanism.

Transposed from `unified-library-ts/src/plugins/mcp/transport.ts` and
`base-transport.ts`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import McpError, McpErrorCode

#: Called when a long-lived stream finishes -- with the error that killed it, or
#: None for a clean server-side teardown.
OnStreamEnd = Callable[[BaseException | None], None]

#: Called with a long-lived request's id BEFORE it is sent. The hook exists for
#: one reason: the server can answer -- with an acknowledgement, or an event --
#: faster than `send_long_lived_request` returns, and the reader thread then
#: routes a frame for a subscription the caller has not been given the id of
#: yet. Registering inside this callback closes that window entirely, because
#: nothing has been sent when it runs.
OnStreamOpen = Callable[[int], None]


@dataclass
class IncomingHandlers:
    """What to do with a message the server started.

    Both optional, and the difference matters: a request without a handler must
    be ANSWERED (with a method-not-found error), because the server is waiting.
    A notification without a handler is simply dropped.
    """

    #: Server->client request. Return the result, or raise `McpError`.
    on_request: Callable[[str, Any], Any] | None = None
    #: Server->client notification. No reply.
    on_notification: Callable[[str, Any], None] | None = None


class McpTransport(Protocol):
    """The wire, as the client sees it."""

    def start(self) -> None: ...

    def request(self, method: str, params: Any = None) -> Any: ...

    def notify(self, method: str, params: Any = None) -> None: ...

    def set_handlers(self, handlers: IncomingHandlers) -> None: ...

    def close(self) -> None: ...


@dataclass
class _Pending:
    """One request waiting for its answer."""

    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class BaseJsonRpcTransport:
    """Correlation, routing and timeouts -- everything above the wire itself."""

    def __init__(self) -> None:
        self._next_id = 0
        self._handlers = IncomingHandlers()
        self._pending: dict[int, _Pending] = {}
        #: Long-lived requests -- `subscriptions/listen` -- by id. Separate from
        #: `_pending` because they are the opposite kind of thing: a pending
        #: request is waiting for an answer, whereas one of these has ALREADY
        #: been answered in the only sense that matters (the stream is open) and
        #: its eventual response means the stream ENDED.
        self._long_lived: dict[int, OnStreamEnd | None] = {}
        # The id counter and both maps are touched by the caller's thread and by
        # the reader thread, so they live under one lock. It is held only around
        # the dict operations, never across a wait or a write.
        self._lock = threading.Lock()

    # -- for the concrete transport ------------------------------------------

    def _send_message(self, message: Mapping[str, Any]) -> None:
        """Write one serialised JSON-RPC object to the peer."""
        raise NotImplementedError

    def _allocate_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    # -- public --------------------------------------------------------------

    def set_handlers(self, handlers: IncomingHandlers) -> None:
        self._handlers = handlers

    # -- routing -------------------------------------------------------------

    def _route_incoming(self, message: Mapping[str, Any]) -> None:
        """Send one parsed inbound message where it belongs.

        The three-way test is on the PRESENCE of `id` and `method`, which is how
        JSON-RPC distinguishes them: a response has an id and no method, a
        request has both, a notification has a method and no id.
        """
        has_id = "id" in message and message["id"] is not None
        has_method = "method" in message and message["method"] is not None
        if has_id and not has_method:
            self._resolve_response(message)
        elif has_id and has_method:
            self._handle_request(message)
        elif has_method:
            handler = self._handlers.on_notification
            if handler is not None:
                handler(str(message["method"]), message.get("params"))

    def _resolve_response(self, message: Mapping[str, Any]) -> None:
        raw_id = message.get("id")
        if not isinstance(raw_id, int):
            return
        with self._lock:
            pending = self._pending.pop(raw_id, None)
        if pending is None:
            # A long-lived request answers only when the server ends the
            # subscription, so its response arrives with no pending entry. That
            # is the end-of-stream signal, not a stray message.
            wire_error = message.get("error")
            self._resolve_long_lived(
                raw_id, McpError.of(wire_error) if isinstance(wire_error, Mapping) else None
            )
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            pending.error = McpError.of(error)
        else:
            pending.result = message.get("result")
        pending.done.set()

    def _handle_request(self, message: Mapping[str, Any]) -> None:
        """Answer a request the server made of us.

        Every path here writes a reply. A server-initiated request that is
        silently dropped leaves the SERVER blocked on a response that will never
        come, which looks from the outside like our own connection hanging.
        """
        request_id = message.get("id")
        method = str(message.get("method") or "")
        handler = self._handlers.on_request
        if handler is None:
            self._send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": McpErrorCode.METHOD_NOT_FOUND,
                        "message": "no request handler registered",
                    },
                }
            )
            return
        try:
            result = handler(method, message.get("params"))
        except McpError as exc:
            self._send_message({"jsonrpc": "2.0", "id": request_id, "error": exc.as_wire()})
        except Exception as exc:  # noqa: BLE001 -- a handler is caller code and may
            # raise anything; the server is waiting either way and must be told.
            self._send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": McpErrorCode.INTERNAL_ERROR, "message": str(exc)},
                }
            )
        else:
            self._send_message(
                {"jsonrpc": "2.0", "id": request_id, "result": result if result is not None else {}}
            )

    # -- request/response ----------------------------------------------------

    def _await_response(
        self, pending: _Pending, request_id: int, method: str, timeout: float
    ) -> Any:
        """Wait for one answer.

        Takes the `_Pending` the caller registered rather than looking it up
        again: between `_register` and here the reader thread may already have
        answered and REMOVED the entry, and a second lookup then raises KeyError
        on a request that in fact succeeded. A narrow window, and a fast local
        server walks straight into it.
        """
        if not pending.done.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise McpError(
                f"MCP request {method!r} timed out after {timeout:g}s",
                code=McpErrorCode.REQUEST_TIMEOUT,
            )
        if pending.error is not None:
            raise pending.error
        return pending.result

    def _register(self, request_id: int) -> _Pending:
        """Claim a slot for an answer, and hand back the thing to wait on."""
        pending = _Pending()
        with self._lock:
            self._pending[request_id] = pending
        return pending

    # -- long-lived requests -------------------------------------------------

    def send_long_lived_request(
        self,
        method: str,
        params: Any = None,
        on_end: OnStreamEnd | None = None,
        on_open: OnStreamOpen | None = None,
    ) -> int:
        """Open a stream and hand back its id, without arming a response timeout.

        The ordinary `request()` path is exactly wrong for `subscriptions/listen`:
        it waits for a response, and the response does not come until the server
        tears the subscription down -- which on a healthy subscription is never.
        So this registers the id with NO timeout and returns immediately.

        Without an entry in `_long_lived` the eventual response would arrive with
        nothing waiting for it, be dropped as an unmatched message, and the
        caller would never learn the stream had ended.

        `on_open` runs with the id BEFORE anything is sent -- see `OnStreamOpen`
        for why that ordering is the whole point.
        """
        request_id = self._allocate_id()
        with self._lock:
            self._long_lived[request_id] = on_end
        if on_open is not None:
            on_open(request_id)
        try:
            self._send_message(self._frame(method, params, request_id))
        except Exception:
            with self._lock:
                self._long_lived.pop(request_id, None)
            raise
        return request_id

    def _resolve_long_lived(self, request_id: int, error: BaseException | None = None) -> bool:
        """Settle a stream from its (late) response. Whether one was waiting."""
        with self._lock:
            if request_id not in self._long_lived:
                return False
            on_end = self._long_lived.pop(request_id)
        # Called OUTSIDE the lock: `on_end` is caller code that may close the
        # subscription, which comes straight back in here.
        if on_end is not None:
            on_end(error)
        return True

    def _fail_all(self, error: McpError) -> None:
        """Wake every in-flight request with the reason the connection ended.

        Without this a caller blocked in `request()` waits out its full timeout
        after the child has already died, and is then told "timed out" -- which
        sends whoever reads it looking for a slow server rather than a crashed
        one.
        """
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
            streams = list(self._long_lived.values())
            self._long_lived.clear()
        for entry in pending:
            entry.error = error
            entry.done.set()
        # Subscriptions die with the connection too. Without this a caller keeps
        # a handle to a stream that will never deliver again and is never told
        # -- which looks exactly like a server where nothing has changed.
        for on_end in streams:
            if on_end is not None:
                on_end(error)

    @staticmethod
    def _frame(method: str, params: Any, request_id: int | None = None) -> dict[str, Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0"}
        if request_id is not None:
            message["id"] = request_id
        message["method"] = method
        if params is not None:
            message["params"] = params
        return message


__all__ = [
    "BaseJsonRpcTransport",
    "IncomingHandlers",
    "McpTransport",
    "OnStreamEnd",
    "OnStreamOpen",
]
