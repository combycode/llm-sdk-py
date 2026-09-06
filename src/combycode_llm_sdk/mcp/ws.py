"""JSON-RPC over a WebSocket: duplex, so both directions are just frames.

**Non-standard, and deliberately kept.** The MCP SDK removed its own WebSocket
transport in `mcp` 2.0.0 on the grounds that it "was never part of the MCP
specification". Ours stays, because an upstream deletion is not our deletion:
this is exported API, removing it would break consumers, and no protocol reason
demands it. Treat it as a supported extra rather than a spec transport -- the
server has to opt into JSON-RPC over a socket, and the standard transports
remain stdio and Streamable HTTP.

What it buys: a socket is duplex by nature, so the server->client direction is
the same channel as everything else. Streamable HTTP needs a second GET held
open to say the same things.

The socket comes from `httpx2`, which this library already depends on, through
its `[ws]` extra -- so this costs no new dependency, only an optional one. A
caller who wants a different client passes `connect=`, which is the same escape
hatch the TypeScript offers when a runtime has no WebSocket global.

Transposed from `unified-library-ts/src/plugins/mcp/transport-ws.ts`.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .errors import McpError, McpErrorCode
from .transport import BaseJsonRpcTransport

DEFAULT_TIMEOUT_SECONDS = 60.0

#: How long one receive waits before the reader loop looks around.
#:
#: The session is NOT touched from two threads at once: a send takes the same
#: lock the reader holds, so the reader has to let go regularly for a send to
#: get in. This is that interval -- the longest a `request()` can wait before
#: its frame goes out, and the price of not having to assume anything about the
#: thread-safety of somebody else's state machine.
POLL_SECONDS = 0.05

#: How long the reader stays OFF the lock between polls.
#:
#: Releasing it is not enough. A Python lock is not fair: a thread that releases
#: and immediately re-acquires usually wins again, so the reader's
#: `release -> continue -> acquire` loop can starve a sender indefinitely -- the
#: handshake then blocks forever in `_send_message` and the process hangs with
#: no error. Measured 2026-09-06: never on a fast desktop, twice on 2-core CI
#: runners, and once the per-test timeout was added it named this exact lock.
#:
#: So the reader sleeps here with the lock RELEASED, which is a window a waiting
#: sender cannot lose rather than one it has to win.
SEND_WINDOW_SECONDS = 0.005


class WsSession(Protocol):
    """The socket, as this transport uses it."""

    def send_text(self, data: str) -> None: ...

    def receive_text(self, timeout: float | None = None) -> str: ...

    def close(self, code: int = 1000, reason: str | None = None) -> None: ...


#: Opens a session and hands back a context manager over it. `httpx2.websocket`
#: satisfies this, and so does anything else a caller wants to bring.
WsConnect = Callable[..., Any]


def default_connect(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """A socket from `httpx2`, the dependency this library already has."""
    try:
        import httpx2
    except ImportError as exc:  # pragma: no cover -- httpx2 is a hard dependency
        raise McpError(
            "MCP WebSocket needs httpx2, which this library depends on",
            code=McpErrorCode.CONNECTION_CLOSED,
        ) from exc
    return httpx2.websocket(
        url,
        headers=dict(headers or {}),
        subprotocols=list(subprotocols) if subprotocols else None,
        timeout=timeout,
    )


def session_is_live(session: Any) -> bool:
    """Whether the socket is still open, for a session that can say.

    `receive_text` raises `TimeoutError` both for "nothing yet" and for "this
    socket is gone", so the exception alone cannot tell them apart. The session
    can: `httpx2` exposes the `wsproto` connection, whose state leaves `OPEN`
    the moment either side closes.

    Read by NAME rather than by importing `wsproto`, so the check costs no
    import and a caller's own client is free to expose the same shape. A
    session that cannot answer is assumed alive, which is the older behaviour
    and the only safe default -- guessing "dead" would tear down a working
    connection.
    """
    connection = getattr(session, "connection", None)
    state = getattr(connection, "state", None)
    if state is None:
        return True
    return str(getattr(state, "name", "")) in {"OPEN", "CONNECTING"}


class WsTransport(BaseJsonRpcTransport):
    """One MCP server, reached over a WebSocket."""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        subprotocols: Sequence[str] | None = None,
        name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connect: WsConnect | None = None,
    ) -> None:
        super().__init__()
        self._url = url
        self._headers = dict(headers or {})
        self._subprotocols = list(subprotocols) if subprotocols else None
        self._name = name or "server"
        self._timeout = timeout
        self._connect = connect or default_connect

        self._session: WsSession | None = None
        self._reader: threading.Thread | None = None
        self._opened = threading.Event()
        self._open_error: BaseException | None = None
        self._closed = False
        #: Guards the SESSION. Distinct from the base class's lock, which
        #: guards the pending map: `close()` holds this one while failing every
        #: pending request, and one lock for both would be a deadlock.
        #: See POLL_SECONDS for why the session gets one lock and not two.
        self._session_lock = threading.RLock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the socket, and do not return until it is open or has failed.

        The connection is a context manager, and a context manager has to be
        entered and left on the SAME thread -- so the reader thread owns the
        whole session lifetime and this call waits for it to say it is up.
        """
        self._closed = False
        self._opened.clear()
        self._open_error = None
        self._reader = threading.Thread(target=self._run, name="mcp-ws", daemon=True)
        self._reader.start()
        if not self._opened.wait(self._timeout):
            self._closed = True
            raise McpError(
                f"MCP ws: {self._url} did not open within {self._timeout:g}s",
                code=McpErrorCode.REQUEST_TIMEOUT,
            )
        if self._open_error is not None:
            raise McpError(
                f"MCP ws: could not open {self._url}: {self._open_error}",
                code=McpErrorCode.CONNECTION_CLOSED,
            ) from self._open_error

    def close(self) -> None:
        """Set the flag and wait for the reader to unwind.

        The socket is closed by the CONTEXT MANAGER the reader is sitting in, so
        there is nothing to close here: the flag is what the reader is watching
        for, and the join is what makes this call mean the socket is actually
        shut rather than about to be. Closing the session here as well would be
        a second owner racing the first for a saving of one poll interval.
        """
        self._closed = True
        self._fail_all(McpError("MCP transport closed", code=McpErrorCode.CONNECTION_CLOSED))
        self._session = None
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            # Bounded: the reader wakes at most one poll interval from now, and
            # a hung peer must not hold up the caller's shutdown.
            reader.join(timeout=self._timeout)

    @property
    def is_open(self) -> bool:
        return self._session is not None and not self._closed

    def set_protocol_version(self, version: str) -> None:
        """Nothing to record: a socket has no per-message headers."""

    def set_era(self, era: str) -> None:
        """Nothing to record, for the same reason."""

    def listen(self) -> None:
        """Already duplex -- server frames arrive on the one channel."""

    # -- sending -------------------------------------------------------------

    def request(self, method: str, params: Any = None) -> Any:
        request_id = self._allocate_id()
        pending = self._register(request_id)
        try:
            self._send_message(self._frame(method, params, request_id))
        except McpError:
            with self._lock:
                self._pending.pop(request_id, None)
            raise
        return self._await_response(pending, request_id, method, self._timeout)

    def notify(self, method: str, params: Any = None) -> None:
        self._send_message(self._frame(method, params))

    def _send_message(self, message: Mapping[str, Any]) -> None:
        session = self._session
        if session is None or self._closed:
            raise McpError(
                "MCP ws transport is not started (or has been closed)",
                code=McpErrorCode.CONNECTION_CLOSED,
            )
        text = json.dumps(message, separators=(",", ":"))
        try:
            with self._session_lock:
                session.send_text(text)
        except Exception as exc:
            # thing to a caller: the message did not go.
            raise McpError(
                f"MCP ws: sending failed: {exc}", code=McpErrorCode.CONNECTION_CLOSED
            ) from exc

    # -- reading -------------------------------------------------------------

    def _run(self) -> None:
        """Own the session for its whole life, and pump frames off it."""
        try:
            with self._connect(
                self._url,
                headers=self._headers,
                subprotocols=self._subprotocols,
                timeout=self._timeout,
            ) as session:
                self._session = session
                self._opened.set()
                self._pump(session)
        except Exception as exc:  # noqa: BLE001 -- reported to whoever is waiting,
            # which is `start()` before the socket is up and `_fail_all` after.
            self._open_error = exc
        finally:
            self._session = None
            # Always set: a `start()` waiting on a socket that failed to open
            # must be woken, or it waits out the full timeout for an answer that
            # already exists.
            self._opened.set()
            if not self._closed:
                self._fail_all(
                    McpError("MCP ws closed", code=McpErrorCode.CONNECTION_CLOSED)
                )

    def _pump(self, session: WsSession) -> None:
        while not self._closed:
            try:
                with self._session_lock:
                    text = session.receive_text(timeout=POLL_SECONDS)
            except TimeoutError:
                # Nothing waiting -- OR the socket is gone, because the same
                # exception says both. So the poll is also where liveness is
                # checked; without it the pump spins forever on a dead peer
                # while every caller waits out its full timeout and is then
                # told it "timed out".
                if not session_is_live(session):
                    return
                # The other half of the short poll: the lock is released here so
                # a sender can get in. The sleep is what makes that reliable --
                # without it the next `acquire` is already queued and the sender
                # may never be scheduled. See SEND_WINDOW_SECONDS.
                time.sleep(SEND_WINDOW_SECONDS)
                continue
            except Exception:  # noqa: BLE001 -- the socket died; the caller learns
                # through `_fail_all` in the finally above.
                return
            self._route_text(text)

    def _route_text(self, text: str) -> None:
        try:
            message = json.loads(text)
        except ValueError:
            # Not JSON. Dropped rather than fatal, for the same reason stdio
            # drops a banner: one bad frame is not a dead connection.
            return
        if isinstance(message, Mapping):
            self._route_incoming(message)

    def __repr__(self) -> str:
        return f"<WsTransport {self._url} open={self.is_open}>"


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "POLL_SECONDS",
    "WsConnect",
    "WsSession",
    "WsTransport",
    "default_connect",
    "session_is_live",
]
