"""The socket, owned by the engine and pumped by one reader thread.

The realtime sibling of the request queue -- except there is no queue. A live
socket has no per-call retry, no rate limit and no idempotency key: the call IS
the connection, and retrying it would mean starting the conversation again. What
the engine still owns is the transport and the observability, which is why the
frames pass through here and not straight from the provider adapter.

**Diverges from the TypeScript by necessity.** There it wraps a browser
`WebSocket` and subscribes with `addEventListener`; Python has no such object,
and `httpx2`'s session is a blocking pull. So the connection owns a reader
thread that pulls frames and fans them out -- the same shape the MCP WebSocket
transport already settled on in `mcp/ws.py`, for the same reason.

Transposed from `unified-library-ts/src/network/realtime-connection.ts`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .types import Frame, WsConnect, WsRequest, WsSession

#: How long one receive waits before the reader looks around. Also the longest a
#: `close()` can take to be noticed, and -- because a send takes the same lock
#: the reader holds -- the longest a frame can wait to go out.
POLL_SECONDS = 0.05

#: How long `open()` waits for the socket before giving up.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0


def default_connect(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    subprotocols: list[str] | None = None,
    timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> Any:
    """A socket from `httpx2`, the dependency this library already has."""
    import httpx2

    return httpx2.websocket(
        url,
        headers=dict(headers or {}),
        subprotocols=list(subprotocols) if subprotocols else None,
        timeout=timeout,
    )


def frame_of(event: Any) -> Frame | None:
    """One `wsproto` event as a normalised frame.

    Duck-typed on the payload rather than on the event class, so `wsproto` is
    never imported here and a caller's own client only has to produce something
    with a `.data`. Anything that is neither text nor bytes -- a ping, a
    close -- is not a frame and returns None.
    """
    data = getattr(event, "data", None)
    if isinstance(data, str):
        return Frame(text=data)
    if isinstance(data, (bytes, bytearray, memoryview)):
        return Frame(binary=bytes(data))
    return None


class RealtimeConnection:
    """One open socket: frames in on a thread, frames out from anywhere."""

    def __init__(
        self,
        request: WsRequest,
        hooks: Any,
        connect: WsConnect | None = None,
        timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.request = request
        self._hooks = hooks
        self._connect = connect or default_connect
        self._timeout = timeout

        self._session: WsSession | None = None
        self._reader: threading.Thread | None = None
        self._opened = threading.Event()
        self._open_error: BaseException | None = None
        self._closed = False
        #: Guards the SESSION. A send takes it, so the reader has to let go
        #: regularly -- see POLL_SECONDS.
        self._lock = threading.RLock()

        self._frame_cbs: list[Callable[[Frame], None]] = []
        self._open_cbs: list[Callable[[], None]] = []
        self._close_cbs: list[Callable[[], None]] = []
        self._error_cbs: list[Callable[[BaseException], None]] = []

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        """Open the socket and wait for it, or raise.

        The socket is a context manager, and a context manager has to be entered
        and left on the SAME thread -- so the reader owns the whole session
        lifetime and this call waits for it to say the socket is up.
        """
        self._closed = False
        self._opened.clear()
        self._open_error = None
        self._reader = threading.Thread(target=self._run, name="realtime-ws", daemon=True)
        self._reader.start()
        if not self._opened.wait(self._timeout):
            self._closed = True
            raise TimeoutError(
                f"realtime: {self.request.url} did not open within {self._timeout:g}s"
            )
        if self._open_error is not None:
            raise ConnectionError(
                f"realtime: could not open {self.request.url}: {self._open_error}"
            ) from self._open_error

    def close(self) -> None:
        """Set the flag and wait for the reader to unwind.

        The socket belongs to the context manager the reader is sitting in, so
        there is nothing to close here: the flag is what it watches for, and the
        join is what makes this call mean the socket is shut rather than about
        to be.
        """
        if self._closed and self._reader is None:
            return
        self._closed = True
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=self._timeout)

    @property
    def is_open(self) -> bool:
        return self._session is not None and not self._closed

    # -- subscribing ---------------------------------------------------------

    def on_frame(self, callback: Callable[[Frame], None]) -> Callable[[], None]:
        return self._subscribe(self._frame_cbs, callback)

    def on_open(self, callback: Callable[[], None]) -> Callable[[], None]:
        return self._subscribe(self._open_cbs, callback)

    def on_close(self, callback: Callable[[], None]) -> Callable[[], None]:
        return self._subscribe(self._close_cbs, callback)

    def on_error(self, callback: Callable[[BaseException], None]) -> Callable[[], None]:
        return self._subscribe(self._error_cbs, callback)

    @staticmethod
    def _subscribe(bucket: list[Any], callback: Any) -> Callable[[], None]:
        bucket.append(callback)

        def unsubscribe() -> None:
            if callback in bucket:
                bucket.remove(callback)

        return unsubscribe

    # -- sending -------------------------------------------------------------

    def send_text(self, data: str) -> None:
        session = self._live_session()
        with self._lock:
            session.send_text(data)
        self._frame_hook("out", "text", len(data.encode("utf-8")))

    def send_bytes(self, data: bytes) -> None:
        session = self._live_session()
        with self._lock:
            session.send_bytes(data)
        self._frame_hook("out", "binary", len(data))

    def _live_session(self) -> WsSession:
        session = self._session
        if session is None or self._closed:
            raise ConnectionError("realtime: the socket is not open")
        return session

    # -- the reader ----------------------------------------------------------

    def _run(self) -> None:
        """Own the session for its whole life, and pump frames off it."""
        try:
            with self._connect(
                self.request.url,
                headers=dict(self.request.headers or {}),
                subprotocols=list(self.request.protocols) if self.request.protocols else None,
                timeout=self._timeout,
            ) as session:
                self._session = session
                self._opened.set()
                self._emit_hook("onRealtimeOpen", {"url": self.request.url})
                for callback in list(self._open_cbs):
                    callback()
                self._pump(session)
        except BaseException as exc:  # noqa: BLE001 -- handed to whoever is waiting:
            # `open()` before the socket is up, `on_error` after.
            self._open_error = exc
            if self._opened.is_set():
                self._report(exc)
        finally:
            self._session = None
            # Always set: an `open()` waiting on a socket that failed must be
            # woken, or it waits out the whole timeout for an answer that
            # already exists.
            self._opened.set()
            self._emit_hook("onRealtimeClose", {"code": None, "reason": None})
            for callback in list(self._close_cbs):
                callback()

    def _pump(self, session: WsSession) -> None:
        while not self._closed:
            try:
                with self._lock:
                    event = session.receive(timeout=POLL_SECONDS)
            except TimeoutError:
                # Nothing waiting. Also where the lock is released so a sender
                # can get in -- the other half of the short poll.
                continue
            except Exception as exc:  # noqa: BLE001 -- a closed socket raises here,
                # and so does a network fault; both end the session.
                if not self._closed:
                    self._report(exc)
                return
            frame = frame_of(event)
            if frame is None:
                continue
            size = len(frame.text.encode("utf-8")) if frame.text is not None else len(
                frame.binary or b""
            )
            self._frame_hook("in", "text" if frame.text is not None else "binary", size)
            for callback in list(self._frame_cbs):
                callback(frame)

    def _report(self, exc: BaseException) -> None:
        self._emit_hook("onRealtimeError", {"error": exc})
        for callback in list(self._error_cbs):
            callback(exc)

    # -- hooks ---------------------------------------------------------------

    def _frame_hook(self, direction: str, kind: str, size: int) -> None:
        self._emit_hook(
            "onRealtimeFrame", {"direction": direction, "kind": kind, "bytes": size}
        )

    def _emit_hook(self, name: str, extra: Mapping[str, Any]) -> None:
        if self._hooks is None:
            return
        ctx: dict[str, Any] = {
            "provider": self.request.provider,
            "model": self.request.model,
        }
        ctx.update(extra)
        # emit_sync, never emit: this runs on the reader thread, which has no
        # event loop to await on, and a frame hook must not hold up the pump.
        self._hooks.emit_sync(name, ctx)

    def __repr__(self) -> str:
        return f"<RealtimeConnection {self.request.url} open={self.is_open}>"


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "POLL_SECONDS",
    "RealtimeConnection",
    "default_connect",
    "frame_of",
]
