"""One session over two protocols, delivered as an iterator.

**Diverges from the TypeScript on purpose**, and the API contract asks for it:
there a caller subscribes with `session.on('text', cb)`, here they write

    for event in session:
        if event.type == "text":
            ...

which is the loop `LLM.stream()` already asks for. A realtime session is a
different transport, not a different idea, so it should not be a different
shape. The events are the same frozen dataclasses.

Readiness is not the same as the socket being open. OpenAI accepts content the
moment the socket connects; Gemini must first answer `setupComplete`. So the
base class does NOT emit `open` when the socket does -- the subclass says when,
and a `send()` before then is buffered rather than lost.

Transposed from `unified-library-ts/src/llm/realtime/session.ts`.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any, Self

from ..events import Event, SessionOpenEvent
from .types import Connection, Frame, SessionConfig, Turn

#: Sentinel pushed onto the queue when the socket closes, so an iterator
#: blocked on `get()` stops rather than waiting for an event that cannot come.
_END = object()


class BaseSession:
    """The half both providers share: readiness, buffering, and iteration."""

    def __init__(self, connection: Connection, config: SessionConfig) -> None:
        self.config = config
        self._conn = connection
        self._events: queue.Queue[Any] = queue.Queue()
        self._ready = threading.Event()
        #: Sends made before the session was ready, in order.
        self._outbox: list[Callable[[], None]] = []
        self._outbox_lock = threading.Lock()
        self._closed = False

        connection.on_frame(self._on_frame)
        on_open = getattr(connection, "on_open", None)
        if on_open is not None:
            on_open(self._on_open)
        # A socket error is an event a caller can see, not an exception thrown
        # on a thread they do not own -- where it would be unhandleable.
        on_error = getattr(connection, "on_error", None)
        if on_error is not None:
            on_error(lambda exc: self.emit(self._error_event(exc)))
        on_close = getattr(connection, "on_close", None)
        if on_close is not None:
            on_close(self._on_close)

    # -- what a subclass provides --------------------------------------------

    def on_open(self) -> None:
        """The provider handshake, sent once the socket opens."""
        raise NotImplementedError

    def on_frame(self, frame: Frame) -> None:
        """Map one inbound frame onto zero or more events."""
        raise NotImplementedError

    def build_turn_frames(self, turn: Turn, turn_complete: bool) -> list[Any]:
        """The frames for one input turn."""
        raise NotImplementedError

    # -- the caller's side ---------------------------------------------------

    def send(
        self,
        text: str | None = None,
        *,
        audio: bytes | None = None,
        turn_complete: bool = True,
    ) -> None:
        """Send a turn.

        `turn_complete=False` leaves the turn open, so several sends make one
        turn -- which is how audio is streamed in. The default asks for a reply
        now, because that is what a caller writing one line means.
        """
        if text is None and audio is None:
            raise ValueError("send: give it text, audio, or both")
        turn = Turn(text=text, audio=audio)

        def dispatch() -> None:
            for frame in self.build_turn_frames(turn, turn_complete):
                self.send_json(frame)

        self._when_ready(dispatch)

    def __iter__(self) -> Iterator[Event]:
        """Events, in arrival order, until the socket closes.

        Ends when the CONNECTION does, not when the model stops talking: a turn
        ending is an event (`turn_end`), and a caller who wants one turn breaks
        on it.
        """
        while True:
            item = self._events.get()
            if item is _END:
                return
            yield item

    def close(self) -> None:
        self._closed = True
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the machinery -------------------------------------------------------

    def emit(self, event: Event | None) -> None:
        if event is not None:
            self._events.put(event)

    def send_json(self, payload: Any) -> None:
        """Both providers speak JSON; only the frame type differs."""
        self._conn.send_text(json.dumps(payload, separators=(",", ":")))

    def mark_ready(self) -> None:
        """Flush anything buffered and let sends through. Idempotent.

        Called by the subclass when its handshake is done -- for OpenAI as soon
        as the socket opens, for Gemini not until `setupComplete` arrives.
        """
        if self._ready.is_set():
            return
        self._ready.set()
        with self._outbox_lock:
            pending, self._outbox = self._outbox, []
        for run in pending:
            run()
        self.emit(SessionOpenEvent(type="open"))

    def _when_ready(self, run: Callable[[], Any]) -> None:
        """Run a send now if the session is ready, else hold it until it is.

        Without this a caller would have to wait for an event before their first
        `send()`, and the natural thing to write -- open, send, read -- would
        drop the first turn on whichever provider is slower to handshake.
        """
        if self._ready.is_set():
            run()
            return
        with self._outbox_lock:
            if self._ready.is_set():
                run()
                return
            self._outbox.append(lambda: run())

    def _on_open(self) -> None:
        try:
            self.on_open()
        except Exception as exc:  # noqa: BLE001 -- the reader thread again:
            # a failed handshake becomes an event rather than a dead thread.
            self.emit(self._error_event(exc))

    def _on_frame(self, frame: Frame) -> None:
        try:
            self.on_frame(frame)
        except Exception as exc:  # noqa: BLE001 -- this is the reader thread; an
            # exception here would kill the pump silently. It becomes an event.
            self.emit(self._error_event(exc))

    def _on_close(self) -> None:
        self._events.put(_END)

    @staticmethod
    def _error_event(exc: BaseException) -> Event:
        from ..events import RealtimeErrorEvent

        return RealtimeErrorEvent(type="error", message=str(exc))

    @staticmethod
    def json_of(frame: Frame) -> Any | None:
        """The frame's JSON, whichever kind of frame carried it.

        A frame that is not JSON is dropped rather than fatal: one bad frame is
        not a dead session, and both providers send keepalives this will not
        parse.
        """
        raw = frame.text if frame.text is not None else (frame.binary or b"").decode(
            "utf-8", "replace"
        )
        try:
            return json.loads(raw)
        except ValueError:
            return None


__all__ = ["BaseSession"]
