"""Streamable HTTP: an MCP server that is a URL rather than a process.

Every call is a POST through the engine's fetch, so it rides the same queue,
retry policy and telemetry as everything else this library sends. Never a
side-fetch -- a transport that opened its own connection would be invisible to
the rate limiter that exists to keep the whole fleet under a provider's ceiling.

Three things make this more than "POST some JSON":

- **A response is one of two media types.** `application/json` carries a single
  JSON-RPC message; `text/event-stream` carries a batch of them, which is how a
  server answers a request that produced notifications on the way. Both have to
  be read, and the answer picked out BY ID -- a server may interleave.
- **A 4xx can still carry a JSON-RPC body, and for negotiation that body is the
  whole point.** `-32022` is what names the versions a server speaks. Collapsing
  every 4xx into "connection closed" throws that away and makes a modern-only
  server look like a dead socket.
- **The server->client channel is a separate GET**, held open, resumed with
  `Last-Event-ID` after a drop. Without the cursor a reconnect silently starts
  from now and the caller never learns what it missed.

`subscriptions/listen` is a fourth thing again: a POST whose RESPONSE BODY is
the stream, held open for the life of the subscription. It cannot go through the
buffered path, which would surface every frame at once after the subscription
had already ended.

Transposed from `unified-library-ts/src/plugins/mcp/transport-http.ts`.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from typing import Any

from ..llm.wire_transforms import make_registry
from ..network.errors import LLMError
from ..wire.interpreter import build_from_spec
from .errors import McpError, McpErrorCode
from .transport import BaseJsonRpcTransport, OnStreamEnd, OnStreamOpen
from .wire_rules import mcp_spec, mcp_wire_registry

DEFAULT_TIMEOUT_SECONDS = 60.0

#: How many times a dropped event stream is resumed before the caller is simply
#: told it ended. Bounded on purpose: retrying forever against a server that is
#: gone is indistinguishable from a hang.
MAX_RESUME_ATTEMPTS = 5
RESUME_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 30.0

_FRAME = re.compile(r"\r?\n\r?\n")
_LINE = re.compile(r"\r?\n")

_REGISTRY = mcp_wire_registry(make_registry({}))


def media_type_essence(content_type: str) -> str:
    """`type/subtype`, lowercased, parameters stripped.

    Compared rather than substring-tested: a substring check misroutes anything
    that merely CONTAINS the token -- `application/json` with a vendor parameter
    that mentions it -- while still having to cope with the ordinary
    `text/event-stream; charset=utf-8`.
    """
    return (content_type.split(";", 1)[0] or "").strip().lower()


def sse_messages(text: str) -> list[dict[str, Any]]:
    """Every JSON-RPC message in an SSE body.

    A frame's `data:` lines are joined, because a message split across several
    of them is one message -- reading only the first silently truncates it.
    """
    out: list[dict[str, Any]] = []
    for frame in _FRAME.split(text):
        data = "\n".join(
            line[5:].strip() for line in _LINE.split(frame) if line.startswith("data:")
        )
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue  # a comment or keep-alive frame
        if isinstance(parsed, Mapping):
            out.append(dict(parsed))
    return out


def pick_response(content_type: str, text: str, request_id: int) -> dict[str, Any] | None:
    """The JSON-RPC message answering `request_id`, from either media type."""
    messages: list[dict[str, Any]]
    if media_type_essence(content_type) == "text/event-stream":
        messages = sse_messages(text)
    else:
        trimmed = text.strip()
        if not trimmed:
            return None
        try:
            parsed = json.loads(trimmed)
        except ValueError:
            return None
        if isinstance(parsed, list):
            messages = [dict(m) for m in parsed if isinstance(m, Mapping)]
        elif isinstance(parsed, Mapping):
            messages = [dict(parsed)]
        else:
            return None
    for message in messages:
        if message.get("id") == request_id:
            return message
    return None


class HttpTransport(BaseJsonRpcTransport):
    """One MCP server, reached over Streamable HTTP."""

    def __init__(
        self,
        url: str,
        *,
        fetch: Any,
        fetch_stream: Any = None,
        headers: Mapping[str, str] | None = None,
        name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        queue_name: str | None = None,
        auth_headers: Callable[[], Mapping[str, str]] | None = None,
        on_unauthorized: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__()
        self._url = url
        self._fetch = fetch
        self._fetch_stream = fetch_stream
        self._headers = dict(headers or {})
        self._name = name or "server"
        self._timeout = timeout
        self._queue_name = queue_name
        self._auth_headers = auth_headers
        self._on_unauthorized = on_unauthorized

        self._session_id: str | None = None
        self._protocol_version: str | None = None
        #: Handshake until negotiation says otherwise: an un-negotiated
        #: connection must behave exactly as it did before 2026 support existed.
        self._era = "handshake"
        self._last_event_id: str | None = None
        self._listening = threading.Event()
        self._events: threading.Thread | None = None
        #: One stop flag per open `subscriptions/listen` stream. A closing
        #: transport has to reach INTO those threads: they are blocked reading a
        #: socket the server is deliberately holding open, so nothing else would
        #: ever wake them.
        self._streams: dict[int, threading.Event] = {}
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Nothing to open: the session is established by `initialize` itself."""

    def set_protocol_version(self, version: str) -> None:
        self._protocol_version = version

    def set_era(self, era: str) -> None:
        self._era = era

    @property
    def session_id(self) -> str | None:
        """The server's session id, once it has issued one."""
        return self._session_id

    def listen(self) -> None:
        """Open the server->client GET stream, in the background.

        Best-effort by design: a server that answers 405 is request/response
        only, which is not an error and must not fail the connection.
        """
        if self._events is not None or self._fetch_stream is None:
            return
        self._closed = False
        self._events = threading.Thread(target=self._event_loop, name="mcp-http", daemon=True)
        self._events.start()

    def close(self) -> None:
        self._closed = True
        self._listening.set()
        for stop in list(self._streams.values()):
            stop.set()
        self._fail_all(McpError("MCP transport closed", code=McpErrorCode.CONNECTION_CLOSED))
        if self._session_id is None:
            return
        try:
            self._fetch(self._build("mcp/http.close", {}, response_type="text"))
        except Exception:  # noqa: BLE001, S110 -- a server that does not support
            # session termination answers 405, and a caller shutting down cannot
            # act on that. Raising here would turn a tidy close into a failure.
            pass
        self._session_id = None

    # -- sending -------------------------------------------------------------

    def request(self, method: str, params: Any = None) -> Any:
        request_id = self._allocate_id()
        call: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            call["params"] = params

        status, headers, text = self._post("mcp/http.request", call)

        # A 401 is answered once, after the caller has had a chance to re-auth.
        if status == 401 and self._on_unauthorized is not None and self._on_unauthorized():
            status, headers, text = self._post("mcp/http.request", call)

        if status >= 400:
            # A 4xx MAY still carry a JSON-RPC error, and for negotiation it is
            # the whole point -- see the module docstring. Matched by id when
            # the server echoed ours, and otherwise taken as the only message
            # in the body: a server rejecting the request outright answers with
            # its own id (DeepWiki says `"server-error"`), and refusing to read
            # that would throw away the one thing it told us.
            body = pick_response(headers.get("content-type", ""), text, request_id) or (
                _sole_error(headers.get("content-type", ""), text)
            )
            error = body.get("error") if body else None
            if isinstance(error, Mapping):
                raise McpError.of(error)
            raise McpError(
                f"MCP HTTP {status} for {method!r}", code=McpErrorCode.CONNECTION_CLOSED
            )

        message = pick_response(headers.get("content-type", ""), text, request_id)
        if message is None:
            raise McpError(
                f"MCP: no JSON-RPC response for {method!r}", code=McpErrorCode.INTERNAL_ERROR
            )
        error = message.get("error")
        if isinstance(error, Mapping):
            raise McpError.of(error)
        return message.get("result")

    def notify(self, method: str, params: Any = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        self._post("mcp/http.notify", payload)

    def send_long_lived_request(
        self,
        method: str,
        params: Any = None,
        on_end: OnStreamEnd | None = None,
        on_open: OnStreamOpen | None = None,
    ) -> int:
        """Open a request whose RESPONSE BODY is the event stream.

        This is how `subscriptions/listen` works on Streamable HTTP. Unlike an
        ordinary call, the POST does not answer with a single JSON-RPC message:
        the server holds the response open and writes notification frames as
        they occur, closing it only when the subscription ends. So it goes
        through the STREAMING fetch rather than the buffered `_post`, which
        would surface every frame at once after the stream had already closed.

        Frames are routed exactly like the GET channel's, so the client cannot
        tell the two apart. The eventual JSON-RPC response has no pending entry
        to settle -- its only meaning is "the stream ended", which is what
        `on_end` reports.
        """
        if self._fetch_stream is None:
            raise McpError(
                "MCP subscriptions/listen needs a streaming fetch; this transport was "
                "built without one, so the response would only arrive after the "
                "subscription had already ended.",
                code=McpErrorCode.CONNECTION_CLOSED,
            )
        request_id = self._allocate_id()
        with self._lock:
            self._long_lived[request_id] = on_end
        # Before the thread starts, so a server that acknowledges instantly
        # cannot deliver a frame for a subscription nobody has registered yet.
        if on_open is not None:
            on_open(request_id)
        stop = threading.Event()
        self._streams[request_id] = stop
        thread = threading.Thread(
            target=self._long_lived_loop,
            args=(request_id, method, params, stop),
            name=f"mcp-listen-{request_id}",
            daemon=True,
        )
        thread.start()
        return request_id

    def _long_lived_loop(
        self, request_id: int, method: str, params: Any, stop: threading.Event
    ) -> None:
        """Drain one long-lived stream, resuming it while the server allows.

        The caller already has the id and is receiving frames; this thread only
        exists so that opening the stream does not block until the subscription
        ENDS, which is the one thing a healthy subscription never does.
        """
        failure: BaseException | None = None
        #: The last id seen on THIS stream -- its resumption cursor. Kept apart
        #: from `_last_event_id`, which the standalone GET channel also writes:
        #: sharing one would make each stream resume from the other's position.
        cursor: str | None = None
        try:
            payload: dict[str, Any] = {"id": request_id, "method": method}
            if params is not None:
                payload["params"] = params
            cursor = self._drain(
                self._fetch_stream(self._build("mcp/http.longLived", payload, "stream")),
                stop,
                cursor,
            )[1]

            # The stream ended without the subscription being torn down. Given
            # event ids we can ask the server to replay what we missed, exactly
            # as the reference client does. Without this a single blip kills a
            # subscription permanently -- and on the 2026-07-28 wire this is the
            # ONLY notification channel, so "permanently" means the caller stops
            # seeing changes and is never told why.
            attempt = 0
            while (
                cursor
                and not stop.is_set()
                and self._stream_is_open(request_id)
                and attempt < MAX_RESUME_ATTEMPTS
            ):
                if stop.wait(min(RESUME_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)):
                    break
                try:
                    delivered, cursor = self._drain(
                        self._fetch_stream(
                            self._build("mcp/http.events", {"lastEventId": cursor}, "stream")
                        ),
                        stop,
                        cursor,
                    )
                except Exception as exc:  # noqa: BLE001 -- reported through
                    # `on_end` once the budget runs out; a reconnect that throws
                    # is a failed attempt, not a reason to stop trying.
                    failure = exc
                    attempt += 1
                    continue
                # A reconnect that delivered frames earns a fresh budget; one
                # that opened and died immediately counts against it, so a
                # flapping server still terminates.
                attempt = 0 if delivered else attempt + 1
        except Exception as exc:  # noqa: BLE001 -- a rejected subscription (4xx)
            # or a dropped connection. Reported rather than swallowed: a caller
            # holding a subscription that silently stopped delivering has no way
            # to find out.
            if not stop.is_set():
                failure = exc
        finally:
            self._streams.pop(request_id, None)
            # Always settles, whether we resumed and later gave up or never
            # could. `_resolve_long_lived` is a no-op when the server's own
            # response already ended it, so `on_end` fires exactly once.
            self._resolve_long_lived(request_id, failure)

    def _stream_is_open(self, request_id: int) -> bool:
        """Whether this stream is still unsettled.

        Read before resuming: once the server's JSON-RPC response has arrived
        the subscription is genuinely over, and reconnecting would reopen a
        channel for a subscription that no longer exists.
        """
        with self._lock:
            return request_id in self._long_lived

    def _drain(
        self, stream: Any, stop: threading.Event, cursor: str | None
    ) -> tuple[bool, str | None]:
        """Route every frame off one stream. Whether it delivered, and where it reached."""
        delivered = False
        for event in stream:
            if stop.is_set() or self._closed:
                break
            event_id = _attr(event, "id")
            if event_id:
                self._last_event_id = str(event_id)
                cursor = str(event_id)
            data = _attr(event, "data")
            if not data:
                continue
            delivered = True
            try:
                message = json.loads(data)
            except ValueError:
                # A comment or keep-alive frame. Not JSON and not a failure.
                continue
            if isinstance(message, Mapping):
                self._route_incoming(message)
        return delivered, cursor

    def _send_message(self, message: Mapping[str, Any]) -> None:
        """A reply to a server-initiated request, posted back."""
        self._post("mcp/http.message", {"message": dict(message)})

    # -- the wire ------------------------------------------------------------

    def _build(
        self, spec_id: str, payload: Mapping[str, Any], response_type: str = "text"
    ) -> dict[str, Any]:
        """One MCP request, built from its spec.

        Everything that varies -- the negotiated era, the session, the declared
        version, a resolved bearer, the resumption cursor -- is passed IN, so the
        SPEC decides which headers those facts produce rather than this method
        deciding and the spec recording the result.
        """
        auth = dict(self._auth_headers()) if self._auth_headers is not None else None
        built = build_from_spec(
            mcp_spec(spec_id),
            {
                **dict(payload),
                "era": self._era,
                "sessionId": self._session_id,
                "protocolVersion": self._protocol_version,
                "authHeaders": auth,
            },
            _REGISTRY,
            "mcp",
            None,
            {"url": self._url, "headers": self._headers},
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            # Not part of the wire: these route and queue the call inside the
            # network engine.
            "provider": "mcp",
            "model": self._name,
            "responseType": response_type,
            # SECONDS on this library's public surface, MILLISECONDS on the
            # wire. Passing the seconds straight through made `timeout=120` mean
            # 120ms, so every request to a real server timed out before it had
            # finished connecting.
            "timeout": self._timeout * 1000,
        }
        if not getattr(built, "no_body", False):
            request["body"] = built.body
        return request

    def _post(self, spec_id: str, payload: Mapping[str, Any]) -> tuple[int, dict[str, str], str]:
        try:
            response = self._fetch(self._build(spec_id, payload, response_type="text"))
        except LLMError as exc:
            # The executor classifies a 4xx as a failure, which is right for a
            # completion and wrong here: an MCP 4xx MAY carry a JSON-RPC body,
            # and during negotiation that body is the entire answer. DeepWiki
            # answers the 2026 probe with `400 Unsupported protocol version:
            # ... Supported versions: ...`, and reading it is what lets the
            # handshake fall back instead of the connection failing.
            #
            # Anything with no status at all is a real transport failure -- a
            # dead socket, a DNS failure -- and is re-raised untouched.
            if exc.status is None:
                raise
            return exc.status, _headers_of(exc.raw), _text_of(exc.raw, exc.message)
        status, headers, text = _read(response)
        # A server issues its session id on the first answer and expects it back
        # on every later request; missing it turns turn two into a new session.
        session = headers.get("mcp-session-id")
        if session:
            self._session_id = session
        return status, headers, text

    # -- the server->client stream -------------------------------------------

    def _event_loop(self) -> None:
        """Hold the GET stream open, resuming after a drop."""
        attempt = 0
        while not self._closed:
            opened = self._run_event_stream()
            if self._closed:
                break
            # Never opened on the FIRST try means the server has no GET channel
            # (405), which is a valid server and not a failure to retry.
            if not opened and attempt == 0:
                break
            attempt = 0 if opened else attempt + 1
            if attempt > MAX_RESUME_ATTEMPTS:
                break
            backoff = min(RESUME_BACKOFF_SECONDS * (2 ** max(0, attempt - 1)), MAX_BACKOFF_SECONDS)
            if self._listening.wait(backoff):
                break

    def _run_event_stream(self) -> bool:
        """One GET session. Whether it opened at all."""
        opened = False
        try:
            stream = self._fetch_stream(
                self._build(
                    "mcp/http.events",
                    {"lastEventId": self._last_event_id},
                    response_type="stream",
                )
            )
            for event in stream:
                opened = True
                if self._closed:
                    break
                event_id = _attr(event, "id")
                if event_id:
                    # The resumption cursor. Without it a reconnect starts from
                    # now and the caller never learns what it missed.
                    self._last_event_id = str(event_id)
                data = _attr(event, "data")
                if not data:
                    continue
                try:
                    message = json.loads(data)
                except ValueError:
                    continue
                if isinstance(message, Mapping):
                    self._route_incoming(message)
        except Exception:  # noqa: BLE001, S110 -- a 405, an abort or a dropped
            # socket. Which of those it was is exactly what `opened` records, and
            # the reconnect loop above is what decides between them.
            pass
        return opened

    def __repr__(self) -> str:
        return f"<HttpTransport {self._url} session={self._session_id}>"


def _sole_error(content_type: str, text: str) -> dict[str, Any] | None:
    """The single JSON-RPC error in a body, whatever id it carries."""
    if media_type_essence(content_type) == "text/event-stream":
        messages = sse_messages(text)
    else:
        try:
            parsed = json.loads(text.strip())
        except ValueError:
            return None
        messages = [dict(parsed)] if isinstance(parsed, Mapping) else []
    errors = [m for m in messages if isinstance(m.get("error"), Mapping)]
    return errors[0] if len(errors) == 1 else None


def _headers_of(raw: Any) -> dict[str, str]:
    headers = raw.get("headers") if isinstance(raw, Mapping) else None
    if not isinstance(headers, Mapping):
        return {}
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _text_of(raw: Any, fallback: str) -> str:
    """The body the executor was holding, or the message it built from it."""
    body = raw.get("body") if isinstance(raw, Mapping) else None
    if isinstance(body, str):
        return body
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    if body is not None:
        return json.dumps(body)
    # `classify_error` puts the body text into the message, so even a raw the
    # executor did not keep still leaves the JSON-RPC error readable.
    return fallback


def _attr(event: Any, name: str) -> Any:
    """One field of an SSE event, however the engine shaped it."""
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def _read(response: Any) -> tuple[int, dict[str, str], str]:
    """Status, lower-cased headers and body text, from whatever the fetch answered."""
    if isinstance(response, Mapping):
        status = response.get("status")
        headers = response.get("headers")
        body = response.get("body")
    else:
        status = getattr(response, "status", None)
        headers = getattr(response, "headers", None)
        body = getattr(response, "body", None)
    lowered = (
        {str(k).lower(): str(v) for k, v in dict(headers).items()}
        if isinstance(headers, Mapping)
        else {}
    )
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", "replace")
    elif isinstance(body, str):
        text = body
    elif body is None:
        text = ""
    else:
        # A fetch that already parsed the JSON for us. Re-serialised rather than
        # refused: which one happens depends on the engine's response type, and
        # the caller did not choose it.
        text = json.dumps(body)
    return (int(status) if isinstance(status, int) else 0, lowered, text)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RESUME_ATTEMPTS",
    "HttpTransport",
    "media_type_essence",
    "pick_response",
    "sse_messages",
]
