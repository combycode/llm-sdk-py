"""The only file here that knows what a socket is.

Bytes in, `HttpRequest` out; `HttpResponse` in, bytes out. Nothing else. Every
decision worth testing -- routing, auth, error mapping, token accounting --
lives in `app.py` and is reachable by calling a function, and what is left here
is the part that genuinely needs a port to exercise.

Two shapes, because deployments differ: `serve()` for `python -m` and a local
port, and `wsgi_app()` for putting the same handler behind gunicorn, uwsgi or
anything else that speaks WSGI. Both are thin, and both go through the same
`handle()`.

Stdlib only. A framework here would be a dependency bought for two functions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .app import OaiServer
from .types import HttpRequest, HttpResponse

#: Refused before it is read. A body this large is a mistake or an attack, and
#: either way reading it first is how the process runs out of memory.
MAX_BODY_BYTES = 8 * 1024 * 1024


def _is_frame_stream(body: Any) -> bool:
    """Is this body a lazy sequence of already-formatted frames?

    A generator or iterator, but never a str/bytes/mapping/list -- those are
    ordinary bodies that happen to be iterable, and treating one as a stream
    would serialise a dict as its keys.
    """
    if isinstance(body, (str, bytes, bytearray, Mapping, list, tuple)) or body is None:
        return False
    return hasattr(body, "__iter__") or hasattr(body, "__next__")


def parse_body(raw: bytes) -> Any:
    """The body as a value.

    Non-JSON is passed through as text rather than rejected here: what counts as
    a valid body is the route's business, and this layer refusing first would
    turn a clear "messages must be an array" into a bare parse error.
    """
    if not raw:
        return None
    text = raw.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def request_from(
    method: str, target: str, headers: Mapping[str, str], raw_body: bytes
) -> HttpRequest:
    """One wire request as the value `handle()` takes."""
    split = urlsplit(target)
    return HttpRequest(
        method=method.upper(),
        path=split.path or "/",
        headers=dict(headers),
        body=parse_body(raw_body),
        query={k: v[0] for k, v in parse_qs(split.query).items()},
    )


def encode(response: HttpResponse) -> tuple[int, dict[str, str], bytes]:
    """One response as a status, headers and bytes."""
    if _is_frame_stream(response.body):
        # Drained here, because `encode` promises a complete response with a
        # content-length. `write_stream` is the path that does not: it writes
        # each frame as it comes, which is what a socket client wants and what
        # WSGI cannot express without chunked transfer.
        payload = b"".join(_as_bytes(frame) for frame in response.body)
    elif response.body is None:
        payload = b""
    elif isinstance(response.body, (bytes, bytearray)):
        payload = bytes(response.body)
    elif isinstance(response.body, str):
        payload = response.body.encode("utf-8")
    else:
        payload = json.dumps(response.body).encode("utf-8")
    headers = {**dict(response.headers), "content-length": str(len(payload))}
    return response.status, headers, payload


def write_stream(response: HttpResponse, write: Callable[[bytes], Any]) -> None:
    """Write a frame-stream body one frame at a time.

    The whole point of the streaming path: a client sees the first token before
    the last one exists. Draining into one buffer and writing that would satisfy
    every test and defeat the feature, which is why this is separate from
    `encode` rather than a flag on it.

    No content-length is sent -- the length is not known when the headers go
    out, and that is precisely what makes it a stream. The connection closes to
    mark the end, and `[DONE]` marks it for the client.
    """
    for frame in response.body:
        write(_as_bytes(frame))


def _as_bytes(frame: Any) -> bytes:
    return frame.encode("utf-8") if isinstance(frame, str) else bytes(frame)


def wsgi_app(server: OaiServer) -> Callable[[dict[str, Any], Any], Iterable[bytes]]:
    """The server as a WSGI application."""

    def application(environ: dict[str, Any], start_response: Any) -> Iterable[bytes]:
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(min(length, MAX_BODY_BYTES)) if length else b""
        headers = {
            key[5:].replace("_", "-").lower(): str(value)
            for key, value in environ.items()
            if key.startswith("HTTP_")
        }
        if environ.get("CONTENT_TYPE"):
            headers["content-type"] = str(environ["CONTENT_TYPE"])
        target = environ.get("PATH_INFO", "/")
        if environ.get("QUERY_STRING"):
            target = f"{target}?{environ['QUERY_STRING']}"
        response = server.handle(
            request_from(environ.get("REQUEST_METHOD", "GET"), target, headers, raw)
        )
        if _is_frame_stream(response.body):
            # Handed over as an ITERABLE, which is what WSGI is built for: the
            # server writes each frame as the generator yields it, and no
            # content-length is sent because none is known yet.
            start_response(f"{response.status} ", list(dict(response.headers).items()))
            return (_as_bytes(frame) for frame in response.body)
        status, out_headers, payload = encode(response)
        start_response(f"{status} ", list(out_headers.items()))
        return [payload]

    return application


class _Handler(BaseHTTPRequestHandler):
    """Adapts one stdlib request. Holds no logic of its own."""

    server_version = "combycode-llm-sdk"
    #: Set by `serve()`.
    oai: OaiServer

    def _respond(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_BODY_BYTES:
            self._write(
                HttpResponse(
                    status=413,
                    body={"error": {"message": "request body too large", "type": "invalid_request_error"}},
                    headers={"content-type": "application/json"},
                )
            )
            return
        raw = self.rfile.read(length) if length else b""
        request = request_from(self.command, self.path, dict(self.headers), raw)
        self._write(self.oai.handle(request))

    def _write(self, response: HttpResponse) -> None:
        if _is_frame_stream(response.body):
            self._write_stream(response)
            return
        status, headers, payload = encode(response)
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _write_stream(self, response: HttpResponse) -> None:
        """Headers first, then a flush after every frame.

        The flush is the feature. Without it the frames sit in the socket's
        buffer and arrive together, which passes every test that reads the whole
        body and delivers nothing a user would call streaming.

        No content-length: it is not known when the headers go out. HTTP/1.1
        would want chunked transfer for that, so the protocol is pinned to 1.0
        for this response and the CLOSED CONNECTION delimits the body -- which
        is what an SSE client expects anyway, since it reads until the stream
        ends.
        """
        self.protocol_version = "HTTP/1.0"
        # Said explicitly as well as implied by the version: this is what
        # delimits the body, and a keep-alive here would hang every reader.
        self.close_connection = True
        self.send_response(response.status)
        for key, value in dict(response.headers).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            write_stream(response, self.wfile.write)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up mid-stream. Ordinary for a stream a reader
            # stopped reading, and not something to log as a server fault.
            return

    do_GET = _respond
    do_POST = _respond
    do_OPTIONS = _respond
    do_DELETE = _respond

    def log_message(self, format: str, *args: Any) -> None:
        """Silent by default: the hook bus is where a server's events belong."""


def make_http_server(
    server: OaiServer, host: str = "127.0.0.1", port: int = 4000
) -> ThreadingHTTPServer:
    """A bound stdlib server, not yet serving.

    Returned rather than started, so the caller owns the loop and the shutdown.
    A helper that started a thread here would be a helper that leaks one.
    """
    handler = type("_BoundHandler", (_Handler,), {"oai": server})
    return ThreadingHTTPServer((host, port), handler)


def serve(server: OaiServer, host: str = "127.0.0.1", port: int = 4000) -> None:
    """Bind and serve until interrupted."""
    httpd = make_http_server(server, host, port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


__all__ = [
    "MAX_BODY_BYTES",
    "encode",
    "make_http_server",
    "parse_body",
    "request_from",
    "serve",
    "wsgi_app",
]
