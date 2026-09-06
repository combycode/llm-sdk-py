"""A request and a response, with the socket already taken out of them.

`HttpRequest` is what the handler takes and `HttpResponse` is what it returns,
and neither knows how bytes get in or out. That is the decision this whole
subsystem rests on: `OaiServer.handle` is a pure function, so routing, auth and
error mapping can be exercised by calling it -- no port, no framework, no
lifecycle. A server whose auth could only be observed through a listening socket
is one where auth is never tested.

`http_shell.py` is the other half: it turns bytes into one of these and one of
these back into bytes, and it is the only file in the package that knows what a
socket is.

Transposed from `unified-library-ts/src/server/oai-types.ts`, whose handler
takes a `Request` and parses inside; splitting the parse out is what makes the
Python side callable without a runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HttpRequest:
    """One parsed request.

    `headers` are lower-cased on the way in, because HTTP header names are
    case-insensitive and every caller that looks one up would otherwise have to
    remember that.
    """

    method: str
    path: str
    headers: Mapping[str, str] = field(default_factory=dict)
    #: The parsed JSON body, or None. Parsed, not raw: the shell owns decoding.
    body: Any = None
    query: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "headers", {str(k).lower(): str(v) for k, v in dict(self.headers).items()}
        )

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


@dataclass(frozen=True)
class HttpResponse:
    """One response, as a status and a value that is still a value.

    `body` is the object, not its serialisation. The example reads
    `response.body["choices"][0]` and a shell serialises it once at the edge --
    encoding here would force every reader to decode it back.
    """

    status: int
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        """The body, under the name a caller reaches for out of habit."""
        return self.body


def cors_headers() -> dict[str, str]:
    """Permissive by design: this is a local shim in front of your own models."""
    return {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET, POST, OPTIONS, DELETE",
        "access-control-allow-headers": "authorization, content-type",
    }


def json_response(body: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status, body=body, headers={"content-type": "application/json", **cors_headers()}
    )


def sse_response(frames: Any, status: int = 200) -> HttpResponse:
    """A response whose body is an ITERATOR of already-formatted SSE frames.

    The body stays a value, as every other response's does -- it is simply a
    lazy one, so the shell can write each frame as it comes instead of holding
    the whole stream in memory. `handle()` remains a pure function of the
    request; nothing here knows about a socket.

    `cache-control: no-cache` is not optional politeness: a proxy that buffers
    an event stream turns it back into one blob, which is the failure this whole
    path exists to avoid.

    There is deliberately NO `connection: keep-alive`. The body has no
    content-length -- it cannot, the length is unknown when the headers go out
    -- so the CLOSED CONNECTION is what delimits it. Announcing keep-alive
    while planning to close is a lie to the client and, in the stdlib server,
    to the handler: `send_header('Connection', 'keep-alive')` sets
    `close_connection = False`, the socket stays open, and a client reading to
    the end of the stream waits forever on a stream that already ended. Found by
    a test that read the socket rather than the body.
    """
    return HttpResponse(
        status=status,
        body=frames,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            **cors_headers(),
        },
    )


__all__ = [
    "HttpRequest",
    "HttpResponse",
    "cors_headers",
    "json_response",
    "sse_response",
]
