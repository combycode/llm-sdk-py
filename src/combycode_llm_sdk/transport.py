"""The HTTP boundary: what a request looks like on its way out, and how to swap it.

Python-native. TypeScript injects `globalThis.fetch` and needs no such type; the
Python API contract (`examples/README.md`) instead makes the transport a plain
callable the caller can replace, and ten examples exercise the library with no
network at all by passing one::

    def stub(request):
        return TransportResponse(status=200, body=RECORDED)

    LLM(model="openai/gpt-5.4-nano", api_key="k", transport=stub)

A transport is therefore just `(TransportRequest) -> TransportResponse`. Its
async twin is `(TransportRequest) -> Awaitable[TransportResponse]`. Both
dataclasses are frozen and snake_case, per the contract's rule that returned
objects read like Python.

The dataclasses stop at this boundary. Inside, the client speaks the wire dicts
the specs and the recorded corpus are written against, and `as_fetch` /
`as_async_fetch` are the only crossing -- one place, so a field cannot quietly
stop being carried in six.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .network.sse import aparse_sse_stream, parse_sse_stream
from .network.types import HttpRequest, HttpResponse

#: Generous by design: a long reasoning completion legitimately takes minutes,
#: and a transport that times out under the model is worse than no timeout.
DEFAULT_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class TransportRequest:
    """One outbound HTTP request, as a transport sees it."""

    url: str
    method: str = "POST"
    headers: Mapping[str, str] = field(default_factory=dict)
    #: The built body -- a JSON-shaped object, or bytes when `raw_body` is set.
    body: Any = None
    #: Milliseconds, as the caller passed it. `None` means the transport decides.
    timeout: float | None = None
    #: True when the caller wants the body streamed rather than read.
    stream: bool = False
    #: Carried for observability, not routing: the queue key is the engine's.
    provider: str = ""
    model: str = ""
    #: `json` (default), `arraybuffer`, `text`, or `stream`.
    response_type: str = "json"
    #: The body is already bytes and must not be serialised again.
    raw_body: bool = False

    @staticmethod
    def from_wire(req: Mapping[str, Any]) -> TransportRequest:
        return TransportRequest(
            url=req["url"],
            method=req.get("method") or "POST",
            headers=req.get("headers") or {},
            body=req.get("body"),
            timeout=req.get("timeout"),
            stream=bool(req.get("stream")),
            provider=req.get("provider") or "",
            model=req.get("model") or "",
            response_type=req.get("responseType") or "json",
            raw_body=bool(req.get("rawBody")),
        )


@dataclass(frozen=True)
class TransportResponse:
    """One HTTP response, as a transport returns it.

    `status` defaults to 200 and `headers` to empty, so a stub can be one line:
    `TransportResponse(body=...)`. A stub that returns a body and says nothing
    about the status is saying the call worked, which is the reading every
    caller already has -- and requiring it spelled out made a reviewed example
    fail with a TypeError from inside the retry layer, three frames from the
    line that omitted it.
    """

    status: int = 200
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def to_wire(self) -> HttpResponse:
        return {"status": self.status, "headers": dict(self.headers), "body": self.body}


#: `(TransportRequest) -> TransportResponse`.
Transport = Callable[[TransportRequest], TransportResponse]

#: `(TransportRequest) -> Awaitable[TransportResponse]`.
AsyncTransport = Callable[[TransportRequest], Awaitable[TransportResponse]]


def as_fetch(transport: Transport) -> Callable[..., HttpResponse]:
    """A transport -> the `fetch` the sync client injects."""

    def fetch(req: HttpRequest, options: Any = None) -> HttpResponse:
        return transport(TransportRequest.from_wire(req)).to_wire()

    return fetch


def as_async_fetch(transport: AsyncTransport) -> Callable[..., Awaitable[HttpResponse]]:
    """An async transport -> the `fetch` the async client injects.

    A SYNC transport is accepted too. Handing one to an async client is an
    ordinary thing to do -- a stub, an in-memory fake, a recorded fixture -- and
    `await`ing its plain return value fails with a TypeError from inside the
    transport layer, which reads as a library bug rather than as the one-line
    mismatch it is.
    """

    async def fetch(req: HttpRequest, options: Any = None) -> HttpResponse:
        answer = transport(TransportRequest.from_wire(req))
        response = await answer if inspect.isawaitable(answer) else answer
        return response.to_wire()

    return fetch


def as_fetch_stream(transport: Transport) -> Callable[..., Iterator[dict[str, Any]]]:
    """A transport -> the `fetch_stream` the sync client injects.

    The transport is asked for a `stream` response and returns an iterator of
    `bytes` as the body; SSE framing happens here, once, rather than in every
    transport.
    """

    def fetch_stream(req: HttpRequest, options: Any = None) -> Iterator[dict[str, Any]]:
        response = transport(_stream_request(req))
        _refuse_error_stream(response)
        return parse_sse_stream(response.body)

    return fetch_stream


def as_async_fetch_stream(
    transport: AsyncTransport,
) -> Callable[..., AsyncIterator[dict[str, Any]]]:
    """An async transport -> the `fetch_stream` the async client injects."""

    async def frames(req: HttpRequest, options: Any) -> AsyncIterator[dict[str, Any]]:
        response = await transport(_stream_request(req))
        _refuse_error_stream(response)
        async for event in aparse_sse_stream(response.body):
            yield event

    def open_stream(req: HttpRequest, options: Any = None) -> AsyncIterator[dict[str, Any]]:
        # Returned, not awaited: the client writes `async for ... in
        # fetch_stream(...)`, which needs the iterator itself rather than a
        # coroutine that eventually produces one.
        return frames(req, options)

    return open_stream


def _stream_request(req: HttpRequest) -> TransportRequest:
    return TransportRequest.from_wire({**req, "stream": True, "responseType": "stream"})


def _refuse_error_stream(response: TransportResponse) -> None:
    """A failed stream is an error body, not events.

    Feeding a 400's JSON to the SSE framer yields nothing at all -- no frames, no
    exception, an empty completion -- so the status is checked before the body is
    ever treated as a stream.
    """
    if response.status >= 400:
        raise RuntimeError(f"stream request failed ({response.status})")


def _prepare(request: TransportRequest) -> tuple[Any, dict[str, str]]:
    """Body and headers as the HTTP client wants them.

    Compact separators, not `json.dumps`'s default spacing: the body a provider
    receives has to match what the request corpus recorded.
    """
    headers = dict(request.headers)
    if request.raw_body or request.body is None or isinstance(request.body, (bytes, str)):
        return request.body, headers
    return json.dumps(request.body, separators=(",", ":")), headers


def _timeout_s(request: TransportRequest) -> float | None:
    """The caller's timeout is MILLISECONDS (the TypeScript's unit, and what the
    request carries); httpx wants seconds."""
    return request.timeout / 1000 if request.timeout else None


def http_transport(client: Any = None) -> Transport:
    """The default synchronous transport, over httpx2.

    Pass an `httpx2.Client` to control timeouts, proxies or verification -- and
    to REUSE connections, which is the reason to hold one: a transport that
    builds its own client per call pays a TLS handshake per request.
    """
    import httpx2

    http = client or httpx2.Client(timeout=DEFAULT_TIMEOUT_S)

    def transport(request: TransportRequest) -> TransportResponse:
        body, headers = _prepare(request)
        if request.response_type == "stream":
            # `send(..., stream=True)` leaves the socket open; the body iterator
            # closes it when it is exhausted or abandoned.
            built = http.build_request(request.method, request.url, headers=headers, content=body)
            res = http.send(built, stream=True)
            return TransportResponse(
                status=res.status_code, headers=dict(res.headers), body=_closing(res)
            )
        res = http.request(
            request.method,
            request.url,
            headers=headers,
            content=body,
            timeout=_timeout_s(request),
        )
        return TransportResponse(
            status=res.status_code, headers=dict(res.headers), body=_decode(res, request)
        )

    return transport


def ahttp_transport(client: Any = None) -> AsyncTransport:
    """The default asynchronous transport, over httpx2.

    Pass an `httpx2.AsyncClient` to reuse connections across calls. One is bound
    to the event loop it was created on, so build it inside the loop that will
    use it.
    """
    import httpx2

    http = client or httpx2.AsyncClient(timeout=DEFAULT_TIMEOUT_S)

    async def transport(request: TransportRequest) -> TransportResponse:
        body, headers = _prepare(request)
        if request.response_type == "stream":
            built = http.build_request(request.method, request.url, headers=headers, content=body)
            res = await http.send(built, stream=True)
            return TransportResponse(
                status=res.status_code, headers=dict(res.headers), body=_aclosing(res)
            )
        res = await http.request(
            request.method,
            request.url,
            headers=headers,
            content=body,
            timeout=_timeout_s(request),
        )
        return TransportResponse(
            status=res.status_code, headers=dict(res.headers), body=_decode(res, request)
        )

    return transport


def _decode(res: Any, request: TransportRequest) -> Any:
    """The body, read as the caller asked for it.

    A non-JSON body comes back as TEXT rather than raising: the status is what
    the caller branches on, and a provider answering a 500 with an HTML error
    page should surface that page, not a decode error about it.
    """
    if request.response_type == "arraybuffer":
        return res.content
    if request.response_type == "text":
        return res.text
    try:
        return res.json()
    except ValueError:
        return res.text


def _closing(res: Any) -> Iterator[bytes]:
    """Bytes, and the response closed after them -- including on an early break."""
    try:
        yield from res.iter_bytes()
    finally:
        res.close()


async def _aclosing(res: Any) -> AsyncIterator[bytes]:
    try:
        async for chunk in res.aiter_bytes():
            yield chunk
    finally:
        await res.aclose()


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "AsyncTransport",
    "Transport",
    "TransportRequest",
    "TransportResponse",
    "ahttp_transport",
    "as_async_fetch",
    "as_async_fetch_stream",
    "as_fetch",
    "as_fetch_stream",
    "http_transport",
]
