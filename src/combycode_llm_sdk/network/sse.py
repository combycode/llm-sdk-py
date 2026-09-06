"""Server-Sent Events parser. One implementation for all providers.

Transposed from `unified-library-ts/src/network/sse.ts`.

The TypeScript reads a `ReadableStream<Uint8Array>` through a reader it must
remember to cancel; Python iterates an async byte iterator, and closing it is the
iterator's own business. What is transposed exactly is the FRAMING -- where a
message ends, which lines carry meaning, and the two payloads that yield no event
at all -- because that is the part every provider depends on and the part a
plausible-looking rewrite gets subtly wrong.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

#: A message ends at a blank line, in any of the three line-ending conventions.
#: Providers do mix them: Anthropic sends `\n\n`, and a proxy in the middle may
#: rewrite to `\r\n\r\n`.
_MESSAGE_BOUNDARY = re.compile(r"\n\n|\r\n\r\n|\r\r")

_LINE_BOUNDARY = re.compile(r"\n|\r\n|\r")


class _Framer:
    """The framing state a stream carries: a decoder and a partial message.

    Its own object so the sync and async readers below are each four lines and
    share every decision about where a message ends -- the part that is subtle
    and that both must agree on.
    """

    def __init__(self) -> None:
        self._decoder = _IncrementalDecoder()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        """Absorb a chunk, return whatever COMPLETE events it finished."""
        self._buffer += self._decoder.decode(chunk)
        parts = _MESSAGE_BOUNDARY.split(self._buffer)
        # The last part is whatever came after the final boundary -- an
        # incomplete message, kept for the next chunk.
        self._buffer = parts.pop() if parts else ""
        return [e for e in (parse_sse_message(p) for p in parts) if e is not None]

    def flush(self) -> list[dict[str, Any]]:
        """A final message with no trailing blank line. Providers do end this way."""
        if not self._buffer.strip():
            return []
        event = parse_sse_message(self._buffer)
        self._buffer = ""
        return [event] if event is not None else []


def parse_sse_stream(body: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Decode a byte stream into SSE events.

    Decoding is incremental and buffered: a multi-byte character can be split
    across two chunks, and decoding each chunk independently would corrupt it.
    """
    framer = _Framer()
    for chunk in body:
        yield from framer.feed(chunk)
    yield from framer.flush()


async def aparse_sse_stream(body: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """The async twin of `parse_sse_stream`."""
    framer = _Framer()
    async for chunk in body:
        for event in framer.feed(chunk):
            yield event
    for event in framer.flush():
        yield event


def parse_sse_message(raw: str) -> dict[str, Any] | None:
    """One SSE message block -> `{event?, data, id?}`, or None.

    None for two distinct reasons, and both matter: a block with no `data:` line
    at all (a comment or a bare keep-alive), and the `[DONE]` sentinel, which is
    a terminator rather than a payload -- handing it to a provider parser would
    fail a JSON decode on every OpenAI-compatible stream.
    """
    event: str | None = None
    event_id: str | None = None
    data = ""
    has_data = False

    for line in _LINE_BOUNDARY.split(raw):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("data:"):
            # Multiple `data:` lines in one message join with a newline; only the
            # LEADING space of each is optional padding, so lstrip, not strip --
            # trailing whitespace can be part of a token.
            if has_data:
                data += "\n"
            data += line[5:].lstrip()
            has_data = True
        # A line starting with `:` is an SSE comment -- ignored.

    if not has_data:
        return None
    if data == "[DONE]":
        return None

    out: dict[str, Any] = {"data": data}
    if event is not None:
        out["event"] = event
    if event_id is not None:
        out["id"] = event_id
    return out


class _IncrementalDecoder:
    """`new TextDecoder()` with `{stream: true}`.

    Holds the tail bytes of an incomplete UTF-8 sequence until the rest arrives.
    Python's `codecs.getincrementaldecoder` does exactly this; it is wrapped only
    to keep the call site reading like the TypeScript.
    """

    def __init__(self) -> None:
        import codecs

        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def decode(self, chunk: bytes) -> str:
        return self._decoder.decode(chunk)


__all__ = ["aparse_sse_stream", "parse_sse_message", "parse_sse_stream"]
