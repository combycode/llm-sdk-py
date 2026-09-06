"""Streamed events as frozen dataclasses, so `event.type` and `match` both work.

Python-native, and the API contract fixes both halves of the shape::

    Events are frozen dataclasses with a `type` discriminator, so
    `event.type == "text"` reads the same as the TypeScript. They also support
    structural pattern matching for callers who prefer it.

A class per kind, unlike `results.Part`: matching is the point here, and
`case TextEvent(text=t)` needs a class to match on. The `type` field is kept
alongside so the string comparison the TypeScript user already knows still
works, and so an event kind this build has never seen still arrives -- as
`UnknownEvent`, carrying its payload, rather than raising or vanishing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .results import BuiltinToolCall, Citation, FileOutput, Usage


@dataclass(frozen=True)
class Event:
    """The base every streamed event shares."""

    type: str


@dataclass(frozen=True)
class TextEvent(Event):
    text: str = ""
    #: Which output item this delta belongs to, when the provider says. A turn
    #: can interleave deltas from several items.
    item_id: str | None = None
    #: `commentary` or `final_answer`, on models that distinguish them.
    phase: str | None = None


@dataclass(frozen=True)
class ThinkingEvent(Event):
    text: str = ""
    item_id: str | None = None


@dataclass(frozen=True)
class ToolCallStartEvent(Event):
    id: str = ""
    name: str = ""


@dataclass(frozen=True)
class ToolCallDeltaEvent(Event):
    id: str = ""
    #: A JSON *string* fragment, not an object -- arguments arrive in pieces and
    #: are only parseable once `tool_call_end` has been seen.
    arguments: str = ""


@dataclass(frozen=True)
class ToolCallEndEvent(Event):
    id: str = ""


@dataclass(frozen=True)
class UsageEvent(Event):
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class DoneEvent(Event):
    finish_reason: str = "stop"


@dataclass(frozen=True)
class ErrorEvent(Event):
    error: Any = None


@dataclass(frozen=True)
class MediaStartEvent(Event):
    media_type: str = ""
    mime_type: str = ""


@dataclass(frozen=True)
class MediaChunkEvent(Event):
    data: str = ""
    progress: float | None = None


@dataclass(frozen=True)
class MediaEndEvent(Event):
    media_id: str | None = None


@dataclass(frozen=True)
class FileEvent(Event):
    """A hosted-tool output file became available.

    Carries the DESCRIPTOR, not the bytes -- fetch those with
    `llm.retrieve_file(event.file)`.
    """

    file: FileOutput = field(default_factory=FileOutput)


@dataclass(frozen=True)
class CitationEvent(Event):
    """The answer cited a source.

    Emitted when the citation arrives, which is not when the search ran: a
    provider searches early and cites while it writes, so these interleave with
    text deltas.
    """

    citation: Citation = field(default_factory=lambda: Citation(url=""))


@dataclass(frozen=True)
class BuiltinToolStartEvent(Event):
    tool: str = ""
    id: str | None = None


@dataclass(frozen=True)
class BuiltinToolEndEvent(Event):
    call: BuiltinToolCall = field(default_factory=lambda: BuiltinToolCall(tool=""))


@dataclass(frozen=True)
class ModerationEvent(Event):
    phase: str = "output"
    result: Mapping[str, Any] = field(default_factory=dict)
    #: `native` when the provider produced it, `emulated` when we did.
    source: str = "native"


# ── realtime ────────────────────────────────────────────────────────────────
#
# A live session yields these alongside `TextEvent`, which it reuses unchanged:
# a text delta is a text delta whichever transport carried it. The three below
# have no counterpart on the streaming path.


@dataclass(frozen=True)
class AudioEvent(Event):
    """A chunk of the model's spoken answer, decoded.

    Bytes rather than the base64 `MediaChunkEvent` carries: realtime audio is a
    continuous stream a caller writes to a device or a file, and making every
    consumer decode it would be a step nobody wants and everyone repeats.
    """

    audio: bytes = b""
    mime_type: str = ""
    #: Samples per second, when the provider says. Both providers here stream
    #: PCM16 at 24 kHz.
    sample_rate: int | None = None


@dataclass(frozen=True)
class SessionOpenEvent(Event):
    """The live session is ready to send.

    Not the same as the socket connecting: OpenAI accepts content immediately,
    Gemini only after it answers `setupComplete`. A caller does not have to wait
    for this -- `send()` before it is buffered, not lost -- but the event exists
    so code ported from the TypeScript, where it is the `open` event, still has
    something to see.

    There is no matching `close`: the socket closing ENDS the iteration, which
    is how Python already says it.
    """


@dataclass(frozen=True)
class TurnEndEvent(Event):
    """The model finished its turn.

    Not the end of the session: the socket stays open for the next turn, which
    is why a caller reading one answer breaks on this rather than waiting for
    the iterator to end.
    """


@dataclass(frozen=True)
class RealtimeErrorEvent(Event):
    """The provider reported an error on a live session.

    Distinct from `ErrorEvent`, which carries an exception object: on a socket
    the failure arrives as a message from the other end, and there is no
    exception until a caller decides to raise one.
    """

    message: str = ""


@dataclass(frozen=True)
class UnknownEvent(Event):
    """An event kind this build does not model.

    It reaches the caller rather than being dropped: the unified event set is
    open, and a provider adding a kind must not make the reply silently lose a
    piece. `payload` is the whole event as it arrived.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)


def to_event(raw: Mapping[str, Any]) -> Event:
    """One wire event -> its dataclass.

    The wire vocabulary is camelCase and the Python one is snake_case, and this
    is the only place they meet on the streaming path.
    """
    kind = str(raw.get("type") or "")
    builder = _BUILDERS.get(kind)
    if builder is None:
        return UnknownEvent(type=kind, payload=raw)
    event: Event = builder(raw)
    return event


_BUILDERS: dict[str, Any] = {
    "text": lambda r: TextEvent(
        type="text", text=r.get("text") or "", item_id=r.get("itemId"), phase=r.get("phase")
    ),
    "thinking": lambda r: ThinkingEvent(
        type="thinking", text=r.get("text") or "", item_id=r.get("itemId")
    ),
    "tool_call_start": lambda r: ToolCallStartEvent(
        type="tool_call_start", id=r.get("id") or "", name=r.get("name") or ""
    ),
    "tool_call_delta": lambda r: ToolCallDeltaEvent(
        type="tool_call_delta", id=r.get("id") or "", arguments=r.get("arguments") or ""
    ),
    "tool_call_end": lambda r: ToolCallEndEvent(type="tool_call_end", id=r.get("id") or ""),
    "usage": lambda r: UsageEvent(type="usage", usage=Usage.of(r.get("usage"))),
    "done": lambda r: DoneEvent(type="done", finish_reason=r.get("finishReason") or "stop"),
    "error": lambda r: ErrorEvent(type="error", error=r.get("error")),
    "media_start": lambda r: MediaStartEvent(
        type="media_start", media_type=r.get("mediaType") or "", mime_type=r.get("mimeType") or ""
    ),
    "media_chunk": lambda r: MediaChunkEvent(
        type="media_chunk", data=r.get("data") or "", progress=r.get("progress")
    ),
    "media_end": lambda r: MediaEndEvent(type="media_end", media_id=r.get("mediaId")),
    "file": lambda r: FileEvent(type="file", file=FileOutput.of(r.get("file") or {})),
    "citation": lambda r: CitationEvent(
        type="citation", citation=Citation.of(r.get("citation") or {})
    ),
    "builtin_tool_start": lambda r: BuiltinToolStartEvent(
        type="builtin_tool_start", tool=r.get("tool") or "", id=r.get("id")
    ),
    "builtin_tool_end": lambda r: BuiltinToolEndEvent(
        type="builtin_tool_end", call=BuiltinToolCall.of(r)
    ),
    "moderation": lambda r: ModerationEvent(
        type="moderation",
        phase=r.get("phase") or "output",
        result=r.get("result") or {},
        source=r.get("source") or "native",
    ),
}

#: Every kind `to_event` models. Asserted against the stream specs by a test, so
#: a kind the parsers can emit cannot quietly fall through to `UnknownEvent`.
KNOWN_EVENT_TYPES: frozenset[str] = frozenset(_BUILDERS)

__all__ = [
    "KNOWN_EVENT_TYPES",
    "AudioEvent",
    "BuiltinToolEndEvent",
    "BuiltinToolStartEvent",
    "CitationEvent",
    "DoneEvent",
    "ErrorEvent",
    "Event",
    "FileEvent",
    "MediaChunkEvent",
    "MediaEndEvent",
    "MediaStartEvent",
    "ModerationEvent",
    "RealtimeErrorEvent",
    "SessionOpenEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
    "TurnEndEvent",
    "UnknownEvent",
    "UsageEvent",
]
