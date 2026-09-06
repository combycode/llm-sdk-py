"""What a live session is made of, before any provider is involved.

Two providers, two protocols that look nothing alike -- OpenAI a typed event
stream over text frames, Gemini a turn-based bidi protocol over binary ones --
and one session API over both. These are the pieces that API is built from.

Transposed from `unified-library-ts/src/llm/realtime/types.py` and
`src/network/types.ts` (the socket half, which the network module said would
"land with the realtime area").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Modality = Literal["text", "audio"]


@dataclass(frozen=True)
class SessionConfig:
    """What to ask the provider for when the session opens."""

    #: Bare model id. The provider is fixed by the adapter.
    model: str
    #: Default `("text",)`. Some models are audio-native and answer in audio
    #: whatever is asked -- and Gemini Live REFUSES a text-only session outright.
    modalities: tuple[Modality, ...] = ("text",)
    voice: str | None = None
    instructions: str | None = None

    def as_input(self) -> dict[str, Any]:
        """The camelCase shape the wire specs read."""
        return {
            "model": self.model,
            "modalities": list(self.modalities),
            "voice": self.voice,
            "instructions": self.instructions,
        }


@dataclass(frozen=True)
class Turn:
    """One input turn, or a piece of one."""

    text: str | None = None
    #: Raw audio bytes in whatever encoding the provider expects (PCM16 for
    #: both of the providers here).
    audio: bytes | None = None

    def as_input(self, turn_complete: bool) -> dict[str, Any]:
        return {"text": self.text, "audio": self.audio, "turnComplete": turn_complete}


# ── the socket ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WsRequest:
    """Where to open the socket, and who it belongs to.

    `provider` and `model` never reach the wire; they are what the observability
    hooks report, so a frame in a log can be attributed without parsing a URL.
    """

    url: str
    protocols: tuple[str, ...] | None = None
    provider: str = ""
    model: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """One inbound frame, normalised.

    Exactly one of `text`/`binary` is set. Both providers send JSON, but OpenAI
    sends it as text frames and Gemini as binary ones, so the difference has to
    survive as far as the adapter that knows which to expect.
    """

    text: str | None = None
    binary: bytes | None = None


class WsSession(Protocol):
    """The socket, as this module uses it.

    A subset of `httpx2`'s session: enough to be satisfied by a caller's own
    client, and small enough that a test double is a few lines.
    """

    def send_text(self, data: str) -> None: ...

    def send_bytes(self, data: bytes) -> None: ...

    def receive(self, timeout: float | None = None) -> Any: ...

    def close(self, code: int = 1000, reason: str | None = None) -> None: ...


#: Opens a socket and hands back a context manager over a `WsSession`.
#: `httpx2.websocket` satisfies this, and so does anything a caller brings.
WsConnect = Callable[..., Any]


class Connection(Protocol):
    """An open socket, with frames arriving on a reader thread."""

    def send_text(self, data: str) -> None: ...

    def send_bytes(self, data: bytes) -> None: ...

    def on_frame(self, callback: Callable[[Frame], None]) -> Callable[[], None]: ...

    def close(self) -> None: ...


#: What an adapter is handed to open its socket: `Engine.connect`.
EngineConnect = Callable[[WsRequest], Connection]


class ProviderAdapter(Protocol):
    """One provider's realtime protocol."""

    name: str

    def build_connect_request(self, config: SessionConfig) -> WsRequest: ...

    def connect(self, config: SessionConfig, connect: EngineConnect) -> Any: ...


def modalities_of(value: Sequence[str] | None) -> tuple[Modality, ...]:
    """Normalise a caller's modality list, defaulting to text.

    Accepts any sequence of strings because that is what a caller writes --
    `["audio"]` -- and returns a tuple so a config stays hashable and cannot be
    mutated behind the session's back.
    """
    if not value:
        return ("text",)
    out: list[Modality] = []
    for item in value:
        name = str(item).lower()
        if name not in ("text", "audio"):
            raise ValueError(
                f"unknown modality {item!r}. Realtime sessions carry 'text' and 'audio'."
            )
        out.append(name)  # type: ignore[arg-type]
    return tuple(out)


__all__ = [
    "Connection",
    "EngineConnect",
    "Frame",
    "Modality",
    "ProviderAdapter",
    "SessionConfig",
    "Turn",
    "WsConnect",
    "WsRequest",
    "WsSession",
    "modalities_of",
]
