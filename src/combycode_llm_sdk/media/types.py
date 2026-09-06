"""What a generated asset is, and where it goes.

Media is the one subsystem whose result is BYTES rather than text, and that
single fact shapes everything here. Bytes do not fit in a completion, cannot be
logged, and must not be held in memory by default -- so a generated asset is
saved to a `MediaStore` and the caller gets an ID plus metadata. `MediaResult`
is deliberately a receipt, not a payload.

Transposed from `unified-library-ts/src/plugins/media/types.ts`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

MediaType = Literal["image", "audio", "video"]

#: How long a video generation is polled, and how often. Video is the one media
#: kind that is asynchronous everywhere: the provider takes minutes and answers
#: with an operation id.
MEDIA_OUTPUT_DEFAULTS: Mapping[str, float] = {
    "poll_interval_seconds": 5.0,
    "max_poll_wait_seconds": 600.0,
}


@dataclass(frozen=True)
class MediaMeta:
    """Everything known about one stored asset except its bytes."""

    id: str
    type: MediaType
    mime_type: str
    size: int
    provider: str
    created_at: float = field(default_factory=time.time)
    model: str | None = None
    prompt: str | None = None
    #: What the provider actually generated from, when it rewrote the prompt.
    #: OpenAI does this routinely, and a caller comparing outputs needs to know
    #: the prompt they wrote is not the prompt that ran.
    revised_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: float | None = None
    sample_rate: int | None = None
    params: Mapping[str, Any] | None = None
    #: A provider-hosted URL, when the bytes live remotely. Async video often
    #: lands in a bucket that refuses a programmatic byte fetch, so this may be
    #: the only way to play the result.
    source_url: str | None = None

    def as_row(self) -> dict[str, Any]:
        """The metadata as a JSON-shaped dict, for a store that persists it."""
        return {
            "id": self.id,
            "type": self.type,
            "mimeType": self.mime_type,
            "size": self.size,
            "createdAt": self.created_at,
            "provider": self.provider,
            "model": self.model,
            "prompt": self.prompt,
            "revisedPrompt": self.revised_prompt,
            "width": self.width,
            "height": self.height,
            "durationMs": self.duration_ms,
            "sampleRate": self.sample_rate,
            "params": dict(self.params) if self.params else None,
            "sourceUrl": self.source_url,
        }

    @staticmethod
    def of(row: Mapping[str, Any]) -> MediaMeta:
        return MediaMeta(
            id=str(row.get("id") or ""),
            type=row.get("type") or "image",
            mime_type=str(row.get("mimeType") or ""),
            size=int(row.get("size") or 0),
            provider=str(row.get("provider") or ""),
            created_at=float(row.get("createdAt") or time.time()),
            model=row.get("model"),
            prompt=row.get("prompt"),
            revised_prompt=row.get("revisedPrompt"),
            width=row.get("width"),
            height=row.get("height"),
            duration_ms=row.get("durationMs"),
            sample_rate=row.get("sampleRate"),
            params=row.get("params"),
            source_url=row.get("sourceUrl"),
        )


@dataclass(frozen=True)
class MediaResult:
    """A receipt: what was made, and the id to fetch it back by."""

    id: str
    type: MediaType
    mime_type: str
    meta: MediaMeta


@dataclass
class RawMediaResult:
    """One asset as the adapter parsed it, before it is stored."""

    data: bytes
    mime_type: str
    #: Set when the bytes live remotely and `data` may be empty.
    source_url: str | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: float | None = None
    sample_rate: int | None = None
    revised_prompt: str | None = None
    #: Token usage, for the media that is token-priced (gpt-image, gemini-tts).
    #: Absent for unit-priced media, which is a different pricing path and not
    #: a zero.
    usage: Mapping[str, Any] | None = None
    provider_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class VideoStatus:
    """Where an asynchronous video generation has got to."""

    status: Literal["pending", "processing", "completed", "failed"]
    progress: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class MediaCapabilities:
    """What one provider can actually make.

    Read before every call, so a provider that cannot do the thing is refused
    by name here rather than by a 404 from somewhere inside its API.
    """

    image_generation: bool = False
    image_editing: bool = False
    audio_generation: bool = False
    video_generation: bool = False
    audio_streaming: bool = False
    #: Extending or editing an EXISTING clip, which is a different endpoint
    #: from generation on every provider that offers it.
    video_extension: bool = False


class MediaStore(Protocol):
    """Where the bytes live."""

    def save(self, media_id: str, data: bytes, meta: MediaMeta) -> None: ...

    def load(self, media_id: str) -> tuple[bytes, MediaMeta] | None: ...

    def get_meta(self, media_id: str) -> MediaMeta | None: ...

    def delete(self, media_id: str) -> None: ...

    def list(
        self, media_type: MediaType | None = None, provider: str | None = None
    ) -> Sequence[str]: ...

    def has(self, media_id: str) -> bool: ...


class MediaAdapter(Protocol):
    """One provider's media endpoints.

    Every method takes the fetch rather than holding one: media HTTP goes
    through the engine's queue like everything else, and an adapter that kept
    its own client would be invisible to the rate limiter.
    """

    name: str

    def capabilities(self) -> MediaCapabilities: ...


__all__ = [
    "MEDIA_OUTPUT_DEFAULTS",
    "MediaAdapter",
    "MediaCapabilities",
    "MediaMeta",
    "MediaResult",
    "MediaStore",
    "MediaType",
    "RawMediaResult",
    "VideoStatus",
]
