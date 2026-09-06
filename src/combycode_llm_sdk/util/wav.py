"""Making raw PCM playable.

Gemini's TTS returns bare little-endian 16-bit samples under `audio/l16`, with
no container at all. Written to a file that way it plays in nothing: no header
means no player knows the sample rate. So a 44-byte WAV header goes in front,
and the result opens everywhere.

Only for the formats that need it. Passing an mp3 through this would corrupt it,
which is why `ensure_playable_audio` checks the mime rather than wrapping
unconditionally.

Transposed from `unified-library-ts/src/util/wav.ts`.
"""

from __future__ import annotations

import re
import struct

_PCM_MIME = re.compile(r"\b(l16|pcm)\b", re.IGNORECASE)
_RATE = re.compile(r"rate=(\d+)")
_CHANNELS = re.compile(r"channels=(\d+)")

#: What Gemini uses when its mime says nothing.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1


def is_raw_pcm_mime(mime: str) -> bool:
    """Whether this mime means headerless samples."""
    return bool(_PCM_MIME.search(mime or ""))


def parse_pcm_params(mime: str) -> tuple[int, int]:
    """`rate=` and `channels=` out of an L16 mime, with the usual defaults."""
    rate = _RATE.search(mime or "")
    channels = _CHANNELS.search(mime or "")
    return (
        int(rate.group(1)) if rate else DEFAULT_SAMPLE_RATE,
        int(channels.group(1)) if channels else DEFAULT_CHANNELS,
    )


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """A 44-byte RIFF/WAVE header in front of 16-bit little-endian samples."""
    bits_per_sample = 16
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # audio format: PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(pcm),
    )
    return header + pcm


def ensure_playable_audio(data: bytes, mime: str) -> tuple[bytes, str]:
    """The audio and its mime, wrapped in WAV only if it needed to be."""
    if not is_raw_pcm_mime(mime):
        return data, mime
    sample_rate, channels = parse_pcm_params(mime)
    return pcm_to_wav(data, sample_rate, channels), "audio/wav"


__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_SAMPLE_RATE",
    "ensure_playable_audio",
    "is_raw_pcm_mime",
    "parse_pcm_params",
    "pcm_to_wav",
]
