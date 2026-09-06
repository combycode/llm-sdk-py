"""Detect an audio clip's MIME type from its leading magic bytes.

Transposed from `unified-library-ts/src/util/audio-mime.ts`.

Same problem as `sniff_image_mime`, from the opposite direction. OpenAI's chat
completions return `message.audio` as `{id, data, expires_at, transcript}` and NO
`format` field, so the adapter's `audio/{format or 'wav'}` fell through to the
default every single time: request mp3, receive mp3 bytes (`ID3...`), and get
told it is `audio/wav`. Anything that trusts the label -- an <audio> element, a
file written to disk, a follow-up upload with a strict validator -- is then
working from a lie.

The bytes are the only honest source here, and the caller's requested format is
not available at parse time.

Returns the CONTAINER, which is what a player needs: Opus arrives inside Ogg and
is reported as `audio/ogg`.
"""

from __future__ import annotations


def sniff_audio_mime(b: bytes) -> str | None:
    # MP3 with an ID3 tag: "ID3"
    if len(b) >= 3 and b[0] == 0x49 and b[1] == 0x44 and b[2] == 0x33:
        return "audio/mpeg"
    # WAV: "RIFF"????"WAVE"
    if len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE":
        return "audio/wav"
    # Ogg (Opus / Vorbis): "OggS"
    if len(b) >= 4 and b[0:4] == b"OggS":
        return "audio/ogg"
    # FLAC: "fLaC"
    if len(b) >= 4 and b[0:4] == b"fLaC":
        return "audio/flac"
    # AAC in an ADTS frame: FF F1 (MPEG-4) or FF F9 (MPEG-2). Checked BEFORE the
    # bare MPEG sync below, because ADTS also satisfies that mask.
    if len(b) >= 2 and b[0] == 0xFF and b[1] in (0xF1, 0xF9):
        return "audio/aac"
    # MP3 with no ID3 tag: an MPEG frame sync, eleven set bits.
    if len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return None


__all__ = ["sniff_audio_mime"]
