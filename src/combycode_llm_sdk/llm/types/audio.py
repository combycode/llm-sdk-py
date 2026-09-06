"""Shared audio types.

Transposed from `unified-library-ts/src/llm/types/audio.ts`.

One shape for voice/format options and raw audio input across `complete()` /
`generate_audio()` / `create_realtime()` / `transcribe()`. Shapes as camelCase
dicts, per `messages.py`:

- `AudioOptions`  `{voice?, format?, sampleRate?}` -- output controls. `voice`
  accepts a provider voice id OR a unified alias (see `resolve_voice`).
  `sampleRate` is Hz, for raw/PCM output.
- `AudioInput`    `{data, mimeType?, sampleRate?}` -- input source. A file path
  is MIME-detected; raw bytes should declare `mimeType` (and `sampleRate` for
  PCM).
- `TranscriptLanguage` `{code}` -- a language a provider reports detecting.
  An object rather than a bare `str` on purpose: providers already annotate
  detections and will annotate them further (confidence, spans). A `list[str]`
  could only grow by becoming a different type, which is exactly the breaking
  change CONSTITUTION.md R3 forbids.
- `TranscriptWord` `{word, start, end}` -- timings in seconds from the start.
- `TranscriptSegment` `{id?, start, end, text, speaker?}` -- `speaker` is
  present only when diarization ran; no provider surface currently returns
  speakers and word timings together (see `transcribe()`), so a segment carries
  whichever the chosen model produces. `id` is normalised to a string (whisper
  numbers them, the diarizing models use `seg_N`).
"""

from __future__ import annotations

from typing import Any, Literal

#: `type AudioFormat` (audio.ts:4).
AudioFormat = Literal["wav", "mp3", "pcm16", "opus", "flac", "aac"]

#: `interface AudioOptions` (audio.ts:8).
AudioOptions = dict[str, Any]

#: `interface AudioInput` (audio.ts:16).
AudioInput = dict[str, Any]

#: `interface TranscriptLanguage` (audio.ts:29).
TranscriptLanguage = dict[str, Any]

#: `interface TranscriptWord` (audio.ts:35).
TranscriptWord = dict[str, Any]

#: `interface TranscriptSegment` (audio.ts:45).
TranscriptSegment = dict[str, Any]

__all__ = [
    "AudioFormat",
    "AudioInput",
    "AudioOptions",
    "TranscriptLanguage",
    "TranscriptSegment",
    "TranscriptWord",
]
