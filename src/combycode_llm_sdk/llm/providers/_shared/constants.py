"""Shared constants used across provider adapters.

Transposed from `unified-library-ts/src/llm/providers/_shared/constants.ts`.
"""

from __future__ import annotations

#: PCM16 mono audio sample rate emitted by Gemini Live and OpenAI Realtime.
AUDIO_PCM16_SAMPLE_RATE_HZ = 24000

#: Default max_tokens / max_completion_tokens when the caller omits the field.
DEFAULT_MAX_TOKENS = 4096

__all__ = ["AUDIO_PCM16_SAMPLE_RATE_HZ", "DEFAULT_MAX_TOKENS"]
