"""Hybrid voice resolution (A3): a small per-provider alias table maps a unified
alias to that provider's voice id; any unrecognized string passes through
unchanged, so raw provider voice ids always work.

Transposed from `unified-library-ts/src/llm/audio/voices.ts`.
"""

from __future__ import annotations

#: Unified voice aliases. Values are real provider voice ids (verified against the
#: provider SDKs / docs). Unknown voices are passed through verbatim.
_VOICE_ALIASES: dict[str, dict[str, str]] = {
    "openai": {"neutral": "alloy", "warm": "coral", "bright": "shimmer", "deep": "echo"},
    "google": {"neutral": "Kore", "warm": "Aoede", "bright": "Zephyr", "deep": "Charon"},
    # xai has no first-party TTS voices today.
}

VOICE_ALIASES_LIST = ("neutral", "warm", "bright", "deep")


def resolve_voice(provider: str, voice: str | None) -> str | None:
    """Map an alias to the provider's voice id, else return the input unchanged."""
    if not voice:
        return None
    return _VOICE_ALIASES.get(provider, {}).get(voice, voice)


__all__ = ["VOICE_ALIASES_LIST", "resolve_voice"]
