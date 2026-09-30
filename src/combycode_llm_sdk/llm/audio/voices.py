"""Hybrid voice resolution (A3): a small per-provider alias table maps a unified
alias to that provider's voice id; any unrecognized string passes through
unchanged, so raw provider voice ids always work.

Transposed from `unified-library-ts/src/llm/audio/voices.ts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Unified voice aliases. Values are real provider voice ids (verified against the
#: provider SDKs / docs). Unknown voices are passed through verbatim.
_VOICE_ALIASES: dict[str, dict[str, str]] = {
    "openai": {"neutral": "alloy", "warm": "coral", "bright": "shimmer", "deep": "echo"},
    "google": {"neutral": "Kore", "warm": "Aoede", "bright": "Zephyr", "deep": "Charon"},
    # xai has no first-party TTS voices today.
}

VOICE_ALIASES_LIST = ("neutral", "warm", "bright", "deep")


def resolve_voice(provider: str, voice: Any) -> str | None:
    """Map an alias to the provider's voice id, else return the input unchanged.

    A `{"id": ...}` is a voice the caller OWNS -- a custom voice whose id is not
    a name anyone could guess. It is never alias-mapped: the aliases translate
    our four adjectives into a provider's catalog names, and a custom id is
    already the final answer. Returned as-is.
    """
    if not voice:
        return None
    if isinstance(voice, Mapping):
        found = voice.get("id")
        return found if isinstance(found, str) and found else None
    if not isinstance(voice, str):
        return None
    return _VOICE_ALIASES.get(provider, {}).get(voice, voice)


def is_owned_voice(voice: Any) -> bool:
    """Is this a voice the caller owns, rather than one from a catalog?

    Decides which wire field carries it. Measured 2026-09-30: Google's flat
    `voiceConfig.voice` accepts BOTH a catalog name and a custom id (a bogus one
    returns `404 The voice was not found or the caller does not have permission
    to access it`), while `prebuiltVoiceConfig.voiceName` is the field we have
    always sent. So a plain string keeps its existing path byte-for-byte and
    only a `{"id": ...}` takes the new one.
    """
    return isinstance(voice, Mapping) and isinstance(voice.get("id"), str) and bool(voice["id"])


__all__ = ["VOICE_ALIASES_LIST", "resolve_voice"]
