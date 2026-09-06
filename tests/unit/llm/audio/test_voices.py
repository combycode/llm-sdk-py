"""Translated from `unified-library-ts/tests/unit/llm/audio/voices.test.ts`."""

from __future__ import annotations

from combycode_llm_sdk.llm.audio.voices import resolve_voice


class TestResolveVoice:
    def test_maps_a_known_alias_to_the_provider_voice_id(self) -> None:
        # voices.test.ts:8-11
        assert resolve_voice("openai", "warm") == "coral"
        assert resolve_voice("openai", "neutral") == "alloy"
        assert resolve_voice("google", "neutral") == "Kore"
        assert resolve_voice("google", "bright") == "Zephyr"

    def test_passes_a_raw_provider_voice_id_through_unchanged(self) -> None:
        # voices.test.ts:15-16
        assert resolve_voice("openai", "shimmer") == "shimmer"
        assert resolve_voice("google", "Puck") == "Puck"

    def test_passes_through_for_an_unknown_provider(self) -> None:
        # voices.test.ts:20
        assert resolve_voice("xai", "whatever") == "whatever"

    def test_returns_none_for_no_voice(self) -> None:
        # voices.test.ts:24
        assert resolve_voice("openai", None) is None
