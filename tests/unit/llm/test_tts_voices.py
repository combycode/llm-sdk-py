"""A voice you own, and a conversation between two of them.

Transposed from `unified-library-ts/tests/unit/llm/tts-voices.test.ts`.

`voice` was a string: an alias we map (``"warm"``) or a provider catalog name
(``"Kore"``). A custom voice has neither -- its id (``voice_...``) is issued when
the voice is created and is not a name anyone could guess. So the type widens to
``str | {"id": ...}``, additively: a string means exactly what it always meant,
and the request it produces is byte-for-byte the one it produced before.

Which wire field carries it is measured, not assumed (2026-09-30):

* ``prebuiltVoiceConfig.voiceName`` is what we have always sent, and still is
  for a catalog name.
* the flat ``voiceConfig.voice`` is what Google validates custom ids against --
  a bogus one returns ``404 The voice was not found or the caller does not have
  permission to access it``, the right answer arriving at the right validator.

**Multi-speaker is one feature, not two.** A request carrying
``multiSpeakerVoiceConfig`` without a speaker on every text part is refused:
*"Multi-speaker generation requests must specify speech_metadata.speaker for
each text part in the contents."* So a caller gives ``segments`` once and the
adapter derives both halves from them.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.audio.voices import is_owned_voice, resolve_voice
from combycode_llm_sdk.media.google import GoogleMediaAdapter


def tts(params: dict[str, Any], text: str = "hello") -> dict[str, Any]:
    built = GoogleMediaAdapter("k").build_audio_request(text, "gemini-3.8-flash-tts", params)
    body = built["body"]
    return {
        "parts": body["contents"][0]["parts"],
        "speech": body["generationConfig"]["speechConfig"],
    }


class TestResolveVoiceWithTheWidenedType:
    def test_it_still_maps_an_alias_and_passes_a_catalog_name(self) -> None:
        assert resolve_voice("google", "warm") == "Aoede"
        assert resolve_voice("google", "Kore") == "Kore"
        assert resolve_voice("google", None) is None

    def test_it_returns_an_owned_id_untouched(self) -> None:
        # Never alias-mapped: a custom id is already the final answer.
        assert resolve_voice("google", {"id": "voice_abc123"}) == "voice_abc123"

    def test_an_empty_id_is_no_voice_at_all(self) -> None:
        assert resolve_voice("google", {"id": ""}) is None
        assert is_owned_voice({"id": ""}) is False
        assert is_owned_voice("Kore") is False
        assert is_owned_voice({"id": "voice_x"}) is True


class TestOneVoice:
    def test_it_keeps_the_existing_wire_shape_for_a_catalog_name(self) -> None:
        # The compatibility guard. This request must not change.
        assert tts({"voice": "Kore"})["speech"] == {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
        }

    def test_it_resolves_an_alias_down_the_same_path(self) -> None:
        assert tts({"voice": "warm"})["speech"] == {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}
        }

    def test_it_falls_back_to_the_default(self) -> None:
        assert tts({})["speech"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}

    def test_it_sends_an_owned_voice_on_the_validating_field(self) -> None:
        assert tts({"voice": {"id": "voice_abc123"}})["speech"] == {
            "voiceConfig": {"voice": "voice_abc123"}
        }

    def test_it_carries_the_input_as_a_single_part(self) -> None:
        assert tts({"voice": "Kore"}, "read this")["parts"] == [{"text": "read this"}]


CAST = [
    {"name": "Ada", "voice": "Kore"},
    {"name": "Grace", "voice": {"id": "voice_grace"}},
]
SCRIPT = [
    {"speaker": "Ada", "text": "The meeting is Tuesday.", "style": "brisk"},
    {"speaker": "Grace", "text": "I will be there."},
]


class TestSeveralVoices:
    def test_it_builds_the_speaker_configs(self) -> None:
        assert tts({"speakers": CAST, "segments": SCRIPT})["speech"] == {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {"speaker": "Ada", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
                    {"speaker": "Grace", "voiceConfig": {"voice": "voice_grace"}},
                ]
            }
        }

    def test_it_puts_the_speaker_on_every_part(self) -> None:
        # Without this the request is refused outright, so it is not optional
        # polish -- it is the other half of the same feature.
        assert tts({"speakers": CAST, "segments": SCRIPT})["parts"] == [
            {"text": "The meeting is Tuesday.", "speechMetadata": {"speaker": "Ada", "style": "brisk"}},
            {"text": "I will be there.", "speechMetadata": {"speaker": "Grace"}},
        ]

    def test_it_lets_a_cast_win_over_a_single_voice(self) -> None:
        # Asking for both is a contradiction; the cast is more specific.
        speech = tts({"voice": "Kore", "speakers": CAST, "segments": SCRIPT})["speech"]
        assert "multiSpeakerVoiceConfig" in speech
        assert "voiceConfig" not in speech

    def test_it_ignores_a_speaker_with_no_name(self) -> None:
        speech = tts({"speakers": [{"name": "", "voice": "Kore"}, *CAST], "segments": SCRIPT})["speech"]
        assert len(speech["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"]) == 2

    def test_it_falls_back_to_plain_input_when_segments_are_empty(self) -> None:
        assert tts({"segments": []}, "just this")["parts"] == [{"text": "just this"}]

    def test_it_drops_a_segment_with_no_text(self) -> None:
        parts = tts({"speakers": CAST, "segments": [{"speaker": "Ada"}, *SCRIPT]})["parts"]
        assert len(parts) == 2
