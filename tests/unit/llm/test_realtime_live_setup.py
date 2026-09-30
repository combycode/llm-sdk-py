"""A translator model we shipped and could not ask to translate.

Transposed from `unified-library-ts/tests/unit/llm/realtime-live-setup.test.ts`.

``gemini-3.5-live-translate`` has been in the catalog, connectable, and useless:
the Live setup frame carried modalities, a prebuilt voice and a system
instruction, and nothing else. There was no way to say which language to
translate INTO, so the model had nothing to do.

Proved by streaming the same English sentence into two sessions (2026-09-30),
which is the only way to tell -- a bogus target language still returns
``setupComplete``, so acceptance says nothing:

===========================  =========================================
setup                        output transcription
===========================  =========================================
no ``translationConfig``     ``""``
``targetLanguageCode: es``   ``"Buenos días. La reunión se ha"``
===========================  =========================================

The difference between no output at all and Spanish.

``echoTargetLanguage`` decides what happens when the target language is ALREADY
being spoken -- parrot it back, or stay quiet. ``False`` is the interesting value
and the one a truthy gate would eat.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.realtime.providers import REALTIME_ADAPTERS
from combycode_llm_sdk.realtime.types import SessionConfig, modalities_of


def setup(**config: Any) -> dict[str, Any]:
    """The `setup` frame the adapter builds for a session.

    `build_open_frame` is the same entry point the frozen cross-language frame
    corpus uses, so this test and that corpus cannot drift apart.
    """
    adapter = REALTIME_ADAPTERS["google"](api_key="k", base_url=None)
    cfg = SessionConfig(
        model="gemini-3.5-live-translate-preview",
        modalities=modalities_of(config.pop("modalities", None)),
        **config,
    )
    frame = adapter.build_open_frame(cfg) or {}
    return dict(frame.get("setup") or {})


def gen(s: dict[str, Any]) -> dict[str, Any]:
    return dict(s.get("generationConfig") or {})


class TestLiveTranslation:
    def test_it_is_absent_when_no_target_language_was_named(self) -> None:
        # An empty translationConfig would be asking to translate into nothing.
        assert "translationConfig" not in gen(setup())
        assert "translationConfig" not in gen(setup(translation={}))

    def test_it_carries_the_target_language(self) -> None:
        cfg = gen(setup(translation={"targetLanguageCode": "es"}))["translationConfig"]
        assert cfg == {"targetLanguageCode": "es"}

    def test_it_sends_echo_false_which_is_the_useful_value(self) -> None:
        # The regression guard. A truthy gate drops exactly this and the session
        # parrots back speech already in the target language.
        cfg = gen(
            setup(translation={"targetLanguageCode": "es", "echoTargetLanguage": False})
        )["translationConfig"]
        assert cfg == {"targetLanguageCode": "es", "echoTargetLanguage": False}

    def test_it_sends_true_as_readily(self) -> None:
        cfg = gen(
            setup(translation={"targetLanguageCode": "fr", "echoTargetLanguage": True})
        )["translationConfig"]
        assert cfg == {"targetLanguageCode": "fr", "echoTargetLanguage": True}

    def test_it_omits_echo_when_the_caller_said_nothing(self) -> None:
        cfg = gen(setup(translation={"targetLanguageCode": "es"}))["translationConfig"]
        assert "echoTargetLanguage" not in cfg


class TestTheOtherSetupFields:
    def test_it_sends_affective_dialog_either_way(self) -> None:
        assert gen(setup(affective_dialog=True))["enableAffectiveDialog"] is True
        # False is a real choice: it turns the behaviour off explicitly.
        assert gen(setup(affective_dialog=False))["enableAffectiveDialog"] is False
        assert "enableAffectiveDialog" not in gen(setup())

    def test_it_sends_input_transcription_only_with_a_mode(self) -> None:
        assert setup(input_transcription={"mode": "VERBATIM"})["inputAudioTranscription"] == {
            "mode": "VERBATIM"
        }
        assert "inputAudioTranscription" not in setup()
        assert "inputAudioTranscription" not in setup(input_transcription={})


class TestTheVoiceSplitTheSameWayAsTts:
    def test_it_keeps_prebuilt_for_a_catalog_name(self) -> None:
        assert gen(setup(voice="Kore"))["speechConfig"] == {
            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
        }

    def test_it_uses_the_flat_voice_for_one_the_caller_owns(self) -> None:
        assert gen(setup(voice="voice_abc", voice_owned=True))["speechConfig"] == {
            "voiceConfig": {"voice": "voice_abc"}
        }

    def test_it_sends_no_speech_config_without_a_voice(self) -> None:
        assert "speechConfig" not in gen(setup())


class TestWhatWasAlreadyThereStillIs:
    def test_it_keeps_the_model_modalities_and_instruction(self) -> None:
        s = setup(modalities=["audio"], instructions="be brief")
        assert s["model"] == "models/gemini-3.5-live-translate-preview"
        assert gen(s)["responseModalities"] == ["AUDIO"]
        assert s["systemInstruction"] == {"parts": [{"text": "be brief"}]}
