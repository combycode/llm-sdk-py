"""A transcript that came back empty, and a mode that only works on one model.

Transposed from `unified-library-ts/tests/unit/llm/transcription-mode.test.ts`.

Two findings from the same probe, on deliberately disfluent audio
("Um, so, I was -- I was thinking, uh, maybe we could...").

**The transcript was being dropped.** ``gemini-3.5-transcribe`` does not answer
with ``parts[].text``. It answers with ``parts[].audioTranscription.text``, and
this library read only the former -- so a successful, billed transcription
returned an EMPTY string. Nothing errored; the caller just got nothing.

**``mode`` is honoured, but not everywhere.** Measured 2026-09-30:

=========================  ================  =========================
model                      VERBATIM          SMART
=========================  ================  =========================
``gemini-3.5-transcribe``  4 fillers kept    **0** -- all removed
``gemini-3.1-flash-lite``  4                 4 -- no different at all
=========================  ================  =========================

The noise floor matters for reading that: two runs of the SAME config on the
dedicated model were byte-identical, so the difference there is signal. On
flash-lite two plain runs already differed by punctuation, so the "difference"
between modes there was nothing.

An invalid value is a 400 naming ``audio_transcription_config.mode`` on both,
which is why "it was accepted" proves nothing on its own.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter

#: What `gemini-3.5-transcribe` actually returns.
TRANSCRIPT: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"audioTranscription": {"text": "Um, so I was thinking uh maybe Tuesday."}}],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {"promptTokenCount": 261, "totalTokenCount": 261},
}


def _adapter() -> GoogleAdapter:
    return GoogleAdapter({"apiKey": "k"})


def _parse(raw: dict[str, Any]) -> dict[str, Any]:
    return _adapter().parse_response(raw, 0)


class TestATranscriptionModelResponse:
    def test_it_is_read_not_dropped(self) -> None:
        # Was `""`: the part has no `text` key at all.
        r = _parse(TRANSCRIPT)
        assert r["text"] == "Um, so I was thinking uh maybe Tuesday."
        assert r["content"] == [{"type": "text", "text": "Um, so I was thinking uh maybe Tuesday."}]

    def test_it_finishes_cleanly(self) -> None:
        assert _parse(TRANSCRIPT)["finishReason"] == "stop"

    def test_it_does_not_disturb_an_ordinary_text_part(self) -> None:
        r = _parse(
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {},
            }
        )
        assert r["text"] == "hello"

    def test_it_prefers_the_transcript_when_a_part_carries_both(self) -> None:
        r = _parse(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "ignored", "audioTranscription": {"text": "kept"}}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            }
        )
        assert r["text"] == "kept"

    def test_it_ignores_an_audio_transcription_with_no_text(self) -> None:
        r = _parse(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"audioTranscription": {}}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            }
        )
        assert r["text"] == ""


class TestTheModeReachesTheWire:
    def body(self, provider_options: dict[str, Any] | None = None) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": "gemini-3.5-transcribe",
            "messages": [{"role": "user", "content": "transcribe"}],
        }
        if provider_options is not None:
            req["providerOptions"] = provider_options
        return dict(_adapter().build_request(req).body)

    def test_it_lands_in_generation_config(self) -> None:
        cfg = self.body({"audioTranscriptionConfig": {"mode": "SMART"}})["generationConfig"]
        assert cfg["audioTranscriptionConfig"] == {"mode": "SMART"}

    def test_it_carries_verbatim_just_as_readily(self) -> None:
        cfg = self.body({"audioTranscriptionConfig": {"mode": "VERBATIM"}})["generationConfig"]
        assert cfg["audioTranscriptionConfig"] == {"mode": "VERBATIM"}

    def test_it_is_absent_when_the_caller_asked_for_nothing(self) -> None:
        # An empty config on every completion would be noise on requests that
        # have no audio in them at all.
        assert "audioTranscriptionConfig" not in self.body().get("generationConfig", {})
        assert "audioTranscriptionConfig" not in self.body({}).get("generationConfig", {})
