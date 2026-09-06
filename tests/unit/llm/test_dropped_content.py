"""Saying so when a content part cannot be carried.

Every adapter has kinds it has no field for, and each already did the right
thing with the content -- a placeholder, or leaving the part out so the request
stays valid. None of them said anything about it, and that silence is the bug
these tests pin: a caller who attaches a wav to Anthropic gets back "I'm unable
to listen to audio files" and blames the model, when the audio was replaced with
`[unsupported: audio]` inside this library.

Measured against the WIRE, not against a mock of it: the assertion is that the
request really does go out without the audio, and that a warning really does
come back on both channels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk import Engine, complete
from combycode_llm_sdk.llm.providers._shared.dropped import note_dropped, note_replaced
from combycode_llm_sdk.transport import TransportResponse

WAV = Path(__file__).resolve().parents[3] / "examples" / "fixtures" / "hello.wav"

ANTHROPIC_BODY = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
)
OPENAI_BODY = json.dumps(
    {
        "id": "resp_1",
        "object": "response",
        "model": "gpt",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
)
GOOGLE_BODY = json.dumps(
    {
        "candidates": [
            {"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }
)
#: Chat Completions has its own response shape, and OpenRouter is how a request
#: reaches that adapter without an audio part forcing the route.
COMPLETIONS_BODY = json.dumps(
    {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
)
BODIES = {
    "anthropic": ANTHROPIC_BODY,
    "openai": OPENAI_BODY,
    "google": GOOGLE_BODY,
    "openrouter": COMPLETIONS_BODY,
}


def run(model: str, **options: Any) -> tuple[Any, dict[str, Any]]:
    """One completion against a stub transport, with the built body handed back."""
    seen: dict[str, Any] = {}

    def stub(request: Any) -> TransportResponse:
        seen["body"] = request.body
        return TransportResponse(status=200, body=BODIES[request.provider])

    result = complete(model=model, api_key="k", max_tokens=16, transport=stub, **options)
    return result, seen


def codes(result: Any) -> list[str]:
    return [w.code for w in result.warnings]


def messages(result: Any) -> str:
    return " | ".join(w.message for w in result.warnings)


class TestTheNoteHelper:
    def test_it_names_the_provider_the_kind_and_the_consequence(self) -> None:
        notes: list[str] = []
        note_dropped(notes, "anthropic", "audio")
        assert notes == [
            (
                "anthropic: audio content is not supported by this API, "
                "so the request was sent without it"
            )
        ]

    def test_replaced_and_dropped_read_differently(self) -> None:
        # Not pedantry: with a placeholder the model still sees something where
        # the content was, so an answer that mentions it is the placeholder
        # talking. Dropped means the message is simply shorter.
        replaced: list[str] = []
        dropped: list[str] = []
        note_replaced(replaced, "openai", "video")
        note_dropped(dropped, "openai", "video")
        assert replaced != dropped
        assert "placeholder" in replaced[0]
        assert "without it" in dropped[0]

    def test_one_note_per_kind_however_many_parts(self) -> None:
        # Three dropped audio parts are one fact about the request, and three
        # identical warnings would bury it.
        notes: list[str] = []
        for _ in range(3):
            note_dropped(notes, "google", "audio")
        assert len(notes) == 1

    def test_two_different_kinds_are_two_notes(self) -> None:
        notes: list[str] = []
        note_dropped(notes, "google", "audio")
        note_dropped(notes, "google", "video")
        assert len(notes) == 2

    def test_no_sink_is_not_an_error(self) -> None:
        # A build outside a request has nobody collecting; the note is discarded
        # rather than forcing every call site to make a list it will not read.
        note_dropped(None, "openai", "audio")

    def test_an_unnamed_kind_writes_nothing(self) -> None:
        # "None content is not supported" tells a reader nothing at all.
        notes: list[str] = []
        note_dropped(notes, "openai", None)
        note_dropped(notes, "openai", "")
        assert notes == []


class TestAnthropicSubstitutesAndSaysSo:
    def test_the_audio_does_not_reach_the_wire(self) -> None:
        _, seen = run(
            "anthropic/claude-haiku-4.5", prompt="What word is spoken?", attachments=[WAV]
        )
        body = json.dumps(seen["body"])
        # hello.wav is ~48KB, so ~64KB of base64. A body this small is the proof
        # the clip is not in it.
        assert len(body) < 1000
        assert "base64" not in body
        assert "[unsupported: audio]" in body

    def test_and_the_caller_is_told(self) -> None:
        result, _ = run(
            "anthropic/claude-haiku-4.5", prompt="What word is spoken?", attachments=[WAV]
        )
        assert codes(result) == ["request_adjusted"]
        assert "audio" in messages(result)
        assert "anthropic" in messages(result)

    def test_the_hook_hears_it_too(self) -> None:
        # Both channels, not one: a subscriber wants it as it happens, and a
        # caller holding only the result must not have had to subscribe.
        heard: list[tuple[Any, Any]] = []
        engine = Engine()
        engine.hooks.on("on_warning", lambda ctx: heard.append((ctx.get("code"), ctx.get("message"))))
        result, _ = run(
            "anthropic/claude-haiku-4.5",
            prompt="What word is spoken?",
            attachments=[WAV],
            engine=engine,
        )
        assert heard
        assert heard[0][0] == "request_adjusted"
        # The same words in both places, so the two never describe one
        # adjustment differently.
        assert heard[0][1] == result.warnings[0].message

    def test_text_alone_warns_about_nothing(self) -> None:
        result, _ = run("anthropic/claude-haiku-4.5", prompt="hello")
        assert codes(result) == []


class TestProvidersThatCanCarryItStaySilent:
    """The other half, and the one that makes the warning worth trusting.

    A warning that fires whenever audio is attached would be noise; these two
    genuinely send the clip, so they must say nothing.
    """

    @pytest.mark.parametrize(
        ("model", "carrier"),
        [
            # Each names the field that provider actually carries audio in --
            # `input_audio` on Chat Completions, `inlineData` on generateContent.
            ("openai/gpt-audio", "input_audio"),
            ("google/gemini-3.1-flash-lite", "inlineData"),
        ],
    )
    def test_the_audio_is_on_the_wire_and_no_warning_is_raised(
        self, model: str, carrier: str
    ) -> None:
        result, seen = run(model, prompt="What word is spoken?", attachments=[WAV])
        body = json.dumps(seen["body"])
        # ~64KB of base64 for a ~48KB clip: the size IS the evidence it is there.
        assert len(body) > 50_000, f"{model} should carry the clip"
        assert carrier in body
        assert codes(result) == []


class TestTheOtherAdaptersReportTheirOwnDrops:
    """Each adapter has its own shape of "cannot carry this", and each says so.

    Anthropic's is the loud one, but it is not the only one: the Responses API
    has no audio form at all, Chat Completions has one only for base64, and
    Google's media branch takes four source kinds and no more. All three used to
    be silent.
    """

    def test_the_responses_api_reports_a_kind_it_has_no_branch_for(self) -> None:
        # The worst of the three shapes: Responses builds a parts list and then
        # omits an item that ended up empty, so the WHOLE MESSAGE disappears
        # from the request. Silently, until now.
        result, seen = run(
            "openai/gpt-5.4-nano",
            messages=[{"role": "user", "content": [{"type": "hologram", "text": "x"}]}],
        )
        body = json.dumps(seen["body"])
        assert "hologram" not in body
        assert codes(result) == ["request_adjusted"]
        assert "hologram" in messages(result)

    def test_chat_completions_carries_audio_only_as_base64(self) -> None:
        # A URL source has no `input_audio` form, so it becomes a placeholder --
        # and the message names the SOURCE, since the kind alone would look like
        # "openai cannot do audio", which is untrue and would mislead.
        # No `api=` needed: an audio part routes an OpenAI request to Chat
        # Completions on its own, because Responses has no form for one.
        result, _ = run(
            "openai/gpt-audio",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "source": {"type": "url", "url": "https://x/a.wav"}}
                    ],
                }
            ],
        )
        assert codes(result) == ["request_adjusted"]
        assert "audio (url source)" in messages(result)

    def test_an_entirely_unknown_kind_is_reported_by_completions(self) -> None:
        # Through OpenRouter, whose default api IS Chat Completions. Routing an
        # OpenAI model here instead would reach Responses, and this test would
        # then pass on the Responses branch while the Completions one rotted --
        # which is exactly what a mutation sweep caught it doing.
        result, seen = run(
            "openrouter/openai/gpt-5.4-nano",
            messages=[{"role": "user", "content": [{"type": "hologram", "text": "x"}]}],
        )
        assert "[unsupported: hologram]" in json.dumps(seen["body"])
        assert codes(result) == ["request_adjusted"]
        assert "hologram" in messages(result)

    def test_google_reports_a_media_source_it_has_no_field_for(self) -> None:
        # Google carries every media KIND, so a drop here is always about the
        # source: `buffer` never became a fileData or an inlineData.
        result, _ = run(
            "google/gemini-3.1-flash-lite",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "buffer", "mimeType": "image/png"}}
                    ],
                }
            ],
        )
        assert codes(result) == ["request_adjusted"]
        assert "google" in messages(result)

    def test_google_reports_an_unknown_kind(self) -> None:
        result, _ = run(
            "google/gemini-3.1-flash-lite",
            messages=[{"role": "user", "content": [{"type": "hologram", "text": "x"}]}],
        )
        assert codes(result) == ["request_adjusted"]
        assert "hologram" in messages(result)
