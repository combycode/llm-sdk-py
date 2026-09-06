"""Audio in, transcript out -- and what the model could not tell you.

The failure worth guarding is an empty list standing in for a limitation:
"there were no words" and "this model cannot report words" are different
answers, and only one of them means the transcript is complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk import Engine, TransportResponse, transcribe
from combycode_llm_sdk.llm.wire_multipart import encode_multipart
from combycode_llm_sdk.transcription import parse_transcription

VERBOSE = {
    "text": "hello world",
    "language": "en",
    "duration": 1.2,
    "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "hello world"}],
    "words": [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.2},
    ],
}


def stub(body: Any = None, seen: list[Any] | None = None) -> Any:
    def send(request: Any) -> TransportResponse:
        if seen is not None:
            seen.append(request)
        return TransportResponse(body=body if body is not None else VERBOSE)

    return send


class TestTheResult:
    def test_text_is_the_guarantee(self) -> None:
        result = transcribe(
            model="openai/whisper-1", api_key="k", audio=b"\x00", transport=stub()
        )
        assert result.text == "hello world"

    def test_the_extras_are_typed_values_not_raw_dicts(self) -> None:
        result = transcribe(
            model="openai/whisper-1",
            api_key="k",
            audio=b"\x00",
            timestamps="word",
            transport=stub(),
        )
        assert result.language == "en"
        assert result.segments is not None
        assert result.words is not None
        assert result.words[0].start == 0.0
        assert result.duration_seconds == 1.2

    def test_absence_is_none_not_an_empty_list(self) -> None:
        # An empty list says "there were none", which is a different answer
        # from "this model cannot tell you".
        plain = parse_transcription({"text": "hi"})
        assert plain.words is None
        assert plain.segments is None
        assert plain.languages is None

    def test_a_zero_segment_id_survives(self) -> None:
        # Whisper numbers segments from 0, so `id: 0` is real -- and falsy.
        parsed = parse_transcription(VERBOSE)
        assert parsed.segments is not None
        assert parsed.segments[0].id == "0"

    def test_a_malformed_word_is_dropped_rather_than_crashing(self) -> None:
        parsed = parse_transcription({"text": "x", "words": [{"word": "a"}, {"start": 1}]})
        assert parsed.words is None

    def test_duration_is_read_from_a_usage_object_too(self) -> None:
        # The token-billed models report it only there.
        parsed = parse_transcription({"text": "x", "usage": {"type": "duration", "seconds": 3}})
        assert parsed.duration_seconds == 3.0

    def test_several_detected_languages_are_kept(self) -> None:
        parsed = parse_transcription(
            {"text": "x", "languages": [{"code": "en"}, {"code": "fr"}]}
        )
        assert parsed.languages == ("en", "fr")


class TestRouting:
    def test_a_transcription_model_uses_the_dedicated_endpoint(self) -> None:
        seen: list[Any] = []
        transcribe(model="openai/whisper-1", api_key="k", audio=b"\x00",
                   transport=stub(seen=seen))
        assert seen[0].url.endswith("/v1/audio/transcriptions")

    def test_a_chat_model_listens_through_a_completion(self) -> None:
        # Routing on the PROVIDER instead sent OpenAI's default chat model to
        # the transcriptions endpoint, which answers 404.
        seen: list[Any] = []
        transcribe(
            model="openai/gpt-4.1",
            api_key="k",
            audio=b"\x00",
            transport=stub(
                body={
                    "id": "r",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                seen=seen,
            ),
        )
        assert not seen[0].url.endswith("/v1/audio/transcriptions")

    def test_a_non_openai_provider_listens_through_a_completion(self) -> None:
        seen: list[Any] = []
        result = transcribe(
            model="google/gemini-2.5-flash",
            api_key="k",
            audio=b"\x00",
            transport=stub(
                body={
                    "candidates": [
                        {"content": {"parts": [{"text": "spoken"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                },
                seen=seen,
            ),
        )
        assert result.text == "spoken"

    def test_asking_a_provider_for_what_it_cannot_do_warns(self) -> None:
        # A caller who asked for speakers must learn they are not coming,
        # instead of quietly receiving plain text.
        engine = Engine(register_as_default=False, api_keys={"google": "k"})
        warnings: list[Any] = []
        engine.hooks.on("onWarning", warnings.append)
        transcribe(
            model="google/gemini-2.5-flash",
            audio=b"\x00",
            diarization=True,
            engine=engine,
            transport=stub(
                body={
                    "candidates": [
                        {"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
                }
            ),
        )
        codes = [w["code"] for w in warnings]
        assert "transcription_option_unsupported" in codes


class TestRefusals:
    def test_timestamps_and_diarization_together_are_refused_here(self) -> None:
        # They select different response formats and no model serves both, so
        # this is refused before it becomes a 400.
        with pytest.raises(ValueError, match="no model serves both"):
            transcribe(
                model="openai/whisper-1",
                api_key="k",
                audio=b"\x00",
                timestamps="word",
                diarization=True,
                transport=stub(),
            )

    def test_a_bare_model_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            transcribe(model="whisper-1", api_key="k", audio=b"\x00", transport=stub())

    def test_no_key_is_refused_before_any_request(self) -> None:
        with pytest.raises(ValueError, match="no API key"):
            transcribe(model="openai/whisper-1", audio=b"\x00", transport=stub())


class TestTheAudioUnion:
    def test_a_path_is_read_from_disk(self, tmp_path: Path) -> None:
        # The same `str | Path | bytes` union `attachments=` takes.
        target = tmp_path / "hello.wav"
        target.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        assert transcribe(
            model="openai/whisper-1", api_key="k", audio=target, transport=stub()
        ).text == "hello world"

    def test_raw_bytes_are_used_directly(self) -> None:
        assert transcribe(
            model="openai/whisper-1", api_key="k", audio=b"\x00\x01", transport=stub()
        ).text == "hello world"


class TestMultipartEncoding:
    def test_the_boundary_appears_in_both_the_body_and_the_header(self) -> None:
        # Whoever builds one must build the other, which is why this is encoded
        # here rather than left to the transport.
        body, content_type = encode_multipart([("model", "whisper-1")])
        boundary = content_type.split("boundary=", 1)[1]
        assert boundary.encode() in body

    def test_a_file_part_carries_its_filename_and_type(self) -> None:
        body, _ = encode_multipart([("file", ("a.wav", b"\x00\x01", "audio/wav"))])
        assert b'filename="a.wav"' in body
        assert b"Content-Type: audio/wav" in body
        assert b"\x00\x01" in body

    def test_the_body_is_crlf_delimited(self) -> None:
        # An LF-only multipart body is rejected by some servers and accepted by
        # others, which makes it the kind of bug that appears on one provider.
        body, _ = encode_multipart([("model", "m")])
        assert b"\r\n" in body
        assert body.endswith(b"--\r\n")

    def test_the_request_declares_a_raw_body(self) -> None:
        # `rawBody` is how the engine is told not to re-serialise a body that
        # already is one; without it the request went out empty.
        seen: list[Any] = []
        transcribe(model="openai/whisper-1", api_key="k", audio=b"\x00",
                   transport=stub(seen=seen))
        assert isinstance(seen[0].body, bytes)
