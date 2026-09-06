"""Generating media, and putting the bytes somewhere.

The requests are built from the REAL specs, so an assertion about a URL or a
body is an assertion about what would go out. Only the transport is stubbed.

The three things worth pinning are the ones a provider difference hides:
`MediaOutput` refuses to invent a place for the bytes, Gemini's headerless PCM
gets a WAV header before anyone tries to play it, and a provider that cannot do
the thing is refused by name here rather than by a 404 from inside its API.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk import Engine
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.media import (
    FileMediaStore,
    GoogleMediaAdapter,
    MediaOutput,
    MemoryMediaStore,
    OpenAIMediaAdapter,
    OpenRouterMediaAdapter,
    XAIMediaAdapter,
    media_adapter,
)
from combycode_llm_sdk.media.types import MediaMeta
from combycode_llm_sdk.util.wav import (
    ensure_playable_audio,
    is_raw_pcm_mime,
    parse_pcm_params,
    pcm_to_wav,
)

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 40).decode()
JPEG = base64.b64encode(b"\xff\xd8\xff" + b"y" * 40).decode()


def fetcher(*bodies: Any) -> tuple[Any, list[dict[str, Any]]]:
    """A fetch that answers each call from `bodies` in turn, recording requests."""
    seen: list[dict[str, Any]] = []
    queue = list(bodies)

    def fetch(request: dict[str, Any], _options: Any = None) -> dict[str, Any]:
        seen.append(request)
        body = queue.pop(0) if queue else {}
        return {"status": 200, "headers": {}, "body": body}

    return fetch, seen


class TestWavWrapping:
    """Gemini TTS returns bare samples; without a header nothing plays them."""

    def test_it_recognises_the_headerless_mimes(self) -> None:
        assert is_raw_pcm_mime("audio/l16; rate=24000")
        assert is_raw_pcm_mime("audio/pcm")
        assert not is_raw_pcm_mime("audio/mp3")
        assert not is_raw_pcm_mime("audio/wav")

    def test_it_reads_the_rate_the_provider_declared(self) -> None:
        assert parse_pcm_params("audio/l16; rate=48000; channels=2") == (48000, 2)

    def test_and_falls_back_to_geminis_own_default(self) -> None:
        assert parse_pcm_params("audio/l16") == (24000, 1)

    def test_the_header_is_a_real_riff_wave_header(self) -> None:
        wav = pcm_to_wav(b"\x00\x01" * 100, 24000, 1)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert len(wav) == 44 + 200

    def test_an_mp3_is_left_alone(self) -> None:
        # Wrapping one would corrupt it, which is why the mime is checked
        # rather than the audio being wrapped unconditionally.
        data, mime = ensure_playable_audio(b"ID3xxxx", "audio/mp3")
        assert data == b"ID3xxxx"
        assert mime == "audio/mp3"

    def test_raw_pcm_comes_back_playable(self) -> None:
        data, mime = ensure_playable_audio(b"\x00\x01" * 10, "audio/l16; rate=24000")
        assert mime == "audio/wav"
        assert data[:4] == b"RIFF"


class TestTheStores:
    def test_memory_round_trips_bytes_and_meta(self) -> None:
        store = MemoryMediaStore()
        meta = MediaMeta(id="img_1", type="image", mime_type="image/png", size=3, provider="openai")
        store.save("img_1", b"abc", meta)
        loaded = store.load("img_1")
        assert loaded is not None
        assert loaded[0] == b"abc"
        assert loaded[1].provider == "openai"
        assert store.has("img_1")

    def test_memory_filters_a_listing(self) -> None:
        store = MemoryMediaStore()
        store.save("img_1", b"a", MediaMeta("img_1", "image", "image/png", 1, "openai"))
        store.save("aud_1", b"b", MediaMeta("aud_1", "audio", "audio/mp3", 1, "google"))
        assert list(store.list(media_type="image")) == ["img_1"]
        assert list(store.list(provider="google")) == ["aud_1"]

    def test_a_missing_asset_is_none_not_an_error(self) -> None:
        assert MemoryMediaStore().load("nope") is None

    def test_file_store_writes_the_bytes_and_a_readable_sidecar(self, tmp_path: Path) -> None:
        store = FileMediaStore(tmp_path)
        meta = MediaMeta(
            id="img_1", type="image", mime_type="image/png", size=3, provider="openai",
            prompt="a red circle",
        )
        store.save("img_1", b"abc", meta)
        # The extension matters: a generated PNG called `.bin` is one no image
        # viewer will open, and the caller here is usually a person.
        assert (tmp_path / "img_1.png").read_bytes() == b"abc"
        sidecar = json.loads((tmp_path / "img_1.meta.json").read_text(encoding="utf-8"))
        assert sidecar["prompt"] == "a red circle"

    def test_file_store_reads_back_what_it_wrote(self, tmp_path: Path) -> None:
        store = FileMediaStore(tmp_path)
        store.save("aud_1", b"snd", MediaMeta("aud_1", "audio", "audio/wav", 3, "google"))
        loaded = store.load("aud_1")
        assert loaded is not None
        assert loaded[0] == b"snd"
        assert loaded[1].mime_type == "audio/wav"
        assert list(store.list()) == ["aud_1"]

    def test_deleting_removes_both_halves(self, tmp_path: Path) -> None:
        store = FileMediaStore(tmp_path)
        store.save("img_1", b"abc", MediaMeta("img_1", "image", "image/png", 3, "openai"))
        store.delete("img_1")
        assert not store.has("img_1")
        assert list(tmp_path.iterdir()) == []

    def test_a_truncated_sidecar_reads_as_absent(self, tmp_path: Path) -> None:
        # A crash mid-write. The caller asked whether the asset is there, and
        # the answer is no -- not a JSON decode error from three frames down.
        (tmp_path / "img_1.meta.json").write_text("{not json", encoding="utf-8")
        assert FileMediaStore(tmp_path).get_meta("img_1") is None


class TestTheOpenAIAdapter:
    def test_gpt_image_and_dall_e_take_different_specs(self) -> None:
        # A real fork: gpt-image always returns b64 and REJECTS response_format,
        # while dall-e still requires it.
        adapter = OpenAIMediaAdapter("k")
        gpt = adapter.build_generate_image_request("x", "gpt-image-1")
        dalle = adapter.build_generate_image_request("x", "dall-e-3")
        assert "response_format" not in json.dumps(gpt["body"])
        assert "response_format" in json.dumps(dalle["body"])

    def test_image_generation_addresses_the_documented_endpoint(self) -> None:
        request = OpenAIMediaAdapter("k").build_generate_image_request("x", "gpt-image-1")
        assert request["url"] == "https://api.openai.com/v1/images/generations"
        assert request["responseType"] == "json"

    def test_speech_asks_for_bytes_rather_than_json(self) -> None:
        # Transport decoding, not wire shape -- which is why it is set by the
        # adapter and not by the spec.
        request = OpenAIMediaAdapter("k").build_audio_request("hi", "gpt-4o-mini-tts")
        assert request["responseType"] == "arraybuffer"
        assert request["url"].endswith("/v1/audio/speech")

    def test_the_batch_usage_rides_on_the_first_image_only(self) -> None:
        # Four images are billed once; four usages would be priced four times.
        fetch, _ = fetcher(
            {
                "data": [{"b64_json": PNG}, {"b64_json": PNG}],
                "usage": {"input_tokens": 10, "output_tokens": 90, "total_tokens": 100},
            }
        )
        results = OpenAIMediaAdapter("k").generate_image("x", "gpt-image-1", fetch)
        assert len(results) == 2
        assert results[0].usage is not None
        assert results[1].usage is None

    def test_a_revised_prompt_is_carried(self) -> None:
        # OpenAI rewrites prompts routinely, and a caller comparing outputs
        # needs to know the prompt they wrote is not the one that ran.
        fetch, _ = fetcher({"data": [{"b64_json": PNG, "revised_prompt": "a crimson circle"}]})
        results = OpenAIMediaAdapter("k").generate_image("x", "gpt-image-1", fetch)
        assert results[0].revised_prompt == "a crimson circle"

    def test_speech_needs_a_model(self) -> None:
        with pytest.raises(ValueError, match="needs a model"):
            OpenAIMediaAdapter("k").generate_audio("hi", "", lambda *_a: None)


class TestTheGoogleAdapter:
    def test_imagen_and_gemini_take_different_endpoints(self) -> None:
        # Each rejects the other's models, so the fork is not cosmetic.
        adapter = GoogleMediaAdapter("k")
        assert ":predict" in adapter.build_generate_image_request("x", "imagen-4.0-generate-001")["url"]
        assert ":generateContent" in adapter.build_generate_image_request("x", "gemini-3-flash-image")["url"]

    def test_imagen_predictions_are_decoded(self) -> None:
        fetch, _ = fetcher({"predictions": [{"bytesBase64Encoded": PNG, "mimeType": "image/png"}]})
        results = GoogleMediaAdapter("k").generate_image("x", "imagen-4.0-generate-001", fetch)
        assert results[0].data.startswith(b"\x89PNG")

    def test_gemini_inline_parts_are_decoded(self) -> None:
        fetch, _ = fetcher(
            {
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}}
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 100},
            }
        )
        results = GoogleMediaAdapter("k").generate_image("x", "gemini-3-flash-image", fetch)
        assert results[0].mime_type == "image/png"
        assert results[0].usage == {"inputTokens": 5, "outputTokens": 100, "totalTokens": 0}

    def test_tts_comes_back_playable(self) -> None:
        # The whole reason the WAV wrapper exists: Gemini answers with bare
        # little-endian samples that no player will open.
        pcm = base64.b64encode(b"\x00\x01" * 50).decode()
        fetch, _ = fetcher(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"inlineData": {"mimeType": "audio/l16; rate=24000", "data": pcm}}
                            ]
                        }
                    }
                ]
            }
        )
        result = GoogleMediaAdapter("k").generate_audio("hello", "gemini-2.5-flash-preview-tts", fetch)
        assert result.mime_type == "audio/wav"
        assert result.data[:4] == b"RIFF"

    def test_tts_that_returned_no_audio_says_so(self) -> None:
        fetch, _ = fetcher({"candidates": [{"content": {"parts": [{"text": "sorry"}]}}]})
        with pytest.raises(RuntimeError, match="no audio part"):
            GoogleMediaAdapter("k").generate_audio("hi", "gemini-2.5-flash-preview-tts", fetch)


class TestTheXAIAdapter:
    def test_an_inline_image_has_its_mime_sniffed_not_assumed(self) -> None:
        # grok-imagine returns JPEG bytes with no mime. Calling it PNG breaks
        # the image-to-video path that feeds a generated image straight back.
        fetch, _ = fetcher({"data": [{"b64_json": JPEG}]})
        results = XAIMediaAdapter("k").generate_image("x", "grok-imagine-image", fetch)
        assert results[0].mime_type == "image/jpeg"

    def test_a_url_image_is_downloaded(self) -> None:
        fetch, seen = fetcher({"data": [{"url": "https://x.ai/i.png"}]}, b"rawbytes")
        results = XAIMediaAdapter("k").generate_image("x", "grok-imagine-image", fetch)
        assert len(seen) == 2, "the second call is the download"
        assert results[0].data == b"rawbytes"
        assert results[0].source_url == "https://x.ai/i.png"

    def test_it_is_the_one_provider_that_can_extend_a_video(self) -> None:
        assert XAIMediaAdapter("k").capabilities().video_extension is True
        assert OpenAIMediaAdapter("k").capabilities().video_extension is False


class TestTheOpenRouterAdapter:
    def test_images_go_through_chat_completions(self) -> None:
        # OpenRouter has no media endpoints at all.
        request = OpenRouterMediaAdapter("k").build_generate_image_request("x", "google/gemini")
        assert request["url"].endswith("/api/v1/chat/completions")

    def test_it_reports_no_video(self) -> None:
        assert OpenRouterMediaAdapter("k").capabilities().video_generation is False

    def test_a_data_url_carries_its_own_mime(self) -> None:
        fetch, _ = fetcher(
            {
                "choices": [
                    {"message": {"images": [{"image_url": {"url": f"data:image/webp;base64,{PNG}"}}]}}
                ]
            }
        )
        results = OpenRouterMediaAdapter("k").generate_image("x", "m", fetch)
        assert results[0].mime_type == "image/webp"

    def test_image_config_only_carries_what_was_set(self) -> None:
        # OpenRouter rejects an image_config full of nulls.
        adapter = OpenRouterMediaAdapter("k")
        assert adapter.image_config({"aspectRatio": "1:1"}) == {"aspect_ratio": "1:1"}
        assert adapter.image_config({}) == {}
        assert adapter.image_config(None) == {}


class TestPickingAnAdapter:
    def test_every_media_provider_has_one(self) -> None:
        for provider in ("openai", "google", "xai", "openrouter"):
            assert media_adapter(provider, "k").name == provider

    def test_anthropic_generates_no_media_at_all(self) -> None:
        with pytest.raises(ValueError, match="generates no media"):
            media_adapter("anthropic", "k")


#: Google answers generateContent, not OpenAI's `data[].b64_json`.
GOOGLE_IMAGE = {
    "candidates": [
        {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}}
    ]
}


class TestTheModelIdOnTheWire:
    """Our slug is not always what the provider answers to.

    Every other subsystem -- LLM, embeddings, batch, realtime, transcription --
    translates through the catalog before sending. Media did not, so a handle
    built on a normalised slug produced a 404 with our own spelling in it.
    """

    def resolved(self, model: str, **over: Any) -> str | None:
        store = MemoryMediaStore()
        body = GOOGLE_IMAGE if model.startswith("google/") else {"data": [{"b64_json": PNG}]}
        fetch, _ = fetcher(body)
        media = MediaOutput(
            model=model, api_key="k", store=store, fetch=fetch,
            catalog=resolve_catalog(None), **over,
        )
        return media.generate_image(prompt="x")[0].meta.model

    def test_a_slug_is_translated_to_the_callable_id(self) -> None:
        # The case that fails live: Google answers to the -preview id only.
        assert self.resolved("google/gemini-3.1-flash-tts") == "gemini-3.1-flash-tts-preview"

    def test_an_id_the_provider_already_accepts_is_untouched(self) -> None:
        # Rewriting a deliberate choice would pin a caller who asked to float.
        assert self.resolved("openai/gpt-image-1") == "gpt-image-1"

    def test_a_model_we_have_never_heard_of_still_goes_out(self) -> None:
        # A catalog we have not updated must not block a model that exists.
        assert self.resolved("openai/some-future-image-model") == "some-future-image-model"

    def test_without_a_catalog_the_model_is_sent_as_written(self) -> None:
        store = MemoryMediaStore()
        fetch, _ = fetcher(GOOGLE_IMAGE)
        media = MediaOutput(
            model="google/gemini-3.1-flash-tts", api_key="k", store=store, fetch=fetch
        )
        assert media.generate_image(prompt="x")[0].meta.model == "gemini-3.1-flash-tts"

    def test_the_catalog_is_taken_from_the_engine_when_not_passed(self) -> None:
        # Otherwise the translation depends on the caller remembering to wire a
        # catalog that the engine already has.
        engine = Engine(api_keys={"google": "k"}, register_as_default=False)
        store = MemoryMediaStore()
        fetch, _ = fetcher(GOOGLE_IMAGE)
        media = MediaOutput(
            model="google/gemini-3.1-flash-tts", store=store, fetch=fetch, engine=engine
        )
        assert media.generate_image(prompt="x")[0].meta.model == "gemini-3.1-flash-tts-preview"


class TestMediaOutput:
    def test_it_refuses_to_invent_a_place_for_the_bytes(self) -> None:
        # Guessing a temp directory is how generated files go missing.
        with pytest.raises(ValueError, match="somewhere to put the bytes"):
            MediaOutput(model="openai/gpt-image-1", api_key="k")

    def test_a_generated_image_is_stored_and_a_receipt_returned(self) -> None:
        store = MemoryMediaStore()
        fetch, _ = fetcher({"data": [{"b64_json": PNG}]})
        media = MediaOutput(model="openai/gpt-image-1", api_key="k", store=store, fetch=fetch)
        results = media.generate_image(prompt="a red circle")
        assert len(results) == 1
        # A receipt, not a payload: the caller gets an id, and the bytes are in
        # the store.
        assert results[0].id.startswith("img_")
        assert store.has(results[0].id)
        assert results[0].meta.size == len(base64.b64decode(PNG))

    def test_speech_returns_a_list_like_every_other_generator(self) -> None:
        store = MemoryMediaStore()
        fetch, _ = fetcher(b"mp3bytes")
        media = MediaOutput(model="openai/gpt-4o-mini-tts", api_key="k", store=store, fetch=fetch)
        clips = media.generate_audio(prompt="Hello.")
        assert len(clips) == 1
        assert clips[0].meta.prompt == "Hello.", "the spoken text is the prompt"

    def test_a_provider_that_cannot_is_refused_by_name(self) -> None:
        store = MemoryMediaStore()
        media = MediaOutput(
            model="openrouter/x", api_key="k", store=store, fetch=lambda *_a: None
        )
        with pytest.raises(ValueError, match="does not support video generation"):
            media.generate_video(prompt="a cat")

    def test_one_event_covers_the_batch(self) -> None:
        # Four images are billed once. Four events would be priced four times.
        events: list[tuple[str, Any]] = []

        class Hooks:
            def emit_sync(self, name: str, payload: Any) -> None:
                events.append((name, payload))

        fetch, _ = fetcher({"data": [{"b64_json": PNG}, {"b64_json": PNG}]})
        media = MediaOutput(
            model="openai/gpt-image-1",
            api_key="k",
            store=MemoryMediaStore(),
            fetch=fetch,
            hooks=Hooks(),
        )
        media.generate_image(prompt="two circles")
        generated = [e for e in events if e[0] == "onMediaGenerated"]
        assert len(generated) == 1
        assert generated[0][1]["count"] == 2

    def test_an_unnamed_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            MediaOutput(
                model="gpt-image-1", api_key="k", store=MemoryMediaStore(), fetch=lambda *_a: None
            ).generate_image(prompt="x")
