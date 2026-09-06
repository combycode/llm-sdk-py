"""Google media: Imagen, Gemini image, Gemini TTS, and Veo video.

Google has two image paths and they are not interchangeable. Imagen models use
`:predict` and answer with `predictions[].bytesBase64Encoded`; the `gemini-*`
image models generate INLINE through `generateContent` with response modalities,
and the bytes come back as an `inlineData` part. Picking by model prefix is what
routes them, because the endpoints reject each other's models.

TTS is the same `generateContent` path with an audio modality, and its output
needs one more step: Gemini returns bare little-endian PCM under `audio/l16`,
which plays in nothing until a WAV header goes in front of it.

Transposed from `unified-library-ts/src/llm/providers/google/media.ts`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.wire_transforms import make_registry
from ..util.wav import ensure_playable_audio
from ..wire.interpreter import build_from_spec
from ..wire.media_specs import media_spec
from .types import MediaCapabilities, RawMediaResult, VideoStatus

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

DEFAULT_IMAGE_MODEL = "imagen-4.0-generate-001"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_EDIT_MODEL = "gemini-2.5-flash-image"


def _usage_of(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        "inputTokens": int(raw.get("promptTokenCount") or 0),
        "outputTokens": int(raw.get("candidatesTokenCount") or 0),
        "totalTokens": int(raw.get("totalTokenCount") or 0),
    }


def _body(response: Any) -> Mapping[str, Any]:
    body = response.get("body") if isinstance(response, Mapping) else None
    if isinstance(body, (str, bytes)):
        try:
            body = json.loads(body)
        except ValueError:
            return {}
    return body if isinstance(body, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [v for v in value if isinstance(v, Mapping)]


class GoogleMediaAdapter:
    """Images, speech and video on Google."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.name = "google"
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URL
        self._registry = make_registry({})

    def capabilities(self) -> MediaCapabilities:
        return MediaCapabilities(
            image_generation=True,
            image_editing=True,
            audio_generation=True,
            video_generation=True,
        )

    # -- request builders ----------------------------------------------------

    def _from_spec(
        self,
        spec_id: str,
        payload: Mapping[str, Any],
        model: str,
        response_type: str = "json",
    ) -> dict[str, Any]:
        built = build_from_spec(
            media_spec(spec_id),
            {**dict(payload), "model": model},
            self._registry,
            "google",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "provider": "google",
            "model": model,
            "responseType": response_type,
        }
        if not getattr(built, "no_body", False):
            request["body"] = built.body
        return request

    def build_generate_image_request(
        self,
        prompt: str,
        model: str = DEFAULT_IMAGE_MODEL,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # The fork that matters: Imagen is `:predict`, gemini-* is inline
        # `generateContent`. Each endpoint rejects the other's models.
        spec = (
            "google/imagen@predict"
            if model.startswith("imagen")
            else "google/gemini-image@generateContent"
        )
        return self._from_spec(spec, {"prompt": prompt, "params": dict(params or {})}, model)

    def build_edit_image_request(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str = DEFAULT_EDIT_MODEL,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._from_spec(
            "google/gemini-image-edit@generateContent",
            {"prompt": prompt, "sourceImage": dict(source_image), "params": dict(params or {})},
            model,
        )

    def build_audio_request(
        self, text: str, model: str = DEFAULT_TTS_MODEL, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "google/gemini-tts@generateContent",
            {"input": text, "params": dict(params or {})},
            model,
        )

    def build_video_request(
        self, prompt: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "google/veo@predictLongRunning",
            {"prompt": prompt, "params": dict(params or {})},
            model,
        )

    def build_video_status_request(self, operation: str) -> dict[str, Any]:
        return self._from_spec("google/media.operation.status", {"operation": operation}, "")

    # -- parsing -------------------------------------------------------------

    def _parse_generate_content(
        self, response: Any
    ) -> tuple[list[tuple[str, str]], dict[str, int] | None]:
        """The `inlineData` parts of a generateContent answer, and its usage."""
        data = _body(response)
        candidates = _rows(data.get("candidates"))
        parts: list[Mapping[str, Any]] = []
        if candidates:
            content = candidates[0].get("content")
            if isinstance(content, Mapping):
                parts = _rows(content.get("parts"))
        items: list[tuple[str, str]] = []
        for part in parts:
            inline = part.get("inlineData")
            if isinstance(inline, Mapping) and inline.get("data"):
                items.append((str(inline.get("mimeType") or ""), str(inline["data"])))
        return items, _usage_of(data.get("usageMetadata"))

    # -- operations ----------------------------------------------------------

    def generate_image(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> list[RawMediaResult]:
        model = model or DEFAULT_IMAGE_MODEL
        if not model.startswith("imagen"):
            items, usage = self._parse_generate_content(
                fetch(self.build_generate_image_request(prompt, model, params))
            )
            return [
                RawMediaResult(
                    data=base64.b64decode(encoded),
                    mime_type=mime or "image/png",
                    usage=usage if index == 0 else None,
                )
                for index, (mime, encoded) in enumerate(items)
            ]

        data = _body(fetch(self.build_generate_image_request(prompt, model, params)))
        return [
            RawMediaResult(
                data=base64.b64decode(str(pred.get("bytesBase64Encoded") or "")),
                mime_type=str(pred.get("mimeType") or "image/png"),
            )
            for pred in _rows(data.get("predictions"))
        ]

    def edit_image(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str,
        fetch: Any,
        params: Mapping[str, Any] | None = None,
    ) -> list[RawMediaResult]:
        items, usage = self._parse_generate_content(
            fetch(self.build_edit_image_request(prompt, source_image, model, params))
        )
        return [
            RawMediaResult(
                data=base64.b64decode(encoded),
                mime_type=mime or "image/png",
                usage=usage if index == 0 else None,
            )
            for index, (mime, encoded) in enumerate(items)
        ]

    def generate_audio(
        self, text: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> RawMediaResult:
        items, usage = self._parse_generate_content(
            fetch(self.build_audio_request(text, model or DEFAULT_TTS_MODEL, params))
        )
        if not items:
            raise RuntimeError(
                "Google TTS: generateContent returned no audio part. The model may "
                "not support an audio response modality."
            )
        mime, encoded = items[0]
        # Bare PCM until a header goes in front of it.
        data, playable_mime = ensure_playable_audio(base64.b64decode(encoded), mime or "audio/wav")
        return RawMediaResult(data=data, mime_type=playable_mime, usage=usage)

    def submit_video(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> str:
        data = _body(fetch(self.build_video_request(prompt, model, params)))
        return str(data.get("name") or "")

    def get_video_status(self, operation: str, fetch: Any) -> VideoStatus:
        response = fetch(self.build_video_status_request(operation))
        status_code = response.get("status") if isinstance(response, Mapping) else 0
        if isinstance(status_code, int) and status_code >= 400:
            return VideoStatus(status="failed", error=f"HTTP {status_code}")
        data = _body(response)
        error = data.get("error")
        if isinstance(error, Mapping):
            return VideoStatus(status="failed", error=str(error.get("message") or "failed"))
        # Google reports completion as `done`, with no progress figure at all.
        if data.get("done") is True:
            return VideoStatus(status="completed")
        return VideoStatus(status="processing")


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_EDIT_MODEL",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_TTS_MODEL",
    "GoogleMediaAdapter",
]
