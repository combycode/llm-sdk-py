"""OpenAI media: images, speech, and Sora video.

Three endpoints with three different response shapes, which is most of what this
adapter is for:

- **Images** answer JSON carrying base64. The spec forks by family and the fork
  is real -- `gpt-image-*` always returns b64 and REJECTS `response_format`,
  while the dall-e models still require it.
- **Speech** answers raw audio bytes, so the request asks the engine for an
  arraybuffer rather than JSON. That is transport decoding, not wire shape,
  which is why it is set here and not in the spec.
- **Video** is asynchronous everywhere: submit, poll, download.

Every request is built from a spec and sent through the injected fetch, so media
rides the same queue and retry policy as everything else.

Transposed from `unified-library-ts/src/llm/providers/openai/media.ts`.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from ..llm.wire_transforms import make_registry
from ..wire.interpreter import build_from_spec
from ..wire.media_specs import media_spec
from .types import (
    MediaCapabilities,
    RawMediaResult,
    VideoStatus,
)

DEFAULT_BASE_URL = "https://api.openai.com"

#: What each requested audio format actually is, once it arrives.
AUDIO_MIME = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
}


def _usage_of(raw: Any) -> dict[str, int] | None:
    """OpenAI's image-token billing as the universal usage shape."""
    if not isinstance(raw, Mapping):
        return None
    return {
        "inputTokens": int(raw.get("input_tokens") or 0),
        "outputTokens": int(raw.get("output_tokens") or 0),
        "totalTokens": int(raw.get("total_tokens") or 0),
    }


def _bytes_of(body: Any) -> bytes:
    """The response body as bytes, whatever the transport handed back."""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    return b""


class OpenAIMediaAdapter:
    """Images, speech and video on OpenAI."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.name = "openai"
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
    # Separate from the methods that fetch, so a request can be asserted on
    # without performing it. Building inline would leave intercepting the
    # network as the only way to see what the adapter sends.

    def _from_spec(
        self, spec_id: str, payload: Mapping[str, Any], model: str, response_type: str
    ) -> dict[str, Any]:
        built = build_from_spec(
            media_spec(spec_id),
            {**dict(payload), "model": model},
            self._registry,
            "openai",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            # Not wire: the engine routes and decodes with these and no
            # provider ever sees them.
            "provider": "openai",
            "model": model,
            "responseType": response_type,
        }
        if not getattr(built, "no_body", False):
            request["body"] = built.body
        return request

    def build_generate_image_request(
        self, prompt: str, model: str = "gpt-image-1", params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        spec = (
            "openai/images.generations"
            if model.startswith("gpt-image-")
            else "openai/images.generations@dall-e"
        )
        return self._from_spec(spec, {"prompt": prompt, "params": dict(params or {})}, model, "json")

    def build_edit_image_request(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str = "gpt-image-1",
        params: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "sourceImage": dict(source_image),
            "params": dict(params or {}),
        }
        if mask:
            payload["mask"] = dict(mask)
        return self._from_spec("openai/images.edits", payload, model, "json")

    def build_audio_request(
        self, text: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        # arraybuffer: the response is audio, not JSON.
        return self._from_spec(
            "openai/audio.speech", {"input": text, "params": dict(params or {})}, model,
            "arraybuffer",
        )

    def build_video_request(
        self, prompt: str, model: str = "sora-2", params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "openai/videos", {"prompt": prompt, "params": dict(params or {})}, model, "json"
        )

    def build_video_status_request(self, video_id: str) -> dict[str, Any]:
        # Routed under no model: the poll is not billed to one.
        return self._from_spec("openai/videos.status", {"videoId": video_id}, "", "json")

    def build_video_content_request(self, video_id: str) -> dict[str, Any]:
        return self._from_spec(
            "openai/videos.content", {"videoId": video_id}, "", "arraybuffer"
        )

    # -- operations ----------------------------------------------------------

    def generate_image(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> list[RawMediaResult]:
        response = fetch(self.build_generate_image_request(prompt, model, params))
        return self._parse_images(_body(response))

    def edit_image(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str,
        fetch: Any,
        params: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
    ) -> list[RawMediaResult]:
        response = fetch(
            self.build_edit_image_request(prompt, source_image, model, params, mask)
        )
        return self._parse_images(_body(response))

    def _parse_images(self, data: Mapping[str, Any]) -> list[RawMediaResult]:
        """The usage rides on the FIRST item only: the batch is billed once."""
        items = data.get("data")
        rows = [i for i in items if isinstance(i, Mapping)] if isinstance(items, list) else []
        usage = _usage_of(data.get("usage"))
        out: list[RawMediaResult] = []
        for index, item in enumerate(rows):
            encoded = item.get("b64_json")
            out.append(
                RawMediaResult(
                    data=base64.b64decode(encoded) if isinstance(encoded, str) else b"",
                    mime_type="image/png",
                    revised_prompt=item.get("revised_prompt"),
                    usage=usage if index == 0 else None,
                )
            )
        return out

    def generate_audio(
        self, text: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> RawMediaResult:
        if not model:
            raise ValueError(
                'OpenAI speech needs a model (e.g. "gpt-4o-mini-tts", "tts-1").'
            )
        response = fetch(self.build_audio_request(text, model, params))
        fmt = str((params or {}).get("format") or "mp3")
        return RawMediaResult(
            data=_bytes_of(_raw_body(response)),
            mime_type=AUDIO_MIME.get(fmt, "audio/mp3"),
        )

    def submit_video(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> str:
        response = fetch(self.build_video_request(prompt, model, params))
        return str(_body(response).get("id") or "")

    def get_video_status(self, video_id: str, fetch: Any) -> VideoStatus:
        response = fetch(self.build_video_status_request(video_id))
        status_code = _status(response)
        if status_code >= 400:
            return VideoStatus(status="failed", error=f"HTTP {status_code}")
        data = _body(response)
        state = str(data.get("status") or "")
        progress = data.get("progress")
        progress = float(progress) if isinstance(progress, (int, float)) else None
        if state == "completed":
            return VideoStatus(status="completed", progress=progress)
        if state == "failed":
            error = data.get("error")
            message = error.get("message") if isinstance(error, Mapping) else None
            return VideoStatus(status="failed", error=str(message or "failed"))
        return VideoStatus(status="processing", progress=progress)

    def download_video(self, video_id: str, fetch: Any) -> RawMediaResult:
        response = fetch(self.build_video_content_request(video_id))
        if _status(response) >= 400:
            raise RuntimeError(f"OpenAI video download failed: HTTP {_status(response)}")
        return RawMediaResult(data=_bytes_of(_raw_body(response)), mime_type="video/mp4")


def _status(response: Any) -> int:
    value = response.get("status") if isinstance(response, Mapping) else None
    return value if isinstance(value, int) else 0


def _raw_body(response: Any) -> Any:
    return response.get("body") if isinstance(response, Mapping) else None


def _body(response: Any) -> Mapping[str, Any]:
    """A JSON body, decoded when the transport handed back text."""
    body = _raw_body(response)
    if isinstance(body, (str, bytes)):
        import json

        try:
            body = json.loads(body)
        except ValueError:
            return {}
    return body if isinstance(body, Mapping) else {}


__all__ = ["AUDIO_MIME", "DEFAULT_BASE_URL", "OpenAIMediaAdapter"]
