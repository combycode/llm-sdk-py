"""xAI media: images, speech and video.

Two things here are xAI's own. Images come back either inline as base64 or as a
URL that has to be fetched separately, and the inline bytes carry NO mime -- so
the format is sniffed from the bytes rather than assumed to be PNG. Assuming
would mislabel a JPEG, and the mislabel breaks the image-to-video path that
takes a generated image straight back as input.

Video is asynchronous and reports a `request_id`, and xAI is the one provider
that can also EXTEND or edit an existing clip.

Transposed from `unified-library-ts/src/llm/providers/xai/media.ts`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.wire_transforms import make_registry
from ..util.image_mime import sniff_image_mime
from ..wire.interpreter import build_from_spec
from ..wire.service_specs import service_spec
from .types import MediaCapabilities, RawMediaResult, VideoStatus

DEFAULT_BASE_URL = "https://api.x.ai"
DEFAULT_IMAGE_MODEL = "grok-imagine-image"
DEFAULT_VIDEO_MODEL = "grok-imagine-video"

AUDIO_MIME = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mulaw": "audio/mulaw",
    "alaw": "audio/alaw",
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


def _raw_bytes(response: Any) -> bytes:
    body = response.get("body") if isinstance(response, Mapping) else None
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    return b""


class XAIMediaAdapter:
    """Images, speech and video on xAI."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.name = "xai"
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URL
        self._registry = make_registry({})

    def capabilities(self) -> MediaCapabilities:
        return MediaCapabilities(
            image_generation=True,
            image_editing=True,
            audio_generation=True,
            video_generation=True,
            audio_streaming=True,
            video_extension=True,
        )

    def _from_spec(
        self,
        spec_id: str,
        payload: Mapping[str, Any],
        model: str,
        response_type: str = "json",
    ) -> dict[str, Any]:
        built = build_from_spec(
            service_spec(spec_id),
            {**dict(payload), "model": model},
            self._registry,
            "xai",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "provider": "xai",
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
        return self._from_spec(
            "xai/images.generations", {"prompt": prompt, "params": dict(params or {})}, model
        )

    def build_edit_image_request(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str = DEFAULT_IMAGE_MODEL,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._from_spec(
            "xai/images.edits",
            {"prompt": prompt, "sourceImage": dict(source_image), "params": dict(params or {})},
            model,
        )

    def build_audio_request(
        self, text: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "xai/tts", {"input": text, "params": dict(params or {})}, model, "arraybuffer"
        )

    def build_video_request(
        self, prompt: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "xai/videos.generations", {"prompt": prompt, "params": dict(params or {})}, model
        )

    def build_video_status_request(self, request_id: str) -> dict[str, Any]:
        return self._from_spec("xai/videos.status", {"requestId": request_id}, "")

    def build_download_request(self, url: str, model: str) -> dict[str, Any]:
        return self._from_spec("xai/media.download", {"url": url}, model, "arraybuffer")

    # -- operations ----------------------------------------------------------

    def generate_image(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> list[RawMediaResult]:
        model = model or DEFAULT_IMAGE_MODEL
        response = fetch(self.build_generate_image_request(prompt, model, params))
        return self._parse_images(_body(response), model, fetch)

    def edit_image(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str,
        fetch: Any,
        params: Mapping[str, Any] | None = None,
    ) -> list[RawMediaResult]:
        model = model or DEFAULT_IMAGE_MODEL
        response = fetch(self.build_edit_image_request(prompt, source_image, model, params))
        return self._parse_images(_body(response), model, fetch)

    def _parse_images(
        self, data: Mapping[str, Any], model: str, fetch: Any
    ) -> list[RawMediaResult]:
        out: list[RawMediaResult] = []
        for item in _rows(data.get("data")):
            encoded = item.get("b64_json")
            if isinstance(encoded, str) and encoded:
                raw = base64.b64decode(encoded)
                out.append(
                    RawMediaResult(
                        data=raw,
                        # SNIFFED, not assumed: grok-imagine returns JPEG with no
                        # mime, and calling it PNG breaks image-to-video later.
                        mime_type=sniff_image_mime(raw) or "image/png",
                        revised_prompt=item.get("revised_prompt"),
                    )
                )
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                downloaded = fetch(self.build_download_request(url, model))
                headers = (
                    downloaded.get("headers") if isinstance(downloaded, Mapping) else None
                ) or {}
                out.append(
                    RawMediaResult(
                        data=_raw_bytes(downloaded),
                        mime_type=str(headers.get("content-type") or "image/png"),
                        revised_prompt=item.get("revised_prompt"),
                        source_url=url,
                    )
                )
        if out and data.get("usage"):
            out[0].provider_meta = {"usage": data["usage"]}
        return out

    def generate_audio(
        self, text: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> RawMediaResult:
        response = fetch(self.build_audio_request(text, model, params))
        fmt = str((params or {}).get("format") or "mp3")
        return RawMediaResult(
            data=_raw_bytes(response), mime_type=AUDIO_MIME.get(fmt, "audio/mp3")
        )

    def submit_video(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> str:
        model = model or DEFAULT_VIDEO_MODEL
        data = _body(fetch(self.build_video_request(prompt, model, params)))
        return str(data.get("request_id") or data.get("id") or "")

    def get_video_status(self, request_id: str, fetch: Any) -> VideoStatus:
        response = fetch(self.build_video_status_request(request_id))
        code = response.get("status") if isinstance(response, Mapping) else 0
        if isinstance(code, int) and code >= 400:
            return VideoStatus(status="failed", error=f"HTTP {code}")
        data = _body(response)
        state = str(data.get("status") or "")
        progress = data.get("progress")
        progress = float(progress) if isinstance(progress, (int, float)) else None
        if state in ("completed", "succeeded", "success"):
            return VideoStatus(status="completed", progress=progress)
        if state in ("failed", "error"):
            return VideoStatus(status="failed", error=str(data.get("error") or "failed"))
        return VideoStatus(status="processing", progress=progress)

    def download_video(self, request_id: str, fetch: Any) -> RawMediaResult:
        status = _body(fetch(self.build_video_status_request(request_id)))
        video = status.get("video")
        url = video.get("url") if isinstance(video, Mapping) else None
        if not isinstance(url, str) or not url:
            raise RuntimeError(
                f"xAI video {request_id} reported completed but carried no URL to fetch."
            )
        response = fetch(self.build_download_request(url, DEFAULT_VIDEO_MODEL))
        return RawMediaResult(
            data=_raw_bytes(response), mime_type="video/mp4", source_url=url
        )


__all__ = ["AUDIO_MIME", "DEFAULT_BASE_URL", "XAIMediaAdapter"]
