"""OpenRouter media: no media endpoints at all.

OpenRouter serves image and audio generation through `POST /chat/completions`
with a `modalities` field, and the output arrives on `message.images[]` or
`message.audio`. So this adapter is the one that does NOT look like the others:
there is no `/images/generations` to call.

Which also means no video. OpenRouter has no video output at all, and
`capabilities()` says so rather than letting a caller discover it from a
confusing chat response.

Transposed from `unified-library-ts/src/llm/providers/openrouter/media.ts`.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm.wire_transforms import make_registry
from ..wire.interpreter import build_from_spec
from ..wire.service_specs import service_spec
from .types import MediaCapabilities, RawMediaResult

DEFAULT_BASE_URL = "https://openrouter.ai"

_DATA_URL_MIME = re.compile(r"^data:(.*?);")


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


def _usage_of(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        "inputTokens": int(raw.get("prompt_tokens") or 0),
        "outputTokens": int(raw.get("completion_tokens") or 0),
        "totalTokens": int(raw.get("total_tokens") or 0),
    }


def _first_message(data: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = _rows(data.get("choices"))
    if not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, Mapping) else {}


class OpenRouterMediaAdapter:
    """Images and speech, both through chat completions."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self.name = "openrouter"
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_BASE_URL
        # The image spec calls back into this adapter for `image_config`, which
        # is a per-provider shape rather than a template -- so the registry is
        # given the adapter itself, exactly as the TypeScript does.
        self._registry = make_registry({"openrouter_media": self})

    def image_config(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """OpenRouter's `image_config`, built from the unified params.

        Only the keys the caller actually set: OpenRouter rejects an
        `image_config` carrying nulls, and an empty one is omitted entirely by
        the spec's own guard.
        """
        params = params or {}
        config: dict[str, Any] = {}
        if params.get("aspectRatio"):
            config["aspect_ratio"] = params["aspectRatio"]
        if params.get("imageSize"):
            config["image_size"] = params["imageSize"]
        if params.get("strength") is not None:
            config["strength"] = params["strength"]
        return config

    def capabilities(self) -> MediaCapabilities:
        return MediaCapabilities(
            image_generation=True,
            image_editing=True,
            # Through `modalities: ['audio']`. No TTS models in the catalog yet,
            # but the path exists.
            audio_generation=True,
            # OpenRouter routes chat, and no chat model produces video.
            video_generation=False,
        )

    def _from_spec(
        self, spec_id: str, payload: Mapping[str, Any], model: str
    ) -> dict[str, Any]:
        built = build_from_spec(
            service_spec(spec_id),
            {**dict(payload), "model": model},
            self._registry,
            "openrouter",
            None,
            {"baseURL": self._base_url, "apiKey": self._api_key},
        )
        request: dict[str, Any] = {
            "url": built.url,
            "method": built.method or "POST",
            "headers": dict(built.headers or {}),
            "provider": "openrouter",
            "model": model,
            "responseType": "json",
        }
        if not getattr(built, "no_body", False):
            request["body"] = built.body
        return request

    def build_generate_image_request(
        self, prompt: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "openrouter/media.image", {"prompt": prompt, "params": dict(params or {})}, model
        )

    def build_edit_image_request(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._from_spec(
            "openrouter/media.imageEdit",
            {"prompt": prompt, "sourceImage": dict(source_image), "params": dict(params or {})},
            model,
        )

    def build_audio_request(
        self, text: str, model: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._from_spec(
            "openrouter/media.audio", {"input": text, "params": dict(params or {})}, model
        )

    # -- operations ----------------------------------------------------------

    def generate_image(
        self, prompt: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> list[RawMediaResult]:
        return self._parse_images(
            _body(fetch(self.build_generate_image_request(prompt, model, params)))
        )

    def edit_image(
        self,
        prompt: str,
        source_image: Mapping[str, Any],
        model: str,
        fetch: Any,
        params: Mapping[str, Any] | None = None,
    ) -> list[RawMediaResult]:
        return self._parse_images(
            _body(fetch(self.build_edit_image_request(prompt, source_image, model, params)))
        )

    def _parse_images(self, data: Mapping[str, Any]) -> list[RawMediaResult]:
        images = _rows(_first_message(data).get("images"))
        usage = _usage_of(data.get("usage"))
        provider_meta = {"usage": data["usage"]} if data.get("usage") else None
        out: list[RawMediaResult] = []
        for index, image in enumerate(images):
            holder = image.get("image_url")
            url = holder.get("url") if isinstance(holder, Mapping) else image.get("url")
            if not isinstance(url, str) or not url:
                continue
            # A data URL carries its own mime; anything after the comma is the
            # payload. A bare base64 string is taken as PNG.
            encoded = url[url.index(",") + 1 :] if "," in url else url
            found = _DATA_URL_MIME.match(url)
            out.append(
                RawMediaResult(
                    data=base64.b64decode(encoded),
                    mime_type=found.group(1) if found else "image/png",
                    # Attached to the FIRST item: the batch is billed once.
                    usage=usage if index == 0 else None,
                    provider_meta=provider_meta if index == 0 else None,
                )
            )
        return out

    def generate_audio(
        self, text: str, model: str, fetch: Any, params: Mapping[str, Any] | None = None
    ) -> RawMediaResult:
        data = _body(fetch(self.build_audio_request(text, model, params)))
        audio = _first_message(data).get("audio")
        encoded = audio.get("data") if isinstance(audio, Mapping) else None
        if not isinstance(encoded, str) or not encoded:
            raise RuntimeError(
                "OpenRouter returned no audio. The model may not serve an audio "
                "modality -- OpenRouter has no dedicated TTS endpoint."
            )
        fmt = str(audio.get("format") or "mp3") if isinstance(audio, Mapping) else "mp3"
        return RawMediaResult(
            data=base64.b64decode(encoded),
            mime_type=f"audio/{fmt}",
            usage=_usage_of(data.get("usage")),
            provider_meta={"usage": data["usage"]} if data.get("usage") else None,
        )


__all__ = ["DEFAULT_BASE_URL", "OpenRouterMediaAdapter"]
