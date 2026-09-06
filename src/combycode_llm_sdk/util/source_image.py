"""Source/reference image handling for media edit + image-to-video.

Normalizes a DataSource (base64 / buffer / url / file-id) into a neutral shape,
then each provider adapter maps it to its own wire field:
  - OpenAI: `{ image_url }` (data-URL) or `{ file_id }`
  - xAI:    `{ url }` (data-URL) or `{ file_id }`
  - Google: `inline_data {mime_type,data}` or `file_data {file_uri}`

Transposed from `unified-library-ts/src/util/source-image.ts`.

A `DataSource` is a plain mapping here, with snake_case keys (`mime_type`,
`file_id`, `ref_id`) to match the reviewed Python examples. Everything on the
PROVIDER side of the boundary -- `inline_data`, `file_uri`, `bytesBase64Encoded`,
`gcsUri`, `image_url` -- is a wire literal and is spelled exactly as the
TypeScript spells it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base64 import base64_to_bytes, bytes_to_base64
from .image_mime import sniff_image_mime

#: The discriminated union `DataSource` from `src/llm/types/messages.ts`, which
#: lands with that file. Until then it is read structurally, as the interpreter
#: reads a request.
DataSource = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NormalizedImageRef:
    """`interface NormalizedImageRef` -- every field optional, as written."""

    #: Raw base64 (no `data:` prefix), when inline.
    base64: str | None = None
    mime_type: str | None = None
    #: A remote URL, when the source is a URL.
    url: str | None = None
    #: A provider Files-API id, when the source is an uploaded file.
    file_id: str | None = None


def _mime_from_base64_prefix(b64: str) -> str | None:
    """Sniff a mime from a base64 string by decoding only its leading magic bytes.

    Decodes a 4-char-aligned prefix and swallows any decode error -> None.
    """
    try:
        n = min(24, len(b64) - (len(b64) % 4))
        if n < 4:
            return None
        return sniff_image_mime(base64_to_bytes(b64[:n]))
    except Exception:  # noqa: BLE001 -- `catch {}`: any decode failure means "no mime"
        return None


def normalize_image_source(src: DataSource) -> NormalizedImageRef:
    """Collapse any DataSource into base64 / url / fileId.

    The declared mime is cross-checked against the actual bytes (and corrected on
    mismatch) so a mislabeled source image -- e.g. JPEG bytes tagged "image/png"
    -- doesn't get rejected by strict validators downstream (Google Veo, OpenAI
    edits).
    """
    kind = src["type"]
    if kind == "base64":
        data = src["data"]
        sniffed = _mime_from_base64_prefix(data)
        return NormalizedImageRef(
            base64=data, mime_type=src.get("mime_type") if sniffed is None else sniffed
        )
    if kind == "buffer":
        data = src["data"]
        sniffed = sniff_image_mime(data)
        return NormalizedImageRef(
            base64=bytes_to_base64(data),
            mime_type=src.get("mime_type") if sniffed is None else sniffed,
        )
    if kind == "url":
        return NormalizedImageRef(url=src["url"])
    if kind == "file":
        return NormalizedImageRef(file_id=src["file_id"])
    if kind == "provider_ref":
        return NormalizedImageRef(file_id=src["ref_id"], mime_type=src.get("mime_type"))
    if kind == "path":
        raise ValueError(
            "media source image: `path` DataSource is not supported here "
            "-- read the file and pass base64/buffer."
        )
    # The TypeScript switch is exhaustive over the union and carries no default,
    # so an unrecognised `type` falls off the end and yields `undefined`. Kept as
    # a fall-through for the same reason: adding a raise here would be a new
    # behaviour, not a transposed one.
    return None  # type: ignore[return-value]


def to_data_url(ref: NormalizedImageRef) -> str:
    """Build a `data:<mime>;base64,...` URL (or pass a plain URL through)."""
    if ref.url:
        return ref.url
    if ref.base64:
        mime = "image/png" if ref.mime_type is None else ref.mime_type
        return f"data:{mime};base64,{ref.base64}"
    raise ValueError("media source image: needs inline base64 or a url (got a file id only).")


def openai_image_ref(ref: NormalizedImageRef) -> dict[str, str]:
    """OpenAI image-ref object (`/v1/images/edits` images[], video input_reference)."""
    return {"file_id": ref.file_id} if ref.file_id else {"image_url": to_data_url(ref)}


def xai_image_ref(ref: NormalizedImageRef) -> dict[str, str]:
    """xAI image-ref object (`/v1/images/edits` image, video image)."""
    return {"file_id": ref.file_id} if ref.file_id else {"url": to_data_url(ref)}


def xai_video_ref(src: DataSource) -> dict[str, str]:
    """xAI video-ref object (`/v1/videos/extensions` + `/v1/videos/edits` `video`
    field) -- `{ url }` (public URL or base64 data-URL) or `{ file_id }`.

    Kept separate from the image path so it doesn't run image mime-sniffing over
    video bytes; the video mime is taken from the DataSource as declared.
    """
    kind = src["type"]
    if kind == "url":
        return {"url": src["url"]}
    if kind == "file":
        return {"file_id": src["file_id"]}
    if kind == "provider_ref":
        return {"file_id": src["ref_id"]}
    if kind == "base64":
        return {"url": f"data:{src['mime_type']};base64,{src['data']}"}
    if kind == "buffer":
        return {"url": f"data:{src['mime_type']};base64,{bytes_to_base64(src['data'])}"}
    if kind == "path":
        raise ValueError(
            "media source video: `path` DataSource is not supported here "
            "-- read the file and pass base64/buffer."
        )
    return None  # type: ignore[return-value]


def google_image_part(ref: NormalizedImageRef) -> dict[str, Any]:
    """Google generateContent image part (inline base64 or Files-API file_uri)."""
    mime_type = "image/png" if ref.mime_type is None else ref.mime_type
    if ref.base64:
        return {"inline_data": {"mime_type": mime_type, "data": ref.base64}}
    uri = ref.file_id if ref.url is None else ref.url
    return {"file_data": {"file_uri": uri, "mime_type": mime_type}}


def google_veo_image(ref: NormalizedImageRef) -> dict[str, Any]:
    """Google Veo instance image (`:predictLongRunning` instances[].image).

    The predict API uses the Image proto -- `bytesBase64Encoded` + `mimeType` --
    NOT the `inlineData` shape of generateContent (Veo rejects inlineData with a
    400). The Gemini Developer API accepts only inline bytes here (no gcsUri /
    file URI), so the URL fallback exists for Vertex-style callers only.
    """
    mime_type = "image/png" if ref.mime_type is None else ref.mime_type
    if ref.base64:
        return {"bytesBase64Encoded": ref.base64, "mimeType": mime_type}
    return {"gcsUri": ref.file_id if ref.url is None else ref.url, "mimeType": mime_type}


__all__ = [
    "DataSource",
    "NormalizedImageRef",
    "google_image_part",
    "google_veo_image",
    "normalize_image_source",
    "openai_image_ref",
    "to_data_url",
    "xai_image_ref",
    "xai_video_ref",
]
