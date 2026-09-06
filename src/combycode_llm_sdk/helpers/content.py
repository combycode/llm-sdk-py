"""Turn a path, a URL or raw bytes into a ready-to-send content part.

Transposed from `unified-library-ts/src/helpers/content.ts`.

The dirty work of reading, fetching, base64-encoding and MIME-detecting, so a
caller can write `attachments=["report.pdf"]` and get a `document` part rather
than assembling one. Accepts, per the API contract:

- a `str` or `pathlib.Path` -- read from disk, or fetched when it is an
  `http(s)://` URL;
- `bytes` -- used directly.

`pathlib.Path` is the Python addition, and the contract asks for it by name:
"`pathlib.Path` is what Python users actually hold. Accepting only strings would
make every caller write `str(path)`."

The MIME decides the PART: pdf and text become `document`, audio becomes
`audio`, video becomes `video`, and anything else is an `image` -- which is why
`load_content` is what attachments use and `load_image_content` is the narrower
one that forces an image.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..util.base64 import bytes_to_base64
from ..util.image_mime import sniff_image_mime

#: Extension -> MIME. Deliberately a small closed table rather than
#: `mimetypes.guess_type`: the stdlib's answer depends on the machine's registry
#: (on Windows it reads HKEY_CLASSES_ROOT), so the same attachment could be sent
#: as `image/png` on one developer's box and `image/x-png` on another's.
EXT_TO_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}

_DEFAULT_MIME = "image/png"


def _ext_of(path: str) -> str:
    """Lowercased extension including the dot, or empty."""
    slash = max(path.rfind("/"), path.rfind("\\"))
    dot = path.rfind(".")
    return path[dot:].lower() if dot > slash else ""


def _sniff(data: bytes) -> str | None:
    """Magic bytes, for a source that carries no name.

    Images only -- `sniff_image_mime` is the shared detector the media layer
    already uses, and a wrong guess for audio would pick the wrong part type
    rather than merely the wrong subtype.
    """
    return sniff_image_mime(data)


def _part_for(mime: str, data: bytes) -> dict[str, Any]:
    source = {"type": "base64", "mimeType": mime, "data": bytes_to_base64(data)}
    if mime in ("application/pdf", "text/plain"):
        return {"type": "document", "source": source}
    if mime.startswith("audio/"):
        return {"type": "audio", "source": source}
    if mime.startswith("video/"):
        return {"type": "video", "source": source}
    return {"type": "image", "source": source}


def _mime_from_url(url: str) -> str | None:
    # Strip a query string before looking at the extension: `?v=2` is not an
    # extension, and `photo.png?v=2` is a real URL shape.
    return EXT_TO_MIME.get(_ext_of(url.split("?")[0].split("#")[0]))


def load_bytes(
    source: str | Path | bytes, options: Mapping[str, Any] | None = None
) -> tuple[bytes, str]:
    """The bytes and the MIME, however the source was named.

    Synchronous. A URL is fetched with the same transport the library uses
    elsewhere rather than a second HTTP client.
    """
    options = options or {}
    override = options.get("mimeType")

    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        return data, override or _sniff(data) or _DEFAULT_MIME

    text = str(source)
    if text.startswith(("http://", "https://")):
        from ..transport import TransportRequest, http_transport

        headers = {"user-agent": options["userAgent"]} if options.get("userAgent") else {}
        response = http_transport()(
            TransportRequest(
                url=text, method="GET", headers=headers, response_type="arraybuffer"
            )
        )
        if response.status >= 400:
            raise RuntimeError(f"load_content: fetch failed ({response.status}) for {text}")
        data = bytes(response.body or b"")
        mime = (
            override
            or _mime_from_url(text)
            or response.headers.get("content-type", "").split(";")[0].strip()
            or _sniff(data)
            or _DEFAULT_MIME
        )
        return data, mime

    data = Path(source).read_bytes()
    return data, override or EXT_TO_MIME.get(_ext_of(text)) or _sniff(data) or _DEFAULT_MIME


def load_content(
    source: str | Path | bytes, options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Any inline part -- image, document, audio or video -- chosen by MIME."""
    data, mime = load_bytes(source, options)
    return _part_for(mime, data)


def load_image_content(
    source: str | Path | bytes, options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """An IMAGE part, whatever the MIME says.

    The narrower twin of `load_content`, for a caller who knows what they have
    and wants it sent as an image regardless of what sniffing concluded.
    """
    data, mime = load_bytes(source, options)
    return {
        "type": "image",
        "source": {"type": "base64", "mimeType": mime, "data": bytes_to_base64(data)},
    }


__all__ = ["EXT_TO_MIME", "load_bytes", "load_content", "load_image_content"]
