"""Detect an image's MIME type from its leading magic bytes.

Providers sometimes return a generated image with the WRONG declared mime (e.g.
xAI hands back JPEG bytes labeled "image/png"). When that image is then forwarded
as a source for image-to-video / image-edit, strict validators like Google Veo
compare the declared mime against the actual bytes and reject the request with a
400. Sniffing the bytes lets us self-correct the label.

Transposed from `unified-library-ts/src/util/image-mime.ts`.

NOTE ON OWNERSHIP: this file is not in the foundation batch's assigned list. It
is ported here because `source_image.py` imports it directly and cannot be tested
-- or used -- without it. Its own TypeScript test is translated alongside, so it
arrives pinned rather than as an unchecked stub.
"""

from __future__ import annotations


def sniff_image_mime(b: bytes) -> str | None:
    # JPEG: FF D8 FF
    if len(b) >= 3 and b[0] == 0xFF and b[1] == 0xD8 and b[2] == 0xFF:
        return "image/jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if len(b) >= 8 and b[0] == 0x89 and b[1] == 0x50 and b[2] == 0x4E and b[3] == 0x47:
        return "image/png"
    # GIF: "GIF8"
    if len(b) >= 4 and b[0] == 0x47 and b[1] == 0x49 and b[2] == 0x46 and b[3] == 0x38:
        return "image/gif"
    # WEBP: "RIFF"????"WEBP"
    if (
        len(b) >= 12
        and b[0] == 0x52
        and b[1] == 0x49
        and b[2] == 0x46
        and b[3] == 0x46
        and b[8] == 0x57
        and b[9] == 0x45
        and b[10] == 0x42
        and b[11] == 0x50
    ):
        return "image/webp"
    return None


__all__ = ["sniff_image_mime"]
