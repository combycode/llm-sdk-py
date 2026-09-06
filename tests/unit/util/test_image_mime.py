"""Translated from `unified-library-ts/tests/unit/util/image-mime.test.ts`."""

from __future__ import annotations

from combycode_llm_sdk.util.image_mime import sniff_image_mime


class TestSniffImageMime:
    def test_detects_jpeg(self) -> None:
        # image-mime.test.ts:6
        assert sniff_image_mime(bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])) == "image/jpeg"

    def test_detects_png(self) -> None:
        # image-mime.test.ts:10
        assert (
            sniff_image_mime(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]))
            == "image/png"
        )

    def test_detects_gif_and_webp(self) -> None:
        # image-mime.test.ts:16-18
        assert sniff_image_mime(bytes([0x47, 0x49, 0x46, 0x38, 0x39, 0x61])) == "image/gif"
        webp = bytes([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4, 0x57, 0x45, 0x42, 0x50])
        assert sniff_image_mime(webp) == "image/webp"

    def test_returns_none_for_unknown_or_too_short_data(self) -> None:
        # image-mime.test.ts:22-23
        assert sniff_image_mime(bytes([0x00, 0x01, 0x02])) is None
        assert sniff_image_mime(b"") is None
