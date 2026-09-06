"""Translated from `unified-library-ts/tests/unit/util/source-image.test.ts`.

Source/reference image + video normalisation.

Every DataSource variant has to reach a provider field, and the failure mode when
one does not is silent: a `provider_ref` that falls through returns `undefined`
and the provider receives `{"file_id": null}` -- a 400 far from the cause, or
worse, a request that renders without the reference image at all. Each case is
asserted here, including the two that throw.

One transposition, applied throughout: a `DataSource` is a plain mapping here and
its keys are snake_case (`mime_type`, `file_id`, `ref_id`), matching the reviewed
Python examples. PROVIDER-facing keys (`inline_data`, `file_uri`,
`bytesBase64Encoded`, `gcsUri`, `image_url`, ...) are wire literals and are left
exactly as the TypeScript writes them.
"""

from __future__ import annotations

import pytest

from combycode_llm_sdk.util.base64 import bytes_to_base64
from combycode_llm_sdk.util.source_image import (
    NormalizedImageRef,
    google_image_part,
    google_veo_image,
    normalize_image_source,
    openai_image_ref,
    to_data_url,
    xai_image_ref,
    xai_video_ref,
)

#: A one-pixel PNG: the 8-byte signature is enough for the mime sniffer.
PNG_BYTES = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13])
PNG_B64 = bytes_to_base64(PNG_BYTES)
#: JPEG magic: FF D8 FF.
JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 0x4A, 0x46, 0x49, 0x46, 0, 1])


class TestNormalizeImageSource:
    def test_base64_sniffs_the_real_mime_and_overrides_a_mislabeled_one(self) -> None:
        # source-image.test.ts:29-35
        jpeg_as_png = normalize_image_source(
            {"type": "base64", "data": bytes_to_base64(JPEG_BYTES), "mime_type": "image/png"}
        )
        assert jpeg_as_png.mime_type == "image/jpeg"
        assert jpeg_as_png.base64 == bytes_to_base64(JPEG_BYTES)

    def test_base64_keeps_the_declared_mime_when_the_bytes_say_nothing(self) -> None:
        # source-image.test.ts:39-42
        ref = normalize_image_source(
            {
                "type": "base64",
                "data": bytes_to_base64(bytes([1, 2, 3, 4, 5, 6, 7, 8])),
                "mime_type": "image/heic",
            }
        )
        assert ref.mime_type == "image/heic"

    def test_base64_undecodable_input_falls_back_to_the_declared_mime(self) -> None:
        # source-image.test.ts:48. Four characters, but not valid base64 -- the
        # decode throws, and the sniff must swallow it rather than take down the
        # whole request build.
        ref = normalize_image_source({"type": "base64", "data": "!!!!", "mime_type": "image/webp"})
        assert ref.mime_type == "image/webp"

    def test_base64_a_payload_shorter_than_one_quantum_is_not_sniffed(self) -> None:
        # source-image.test.ts:54
        ref = normalize_image_source({"type": "base64", "data": "ab", "mime_type": "image/gif"})
        assert ref.mime_type == "image/gif"

    def test_buffer_encodes_to_base64_and_sniffs_the_mime_from_the_bytes(self) -> None:
        # source-image.test.ts:60-62
        r = normalize_image_source(
            {"type": "buffer", "data": PNG_BYTES, "mime_type": "application/octet-stream"}
        )
        assert r.base64 == PNG_B64
        assert r.mime_type == "image/png"

    def test_url_passes_through_untouched(self) -> None:
        # source-image.test.ts:66-68
        assert normalize_image_source({"type": "url", "url": "https://x/y.png"}) == (
            NormalizedImageRef(url="https://x/y.png")
        )

    def test_file_to_file_id(self) -> None:
        # source-image.test.ts:72
        assert normalize_image_source({"type": "file", "file_id": "file-123"}) == (
            NormalizedImageRef(file_id="file-123")
        )

    def test_provider_ref_to_file_id_plus_its_declared_mime(self) -> None:
        # source-image.test.ts:76-78 -- this case used to fall through.
        assert normalize_image_source(
            {"type": "provider_ref", "ref_id": "files/abc", "mime_type": "image/png"}
        ) == NormalizedImageRef(file_id="files/abc", mime_type="image/png")

    def test_path_throws(self) -> None:
        # source-image.test.ts:82-84 -- the one source that cannot be resolved here.
        with pytest.raises(ValueError, match="DataSource is not supported"):
            normalize_image_source({"type": "path", "path": "/tmp/a.png", "mime_type": "image/png"})


class TestToDataUrlAndProviderRefObjects:
    def test_a_url_wins_over_inline_bytes(self) -> None:
        # source-image.test.ts:90
        assert to_data_url(NormalizedImageRef(url="https://x/y.png", base64=PNG_B64)) == (
            "https://x/y.png"
        )

    def test_inline_bytes_become_a_data_url_defaulting_the_mime(self) -> None:
        # source-image.test.ts:94-95
        assert to_data_url(NormalizedImageRef(base64="AAAA")) == "data:image/png;base64,AAAA"
        assert to_data_url(NormalizedImageRef(base64="AAAA", mime_type="image/jpeg")) == (
            "data:image/jpeg;base64,AAAA"
        )

    def test_a_file_id_alone_cannot_become_a_data_url(self) -> None:
        # source-image.test.ts:99
        with pytest.raises(ValueError, match="needs inline base64 or a url"):
            to_data_url(NormalizedImageRef(file_id="file-1"))

    def test_openai_prefers_file_id_else_image_url(self) -> None:
        # source-image.test.ts:103-104
        assert openai_image_ref(NormalizedImageRef(file_id="file-1")) == {"file_id": "file-1"}
        assert openai_image_ref(NormalizedImageRef(url="https://x/y.png")) == (
            {"image_url": "https://x/y.png"}
        )

    def test_xai_prefers_file_id_else_url_not_image_url(self) -> None:
        # source-image.test.ts:108-109 -- a different field name from OpenAI's.
        assert xai_image_ref(NormalizedImageRef(file_id="file-1")) == {"file_id": "file-1"}
        assert xai_image_ref(NormalizedImageRef(url="https://x/y.png")) == (
            {"url": "https://x/y.png"}
        )

    def test_google_generate_content_inline_data_for_bytes_file_data_for_a_ref(self) -> None:
        # source-image.test.ts:113-121
        assert google_image_part(NormalizedImageRef(base64="AAAA", mime_type="image/jpeg")) == {
            "inline_data": {"mime_type": "image/jpeg", "data": "AAAA"}
        }
        assert google_image_part(NormalizedImageRef(url="https://x/y.png")) == {
            "file_data": {"file_uri": "https://x/y.png", "mime_type": "image/png"}
        }
        assert google_image_part(
            NormalizedImageRef(file_id="files/abc", mime_type="image/webp")
        ) == {"file_data": {"file_uri": "files/abc", "mime_type": "image/webp"}}

    def test_google_veo_uses_the_image_proto_never_inline_data(self) -> None:
        # source-image.test.ts:125-132
        assert google_veo_image(NormalizedImageRef(base64="AAAA", mime_type="image/jpeg")) == {
            "bytesBase64Encoded": "AAAA",
            "mimeType": "image/jpeg",
        }
        assert google_veo_image(NormalizedImageRef(url="gs://bucket/x.png")) == {
            "gcsUri": "gs://bucket/x.png",
            "mimeType": "image/png",
        }


class TestXaiVideoRef:
    """Every DataSource variant -- source-image.test.ts:136."""

    def test_url(self) -> None:
        # source-image.test.ts:138
        assert xai_video_ref({"type": "url", "url": "https://x/v.mp4"}) == (
            {"url": "https://x/v.mp4"}
        )

    def test_file(self) -> None:
        # source-image.test.ts:142
        assert xai_video_ref({"type": "file", "file_id": "file-9"}) == {"file_id": "file-9"}

    def test_provider_ref_uses_ref_id(self) -> None:
        # source-image.test.ts:146-148
        assert xai_video_ref({"type": "provider_ref", "ref_id": "file-ref-9"}) == (
            {"file_id": "file-ref-9"}
        )

    def test_base64_keeps_the_declared_mime_no_image_sniffing_on_video_bytes(self) -> None:
        # source-image.test.ts:152-154
        assert xai_video_ref({"type": "base64", "data": "QUJD", "mime_type": "video/mp4"}) == (
            {"url": "data:video/mp4;base64,QUJD"}
        )

    def test_buffer_becomes_a_data_url_with_the_declared_mime(self) -> None:
        # source-image.test.ts:159-160
        assert xai_video_ref(
            {"type": "buffer", "data": bytes([65, 66, 67]), "mime_type": "video/webm"}
        ) == {"url": "data:video/webm;base64,QUJD"}

    def test_path_throws(self) -> None:
        # source-image.test.ts:164
        with pytest.raises(ValueError, match="DataSource is not supported"):
            xai_video_ref({"type": "path", "path": "/tmp/v.mp4", "mime_type": "video/mp4"})
