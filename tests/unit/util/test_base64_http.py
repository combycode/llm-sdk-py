"""Translated from `unified-library-ts/tests/unit/util/base64-http.test.ts`.

PARTIAL by design: that file covers `src/util/base64.ts` AND `src/util/http.ts`.
Only base64 is ported in this batch, so the `anySignal` / `header` /
`parseIntHeader` / `isStreamBody` describes are carried here as skips rather than
dropped -- they are the honest gap, and they land with `src/util/http.ts`.
"""

from __future__ import annotations

import pytest

from combycode_llm_sdk.util.base64 import base64_to_bytes, base64_to_utf8, bytes_to_base64
from combycode_llm_sdk.util.http import header, parse_int_header


class TestBase64:
    def test_round_trips_bytes(self) -> None:
        # base64-http.test.ts:17-18
        data = bytes([0, 1, 2, 250, 251, 255])
        assert base64_to_bytes(bytes_to_base64(data)) == data

    @pytest.mark.skip(
        reason="TS-specific: asserts the btoa fallback taken when globalThis.Buffer is "
        "deleted. Python has a single base64 implementation and no fast path to "
        "disable, so there is no second path to compare against "
        "-- base64-http.test.ts:21"
    )
    def test_the_btoa_fallback_produces_the_same_string(self) -> None:
        raise AssertionError("unreachable")

    def test_handles_bytes_above_0x7f_without_mangling_them(self) -> None:
        # base64-http.test.ts:30-35. The `g.Buffer = undefined` line has no Python
        # analogue (single implementation); the assertion it guards -- that bytes
        # above 0x7f survive a round trip -- transposes unchanged.
        data = bytes([0x80, 0xFF, 0xC3, 0xA9])
        assert base64_to_bytes(bytes_to_base64(data)) == data

    def test_base64_to_utf8_decodes_multi_byte_characters(self) -> None:
        # base64-http.test.ts:39
        assert base64_to_utf8(bytes_to_base64("héllo ✓".encode())) == "héllo ✓"


@pytest.mark.skip(
    reason="not ported: anySignal combines AbortSignals for the fetch layer, and "
    "lands with the network engine that owns one -- base64-http.test.ts:43"
)
class TestAnySignal:
    def test_placeholder(self) -> None:
        raise AssertionError("unreachable")


class TestHttpOddsAndEnds:
    """base64-http.test.ts:72.

    Two of its three cases run: `header` and `parse_int_header` are ported.
    `isStreamBody` is not -- it asks whether a request body is a `ReadableStream`,
    which is a question only the fetch layer has, and it lands with the engine.
    """

    def test_header_lookup_is_case_insensitive(self) -> None:
        # base64-http.test.ts:74-77. Header names are case-insensitive (RFC 9110
        # 5.1) and a plain dict is not, so every response-header read goes
        # through this rather than guessing the casing at the call site.
        assert header({"X-GOOG-UPLOAD-URL": "u"}, "x-goog-upload-url") == "u"
        assert header({"x-goog-upload-url": "u"}, "X-Goog-Upload-Url") == "u"
        assert header({}, "missing") is None

    def test_parse_int_header_returns_none_for_absent_and_non_numeric(self) -> None:
        # base64-http.test.ts:80-84
        assert parse_int_header({"a": "42"}, "a") == 42
        assert parse_int_header({}, "a") is None
        assert parse_int_header({"a": ""}, "a") is None
        assert parse_int_header({"a": "abc"}, "a") is None
