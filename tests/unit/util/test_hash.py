"""Translated from `unified-library-ts/tests/unit/util/hash.test.ts`.

Assertions transposed one for one; each carries the TypeScript line it came from.
"""

from __future__ import annotations

import re

import pytest

from combycode_llm_sdk.util.hash import fnv1a32, fnv1a32_hex


class TestFnv1a32:
    def test_matches_the_published_fnv_1a_32_bit_vectors(self) -> None:
        # hash.test.ts:7-9 -- reference values for the standard offset basis / prime.
        assert fnv1a32("") == 0x811C9DC5
        assert fnv1a32("a") == 0xE40C292C
        assert fnv1a32("foobar") == 0xBF9CF968

    @pytest.mark.parametrize("s", ["", "a", "hello world", "x" * 1000])
    def test_is_deterministic_and_unsigned(self, s: str) -> None:
        # hash.test.ts:13-17
        assert fnv1a32(s) == fnv1a32(s)
        assert fnv1a32(s) >= 0
        assert fnv1a32(s) <= 0xFFFFFFFF

    def test_renders_8_hex_characters_zero_padded(self) -> None:
        # hash.test.ts:21-24
        assert fnv1a32_hex("foobar") == "bf9cf968"
        for s in ["", "a", "zz", "batch"]:
            assert re.fullmatch(r"[0-9a-f]{8}", fnv1a32_hex(s))


class TestFnv1a32Utf16Bridge:
    """PYTHON-ONLY. A JS/Python semantic bridge, in the sense PORTING.md allows.

    TypeScript hashes `charCodeAt(i)` -- UTF-16 code UNITS. Python's `for ch in s`
    yields code POINTS, so an astral character (one code point, two UTF-16 units)
    hashes to a different number under the obvious Python spelling. The value
    reaches the wire through `xaiBatchName`, so two SDKs would name the same batch
    differently. There is nothing to translate here: JavaScript simply behaves this
    way and its suite has no reason to assert it.
    """

    def test_an_astral_character_hashes_as_its_two_utf16_units(self) -> None:
        # U+1F600 == surrogate pair D83D DE00. Hash it the way TypeScript would.
        expected = 0x811C9DC5
        for unit in (0xD83D, 0xDE00):
            expected = ((expected ^ unit) * 0x01000193) & 0xFFFFFFFF
        assert fnv1a32("\U0001f600") == expected

        # …and NOT the way a code-point loop would.
        naive = ((0x811C9DC5 ^ 0x1F600) * 0x01000193) & 0xFFFFFFFF
        assert fnv1a32("\U0001f600") != naive
