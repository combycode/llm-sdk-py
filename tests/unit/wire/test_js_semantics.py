"""The two places Python disagrees with JavaScript on the wire.

PORTING.md allows new tests only for genuinely Python-only surface, and these
two helpers are exactly that: they exist because the transposition crosses a
language boundary where the obvious Python spelling is silently wrong.

Neither is covered by the TypeScript suite -- there is nothing to cover there,
since JavaScript simply behaves this way -- which makes them the least verified
part of the port and the most worth pinning.
"""

from __future__ import annotations

import pytest

from combycode_llm_sdk.wire.interpreter import MISSING, get_path, js_string, js_truthy


class TestJsTruthy:
    """`Boolean(v)`, which decides whether a `truthy:` guard fires."""

    @pytest.mark.parametrize("value", [[], {}, " ", "0", "false", -1, 0.5, object()])
    def test_javascript_says_these_are_truthy(self, value: object) -> None:
        assert js_truthy(value) is True

    def test_an_empty_collection_is_TRUTHY_which_is_the_whole_reason_this_exists(self) -> None:
        # Python's bool([]) is False. A spec guarded by `truthy: tools` with
        # `tools: []` emits the field in TypeScript, so it must emit it here --
        # otherwise the same spec builds two different requests.
        assert js_truthy([]) is True
        assert bool([]) is False, "if Python ever changes this, the helper is redundant"

    @pytest.mark.parametrize("value", [None, False, 0, 0.0, -0.0, "", MISSING, float("nan")])
    def test_javascript_says_these_are_falsy(self, value: object) -> None:
        assert js_truthy(value) is False


class TestJsString:
    """`String(v)`, for values that reach a header, a URL or a joined string."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, "true"),
            (False, "false"),
            (None, "null"),
            (MISSING, "null"),
            ("already", "already"),
            (7, "7"),
            (7.0, "7"),
            (7.5, "7.5"),
        ],
    )
    def test_it_matches_javascript(self, value: object, expected: str) -> None:
        assert js_string(value) == expected

    def test_a_bool_does_not_become_a_python_bool_spelling(self) -> None:
        # The failure this prevents: a header reading `True`, which no provider
        # recognises, from Python's str(True).
        assert js_string(True) != str(True)

    def test_a_whole_float_loses_its_trailing_zero_as_javascript_does(self) -> None:
        # `String(7.0)` is "7" in JavaScript and "7.0" in Python. In a URL path
        # segment or a query value that is a different request.
        assert js_string(7.0) == "7"
        assert str(7.0) == "7.0"


class TestGetPathArrayIndex:
    """`arr["0"]` in JavaScript, `arr[0]` in Python.

    A JS array IS an object, so the TypeScript `cur[part]` resolves a numeric
    segment without knowing it is one. Python has no such coercion, and the
    transposition originally fell through to `getattr`, which returns MISSING --
    silently, with the field simply absent from the built value.

    Latent for the whole request port: not one of the 150 request specs uses a
    numeric segment. Seven of the response and stream specs do.
    """

    def test_a_numeric_segment_indexes_a_list(self) -> None:
        root = {"raw": {"choices": [{"message": {"tool_calls": [1, 2]}}]}}
        assert get_path(root, "raw.choices.0.message.tool_calls") == [1, 2]

    def test_an_out_of_range_index_is_MISSING_not_an_error(self) -> None:
        # JavaScript gives undefined; raising here would take down a whole parse
        # over a response that simply had fewer items than the spec expected.
        assert get_path({"a": [1]}, "a.5") is MISSING

    def test_a_non_numeric_segment_on_a_list_is_MISSING(self) -> None:
        # `arr["length"]` is 1 in JavaScript. Emulating that would be MORE
        # faithful, but no spec uses it, and inventing behaviour nothing needs
        # is how a port grows differences.
        assert get_path({"a": [1, 2]}, "a.length") is MISSING

    def test_a_non_ascii_digit_does_not_index(self) -> None:
        # str.isdigit() is true for other numeral systems and int() accepts
        # them, so "\u0665" would index element 5 from a character no provider
        # ever sent.
        assert get_path({"a": [0, 1, 2, 3, 4, 5, 6]}, "a.\u0665") is MISSING

    def test_a_dict_with_a_numeric_key_still_wins(self) -> None:
        # Mappings are checked first, so a provider that really does key by "0"
        # is read as a mapping rather than indexed as a list.
        assert get_path({"a": {"0": "by-key"}}, "a.0") == "by-key"
