"""SSE framing: where a message ends, and what a data line really contains.

Three faults the block-based parser had, each of which only shows on real
traffic:

1. It re-split the WHOLE accumulated buffer on every chunk -- quadratic in the
   size of one event, and a base64 partial image is exactly that shape.
2. Its boundary pattern knew only LF-LF, CRLF-CRLF and CR-CR, so a MIXED
   terminator was not a boundary and two events arrived as one.
3. `data:` values were `lstrip()`ed, which eats every leading space. The spec
   strips exactly ONE, so a payload beginning with whitespace came back changed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from combycode_llm_sdk.network.sse import parse_sse_stream


def _stream(*chunks: str) -> Iterator[bytes]:
    return iter(c.encode("utf-8") for c in chunks)


def _data(*chunks: str) -> list[str]:
    return [e["data"] for e in parse_sse_stream(_stream(*chunks))]


class TestLineTerminators:
    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("LF", "data: one\n\ndata: two\n\n"),
            ("CRLF", "data: one\r\n\r\ndata: two\r\n\r\n"),
            ("CR", "data: one\r\rdata: two\r\r"),
        ],
    )
    def test_separates_events(self, label: str, body: str) -> None:
        assert _data(body) == ["one", "two"]

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("CRLF then LF", "data: one\r\n\ndata: two\r\n\n"),
            ("LF then CRLF", "data: one\n\r\ndata: two\n\r\n"),
            ("LF then CR", "data: one\n\rdata: two\n\r"),
        ],
    )
    def test_separates_events_on_a_mixed_terminator(self, label: str, body: str) -> None:
        assert _data(body) == ["one", "two"]

    def test_a_crlf_straddling_two_chunks_does_not_end_the_event(self) -> None:
        # The CR ends the line; the LF arriving next is its partner, NOT a blank
        # line. Treating it as one would dispatch after `one` and lose `two`.
        assert _data("data: one\r", "\ndata: two\r\n\r\n") == ["one\ntwo"]

    def test_two_data_lines_with_no_blank_line_are_one_event(self) -> None:
        assert _data("data: one\r\ndata: two\r\n\r\n") == ["one\ntwo"]

    def test_delivers_a_final_event_that_never_got_its_blank_line(self) -> None:
        assert _data("data: last\n") == ["last"]

    def test_events_split_across_chunk_boundaries(self) -> None:
        assert _data('data: {"a":1}\n', '\ndata: {"b":2}\n\n') == ['{"a":1}', '{"b":2}']


class TestExactlyOneLeadingSpace:
    def test_keeps_the_second_space(self) -> None:
        assert _data("data:  two spaces\n\n") == [" two spaces"]

    def test_strips_the_single_conventional_space(self) -> None:
        assert _data("data: normal\n\n") == ["normal"]

    def test_keeps_a_value_with_no_space_at_all(self) -> None:
        assert _data("data:tight\n\n") == ["tight"]

    def test_keeps_leading_whitespace_inside_a_payload(self) -> None:
        assert _data('data:   {"a":1}\n\n') == ['  {"a":1}']


class TestTheThingsThatYieldNoEvent:
    def test_done_is_not_an_event(self) -> None:
        assert _data("data: [DONE]\n\n") == []

    def test_a_comment_is_ignored_even_when_it_contains_a_colon(self) -> None:
        assert _data(": ping: still here\ndata: real\n\n") == ["real"]

    def test_a_message_with_no_data_field_yields_nothing(self) -> None:
        assert _data("event: ping\n\n") == []


class TestALargeEvent:
    def test_reassembles_a_multi_megabyte_line_delivered_in_many_chunks(self) -> None:
        # The old parser re-split the whole buffer per chunk, which is quadratic
        # in one event's size. This asserts the RESULT is intact; the cost is the
        # reason the decoder is incremental.
        payload = "x" * 2_000_000
        chunks = ["data: "]
        chunks += [payload[i : i + 64_000] for i in range(0, len(payload), 64_000)]
        chunks.append("\n\n")
        out = _data(*chunks)
        assert len(out) == 1
        assert len(out[0]) == len(payload)


class TestTheVendoredXaiImageQuality:
    """The wire spec and the catalog are vendored from the TypeScript tree, so
    this asserts the SYNC carried the change rather than re-testing the rule."""

    def test_quality_is_recorded_only_where_the_sdk_says_it_works(self) -> None:
        from combycode_llm_sdk.catalog.catalog import ModelCatalog

        catalog = ModelCatalog()
        catalog.load_provider_defaults()
        two = catalog.get("xai", "grok-imagine-image-2.0")
        if two is None:
            return  # a trimmed catalog build
        quality = dict(two).get("mediaParams", {}).get("quality")
        assert quality is not None
        assert quality["values"] == ["low", "medium"]
        assert quality["default"] == "medium"

        plain = catalog.get("xai", "grok-imagine-image")
        if plain is not None:
            assert dict(plain).get("mediaParams", {}).get("quality") is None
