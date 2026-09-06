"""Exporting a conversation as Markdown, and as an archive.

The two renderings differ only in where media goes, so most of what can break is
silent: a blob inlined as a multi-megabyte data-URL where a file was wanted, an
archive stamped with the wall clock so two identical exports differ, a mime type
from a provider response reaching a path inside something someone extracts.
"""

from __future__ import annotations

import io
import sys
import zipfile
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.helpers.conversation_export import (
    FIXED_TIMESTAMP,
    conversation_to_markdown,
    conversation_to_zip,
    ext_of,
)

PNG = "iVBORw0KGgo="  # not a real image; the bytes are never decoded as one

TALK: list[dict[str, Any]] = [
    {"role": "user", "content": "what is this?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "A chart."},
            {"type": "image", "source": {"type": "base64", "mimeType": "image/png", "data": PNG}},
        ],
    },
]


class TestMarkdown:
    def test_each_turn_is_a_section(self) -> None:
        out = conversation_to_markdown([{"role": "user", "content": "hi"}])
        assert "## user" in out
        assert out.endswith("\n")

    def test_a_title_becomes_the_heading(self) -> None:
        out = conversation_to_markdown([{"role": "user", "content": "hi"}], title="Bug 41")
        assert out.startswith("# Bug 41")

    def test_media_is_inlined_so_one_file_is_the_whole_record(self) -> None:
        out = conversation_to_markdown(TALK)
        assert f"![image](data:image/png;base64,{PNG})" in out

    def test_it_can_be_asked_for_the_transcript_without_the_attachments(self) -> None:
        out = conversation_to_markdown(TALK, inline_media=False)
        assert "[image]" in out
        assert "base64" not in out

    def test_a_tool_call_is_readable_and_its_result_is_not_padded(self) -> None:
        # A call is something a person reads in a transcript; a result is a
        # payload. The TypeScript renders them that way and so does this.
        out = conversation_to_markdown(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "name": "search", "arguments": {"q": "berlin"}},
                        {"type": "tool_result", "content": {"hits": 2}},
                    ],
                }
            ]
        )
        assert '```tool_call search\n{\n  "q": "berlin"\n}\n```' in out
        assert '```tool_result\n{"hits":2}\n```' in out

    def test_generated_media_is_named_rather_than_linked(self) -> None:
        # There is nothing to link to: the bytes live in the media store.
        out = conversation_to_markdown(
            [{"role": "assistant", "content": [{"type": "image_output", "mediaId": "m1"}]}]
        )
        assert "[generated image: m1]" in out

    def test_an_unknown_part_is_skipped_rather_than_fatal(self) -> None:
        out = conversation_to_markdown(
            [{"role": "user", "content": [{"type": "something_new"}, {"type": "text", "text": "x"}]}]
        )
        assert "x" in out


class TestTheArchive:
    def test_it_is_a_zip_that_opens(self) -> None:
        result = conversation_to_zip(TALK)
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            assert archive.testzip() is None
            assert "conversation.md" in archive.namelist()

    def test_media_is_pulled_out_into_files(self) -> None:
        result = conversation_to_zip(TALK)
        assert result.media_count == 1
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            assert "media/media-001.png" in archive.namelist()

    def test_the_markdown_links_the_file_not_the_bytes(self) -> None:
        result = conversation_to_zip(TALK)
        assert "![image](media/media-001.png)" in result.markdown
        assert "base64" not in result.markdown

    def test_the_markdown_is_listed_first(self) -> None:
        # It is what someone opening the archive should see at the top.
        result = conversation_to_zip(TALK)
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            assert archive.namelist()[0] == "conversation.md"

    def test_every_entry_is_stamped_with_the_dos_epoch(self) -> None:
        # Mutation-driven twice over. First it caught a test that proved
        # nothing: two exports taken microseconds apart match even with a
        # wall-clock stamp, because both land in the same second. Then it caught
        # the replacement asserting `== FIXED_TIMESTAMP` -- the very constant
        # the mutation changes, which is a tautology. The literal is the claim.
        #
        # What this actually guards is one character: `writestr("name", data)`
        # stamps the clock, `writestr(ZipInfo(...), data)` does not. Measured: a
        # string name produced a 2026 timestamp.
        result = conversation_to_zip(TALK)
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            stamps = {info.date_time for info in archive.infolist()}
        assert stamps == {(1980, 1, 1, 0, 0, 0)}
        assert FIXED_TIMESTAMP == (1980, 1, 1, 0, 0, 0)

    def test_the_same_conversation_produces_the_same_bytes(self) -> None:
        # An archive that differs between two runs of one input cannot be
        # diffed, cached or checksummed.
        assert conversation_to_zip(TALK).bytes == conversation_to_zip(TALK).bytes

    def test_the_media_stored_is_the_media_given(self) -> None:
        result = conversation_to_zip(TALK)
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            stored = archive.read("media/media-001.png")
        import base64 as b64

        assert stored == b64.b64decode(PNG)

    def test_a_data_url_counts_as_embedded(self) -> None:
        # It arrived as a url but carries the whole blob, and leaving it inline
        # is the multi-megabyte line the archive exists to avoid.
        talk = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": f"data:image/png;base64,{PNG}"},
                    }
                ],
            }
        ]
        result = conversation_to_zip(talk)
        assert result.media_count == 1
        assert "data:image" not in result.markdown

    def test_a_remote_url_is_linked_not_downloaded(self) -> None:
        talk = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
                ],
            }
        ]
        result = conversation_to_zip(talk)
        assert result.media_count == 0
        assert "![image](https://x/y.png)" in result.markdown

    def test_media_files_are_numbered_in_order(self) -> None:
        source = {"type": "base64", "mimeType": "image/png", "data": PNG}
        talk = [{"role": "user", "content": [{"type": "image", "source": source}] * 3}]
        result = conversation_to_zip(talk)
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            assert [n for n in archive.namelist() if n.startswith("media/")] == [
                "media/media-001.png",
                "media/media-002.png",
                "media/media-003.png",
            ]

    def test_the_names_are_configurable(self) -> None:
        result = conversation_to_zip(TALK, markdown_name="talk.md", media_dir="blobs")
        with zipfile.ZipFile(io.BytesIO(result.bytes)) as archive:
            assert set(archive.namelist()) == {"talk.md", "blobs/media-001.png"}


class TestExtensions:
    def test_known_types_get_their_usual_extension(self) -> None:
        assert ext_of("image/jpeg") == "jpg"
        assert ext_of("audio/mpeg") == "mp3"

    def test_parameters_are_ignored(self) -> None:
        assert ext_of("image/png; charset=binary") == "png"

    def test_an_unknown_type_falls_back_to_its_subtype(self) -> None:
        assert ext_of("image/avif") == "avif"

    def test_a_hostile_mime_cannot_write_outside_the_media_folder(self) -> None:
        # It comes off a provider response and ends up in a path someone
        # extracts, so it is sanitised rather than trusted.
        assert ext_of("image/../../etc/passwd") == "etcpasswd"
        assert ext_of("") == "bin"
