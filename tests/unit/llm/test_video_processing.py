"""How the model works through a video, and what each surface can be told.

Transposed from `unified-library-ts/tests/unit/llm/video-processing.test.ts`.

Google offers two ways of reading a video: ``agentic``, where the model
navigates and seeks to what it needs, and ``static``, a fixed frame rate with
every extracted frame placed in the context window. On a long video the
difference is not stylistic -- static sampling is predictably expensive, and the
offsets are how a question about 30 seconds of a two-hour recording costs what
30 seconds should.

The catch this file exists for: **the two Google surfaces take different
shapes**, and one of them cannot express the interesting half.

* ``generateContent`` takes ``Part.mediaProcessing``, a screaming-snake enum
  with exactly two values. No fps, no offsets.
* Interactions takes ``processing``, either the bare mode or an object carrying
  ``fps`` / ``start_offset`` / ``end_offset`` -- snake_case there, camelCase
  here.

Measured against the live API 2026-09-30:

* an invalid enum is refused, naming ``Part.MediaProcessing`` -- so the field is
  read, not ignored;
* ``media_processing`` with no mime type is ``400 mime_type must be set when
  media_processing is specified``;
* with our ``application/octet-stream`` default it is ``400 media_processing can
  only be set on video parts``;
* ``video/mp4`` + ``STATIC`` is 200;
* ``AGENTIC`` on ``gemini-3.1-flash-lite`` is ``400 Agentic video processing is
  not enabled for this model`` -- a per-model gate the provider reports itself.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter
from combycode_llm_sdk.llm.providers.google.interactions import GoogleInteractionsAdapter

VIDEO: dict[str, Any] = {"type": "video", "source": {"type": "url", "url": "https://v.test/c.mp4"}}


def gen_parts(part: Any) -> list[dict[str, Any]]:
    """The `parts[]` generateContent would receive."""
    built = GoogleAdapter({"apiKey": "k"}).build_request(
        {"model": "gemini-3.1-flash-lite", "messages": [{"role": "user", "content": [part]}]}
    )
    return list(built.body["contents"][0]["parts"])


def interaction_parts(part: Any) -> list[dict[str, Any]]:
    """The input-item content Interactions would receive."""
    built = GoogleInteractionsAdapter({"apiKey": "k"}).build_request(
        {"model": "gemini-3.1-flash-lite", "messages": [{"role": "user", "content": [part]}]}
    )
    return list(built.body["input"][0]["content"])


class TestGenerateContentTheEnumSurface:
    def test_it_sends_nothing_when_the_caller_did_not_ask(self) -> None:
        assert "mediaProcessing" not in gen_parts(VIDEO)[0]

    def test_it_maps_the_modes_to_the_screaming_enum(self) -> None:
        assert (
            gen_parts({**VIDEO, "providerOptions": {"processing": "agentic"}})[0]["mediaProcessing"]
            == "AGENTIC"
        )
        assert (
            gen_parts({**VIDEO, "providerOptions": {"processing": "static"}})[0]["mediaProcessing"]
            == "STATIC"
        )

    def test_it_reduces_the_object_form_to_plain_static(self) -> None:
        # The honest outcome: this surface has nowhere to put fps or offsets,
        # so the mode survives and the sampling does not. Asserted so the loss
        # is a decision on the record rather than a surprise.
        part = gen_parts(
            {
                **VIDEO,
                "providerOptions": {
                    "processing": {"type": "static", "fps": 2, "startOffset": "10.5s"}
                },
            }
        )[0]
        assert part["mediaProcessing"] == "STATIC"
        assert "fps" not in part
        assert "start_offset" not in part

    def test_it_does_not_touch_an_image_part(self) -> None:
        image = {"type": "image", "source": {"type": "url", "url": "https://i.test/x.png"}}
        assert "mediaProcessing" not in gen_parts(image)[0]


class TestTheMimeTypeMediaProcessingDemands:
    def test_it_replaces_the_octet_stream_default_on_a_url_source(self) -> None:
        part = gen_parts({**VIDEO, "providerOptions": {"processing": "static"}})[0]
        assert part["fileData"]["mimeType"] == "video/mp4"

    def test_it_supplies_one_for_a_file_source_which_has_none(self) -> None:
        part = gen_parts(
            {
                "type": "video",
                "source": {"type": "file", "fileId": "files/abc"},
                "providerOptions": {"processing": "static"},
            }
        )[0]
        assert part["fileData"]["mimeType"] == "video/mp4"

    def test_it_keeps_a_callers_own_video_mime_type(self) -> None:
        part = gen_parts(
            {
                "type": "video",
                "source": {"type": "provider_ref", "mimeType": "video/webm", "refId": "files/a"},
                "providerOptions": {"processing": "static"},
            }
        )[0]
        assert part["fileData"]["mimeType"] == "video/webm"

    def test_it_leaves_a_video_that_asked_for_nothing_as_it_was(self) -> None:
        # The fix is scoped to requests that opted in, so nothing already
        # working changes shape.
        assert gen_parts(VIDEO)[0]["fileData"]["mimeType"] == "application/octet-stream"


class TestInteractionsTheRichSurface:
    def test_it_sends_nothing_when_the_caller_did_not_ask(self) -> None:
        assert interaction_parts(VIDEO)[0] == {"type": "video", "uri": "https://v.test/c.mp4"}

    def test_it_passes_a_bare_mode_through_lowercase(self) -> None:
        part = interaction_parts({**VIDEO, "providerOptions": {"processing": "agentic"}})[0]
        assert part["processing"] == "agentic"

    def test_it_carries_the_sampling_renaming_the_offsets(self) -> None:
        part = interaction_parts(
            {
                **VIDEO,
                "providerOptions": {
                    "processing": {
                        "type": "static",
                        "fps": 2,
                        "startOffset": "10.5s",
                        "endOffset": "30s",
                    }
                },
            }
        )[0]
        assert part["processing"] == {
            "type": "static",
            "fps": 2,
            "start_offset": "10.5s",
            "end_offset": "30s",
        }

    def test_it_omits_the_parts_that_were_not_given(self) -> None:
        # Absent, not None. `fps: None` is a value the server has to reject;
        # leaving it out means "your default".
        part = interaction_parts(
            {**VIDEO, "providerOptions": {"processing": {"type": "static"}}}
        )[0]
        assert part["processing"] == {"type": "static"}

    def test_it_keeps_fps_zero_rather_than_treating_it_as_absent(self) -> None:
        # 0 is falsy and would be dropped by a truthiness check.
        part = interaction_parts(
            {**VIDEO, "providerOptions": {"processing": {"type": "static", "fps": 0}}}
        )[0]
        assert part["processing"]["fps"] == 0

    def test_a_bool_is_not_an_fps(self) -> None:
        # `bool` is an `int` in Python, so the obvious numeric check would send
        # `fps: True`.
        part = interaction_parts(
            {**VIDEO, "providerOptions": {"processing": {"type": "static", "fps": True}}}
        )[0]
        assert "fps" not in part["processing"]

    def test_it_carries_a_name_when_one_was_given(self) -> None:
        part = interaction_parts({**VIDEO, "providerOptions": {"name": "the interview"}})[0]
        assert part["name"] == "the interview"
        assert "name" not in interaction_parts({**VIDEO, "providerOptions": {"name": ""}})[0]
