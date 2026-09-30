"""Asking for a video with no sound, and choosing the voice that speaks.

Transposed from `unified-library-ts/tests/unit/llm/xai-video-audio.test.ts`.

xAI's video models generate an audio track by DEFAULT. So the useful thing
`generateAudio` does is carry ``False`` -- and ``False`` is exactly the value a
presence gate throws away. Mapped with ``presence: 'defined'`` for that reason:
a ``truthy`` gate would have sent the flag only when it agreed with the
default, which is the one case where sending it changes nothing.

`referenceAudios` conditions the generated speech on voices from xAI's own
text-to-speech catalog (``[{"voiceId": "ara"}]``, at most three). The wire shape
is ``AudioUrlContent``, whose ``source`` is a protobuf ``oneof`` -- so an entry
with no voice id is not an empty object the server tolerates, it is a request it
has to reject. Those entries are dropped instead.

Measured against the live API on 2026-09-30, which is the only reason any of
this is known to be right: xAI IGNORES unknown fields and answers 200, so a
successful submission proves nothing on its own. What proves it is the refusals
-- a wrongly typed ``generate_audio`` is a 422 naming the expected type, and an
unknown ``voice_id`` is a 400 listing the whole catalog.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.media.xai import XAIMediaAdapter

MODEL = "grok-imagine-video-1.5"


def body(params: dict[str, Any] | None = None) -> dict[str, Any]:
    adapter = XAIMediaAdapter("k")
    built = adapter.build_video_request("a cat", MODEL, params)
    return dict(built["body"])


class TestGenerateAudio:
    def test_it_is_absent_when_the_caller_did_not_ask(self) -> None:
        # Absent, not True: the provider's default is its own to choose, and
        # restating it would freeze today's default into every request.
        assert "generate_audio" not in body()
        assert "generate_audio" not in body({})

    def test_it_sends_false_the_reason_the_flag_exists(self) -> None:
        # The regression guard. A truthy presence gate drops this silently and
        # the caller gets a video with sound.
        assert body({"generateAudio": False})["generate_audio"] is False

    def test_it_sends_true_when_asked_explicitly(self) -> None:
        assert body({"generateAudio": True})["generate_audio"] is True


class TestReferenceAudios:
    def test_it_becomes_xai_voice_references(self) -> None:
        got = body({"referenceAudios": [{"voiceId": "ara"}, {"voiceId": "rex"}]})
        assert got["reference_audios"] == [{"voice_id": "ara"}, {"voice_id": "rex"}]

    def test_it_is_absent_when_not_asked_for(self) -> None:
        assert "reference_audios" not in body()

    def test_it_is_absent_rather_than_empty_when_nothing_usable_arrived(self) -> None:
        # `reference_audios: []` is a different request from not asking for
        # reference audio, and the empty one has no meaning to send.
        assert "reference_audios" not in body({"referenceAudios": []})
        assert "reference_audios" not in body({"referenceAudios": [{"voiceId": ""}]})
        assert "reference_audios" not in body({"referenceAudios": "ara"})

    def test_it_drops_an_entry_with_no_voice_id(self) -> None:
        # `AudioUrlContent.source` is a protobuf oneof; `{}` selects no branch.
        got = body({"referenceAudios": [{"voiceId": "ara"}, {}]})
        assert got["reference_audios"] == [{"voice_id": "ara"}]


class TestTheRestOfTheVideoRequestIsUntouched:
    def test_it_still_carries_prompt_model_and_the_existing_params(self) -> None:
        got = body(
            {
                "duration": 6,
                "aspectRatio": "16:9",
                "resolution": "1080p",
                "generateAudio": False,
            }
        )
        assert got["prompt"] == "a cat"
        assert got["model"] == MODEL
        assert got["duration"] == 6
        assert got["aspect_ratio"] == "16:9"
        assert got["resolution"] == "1080p"
        assert got["generate_audio"] is False
