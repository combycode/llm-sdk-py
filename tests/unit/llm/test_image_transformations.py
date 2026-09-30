"""An image the model never saw at the size you sent it.

Transposed from `unified-library-ts/tests/unit/llm/image-transformations.test.ts`.

Anthropic scales an over-large image down to fit and says nothing about it. The
model then reasons over dimensions you did not choose, the answer comes back
looking normal, and nothing in the response mentions that the thing you were
asking about was resampled away.

Measured 2026-09-30: a 4000x4000 image sent with
``transformations: {"oversized_image": "error"}`` is refused with

    image dimensions 4000x4000 exceed the maximum image size of a model named
    on this request and would be downsized to 1092x1092; scale the image to at
    most 1092x1092 or set the image's oversized_image setting to "downsize"

1092x1092 is 7% of the pixels that were sent. Without the field, that happens
silently. (Separately, and above this: a dimension over 8000px is refused
outright whatever the setting.)

Per-IMAGE rather than per-request: one oversized screenshot in a long
conversation should not change how every other image in it is handled.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter

PNG: dict[str, Any] = {
    "type": "image",
    "source": {"type": "base64", "mimeType": "image/png", "data": "aGk="},
}


def blocks(parts: list[Any]) -> list[dict[str, Any]]:
    """The content blocks Anthropic would receive for one user message."""
    adapter = AnthropicAdapter({"apiKey": "k"})
    built = adapter.build_request(
        {"model": "claude-haiku-4.5", "messages": [{"role": "user", "content": parts}]}
    )
    return list(built.body["messages"][0]["content"])


class TestTransformationsOnAnImageBlock:
    def test_it_is_absent_when_the_caller_did_not_ask(self) -> None:
        # Absent, not `{"oversized_image": "downsize"}`. Restating the server's
        # default would freeze today's default into every request we send.
        block = blocks([PNG])[0]
        assert block["type"] == "image"
        assert "transformations" not in block

    def test_it_carries_error_the_reason_the_field_exists(self) -> None:
        block = blocks(
            [{**PNG, "providerOptions": {"transformations": {"oversized_image": "error"}}}]
        )[0]
        assert block["transformations"] == {"oversized_image": "error"}

    def test_it_carries_an_explicit_downsize_too(self) -> None:
        # Saying the default out loud is a legitimate choice: it pins the
        # behaviour against a future change of default.
        block = blocks(
            [{**PNG, "providerOptions": {"transformations": {"oversized_image": "downsize"}}}]
        )[0]
        assert block["transformations"] == {"oversized_image": "downsize"}

    def test_an_empty_object_means_nothing_and_is_not_sent(self) -> None:
        # Anthropic documents an empty object as equivalent to omitting it.
        block = blocks([{**PNG, "providerOptions": {"transformations": {}}}])[0]
        assert "transformations" not in block

    def test_it_applies_to_one_image_without_touching_its_neighbours(self) -> None:
        first, second = blocks(
            [
                {**PNG, "providerOptions": {"transformations": {"oversized_image": "error"}}},
                PNG,
            ]
        )
        assert first["transformations"] == {"oversized_image": "error"}
        assert "transformations" not in second

    def test_it_rides_along_with_every_source_kind(self) -> None:
        opts = {"transformations": {"oversized_image": "error"}}
        for source in (
            {"type": "base64", "mimeType": "image/png", "data": "aGk="},
            {"type": "url", "url": "https://a.test/x.png"},
            {"type": "file", "fileId": "file_1"},
        ):
            block = blocks([{"type": "image", "source": source, "providerOptions": opts}])[0]
            assert block["transformations"] == {"oversized_image": "error"}
            # And the source itself still built correctly.
            assert block["source"]

    def test_it_leaves_other_part_types_alone(self) -> None:
        assert blocks([{"type": "text", "text": "hello"}])[0] == {"type": "text", "text": "hello"}
