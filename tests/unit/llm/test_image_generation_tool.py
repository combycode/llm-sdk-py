"""The `image_generation` builtin, and the label on what comes back.

Transposed from `unified-library-ts/tests/unit/llm/image-generation-tool.test.ts`.

The row that led here asked whether xAI's server-side image tool needed its own
parsing. Measured on REST on 2026-10-01 it does not: xAI's output item is
OpenAI's `image_generation_call`, same keys, `result` as base64 -- not the
`{"__type": "image_generation_result"}` envelope its gRPC surface uses. So the
tool was working on both providers already.

Two things were wrong anyway, and neither is what the row was looking for.

**The catalog said no.** `image_generation` was in no provider's builtin list --
including OpenAI's, where the tool has worked all along -- so
`supports_builtin_tool(..., "image_generation")` answered False about a tool that
works, and a caller gating on the catalog refused itself.

**The mime was a guess.** The parser read `output_format` and defaulted to PNG.
OpenAI reports the format (accurately: asking for jpeg returns JPEG bytes). xAI
reports NONE and returns JPEG -- so every xAI image came back labeled
`image/png` with JPEG inside it. A caller writing the file gets the wrong
extension, and a strict validator downstream (Google Veo compares the declared
mime against the bytes) answers 400.
"""

from __future__ import annotations

import base64
import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.catalog.builtin_tools import PROVIDER_BUILTIN_TOOLS
from combycode_llm_sdk.catalog.catalog import ModelCatalog
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

ADAPTER = OpenAIResponsesAdapter({"apiKey": "test-key"})

#: Leading magic bytes, padded so a 16-character slice decodes.
JPEG_B64 = base64.b64encode(bytes([0xFF, 0xD8, 0xFF, 0xE0] + [0] * 8)).decode()
PNG_B64 = base64.b64encode(bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] + [0] * 4)).decode()
UNKNOWN_B64 = base64.b64encode(bytes([1, 2, 3, 4] + [0] * 8)).decode()


def image_of(item: dict[str, Any]) -> dict[str, Any] | None:
    parsed = ADAPTER.parse_response({"id": "resp_1", "output": [item], "usage": {}}, 1)
    return next((p for p in parsed["content"] if p.get("type") == "image_output"), None)


def mime_of(item: dict[str, Any]) -> Any:
    part = image_of(item)
    return part["mimeType"] if part else None


class TestTheMimeOnAGeneratedImage:
    def test_it_takes_the_providers_word_when_it_gives_one(self) -> None:
        # OpenAI reports `output_format`, and reports it accurately.
        for fmt, expected in (("jpeg", "image/jpeg"), ("webp", "image/webp"), ("png", "image/png")):
            assert (
                mime_of(
                    {"type": "image_generation_call", "output_format": fmt, "result": PNG_B64}
                )
                == expected
            )

    def test_it_reads_the_bytes_when_the_provider_says_nothing(self) -> None:
        # Measured 2026-10-01: xAI's item is `[id, type, status, result, prompt]`,
        # no `output_format`, and the bytes are JPEG. This was the mislabel.
        assert mime_of({"type": "image_generation_call", "result": JPEG_B64}) == "image/jpeg"

    def test_it_still_answers_png_when_the_bytes_say_nothing_either(self) -> None:
        # A format we do not recognise must not become a confident wrong answer in
        # the other direction; PNG stays the fallback it always was.
        assert mime_of({"type": "image_generation_call", "result": UNKNOWN_B64}) == "image/png"

    def test_a_declared_format_wins_over_contradicting_bytes(self) -> None:
        # Deliberate. A provider that says png and sends JPEG is a provider bug,
        # and second-guessing a provider that DID answer would make the behaviour
        # depend on which of two wrong things we trust. Only silence is filled in.
        assert (
            mime_of({"type": "image_generation_call", "output_format": "png", "result": JPEG_B64})
            == "image/png"
        )

    def test_it_produces_no_part_without_data(self) -> None:
        assert image_of({"type": "image_generation_call", "status": "in_progress"}) is None


class TestWhatTheCatalogSaysAboutTheTool:
    CATALOG = ModelCatalog.with_provider_defaults()

    def test_it_reports_it_supported_where_it_was_measured(self) -> None:
        # Measured through the library on 2026-10-01: each returned an image on
        # `response.media`.
        for provider, model in (
            ("openai", "gpt-5.6-sol"),
            ("openai", "gpt-5.4-nano"),
            ("xai", "grok-4.6"),
            ("xai", "grok-4.5"),
            ("xai", "grok-4.3"),
        ):
            assert self.CATALOG.supports_builtin_tool(provider, model, "image_generation") is True

    def test_it_does_not_claim_it_where_it_was_not_measured(self) -> None:
        # Anthropic and Google have no such hosted tool, and an unchecked True
        # would be a capability claim nobody made.
        for provider in ("anthropic", "google", "openrouter"):
            assert "image_generation" not in PROVIDER_BUILTIN_TOOLS[provider]

    def test_it_leaves_the_already_listed_tools_alone(self) -> None:
        # The regression that would be easy to miss while editing a shared list.
        assert self.CATALOG.supports_builtin_tool("xai", "grok-4.6", "web_search") is True
        assert self.CATALOG.supports_builtin_tool("xai", "grok-4.6", "code_interpreter") is True
        assert self.CATALOG.supports_builtin_tool("xai", "grok-4.6", "web_fetch") is False


class TestTheToolsParamsReachTheWire:
    def test_they_are_spread_as_siblings_of_type(self) -> None:
        # xAI reads `action` beside `type`. Nested under a `params` key it would be
        # an unknown field, and `action` silently absent rather than refused.
        built = ADAPTER.build_request(
            {
                "model": "grok-4.6",
                "messages": [{"role": "user", "content": "draw a circle"}],
                "tools": [{"type": "image_generation", "params": {"action": "generate"}}],
            }
        )
        assert built.body["tools"][0] == {"type": "image_generation", "action": "generate"}

    def test_the_bare_tool_is_sent_when_no_params_are_given(self) -> None:
        built = ADAPTER.build_request(
            {
                "model": "grok-4.6",
                "messages": [{"role": "user", "content": "draw a circle"}],
                "tools": [{"type": "image_generation"}],
            }
        )
        assert built.body["tools"][0] == {"type": "image_generation"}
