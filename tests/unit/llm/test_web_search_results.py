"""A search that found images, and a response that did not contain them.

Transposed from `unified-library-ts/tests/unit/llm/web-search-results.test.ts`.

Two halves that only work together.

**Asking.** ``web_search_call.results`` is NOT returned by default. Setting
``search_content_types: ["image"]`` alone gets a search that found images and a
response with nothing in it -- the results arrive only when the request also
carries ``include: ["web_search_call.results"]``. Measured 2026-09-30 by
sending the same request twice, with and without: ``with_results=1`` against
``with_results=0``. The adapter derives the include from the tool rather than
exposing a second knob, because the two halves arriving separately is exactly
how you get an empty result set that reads as "no images found".

**Reading.** ``builtin_call_from_responses_item`` read ``action.queries/query/
url`` and nothing else, so both the results and the ``action.sources[]`` behind
an answer were parsed off the wire and thrown away.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers.openai.parse_helpers import (
    builtin_call_from_responses_item,
)
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter


def body(tools: list[Any]) -> dict[str, Any]:
    """The JSON body the adapter would POST, for a given tool list."""
    adapter = OpenAIResponsesAdapter({"apiKey": "k"})
    built = adapter.build_request(
        {
            "model": "gpt-5.4-nano",
            "messages": [{"role": "user", "content": "find pictures of the bridge"}],
            "tools": tools,
        }
    )
    return dict(built.body)


class TestAskingForImageResults:
    def test_it_adds_the_include_that_makes_them_arrive(self) -> None:
        got = body([{"type": "web_search", "params": {"search_content_types": ["image", "text"]}}])
        assert got["include"] == ["web_search_call.results"]

    def test_it_does_not_add_it_for_a_text_only_search(self) -> None:
        # The include is not free -- it is extra payload on every response.
        assert "include" not in body(
            [{"type": "web_search", "params": {"search_content_types": ["text"]}}]
        )
        assert "include" not in body([{"type": "web_search"}])
        assert "include" not in body([{"type": "code_interpreter"}])

    def test_it_does_not_add_it_when_there_are_no_tools(self) -> None:
        assert "include" not in body([])

    def test_it_is_not_confused_by_a_function_tool_named_like_a_builtin(self) -> None:
        fn = {"name": "web_search", "description": "mine", "parameters": {"type": "object"}}
        assert "include" not in body([fn])

    def test_it_still_forwards_the_params_verbatim(self) -> None:
        got = body(
            [
                {
                    "type": "web_search",
                    "params": {
                        "search_content_types": ["image"],
                        "image_settings": {"max_results": 3, "caption": True},
                        "external_web_access": False,
                    },
                }
            ]
        )
        tool = got["tools"][0]
        assert tool["type"] == "web_search"
        assert tool["search_content_types"] == ["image"]
        assert tool["image_settings"] == {"max_results": 3, "caption": True}
        # False has to survive: cache-only is the whole point of the flag.
        assert tool["external_web_access"] is False


class TestReadingWhatCameBack:
    def test_it_keeps_the_image_results_renaming_the_documented_fields(self) -> None:
        call = builtin_call_from_responses_item(
            {
                "type": "web_search_call",
                "id": "ws_1",
                "action": {"type": "search", "queries": ["golden gate at sunset"]},
                "results": [
                    {
                        "image_url": "https://img.test/a.jpg",
                        "source_website_url": "https://news.test/story",
                        "thumbnail_url": "https://img.test/a-thumb.jpg",
                        "caption": "The bridge at dusk",
                    }
                ],
            }
        )
        assert call is not None
        assert call["results"] == [
            {
                "imageUrl": "https://img.test/a.jpg",
                "sourceWebsiteUrl": "https://news.test/story",
                "thumbnailUrl": "https://img.test/a-thumb.jpg",
                "caption": "The bridge at dusk",
            }
        ]
        assert call["query"] == "golden gate at sunset"

    def test_it_carries_through_a_field_we_have_never_seen(self) -> None:
        # A real result also carries `type` (measured 2026-09-30). This list
        # exists because the payload is richer than our type; dropping the
        # unrecognised half would defeat the point.
        call = builtin_call_from_responses_item(
            {
                "type": "web_search_call",
                "action": {"type": "search"},
                "results": [{"image_url": "https://img.test/a.jpg", "type": "image", "rank": 2}],
            }
        )
        assert call is not None
        assert call["results"][0] == {
            "imageUrl": "https://img.test/a.jpg",
            "type": "image",
            "rank": 2,
        }

    def test_it_keeps_the_sources_behind_an_answer(self) -> None:
        call = builtin_call_from_responses_item(
            {
                "type": "web_search_call",
                "action": {
                    "type": "search",
                    "queries": ["q"],
                    "sources": [
                        {"type": "url", "url": "https://a.test/1"},
                        {"type": "url", "url": "https://b.test/2"},
                    ],
                },
            }
        )
        assert call is not None
        assert call["sources"] == ["https://a.test/1", "https://b.test/2"]

    def test_it_is_absent_not_empty_when_the_provider_sent_none(self) -> None:
        call = builtin_call_from_responses_item(
            {"type": "web_search_call", "action": {"type": "search", "queries": ["q"]}}
        )
        assert call is not None
        assert "results" not in call
        assert "sources" not in call

        empty = builtin_call_from_responses_item(
            {"type": "web_search_call", "action": {"type": "search", "sources": []}, "results": []}
        )
        assert empty is not None
        assert "results" not in empty
        assert "sources" not in empty

    def test_it_ignores_junk_rather_than_raising(self) -> None:
        call = builtin_call_from_responses_item(
            {
                "type": "web_search_call",
                "action": {"type": "search", "sources": ["not-a-mapping", {"url": 42}]},
                "results": "nope",
            }
        )
        assert call is not None
        assert "results" not in call
        assert "sources" not in call

    def test_it_leaves_the_existing_open_page_payload_alone(self) -> None:
        call = builtin_call_from_responses_item(
            {"type": "web_search_call", "action": {"type": "open_page", "url": "https://a.test/p"}}
        )
        assert call is not None
        assert call["url"] == "https://a.test/p"
        assert "results" not in call
