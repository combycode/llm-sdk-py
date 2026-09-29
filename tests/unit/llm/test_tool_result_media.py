"""A tool that returns an image sends an IMAGE, on every backend.

Transposed from `unified-library-ts/tests/unit/llm/tool-result-media.test.ts`.

`execute` has always been allowed to return content parts, and the loop always
carried them into the tool result -- then every adapter serialised the list into
the provider's text slot. So the documented way to return a screenshot worked, in
the sense that the request succeeded: the model received a wall of base64 as
prose, was billed for it as prose, and could not see the picture. Nothing failed;
it just did not work.

Each API has somewhere to put this and they disagree about where, so the test is
per-adapter rather than shared. The one invariant across all of them: a STRING
result must build exactly the body it built before this existed.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.llm.providers._shared.tool_result import (
    has_tool_result_media,
    split_tool_result,
)
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter
from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter
from combycode_llm_sdk.llm.providers.google.interactions import GoogleInteractionsAdapter
from combycode_llm_sdk.llm.providers.openai.completions import OpenAIAdapter
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

#: A 1x1 PNG, small enough to read in a failure message.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
    "q842iQAAAABJRU5ErkJggg=="
)

CHART: list[dict[str, Any]] = [
    {"type": "text", "text": "revenue by quarter"},
    {"type": "image", "source": {"type": "base64", "mimeType": "image/png", "data": PNG}},
]


def conversation(result: Any) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "chart the revenue"},
        {
            "role": "assistant",
            "content": [{"type": "tool_call", "id": "call_1", "name": "chart", "arguments": {}}],
        },
        {"role": "tool", "content": [{"type": "tool_result", "id": "call_1", "content": result}]},
    ]


def body(adapter: Any, model: str, result: Any) -> dict[str, Any]:
    req = {"model": model, "messages": conversation(result)}
    return dict(adapter.build_request(req).body)


class TestTheSplitItself:
    def test_it_leaves_a_string_alone(self) -> None:
        assert split_tool_result("done") == ("done", [])

    def test_it_separates_the_text_half_from_the_media_half(self) -> None:
        text, media = split_tool_result(CHART)
        assert text == "revenue by quarter"
        assert [m["type"] for m in media] == ["image"]

    def test_it_joins_several_text_parts_keeping_their_order(self) -> None:
        text, _ = split_tool_result(
            [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        )
        assert text == "first\nsecond"

    def test_it_serialises_a_part_that_is_neither_rather_than_dropping_it(self) -> None:
        # A tool returning something unexpected should reach the model looking
        # odd, not vanish on the way.
        text, media = split_tool_result(
            [{"type": "tool_call", "id": "x", "name": "inner", "arguments": {"a": 1}}]
        )
        assert "inner" in text
        assert media == []

    def test_it_answers_whether_anything_has_to_travel_as_media(self) -> None:
        assert has_tool_result_media("done") is False
        assert has_tool_result_media([{"type": "text", "text": "done"}]) is False
        assert has_tool_result_media(CHART) is True


class TestAnthropicSendsBlocksInsideTheToolResult:
    adapter = AnthropicAdapter({"apiKey": "k"})

    def tool_result(self, result: Any) -> dict[str, Any]:
        msgs = body(self.adapter, "claude-haiku-4.5", result)["messages"]
        blocks = [b for m in msgs for b in m.get("content", []) if isinstance(b, dict)]
        return next(b for b in blocks if b.get("type") == "tool_result")

    def test_a_string_result_stays_a_string(self) -> None:
        assert self.tool_result("done")["content"] == "done"

    def test_the_image_travels_as_an_image_block(self) -> None:
        content = self.tool_result(CHART)["content"]
        assert content == [
            {"type": "text", "text": "revenue by quarter"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        ]

    def test_it_needs_no_extra_message_to_do_it(self) -> None:
        # The whole point of the native slot: history stays one turn per turn.
        with_media = body(self.adapter, "claude-haiku-4.5", CHART)["messages"]
        plain = body(self.adapter, "claude-haiku-4.5", "done")["messages"]
        assert len(with_media) == len(plain)


class TestOpenAIResponsesSendsItInsideFunctionCallOutput:
    adapter = OpenAIResponsesAdapter({"apiKey": "k"})

    def items(self, result: Any) -> list[dict[str, Any]]:
        return list(body(self.adapter, "gpt-5.4-nano", result)["input"])

    def output(self, result: Any) -> Any:
        return next(
            i["output"] for i in self.items(result) if i.get("type") == "function_call_output"
        )

    def test_a_string_result_stays_a_string(self) -> None:
        assert self.output("done") == "done"

    def test_the_image_travels_as_an_input_image_item(self) -> None:
        assert self.output(CHART) == [
            {"type": "input_text", "text": "revenue by quarter"},
            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"},
        ]

    def test_it_needs_no_extra_input_item_to_do_it(self) -> None:
        assert len(self.items(CHART)) == len(self.items("done"))


class TestGooglePutsItInFunctionResponseParts:
    adapter = GoogleAdapter({"apiKey": "k"})

    def fn_response(self, result: Any) -> dict[str, Any]:
        contents = body(self.adapter, "gemini-3-flash", result)["contents"]
        parts = [p for c in contents for p in c.get("parts", [])]
        return next(p["functionResponse"] for p in parts if "functionResponse" in p)

    def test_a_string_result_sits_under_response_result(self) -> None:
        assert self.fn_response("done")["response"] == {"result": "done"}

    def test_the_text_goes_in_response_and_the_image_in_parts(self) -> None:
        # `response` is a JSON object, so media cannot live there -- and a content
        # part list used to be sent AS that object, which is not an object.
        fr = self.fn_response(CHART)
        assert fr["response"] == {"result": "revenue by quarter"}
        assert fr["parts"] == [{"inlineData": {"mimeType": "image/png", "data": PNG}}]

    def test_a_source_it_cannot_inline_is_said_out_loud_not_dropped(self) -> None:
        # `fileData` in a function response is documented Vertex-only, so a URL
        # source has nowhere to go here. Silence would lose the tool's answer.
        fr = self.fn_response(
            [
                {"type": "text", "text": "see attached"},
                {"type": "image", "source": {"type": "url", "url": "https://example.invalid/a.png"}},
            ]
        )
        assert "parts" not in fr
        assert "image omitted" in fr["response"]["result"]


class TestChatCompletionsHasNoSlotSoTheMediaFollows:
    adapter = OpenAIAdapter({"apiKey": "k"})

    def messages(self, result: Any) -> list[dict[str, Any]]:
        return list(body(self.adapter, "gpt-5.4-nano", result)["messages"])

    def test_a_string_result_stays_a_string_in_one_tool_message(self) -> None:
        tool = [m for m in self.messages("done") if m.get("role") == "tool"]
        assert len(tool) == 1
        assert tool[0]["content"] == "done"

    def test_the_text_is_the_tool_message_and_the_image_a_user_message_after_it(self) -> None:
        msgs = self.messages(CHART)
        at = next(i for i, m in enumerate(msgs) if m.get("role") == "tool")
        assert msgs[at]["content"] == "revenue by quarter"
        after = msgs[at + 1]
        assert after["role"] == "user"
        assert after["content"][0]["type"] == "image_url"
        assert after["content"][0]["image_url"]["url"] == f"data:image/png;base64,{PNG}"

    def test_every_call_is_answered_before_the_follow_up(self) -> None:
        # This API rejects a request where a tool call has no answer. The media
        # message therefore goes after the LAST tool message, not after each one.
        msgs = list(
            dict(
                self.adapter.build_request(
                    {
                        "model": "gpt-5.4-nano",
                        "messages": [
                            {"role": "user", "content": "two charts"},
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_call",
                                        "id": "call_1",
                                        "name": "chart",
                                        "arguments": {},
                                    },
                                    {
                                        "type": "tool_call",
                                        "id": "call_2",
                                        "name": "chart",
                                        "arguments": {},
                                    },
                                ],
                            },
                            {
                                "role": "tool",
                                "content": [
                                    {"type": "tool_result", "id": "call_1", "content": CHART},
                                    {"type": "tool_result", "id": "call_2", "content": CHART},
                                ],
                            },
                        ],
                    }
                ).body
            )["messages"]
        )
        assert [m["role"] for m in msgs[-3:]] == ["tool", "tool", "user"]
        assert len(msgs[-1]["content"]) == 2


class TestGoogleInteractionsDoesTheSameForTheSameReason:
    adapter = GoogleInteractionsAdapter({"apiKey": "k"})

    def input_items(self, result: Any) -> list[dict[str, Any]]:
        return list(body(self.adapter, "gemini-3-flash", result)["input"])

    def test_a_string_result_stays_in_the_result_field(self) -> None:
        items = self.input_items("done")
        fr = next(i for i in items if i.get("type") == "function_result")
        assert fr["result"] == "done"

    def test_the_text_is_the_result_and_the_image_a_user_input_after_it(self) -> None:
        items = self.input_items(CHART)
        at = next(i for i, x in enumerate(items) if x.get("type") == "function_result")
        assert items[at]["result"] == "revenue by quarter"
        after = items[at + 1]
        assert after["type"] == "user_input"
        assert after["content"][0] == {"type": "image", "mime_type": "image/png", "data": PNG}
