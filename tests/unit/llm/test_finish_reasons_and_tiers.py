"""Why a turn ended, and which tier actually went on the wire.

Two silences, both of which had already cost a release.

The stream registry carried its own copy of Google's terminal-reason table
holding exactly one entry, `MAX_TOKENS`. Everything else fell through to `stop`,
so the SAME response finished differently depending on whether it was streamed:
a SAFETY block read as a clean finish with no content.

The service-tier map fell back to `auto` for anything it did not recognise.
`fast` arrived in 2026-08 and was downgraded to the project default for a month;
`ultrafast` was on course to repeat it. Both are billing and latency decisions
taken on the caller's behalf, and neither was visible from outside.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter
from combycode_llm_sdk.llm.providers.google.response_registry import GOOGLE_FINISH
from combycode_llm_sdk.llm.providers.openai.completions import OpenAIAdapter
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

KEY = {"apiKey": "test-key"}


def _buffered(reason: str) -> Any:
    return GoogleAdapter(KEY).parse_response(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": reason}
            ],
            "usageMetadata": {},
        },
        1.0,
    )["finishReason"]


def _streamed(reason: str) -> Any:
    parse = GoogleAdapter(KEY).create_stream_parser()
    import json as _json

    events = parse({"data": _json.dumps(
        {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": reason}]}
    )})
    done = [e for e in events if e.get("type") == "done"]
    return done[0].get("finishReason") if done else None


class TestGoogleTerminalReasons:
    def test_too_many_tool_calls_keeps_its_own_name(self) -> None:
        assert _buffered("TOO_MANY_TOOL_CALLS") == "too_many_tool_calls"

    def test_the_reasons_it_already_knew(self) -> None:
        assert _buffered("MAX_TOKENS") == "length"
        assert _buffered("SAFETY") == "content_filter"
        assert _buffered("MALFORMED_FUNCTION_CALL") == "malformed_tool_call"

    def test_an_unknown_reason_still_falls_back_to_stop(self) -> None:
        assert _buffered("SOMETHING_GOOGLE_INVENTED_TODAY") == "stop"


class TestStreamingCannotChangeTheAnswer:
    # The structural assertion: not "these four agree", but EVERY key agrees.
    @pytest.mark.parametrize("reason", sorted(GOOGLE_FINISH))
    def test_every_key_finishes_the_same_either_way(self, reason: str) -> None:
        assert _streamed(reason) == _buffered(reason)

    def test_safety_is_no_longer_a_clean_stop_on_a_stream(self) -> None:
        assert _streamed("SAFETY") == "content_filter"

    def test_an_unknown_reason_agrees_too(self) -> None:
        assert _streamed("NEVER_SEEN") == _buffered("NEVER_SEEN") == "stop"


class TestOpenAIIncompleteSubReason:
    @staticmethod
    def _finish(reason: str) -> Any:
        return OpenAIResponsesAdapter(KEY).parse_response(
            {"id": "r", "status": "incomplete", "incomplete_details": {"reason": reason}, "output": [], "usage": {}},
            1.0,
        )["finishReason"]

    def test_only_max_output_tokens_means_length(self) -> None:
        assert self._finish("max_output_tokens") == "length"

    def test_a_message_cap_is_not_a_token_cap(self) -> None:
        assert self._finish("max_messages") == "max_messages"

    def test_steered_keeps_its_own_name(self) -> None:
        # Steering finishes a response and a successor `response.created` follows
        # automatically, so `length` claimed a truncation that never happened.
        assert self._finish("steered") == "steered"

    def test_content_filter_is_unchanged(self) -> None:
        assert self._finish("content_filter") == "content_filter"


class TestServiceTierPerSurface:
    @staticmethod
    def _build(surface: str, tier: str | None = None) -> tuple[Any, list[str]]:
        adapter = OpenAIResponsesAdapter(KEY) if surface == "responses" else OpenAIAdapter(KEY)
        req: dict[str, Any] = {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]}
        if tier:
            req["serviceTier"] = tier
        built = adapter.build_request(req)
        body = built.get("body") if isinstance(built, dict) else built.body
        notes = (built.get("notes") if isinstance(built, dict) else getattr(built, "notes", None)) or []
        return body.get("service_tier"), list(notes)

    def test_responses_sends_ultrafast(self) -> None:
        tier, notes = self._build("responses", "ultrafast")
        assert tier == "ultrafast"
        assert notes == []

    def test_chat_completions_does_not_and_says_so(self) -> None:
        tier, notes = self._build("chat-completions", "ultrafast")
        assert tier == "auto"
        assert len(notes) == 1
        assert "ultrafast" in notes[0] and "chat-completions" in notes[0]

    def test_chat_completions_still_sends_fast(self) -> None:
        tier, notes = self._build("chat-completions", "fast")
        assert tier == "fast"
        assert notes == []

    @pytest.mark.parametrize("surface", ["responses", "chat-completions"])
    def test_an_unknown_tier_falls_back_audibly(self, surface: str) -> None:
        tier, notes = self._build(surface, "hyperfast")
        assert tier == "auto"
        assert "hyperfast" in notes[0] and surface in notes[0]

    @pytest.mark.parametrize("surface", ["responses", "chat-completions"])
    def test_no_tier_omits_the_field_and_warns_about_nothing(self, surface: str) -> None:
        tier, notes = self._build(surface)
        assert tier is None
        assert notes == []
