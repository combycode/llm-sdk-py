"""Why the prompt cache missed -- one option, two providers, two wire shapes.

Transposed from `unified-library-ts/tests/unit/llm/cache-diagnostics.test.ts`.

Every body below is one that was MEASURED on 2026-09-29 (claude-haiku-4-5 on GA
/v1/messages, gpt-5.6-luna on /v1/responses) with a prompt large enough to be
cached. A short prompt produces nothing to diagnose on either provider and reads
as a broken feature, which is how this would have been mis-specified.

The PARSE side is also proven by the vendored response corpus, which carries a
cell per branch. What lives here is the request side, the warning, and the one
case a corpus cannot show: that Anthropic's silence stays silence.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

from combycode_llm_sdk.llm.cache_diagnostics import (
    anthropic_cache_diagnostics,
    openai_cache_diagnostics,
)
from combycode_llm_sdk.llm.providers.anthropic.messages import AnthropicAdapter
from combycode_llm_sdk.llm.providers.google.generate import GoogleAdapter
from combycode_llm_sdk.llm.providers.openai.responses import OpenAIResponsesAdapter

KEY = {"apiKey": "test-key"}


class TestAnthropicReading:
    def test_it_reports_the_block_that_diverged_with_its_cost(self) -> None:
        raw = {"cache_miss_reason": {"type": "system_changed", "cache_missed_input_tokens": 9197}}
        assert anthropic_cache_diagnostics(raw) == {
            "status": "miss",
            "reason": "system_changed",
            "missedTokens": 9197,
            "raw": raw,
        }

    def test_an_unknown_previous_id_becomes_the_unified_not_found(self) -> None:
        # Measured: HTTP 200, not an error -- so a stored id can be passed
        # without guarding its age.
        d = anthropic_cache_diagnostics({"cache_miss_reason": {"type": "previous_message_not_found"}})
        assert d is not None and d["status"] == "comparison_not_found"
        assert "missedTokens" not in d

    def test_unavailable_passes_through(self) -> None:
        d = anthropic_cache_diagnostics({"cache_miss_reason": {"type": "unavailable"}})
        assert d is not None and d["status"] == "unavailable"

    @pytest.mark.parametrize(
        "body", [None, {"cache_miss_reason": None}, {}, "not-a-mapping"]
    )
    def test_it_says_nothing_when_the_prefix_was_reused(self, body: Any) -> None:
        # The load-bearing measurement: a cache HIT returns `diagnostics: null`,
        # byte for byte what an undiagnosed request returns. Anthropic has no hit
        # variant, so claiming one here would publish our inference as theirs.
        assert anthropic_cache_diagnostics(body) is None

    def test_an_unrecognised_reason_survives(self) -> None:
        d = anthropic_cache_diagnostics(
            {"cache_miss_reason": {"type": "something_new", "cache_missed_input_tokens": 5}}
        )
        assert d is not None
        assert d["status"] == "miss" and d["reason"] == "something_new"


class TestOpenAIReading:
    def test_it_reports_a_hit(self) -> None:
        assert openai_cache_diagnostics({"type": "cache_hit"}) == {
            "status": "hit",
            "raw": {"type": "cache_hit"},
        }

    def test_it_reports_a_miss_with_both_token_counts(self) -> None:
        raw = {
            "type": "cache_miss",
            "reason": "input_changed",
            "cache_missed_tokens": 10052,
            "comparison_reusable_tokens": 10052,
        }
        assert openai_cache_diagnostics(raw) == {
            "status": "miss",
            "reason": "input_changed",
            "missedTokens": 10052,
            "reusableTokens": 10052,
            "raw": raw,
        }

    def test_its_not_found_spelling_unifies_with_anthropics(self) -> None:
        # These two ARE the same fact under two names, so unifying them is not a
        # guess -- unlike the reason vocabularies below.
        d = openai_cache_diagnostics({"type": "comparison_response_not_found"})
        assert d is not None and d["status"] == "comparison_not_found"

    def test_unavailable_is_what_a_small_prompt_gets(self) -> None:
        d = openai_cache_diagnostics({"type": "unavailable"})
        assert d is not None and d["status"] == "unavailable"

    def test_an_unknown_type_is_not_collapsed_onto_a_neighbour(self) -> None:
        # R1: the enum has grown twice already. A value nothing branches on must
        # reach the caller.
        d = openai_cache_diagnostics({"type": "cache_partial_hit"})
        assert d is not None and d["status"] == "cache_partial_hit"

    @pytest.mark.parametrize("body", [None, {}, {"reason": "input_changed"}, 7])
    def test_a_body_with_no_type_is_ignored(self, body: Any) -> None:
        assert openai_cache_diagnostics(body) is None


class TestTheTwoVocabulariesStayApart:
    def test_each_provider_keeps_its_own_reason_word(self) -> None:
        # `system_changed` and `input_changed` are not the same claim, and
        # picking one to stand for the other would be a guess wearing a unified
        # name.
        a = anthropic_cache_diagnostics(
            {"cache_miss_reason": {"type": "system_changed", "cache_missed_input_tokens": 1}}
        )
        o = openai_cache_diagnostics(
            {"type": "cache_miss", "reason": "input_changed", "cache_missed_tokens": 1}
        )
        assert a is not None and a["reason"] == "system_changed"
        assert o is not None and o["reason"] == "input_changed"


class TestWhatGoesOnTheWire:
    @staticmethod
    def _body(adapter: Any, model: str, **extra: Any) -> dict[str, Any]:
        body = adapter.build_request(
            {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}
        ).body
        assert isinstance(body, dict)
        return body

    def test_anthropic_sends_the_id_under_its_own_name(self) -> None:
        body = self._body(
            AnthropicAdapter(KEY), "claude-haiku-4.5",
            cacheDiagnostics={"compareWith": "msg_01abc"},
        )
        assert body["diagnostics"] == {"previous_message_id": "msg_01abc"}

    def test_anthropic_sends_an_explicit_null_when_nothing_is_named(self) -> None:
        # Not omitted: null is how a first turn opts in, and omitting the field
        # opts out entirely -- the difference between a diagnosis and silence.
        body = self._body(AnthropicAdapter(KEY), "claude-haiku-4.5", cacheDiagnostics={})
        assert body["diagnostics"] == {"previous_message_id": None}

    def test_anthropic_sends_nothing_when_it_was_not_asked(self) -> None:
        assert "diagnostics" not in self._body(AnthropicAdapter(KEY), "claude-haiku-4.5")

    def test_openai_sends_it_inside_prompt_cache_options(self) -> None:
        body = self._body(
            OpenAIResponsesAdapter(KEY), "gpt-5.6-luna",
            cacheDiagnostics={"compareWith": "resp_abc"},
        )
        assert body["prompt_cache_options"] == {"comparison_response_id": "resp_abc"}

    def test_openai_merges_with_the_raw_passthrough(self) -> None:
        # Two blocks used to write the same key; whichever ran second won, and a
        # caller who set both silently lost one.
        body = self._body(
            OpenAIResponsesAdapter(KEY), "gpt-5.6-luna",
            cacheDiagnostics={"compareWith": "resp_abc"},
            providerOptions={"promptCacheOptions": {"mode": "explicit"}},
        )
        assert body["prompt_cache_options"] == {
            "comparison_response_id": "resp_abc",
            "mode": "explicit",
        }

    def test_an_explicit_passthrough_id_still_wins(self) -> None:
        body = self._body(
            OpenAIResponsesAdapter(KEY), "gpt-5.6-luna",
            cacheDiagnostics={"compareWith": "resp_unified"},
            providerOptions={"promptCacheOptions": {"comparison_response_id": "resp_raw"}},
        )
        assert body["prompt_cache_options"]["comparison_response_id"] == "resp_raw"

    def test_the_raw_passthrough_alone_is_untouched(self) -> None:
        body = self._body(
            OpenAIResponsesAdapter(KEY), "gpt-5.6-luna",
            providerOptions={"promptCacheOptions": {"mode": "explicit", "ttl": "30m"}},
        )
        assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}

    def test_google_has_no_field_for_it_at_all(self) -> None:
        # Which is what the client's warning is built on: the check is whether
        # the BUILT body carries it, not which provider was asked.
        body = self._body(
            GoogleAdapter(KEY), "gemini-3.1-flash", cacheDiagnostics={"compareWith": "x"}
        )
        assert "diagnostics" not in body
        assert "prompt_cache_options" not in body


class TestTheWarningWhereNoFieldExists:
    """A provider with nowhere to put it says so rather than dropping it.

    Driven through the real client so the check runs against the BUILT body,
    which is what makes it survive a spec change.
    """

    @staticmethod
    async def _warnings(adapter: Any, provider: str, model: str, **options: Any) -> list[str]:
        from combycode_llm_sdk.bus.hook_bus import HookBus
        from combycode_llm_sdk.llm.async_client import AsyncLLMClient

        hooks = HookBus()
        seen: list[dict[str, Any]] = []
        hooks.on("onWarning", lambda ctx: seen.append(dict(ctx)))

        async def fetch(_request: Any, _options: Any = None) -> dict[str, Any]:
            return {"status": 200, "headers": {}, "body": {}}

        client = AsyncLLMClient(
            {
                "provider": provider,
                "model": model,
                "apiKey": "sk-test",
                "adapter": adapter,
                "fetch": fetch,
                "hooks": hooks,
            }
        )
        # The stub body is not a completion, so the parse may raise. The build
        # notes are emitted BEFORE the request goes out, which is the point.
        with contextlib.suppress(Exception):
            await client.complete("hi", options)
        return [w["message"] for w in seen if w.get("code") == "request_adjusted"]

    @pytest.mark.asyncio
    async def test_it_warns_rather_than_dropping_the_request_quietly(self) -> None:
        notes = await self._warnings(
            GoogleAdapter(KEY), "google", "gemini-3.1-flash",
            maxTokens=4, cacheDiagnostics={"compareWith": "x"},
        )
        assert len(notes) == 1
        assert "cacheDiagnostics" in notes[0]

    @pytest.mark.asyncio
    async def test_it_says_nothing_when_it_was_not_asked_for(self) -> None:
        notes = await self._warnings(
            GoogleAdapter(KEY), "google", "gemini-3.1-flash", maxTokens=4
        )
        assert notes == []

    @pytest.mark.asyncio
    async def test_it_stays_silent_on_a_provider_that_did_send_it(self) -> None:
        notes = await self._warnings(
            AnthropicAdapter(KEY), "anthropic", "claude-haiku-4.5",
            maxTokens=4, cacheDiagnostics={"compareWith": "msg_01abc"},
        )
        assert notes == []


class TestStreamingAnswersTheSameQuestion:
    """Measured 2026-09-29: both providers send it in the stream too.

    Without this the same request answered a different question depending on how
    it was fetched -- the failure the `citation` and `file` events already exist
    to prevent.
    """

    def test_anthropic_reports_it_on_message_start(self) -> None:
        # Before a token is generated: it is a fact about the REQUEST.
        events = AnthropicAdapter(KEY).parse_stream_event(
            {
                "event": "message_start",
                "data": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "m",
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                            "diagnostics": {
                                "cache_miss_reason": {
                                    "type": "system_changed",
                                    "cache_missed_input_tokens": 9197,
                                }
                            },
                        },
                    }
                ),
            }
        )
        diag = [e for e in events if e.get("type") == "cache_diagnostics"]
        assert len(diag) == 1
        assert diag[0]["diagnostics"]["reason"] == "system_changed"
        assert diag[0]["diagnostics"]["missedTokens"] == 9197

    def test_anthropic_emits_nothing_when_the_prefix_was_reused(self) -> None:
        events = AnthropicAdapter(KEY).parse_stream_event(
            {
                "event": "message_start",
                "data": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "m",
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                            "diagnostics": None,
                        },
                    }
                ),
            }
        )
        assert not [e for e in events if e.get("type") == "cache_diagnostics"]

    def test_openai_reports_it_on_the_terminal_frame(self) -> None:
        events = OpenAIResponsesAdapter(KEY).parse_stream_event(
            {
                "event": "response.completed",
                "data": json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                            "prompt_cache_diagnostics": {"type": "cache_hit"},
                        },
                    }
                ),
            }
        )
        diag = [e for e in events if e.get("type") == "cache_diagnostics"]
        assert len(diag) == 1 and diag[0]["diagnostics"]["status"] == "hit"

    def test_the_event_reaches_the_public_facade(self) -> None:
        # The seam a typed event union hides: an emitted kind with no builder
        # falls through to UnknownEvent and the caller can never match on it.
        from combycode_llm_sdk.events import CacheDiagnosticsEvent, to_event

        event = to_event({"type": "cache_diagnostics", "diagnostics": {"status": "hit"}})
        assert isinstance(event, CacheDiagnosticsEvent)
        assert event.diagnostics["status"] == "hit"

    def test_the_streamed_final_response_carries_it(self) -> None:
        from combycode_llm_sdk.llm.client_base import StreamAccumulator

        acc = StreamAccumulator()
        acc.absorb({"type": "text", "text": "OK"}, None)
        acc.absorb(
            {"type": "cache_diagnostics", "diagnostics": {"status": "miss", "reason": "x"}}, None
        )
        response = acc.to_response("m", 1.0)
        assert response["cacheDiagnostics"] == {"status": "miss", "reason": "x"}

    def test_it_stays_absent_when_nothing_was_diagnosed(self) -> None:
        from combycode_llm_sdk.llm.client_base import StreamAccumulator

        acc = StreamAccumulator()
        acc.absorb({"type": "text", "text": "OK"}, None)
        assert "cacheDiagnostics" not in acc.to_response("m", 1.0)
