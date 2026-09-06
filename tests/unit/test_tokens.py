"""Counting tokens, and saying what the number is worth.

The failure these guard against is not a wrong count -- it is a right-looking
count that was an estimate while the caller believed it was a measurement.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk import Engine, TransportResponse, count_tokens
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.tokens import (
    DEFAULT_CHARS_PER_TOKEN,
    NON_TEXT_PART_TOKENS,
    HeuristicCounter,
    HybridCounter,
    TiktokenCounter,
    message_chars,
)

TEXT = "The quick brown fox jumps over the lazy dog."


def counting_engine(body: Any, transport_seen: list[Any] | None = None) -> Engine:
    def stub(request: Any) -> TransportResponse:
        if transport_seen is not None:
            transport_seen.append(request)
        return TransportResponse(body=body)

    return Engine(
        catalog="defaults",
        api_keys={"anthropic": "k", "google": "k", "xai": "k"},
        transport=stub,
        register_as_default=False,
    )


class TestTheStrategyIsChosenByTheCatalog:
    def test_openai_asks_for_a_local_tokenizer(self) -> None:
        assert HybridCounter(resolve_catalog("defaults")).strategy_for("openai", "gpt-4.1") == (
            "tiktoken"
        )

    def test_anthropic_asks_for_the_count_endpoint(self) -> None:
        assert HybridCounter(resolve_catalog("defaults")).strategy_for(
            "anthropic", "claude-haiku-4.5"
        ) == "api"

    def test_a_model_with_no_tokenizer_entry_estimates(self) -> None:
        assert HybridCounter(resolve_catalog("defaults")).strategy_for("openai", "unknown") == (
            "estimate"
        )

    def test_no_catalog_at_all_estimates(self) -> None:
        assert HybridCounter(None).strategy_for("openai", "gpt-4.1") == "estimate"


class TestHonesty:
    def test_an_estimate_never_claims_to_be_exact(self) -> None:
        # The whole reason the field exists.
        result = count_tokens(model="openai/gpt-4.1", input=TEXT, exact=False)
        assert result.strategy == "estimate"
        assert result.exact is False

    def test_exact_false_never_calls_anything(self) -> None:
        seen: list[Any] = []
        engine = counting_engine({"input_tokens": 99}, seen)
        count_tokens(
            model="anthropic/claude-haiku-4.5", input=TEXT, exact=False, engine=engine
        )
        assert seen == []

    def test_the_count_endpoint_reports_itself_as_exact(self) -> None:
        engine = counting_engine({"input_tokens": 42})
        result = count_tokens(model="anthropic/claude-haiku-4.5", input=TEXT, engine=engine)
        assert result.tokens == 42
        assert result.strategy == "api"
        assert result.exact is True

    def test_google_reads_its_own_field(self) -> None:
        engine = counting_engine({"totalTokens": 17})
        assert count_tokens(model="google/gemini-2.5-flash", input=TEXT, engine=engine).tokens == 17

    def test_xai_answers_with_a_token_list_and_the_length_is_the_count(self) -> None:
        # Reading it as a number would silently give zero.
        engine = counting_engine({"token_ids": [1, 2, 3, 4]})
        assert count_tokens(model="xai/grok-4.3", input=TEXT, engine=engine).tokens == 4

    def test_a_failing_endpoint_raises_rather_than_guessing(self) -> None:
        def failing(request: Any) -> TransportResponse:
            return TransportResponse(status=500, body={"error": "down"})

        engine = Engine(
            catalog="defaults",
            api_keys={"anthropic": "k"},
            transport=failing,
            register_as_default=False,
        )
        with pytest.raises(Exception, match="count endpoint failed|500"):
            count_tokens(model="anthropic/claude-haiku-4.5", input=TEXT, engine=engine)

    def test_no_key_falls_back_to_the_estimate_and_says_so(self) -> None:
        engine = Engine(catalog="defaults", api_keys={}, register_as_default=False)
        result = count_tokens(model="anthropic/claude-haiku-4.5", input=TEXT, engine=engine)
        assert result.strategy == "estimate"
        assert result.exact is False


class TestTheEstimate:
    def test_it_uses_the_models_own_rate(self) -> None:
        catalog = resolve_catalog("defaults")
        # gpt-4.1 declares 3.8 chars/token; a bare default would give a
        # different number for the same text.
        assert HeuristicCounter(catalog).rate("openai", "gpt-4.1") == 3.8

    def test_an_unknown_model_gets_the_default_rate(self) -> None:
        assert HeuristicCounter(resolve_catalog("defaults")).rate("openai", "nope") == (
            DEFAULT_CHARS_PER_TOKEN
        )

    def test_it_rounds_up(self) -> None:
        # Half a token still occupies one, and rounding down is the direction
        # that overflows a window.
        assert HeuristicCounter(None).count("abcde", "p", "m") == 2


class TestWhatGetsCounted:
    def test_messages_are_flattened_to_their_text(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "world"}]},
        ]
        assert count_tokens(model="openai/gpt-4.1", input=messages, exact=False).tokens > 0

    def test_a_non_text_part_costs_an_allowance_not_zero(self) -> None:
        # Under-counting a window is the failure that matters, because it is the
        # one discovered by a 400 at the far end.
        with_image = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "data": "AA=="}},
                ],
            }
        ]
        text_only = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}]
        richer = count_tokens(model="openai/gpt-4.1", input=with_image, exact=False)
        plain = count_tokens(model="openai/gpt-4.1", input=text_only, exact=False)
        assert richer.tokens - plain.tokens == NON_TEXT_PART_TOKENS

    def test_message_chars_counts_a_tool_call(self) -> None:
        chars = message_chars(
            {"content": [{"type": "tool_call", "name": "f", "arguments": {"a": 1}}]}
        )
        assert chars > len("f")

    def test_a_bare_string_is_counted_directly(self) -> None:
        assert count_tokens(model="openai/gpt-4.1", input="hello", exact=False).tokens > 0


class TestTheOptionalTokenizer:
    def test_the_encoding_comes_from_the_catalog(self) -> None:
        # Two models from one provider can use different encodings, and counting
        # o200k text with cl100k is wrong quietly.
        assert TiktokenCounter(resolve_catalog("defaults")).encoding_for(
            "openai", "gpt-4.1"
        ) == "o200k_base"

    def test_it_counts_exactly_when_the_extra_is_installed(self) -> None:
        # The exact branch, actually executed. Everything else about this class
        # was covered by monkeypatching the tokenizer AWAY, so the code that
        # does the counting never ran and its absence looked like a pass.
        counter = TiktokenCounter(resolve_catalog("defaults"))
        assert counter.available(), "the `tokens` extra is declared in dev"
        exact = counter.count(TEXT, "openai", "gpt-4.1")
        assert exact > 0
        # An exact count is not the estimate. If these agreed to the token the
        # test would pass with the tokenizer wired to the heuristic.
        assert exact != HeuristicCounter(resolve_catalog("defaults")).count(
            TEXT, "openai", "gpt-4.1"
        )

    def test_the_encoding_actually_changes_the_count(self) -> None:
        # Why the encoding is read from the catalog rather than guessed from the
        # provider. Measured, not assumed: on English prose the two encodings
        # return the SAME count, so a wrong encoding survives every ASCII test
        # anyone would think to write and only shows up on other scripts --
        # where it is out by a factor of two.
        import tiktoken

        english = "The quick brown fox jumps over the lazy dog. " * 4
        assert len(tiktoken.get_encoding("o200k_base").encode(english)) == len(
            tiktoken.get_encoding("cl100k_base").encode(english)
        )

        japanese = "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c" * 8
        assert len(tiktoken.get_encoding("o200k_base").encode(japanese)) == 16
        assert len(tiktoken.get_encoding("cl100k_base").encode(japanese)) == 32

    def test_it_degrades_to_the_estimate_when_absent(self, monkeypatch: Any) -> None:
        # Without the extra the local strategy must SAY it estimated, rather
        # than failing or silently lying.
        counter = HybridCounter(resolve_catalog("defaults"))
        monkeypatch.setattr(counter.tiktoken, "available", lambda: False)
        result = counter.count(TEXT, "openai", "gpt-4.1")
        assert result.strategy == "estimate"
        assert result.exact is False


class TestTheProviderIsNamed:
    def test_a_bare_model_without_a_provider_is_refused(self) -> None:
        with pytest.raises(ValueError, match="name the provider"):
            count_tokens(model="gpt-4.1", input=TEXT)
