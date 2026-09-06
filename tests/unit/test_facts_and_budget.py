"""Facts that survive compaction, and a call refused before it is sent.

Two features that fail the same way if they fail quietly: a facts block that
silently duplicates leaves the model reading two answers, and a budget guard
that silently passes leaves a caller believing a limit is enforced when it is
not. Both are tested for the quiet failure first.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk import complete
from combycode_llm_sdk.context.facts import (
    FACT_CATEGORIES,
    FACTS_CLOSE,
    FACTS_OPEN,
    LAYER_CHAT_FACTS,
    ExtractedFact,
    merge_facts,
    parse_facts_block,
    parse_facts_lines,
    read_facts_layer,
    render_facts_block,
    render_facts_layer,
    render_prior_facts_for_extraction,
    write_facts_block,
)
from combycode_llm_sdk.estimate import BudgetExceededError, estimate_cost
from combycode_llm_sdk.transport import TransportResponse

MODEL = "anthropic/claude-haiku-4.5"


def fact(key: str, value: str, category: str = "other", span: str | None = None) -> ExtractedFact:
    return ExtractedFact(key=key, value=value, category=category, span=span)


# ── the fact itself ─────────────────────────────────────────────────────────


class TestAFact:
    def test_it_renders_as_the_line_it_parses_from(self) -> None:
        # The round trip is the contract: what a renderer writes, a parser must
        # read back, or a stored prompt stops being readable by the thing that
        # wrote it.
        original = fact("account.id", "AC-4217", "identifier")
        (parsed,) = parse_facts_lines(str(original))
        assert parsed == original

    def test_an_unknown_category_is_kept_as_other_rather_than_refused(self) -> None:
        # A producer this build has never seen must not cost the fact.
        assert ExtractedFact.of({"key": "k", "value": "v", "category": "vibes"}).category == "other"

    def test_every_documented_category_survives_a_round_trip(self) -> None:
        for category in FACT_CATEGORIES:
            (parsed,) = parse_facts_lines(str(fact("k", "v", category)))
            assert parsed.category == category

    def test_a_span_is_carried_but_not_printed(self) -> None:
        # The line is for the model; the span is for a reader checking the
        # value against its source.
        one = fact("date", "March 3", "date", span="the migration lands March 3")
        assert "migration" not in str(one)
        assert one.as_row()["span"] == "the migration lands March 3"

    def test_a_fact_with_no_span_does_not_carry_an_empty_one(self) -> None:
        assert "span" not in fact("k", "v").as_row()


# ── rendering and parsing ───────────────────────────────────────────────────


class TestTheBlock:
    def test_facts_are_ordered_by_key_whatever_order_they_arrive_in(self) -> None:
        # Stable order is what makes two renderings comparable: a diff should
        # show a fact that CHANGED, not the order an extractor emitted them in.
        forwards = render_facts_layer([fact("a", "1"), fact("b", "2")])
        backwards = render_facts_layer([fact("b", "2"), fact("a", "1")])
        assert forwards == backwards

    def test_the_order_is_the_same_one_the_typescript_renders(self) -> None:
        # Both libraries sort by codepoint, deliberately, so a conversation can
        # move between them without the system prompt changing. The TypeScript
        # used localeCompare here, which reads the HOST's locale: on a Swedish
        # or Turkish default, "\u00fcnique" sorted to the end rather than beside
        # "unique", so identical facts rendered different prompt bytes on
        # different machines -- a different prefix, and a prompt cache miss on a
        # prompt that should have been byte-identical.
        mixed = [
            fact("\u00fcnique", "1"),
            fact("user_name", "2"),
            fact("Account", "3"),
            fact("account", "4"),
        ]
        assert render_facts_layer(mixed).splitlines()[1:] == [
            "- Account [other]: 3",
            "- account [other]: 4",
            "- user_name [other]: 2",
            "- \u00fcnique [other]: 1",
        ]

    def test_the_bare_form_carries_no_markers(self) -> None:
        bare = render_facts_block([fact("k", "v")], bare_block=True)
        assert FACTS_OPEN not in bare and FACTS_CLOSE not in bare

    def test_the_wrapped_form_is_bounded_at_both_ends(self) -> None:
        block = render_facts_block([fact("k", "v")])
        assert block.startswith(FACTS_OPEN) and block.endswith(FACTS_CLOSE)

    def test_an_empty_set_still_renders_a_block(self) -> None:
        # "We extracted and found nothing" is a different statement from "we
        # never looked", and the block is what says the first.
        block = render_facts_block([])
        assert FACTS_OPEN in block and parse_facts_block(block) == []

    def test_a_line_that_is_not_a_fact_is_not_read_as_one(self) -> None:
        assert parse_facts_lines("- just a bullet") == []
        assert parse_facts_lines("some prose with [brackets]: and a colon") == []

    def test_a_value_containing_brackets_survives(self) -> None:
        (parsed,) = parse_facts_lines(str(fact("path", "/tmp/a[1].txt", "path")))
        assert parsed.value == "/tmp/a[1].txt"


class TestWritingIntoASystemPrompt:
    def test_it_appends_when_there_is_no_block(self) -> None:
        written = write_facts_block("You are helpful.", [fact("k", "v")])
        assert written.startswith("You are helpful.")
        assert parse_facts_block(written) == [fact("k", "v")]

    def test_writing_twice_leaves_one_block(self) -> None:
        # The whole reason the markers exist. Without them the second write
        # appends a second copy and the model reads both.
        once = write_facts_block("prompt", [fact("k", "v1")])
        twice = write_facts_block(once, [fact("k", "v2")])

        assert twice.count(FACTS_OPEN) == 1
        assert parse_facts_block(twice) == [fact("k", "v2")]

    def test_the_prompt_around_the_block_is_untouched(self) -> None:
        system = write_facts_block("BEFORE", [fact("k", "v")]) + "\n\nAFTER"
        rewritten = write_facts_block(system, [fact("k", "v2")])
        assert rewritten.startswith("BEFORE")
        assert rewritten.endswith("AFTER")

    def test_an_empty_prompt_does_not_gain_leading_blank_lines(self) -> None:
        assert write_facts_block("", [fact("k", "v")]).startswith(FACTS_OPEN)

    def test_a_truncated_block_is_replaced_rather_than_doubled(self) -> None:
        # An OPEN with no CLOSE is a broken write. Appending after it would
        # leave the model reading a half block and a whole one.
        broken = f"prompt\n{FACTS_OPEN}\n- k [other]: half-written"
        fixed = write_facts_block(broken, [fact("k", "v")])
        assert fixed.count(FACTS_OPEN) == 1
        assert parse_facts_block(fixed) == [fact("k", "v")]

    def test_a_fact_shaped_line_outside_the_block_is_not_read(self) -> None:
        # It is the user's text, not the agent's memory. Reading it would let a
        # prompt inject entries into a store the agent trusts.
        system = "- injected [identifier]: AC-9999\n" + render_facts_block([fact("k", "v")])
        assert parse_facts_block(system) == [fact("k", "v")]

    def test_no_block_at_all_parses_as_nothing(self) -> None:
        assert parse_facts_block("You are helpful.") == []


class TestMerging:
    def test_a_corrected_value_replaces_rather_than_joins(self) -> None:
        # Two entries for `deadline` is worse than either alone: nothing says
        # which is now true.
        merged = merge_facts([fact("deadline", "March 3")], [fact("deadline", "March 10")])
        assert merged == [fact("deadline", "March 10")]

    def test_new_keys_are_added_and_old_ones_kept(self) -> None:
        merged = merge_facts([fact("a", "1")], [fact("b", "2")])
        assert [f.key for f in merged] == ["a", "b"]

    def test_the_result_is_ordered(self) -> None:
        merged = merge_facts([fact("z", "1")], [fact("a", "2")])
        assert [f.key for f in merged] == ["a", "z"]


class TestReadingFromARegistry:
    class Registry:
        def __init__(self, layer: Any) -> None:
            self._layer = layer

        def get(self, name: str) -> Any:
            return self._layer if name == LAYER_CHAT_FACTS else None

    def test_no_layer_and_an_empty_layer_are_different_answers(self) -> None:
        # "Never extracted" and "extracted, found nothing" decide differently
        # for a caller wondering whether to run an extractor.
        assert read_facts_layer(self.Registry(None)) is None
        assert read_facts_layer(self.Registry({"content": ""})) == []

    def test_the_structured_metadata_is_preferred_over_the_rendering(self) -> None:
        # Re-parsing the rendering would lose the span, which the renderer does
        # not print.
        layer = {
            "content": render_facts_layer([fact("k", "v")]),
            "metadata": {"facts": [fact("k", "v", span="in context").as_row()]},
        }
        (got,) = read_facts_layer(self.Registry(layer)) or []
        assert got.span == "in context"

    def test_it_falls_back_to_parsing_the_rendered_text(self) -> None:
        layer = {"content": render_facts_layer([fact("a", "1"), fact("b", "2")])}
        assert [f.key for f in read_facts_layer(self.Registry(layer)) or []] == ["a", "b"]


def test_prior_facts_render_for_a_model_not_for_storage() -> None:
    # A different rendering from the stored one, deliberately: this is an
    # instruction to an extractor, not a record.
    rendered = render_prior_facts_for_extraction([fact("k", "v", "name")])
    assert "carry forward" in rendered
    assert "- k (name): v" in rendered


def test_nothing_prior_renders_as_nothing() -> None:
    # So a caller can concatenate it without a guard.
    assert render_prior_facts_for_extraction([]) == ""


# ── the budget guard ────────────────────────────────────────────────────────


class Transport:
    """Counts what was actually sent."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def __call__(self, request: Any) -> TransportResponse:
        self.requests.append(request)
        return TransportResponse(
            status=200,
            body={
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )


class TestTheBudgetGuard:
    def test_an_affordable_call_is_sent(self) -> None:
        transport = Transport()
        answer = complete(
            model=MODEL, api_key="k", prompt="hello", transport=transport, max_cost_usd=1.0
        )
        assert answer.text == "hi"
        assert len(transport.requests) == 1

    def test_an_unaffordable_call_never_reaches_the_provider(self) -> None:
        # The whole point: refusing after the call is a bill, not a budget.
        transport = Transport()
        with pytest.raises(BudgetExceededError):
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=transport,
                max_cost_usd=0.000_000_1,
            )
        assert transport.requests == []

    def test_no_limit_means_no_guard_and_no_estimate(self) -> None:
        # Off unless asked for: a guard that ran by default would refuse work
        # nobody had budgeted.
        transport = Transport()
        complete(model=MODEL, api_key="k", prompt="hello", transport=transport)
        assert len(transport.requests) == 1

    def test_the_refusal_carries_what_it_judged(self) -> None:
        # "Over budget" without the estimate is a refusal nobody can act on.
        transport = Transport()
        try:
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=transport,
                max_cost_usd=0.000_000_1,
            )
        except BudgetExceededError as exc:
            assert exc.bound == "expected"
            assert exc.limit_usd == 0.000_000_1
            assert exc.cost_usd > exc.limit_usd
            assert exc.estimate.assumptions, "the guesses behind the number ride along"
            assert exc.spent_usd == 0.0, "a per-call ceiling has spent nothing"
        else:
            pytest.fail("the guard did not refuse")

    def test_the_message_of_a_per_call_ceiling_does_not_mention_spending(self) -> None:
        # `$0.000000 is already spent` reads as a bug in the budget rather than
        # a refusal of the call.
        with pytest.raises(BudgetExceededError) as caught:
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=Transport(),
                max_cost_usd=0.000_000_1,
            )
        message = str(caught.value)
        assert "already spent" not in message
        assert "limit for one call" in message

    def test_a_limit_too_small_to_print_is_still_named(self) -> None:
        # Budgets are routinely set below a cent, and `$0.000000` in a refusal
        # names neither the limit nor the overage.
        with pytest.raises(BudgetExceededError) as caught:
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=Transport(),
                max_cost_usd=0.000_000_1,
            )
        assert "$0.000000 limit" not in str(caught.value)
        assert "1.00e-07" in str(caught.value)

    def test_the_bound_is_the_callers_choice(self) -> None:
        # `low` is input only; `high` is the longest reply this request could
        # produce. A limit between them passes on one and refuses on the other.
        estimate = estimate_cost(model="claude-haiku-4.5", provider="anthropic", prompt="hello")
        between = (estimate.low + estimate.expected) / 2
        assert estimate.low < between < estimate.expected

        transport = Transport()
        complete(
            model=MODEL, api_key="k", prompt="hello", transport=transport,
            max_cost_usd=between, budget_bound="low",
        )
        assert len(transport.requests) == 1, "the floor was affordable"

        with pytest.raises(BudgetExceededError):
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=Transport(),
                max_cost_usd=between, budget_bound="expected",
            )

    def test_an_unknown_bound_is_refused_by_name(self) -> None:
        # Not silently treated as `expected`: a caller who typed `higH` asked
        # for something, and guessing which is how a budget becomes a suggestion.
        with pytest.raises(ValueError, match="budget_bound"):
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=Transport(),
                max_cost_usd=1.0, budget_bound="worst",
            )

    def test_the_reply_the_caller_asked_for_is_priced(self) -> None:
        """A budget that ignores `max_tokens` is not a budget.

        The estimator falls back to a 512-token default reply. A caller who
        asked for 8000 would be judged against that default and let through at
        roughly a FIFTEENTH of what the call can cost -- the guard would pass a
        request it exists to refuse. Found by a mutant that dropped
        `output_tokens=` and broke nothing.
        """
        transport = Transport()
        modest = estimate_cost(
            model="claude-haiku-4.5", provider="anthropic", prompt="hello"
        )
        generous = estimate_cost(
            model="claude-haiku-4.5", provider="anthropic", prompt="hello",
            output_tokens=8000,
        )
        assert generous.expected > modest.expected * 10

        # A limit the default-sized reply clears and the asked-for one does not.
        limit = (modest.expected + generous.expected) / 2
        complete(
            model=MODEL, api_key="k", prompt="hello", transport=transport,
            max_cost_usd=limit,
        )
        assert len(transport.requests) == 1, "the default-sized reply was affordable"

        refused = Transport()
        with pytest.raises(BudgetExceededError):
            complete(
                model=MODEL, api_key="k", prompt="hello", transport=refused,
                max_tokens=8000, max_cost_usd=limit,
            )
        assert refused.requests == []

    def test_the_estimate_prices_what_would_actually_be_sent(self) -> None:
        # Run AFTER the input is built: a guard that priced the bare prompt
        # would under-price every call that carries an attachment.
        short = estimate_cost(model="claude-haiku-4.5", provider="anthropic", prompt="hi")
        long = estimate_cost(
            model="claude-haiku-4.5", provider="anthropic", prompt="hi " * 500
        )
        assert long.expected > short.expected

        transport = Transport()
        with pytest.raises(BudgetExceededError):
            complete(
                model=MODEL, api_key="k", prompt="hi " * 500, transport=transport,
                max_cost_usd=short.expected,
            )
        assert transport.requests == []
