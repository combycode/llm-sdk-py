"""The three compaction policies, and the guard that routes pressure to them.

The failures worth guarding here are the ones that look like success: a strategy
that summarises the same range on every measurement because the guard cannot
tell a new crossing from a re-reading, a compaction judged on the count that
triggered it rather than on what it left, and a summariser that returns nothing
being allowed to delete the conversation it failed to summarise.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.history import ConversationHistory
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.context.context_guard import STATE_KEY, ContextGuard
from combycode_llm_sdk.context.facts import ExtractedFact
from combycode_llm_sdk.context.guard import ContextMeasurer
from combycode_llm_sdk.context.strategies import (
    ANCHOR_MARKER,
    AnchoredStrategy,
    LayeredStrategy,
    TruncateStrategy,
    calculate_retain_start_index,
    merge_anchor,
)
from combycode_llm_sdk.context.strategy_types import (
    ReactContext,
    StrategyDecision,
    TriggerLevel,
    highest_crossed_level,
)
from combycode_llm_sdk.context.tools import NoopContextTools, StrategyToolsImpl
from combycode_llm_sdk.tokens import HeuristicCounter


class Tools:
    """A ContextTools that answers predictably."""

    def __init__(self, summary: str = "the state so far", facts: list[Any] | None = None) -> None:
        self.summary = summary
        self.facts = facts if facts is not None else []

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        return self.summary

    async def extract_facts(
        self, content: str, categories: Sequence[str] | None = None
    ) -> list[Any]:
        return list(self.facts)


def _ctx(
    turns: int = 12,
    *,
    level: str = "urgent",
    percentage: float = 0.9,
    window: int = 1000,
    delta: int = 0,
    attempt: int = 0,
    context_tools: Any = None,
    messages: list[Any] | None = None,
) -> tuple[ReactContext, ConversationHistory]:
    history = ConversationHistory("conv")
    for message in messages or [{"role": "user", "content": f"turn {i}"} for i in range(turns)]:
        history.append(message)
    active = history.messages()
    tools = StrategyToolsImpl(
        history=history,
        active_messages=active,
        counter=HeuristicCounter(resolve_catalog("defaults")),
        context_tools=context_tools or Tools(),
        provider="openai",
        model="gpt-4.1",
    )
    return (
        ReactContext(
            level=level,
            percentage=percentage,
            current=int(percentage * window),
            window=window,
            delta=delta,
            provider="openai",
            model="gpt-4.1",
            attempt=attempt,
            tools=tools,
        ),
        history,
    )


class TestTruncate:
    @pytest.mark.asyncio
    async def test_it_keeps_the_recent_turns(self) -> None:
        ctx, history = _ctx(12)
        decision = await TruncateStrategy(keep_recent=4).react(ctx)
        assert decision.action == "compacted"
        assert history.length == 4
        assert next(m["content"] for m in ctx.tools.active_messages) == "turn 8"

    @pytest.mark.asyncio
    async def test_a_conversation_that_already_fits_is_left_alone(self) -> None:
        ctx, history = _ctx(3)
        assert (await TruncateStrategy(keep_recent=4).react(ctx)).action == "none"
        assert history.length == 3

    @pytest.mark.asyncio
    async def test_a_truncation_that_worked_is_not_declined(self) -> None:
        # The count that TRIGGERED compaction is not the count to judge it by.
        # Reading `ctx.current` here refuses work that succeeded: a truncation
        # taking a conversation from 99% of the window to a fraction of it was
        # declined for being "still above 95%", after it had already destroyed
        # the messages. The caller lost the history AND the call.
        ctx, history = _ctx(12, percentage=0.99)
        decision = await TruncateStrategy(keep_recent=4, decline_ceiling=0.95).react(ctx)
        assert decision.action == "compacted", decision
        assert history.length == 4

    @pytest.mark.asyncio
    async def test_it_declines_when_dropping_genuinely_did_not_help(self) -> None:
        # Same ceiling, but what is LEFT really is still over it, so the caller
        # must hear that the cheapest compaction was not enough.
        ctx, _ = _ctx(
            12,
            percentage=0.99,
            messages=[{"role": "user", "content": "x" * 4000} for _ in range(12)],
        )
        decision = await TruncateStrategy(keep_recent=4, decline_ceiling=0.95).react(ctx)
        assert decision.action == "decline"
        assert "95%" in (decision.reason or "")

    @pytest.mark.asyncio
    async def test_an_unknown_window_is_not_read_as_full(self) -> None:
        ctx, _ = _ctx(12, window=0)
        ctx.window = None
        assert (await TruncateStrategy(keep_recent=4).react(ctx)).action == "compacted"


class TestLayered:
    @pytest.mark.asyncio
    async def test_healthy_touches_only_the_oldest_zone(self) -> None:
        ctx, history = _ctx(12, level="healthy")
        decision = await LayeredStrategy(recent_count=6).react(ctx)
        assert decision.action == "compacted"
        assert history.length < 12
        assert "Earlier conversation summary" in str(history.at(0).message["content"])  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_critical_keeps_only_the_last_turns(self) -> None:
        ctx, history = _ctx(12, level="critical")
        decision = await LayeredStrategy().react(ctx)
        assert decision.action == "compacted"
        assert history.length == 3  # one summary + CRITICAL_KEEP_LAST

    @pytest.mark.asyncio
    async def test_critical_with_nothing_left_to_drop_declines(self) -> None:
        ctx, _ = _ctx(2, level="critical")
        decision = await LayeredStrategy().react(ctx)
        assert decision.action == "decline"
        assert "compaction can free" in (decision.reason or "")

    @pytest.mark.asyncio
    async def test_the_facts_survive_the_messages_they_came_from(self) -> None:
        fact = ExtractedFact(key="account", value="ACC-1", category="identifier")
        ctx, history = _ctx(12, level="healthy", context_tools=Tools(facts=[fact]))
        await LayeredStrategy(recent_count=6).react(ctx)
        assert "ACC-1" in history.system

    @pytest.mark.asyncio
    async def test_it_declines_only_after_it_has_actually_tried(self) -> None:
        # Declining on the first look would refuse work compaction was about to
        # make possible.
        first, _ = _ctx(12, level="healthy", percentage=0.95)
        assert (await LayeredStrategy(decline_ceiling=0.9).react(first)).action == "compacted"
        again, _ = _ctx(12, level="healthy", percentage=0.95, attempt=1)
        assert (await LayeredStrategy(decline_ceiling=0.9).react(again)).action == "decline"

    @pytest.mark.asyncio
    async def test_a_large_jump_escalates_a_rung(self) -> None:
        # Arriving at a level gradually and arriving there in one turn are not
        # the same situation; the second may overshoot the window next turn.
        strategy = LayeredStrategy(recent_count=6, jump_escalate_delta=0.3)
        gradual, calm_history = _ctx(12, level="healthy", delta=10)
        await strategy.react(gradual)
        jumped, jumpy_history = _ctx(12, level="healthy", delta=500)
        await strategy.react(jumped)
        # 'healthy' compacts old only; escalating to 'pressure' also takes the
        # middle, so strictly more is collapsed.
        assert jumpy_history.length < calm_history.length

    @pytest.mark.asyncio
    async def test_the_top_of_the_ladder_does_not_escalate_past_itself(self) -> None:
        ctx, _ = _ctx(12, level="critical", delta=900)
        assert (await LayeredStrategy().react(ctx)).action == "compacted"

    @pytest.mark.asyncio
    async def test_an_unrecognised_level_does_nothing(self) -> None:
        ctx, history = _ctx(12, level="not-a-level")
        assert (await LayeredStrategy().react(ctx)).action == "none"
        assert history.length == 12


class TestAnchored:
    @pytest.mark.asyncio
    async def test_it_writes_one_marked_anchor(self) -> None:
        ctx, history = _ctx(12)
        decision = await AnchoredStrategy(keep_recent=4).react(ctx)
        assert decision.action == "compacted"
        head = history.at(0)
        assert head is not None
        assert head.message["role"] == "system"
        assert str(head.message["content"]).startswith(ANCHOR_MARKER)

    @pytest.mark.asyncio
    async def test_the_second_pass_merges_into_the_existing_anchor(self) -> None:
        # The whole point: facts are summarised from raw text once, rather than
        # a summary being re-summarised until it drifts.
        ctx, history = _ctx(12, context_tools=Tools(summary="first pass"))
        await AnchoredStrategy(keep_recent=4).react(ctx)
        for i in range(8):
            history.append({"role": "user", "content": f"later {i}"})
        ctx.tools.active_messages[:] = history.messages()
        ctx.tools.context_tools = Tools(summary="second pass")
        decision = await AnchoredStrategy(keep_recent=4).react(ctx)
        assert decision.action == "compacted"
        assert "existing" in (decision.note or "")
        head = history.at(0)
        assert head is not None
        assert "first pass" in str(head.message["content"])
        assert "second pass" in str(head.message["content"])

    @pytest.mark.asyncio
    async def test_a_summariser_that_says_nothing_must_not_erase_history(self) -> None:
        # Trading a context overflow for silent data loss is strictly worse: one
        # is a failed request, the other is a conversation that continues while
        # quietly missing what it was about.
        ctx, history = _ctx(12, context_tools=Tools(summary="   "))
        decision = await AnchoredStrategy(keep_recent=4).react(ctx)
        assert decision.action == "decline"
        assert history.length == 12

    @pytest.mark.asyncio
    async def test_a_short_conversation_is_left_alone(self) -> None:
        ctx, history = _ctx(4)
        assert (await AnchoredStrategy(keep_recent=4).react(ctx)).action == "none"
        assert history.length == 4


class TestTheRetainBoundary:
    def test_it_refuses_to_split_a_call_from_its_result(self) -> None:
        # A tool_result whose call is gone is rejected outright by some
        # providers and silently misread by the rest.
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "go"})
        history.append(
            {"role": "assistant", "content": [{"type": "tool_call", "name": "f", "arguments": {}}]}
        )
        history.append({"role": "user", "content": [{"type": "tool_result", "content": "ok"}]})
        entries = history.all()
        # A naive boundary would cut at index 2, orphaning the result.
        assert calculate_retain_start_index(entries, 1) == 1

    def test_an_ordinary_boundary_is_left_where_it_falls(self) -> None:
        history = ConversationHistory("c")
        for i in range(5):
            history.append({"role": "user", "content": f"m{i}"})
        assert calculate_retain_start_index(history.all(), 2) == 3

    def test_keeping_more_than_exists_starts_at_the_beginning(self) -> None:
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "only"})
        assert calculate_retain_start_index(history.all(), 99) == 0


class TestMergingTheAnchor:
    def test_it_joins_the_new_text_onto_the_old(self) -> None:
        assert merge_anchor("older", "newer", 100) == "older\nnewer"

    def test_the_first_merge_has_nothing_to_join_to(self) -> None:
        assert merge_anchor("", "first", 100) == "first"

    def test_it_is_bounded_so_the_anchor_cannot_become_the_problem(self) -> None:
        assert len(merge_anchor("x" * 500, "y" * 500, 100)) == 100

    def test_trimming_keeps_the_newer_half(self) -> None:
        # The newer text already subsumes the older state, so the end is the
        # half worth having.
        assert merge_anchor("old" * 50, "NEWEST", 10).endswith("NEWEST")


class TestTheTriggerLadder:
    def test_it_reports_the_furthest_rung_reached(self) -> None:
        ladder = [TriggerLevel("a", 0.5), TriggerLevel("b", 0.7), TriggerLevel("c", 0.9)]
        assert highest_crossed_level(ladder, 0.75) == 1

    def test_below_every_rung_is_minus_one(self) -> None:
        assert highest_crossed_level([TriggerLevel("a", 0.5)], 0.1) == -1

    def test_exactly_on_a_rung_counts_as_reaching_it(self) -> None:
        assert highest_crossed_level([TriggerLevel("a", 0.5)], 0.5) == 0


class Recorder:
    """A strategy that records what it was asked and answers as told."""

    def __init__(self, decision: StrategyDecision | None = None) -> None:
        self.triggers = [TriggerLevel("low", 0.5), TriggerLevel("high", 0.9)]
        self.seen: list[ReactContext] = []
        self.decision = decision or StrategyDecision.none()

    async def react(self, ctx: ReactContext) -> StrategyDecision:
        self.seen.append(ctx)
        return self.decision


def _guard(strategy: Any, **kwargs: Any) -> tuple[ContextGuard, HookBus]:
    hooks = HookBus()
    guard = ContextGuard(
        hooks=hooks,
        measurer=ContextMeasurer(resolve_catalog("defaults")),
        strategies={"demo": strategy},
        default_strategy="demo",
        context_tools=NoopContextTools(),
        **kwargs,
    )
    return guard, hooks


def _measure_ctx(history: ConversationHistory, percentage: float, current: int) -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-4.1",
        "current": current,
        "window": 1000,
        "percentage": percentage,
        "messages": history.messages(),
        "history": history,
        "abort": None,
        "abortReason": None,
    }


class TestTheGuard:
    @pytest.mark.asyncio
    async def test_it_reacts_when_a_new_rung_is_crossed(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "hi"})
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.6, 600))
        assert len(strategy.seen) == 1
        assert strategy.seen[0].level == "low"

    @pytest.mark.asyncio
    async def test_the_same_pressure_re_reported_does_not_react_again(self) -> None:
        # Otherwise the same range is summarised on every measurement while the
        # conversation sits still.
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "hi"})
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.6, 600))
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.6, 600))
        assert len(strategy.seen) == 1

    @pytest.mark.asyncio
    async def test_still_growing_on_the_same_rung_does_react(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "hi"})
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.6, 600))
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.7, 700))
        assert len(strategy.seen) == 2

    @pytest.mark.asyncio
    async def test_below_every_rung_nothing_runs(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.1, 100))
        assert strategy.seen == []

    @pytest.mark.asyncio
    async def test_an_unknown_window_is_not_guessed_at(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        ctx = _measure_ctx(history, 0.9, 900)
        ctx["window"] = None
        ctx["percentage"] = None
        await hooks.emit("onContextMeasure", ctx)
        assert strategy.seen == []

    @pytest.mark.asyncio
    async def test_a_conversation_can_opt_out(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c", strategy=False)
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.95, 950))
        assert strategy.seen == []

    @pytest.mark.asyncio
    async def test_declining_stops_the_request(self) -> None:
        strategy = Recorder(StrategyDecision.decline("too big"))
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        ctx = _measure_ctx(history, 0.95, 950)
        await hooks.emit("onContextMeasure", ctx)
        assert ctx["abort"] is True
        assert ctx["abortReason"] == "too big"

    @pytest.mark.asyncio
    async def test_a_warning_does_not_stop_the_request(self) -> None:
        seen: list[Any] = []
        strategy = Recorder(StrategyDecision.warn("getting full"))
        _, hooks = _guard(strategy)
        hooks.on("onWarning", lambda c: seen.append(dict(c)))
        history = ConversationHistory("c")
        ctx = _measure_ctx(history, 0.95, 950)
        await hooks.emit("onContextMeasure", ctx)
        assert not ctx["abort"]
        assert seen and seen[0]["code"] == "context_pressure"

    @pytest.mark.asyncio
    async def test_it_gives_up_after_the_retry_budget(self) -> None:
        # A strategy that keeps saying "compacted" while nothing shrinks must
        # not loop for ever.
        strategy = Recorder(StrategyDecision.compacted("no-op"))
        _, hooks = _guard(strategy, max_compact_retries=2)
        history = ConversationHistory("c")
        for i in range(40):
            history.append({"role": "user", "content": "x" * 400})
        ctx = _measure_ctx(history, 0.99, 990)
        await hooks.emit("onContextMeasure", ctx)
        assert ctx["abort"] is True
        assert "compaction attempts" in str(ctx["abortReason"])
        assert len(strategy.seen) == 3

    @pytest.mark.asyncio
    async def test_the_strategy_is_told_which_attempt_this_is(self) -> None:
        strategy = Recorder(StrategyDecision.compacted("no-op"))
        _, hooks = _guard(strategy, max_compact_retries=2)
        history = ConversationHistory("c")
        for i in range(40):
            history.append({"role": "user", "content": "x" * 400})
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.99, 990))
        assert [c.attempt for c in strategy.seen] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_a_compaction_that_worked_is_accepted(self) -> None:
        strategy = TruncateStrategy(keep_recent=1, triggers=[TriggerLevel("urgent", 0.5)])
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        for i in range(20):
            history.append({"role": "user", "content": "x" * 400})
        ctx = _measure_ctx(history, 0.9, 900)
        await hooks.emit("onContextMeasure", ctx)
        assert not ctx["abort"]
        # Re-measured on what is LEFT, not on the count that triggered it.
        assert ctx["current"] < 900

    @pytest.mark.asyncio
    async def test_it_remembers_where_it_was_on_the_ladder(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.6, 600))
        assert history.metadata[STATE_KEY]["contextGuard"]["lastLevelIdx"] == 0

    @pytest.mark.asyncio
    async def test_two_conversations_do_not_share_a_place_on_the_ladder(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy)
        first, second = ConversationHistory("a"), ConversationHistory("b")
        await hooks.emit("onContextMeasure", _measure_ctx(first, 0.6, 600))
        await hooks.emit("onContextMeasure", _measure_ctx(second, 0.6, 600))
        assert len(strategy.seen) == 2

    @pytest.mark.asyncio
    async def test_an_unknown_strategy_warns_and_is_skipped(self) -> None:
        seen: list[Any] = []
        strategy = Recorder()
        _, hooks = _guard(strategy)
        hooks.on("onWarning", lambda c: seen.append(dict(c)))
        history = ConversationHistory("c", strategy="nope")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.95, 950))
        assert strategy.seen == []
        assert seen and seen[0]["code"] == "context_unknown_strategy"

    @pytest.mark.asyncio
    async def test_it_warns_once_rather_than_every_request(self) -> None:
        seen: list[Any] = []
        strategy = Recorder()
        _, hooks = _guard(strategy)
        hooks.on("onWarning", lambda c: seen.append(dict(c)))
        history = ConversationHistory("c", strategy="nope")
        for share in (0.95, 0.96, 0.97):
            await hooks.emit("onContextMeasure", _measure_ctx(history, share, int(share * 1000)))
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_it_can_fall_back_to_the_default_instead(self) -> None:
        strategy = Recorder()
        _, hooks = _guard(strategy, on_unknown_strategy="fallback-default")
        history = ConversationHistory("c", strategy="nope")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.95, 950))
        assert len(strategy.seen) == 1

    def test_a_default_that_is_not_registered_is_refused_at_construction(self) -> None:
        # Later is too late: the failure would land on a request, not on setup.
        with pytest.raises(ValueError, match="not in the strategies map"):
            ContextGuard(
                hooks=HookBus(),
                measurer=ContextMeasurer(resolve_catalog("defaults")),
                strategies={"demo": Recorder()},
                default_strategy="missing",
            )

    @pytest.mark.asyncio
    async def test_destroy_unsubscribes(self) -> None:
        strategy = Recorder()
        guard, hooks = _guard(strategy)
        guard.destroy()
        history = ConversationHistory("c")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.95, 950))
        assert strategy.seen == []

    @pytest.mark.asyncio
    async def test_the_ladder_is_sorted_before_it_is_walked(self) -> None:
        # `highest_crossed_level` stops at the first rung it cannot reach, so an
        # unsorted ladder would silently stop looking partway up.
        strategy = Recorder()
        strategy.triggers = [TriggerLevel("high", 0.9), TriggerLevel("low", 0.5)]
        _, hooks = _guard(strategy)
        history = ConversationHistory("c")
        await hooks.emit("onContextMeasure", _measure_ctx(history, 0.95, 950))
        assert strategy.seen and strategy.seen[0].level == "high"


class TestTheMeasurersEmit:
    @pytest.mark.asyncio
    async def test_it_publishes_a_reading_the_guard_can_read(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("onContextMeasure", lambda c: seen.append(dict(c)))
        measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=hooks)
        history = ConversationHistory("c")
        await measurer.measure_and_emit(
            "openai", "gpt-4.1", [{"role": "user", "content": "hello"}], history=history
        )
        assert seen and seen[0]["current"] > 0
        assert seen[0]["history"] is history

    @pytest.mark.asyncio
    async def test_the_system_prompt_is_counted_too(self) -> None:
        hooks = HookBus()
        measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=hooks)
        messages = [{"role": "user", "content": "hello"}]
        bare = await measurer.measure_and_emit("openai", "gpt-4.1", messages)
        withsys = await measurer.measure_and_emit(
            "openai", "gpt-4.1", messages, system="You are a helpful assistant." * 10
        )
        assert withsys["total"] > bare["total"]

    @pytest.mark.asyncio
    async def test_a_listener_that_compacted_changes_the_total_reported(self) -> None:
        # Reporting the pre-compaction number would describe a request that is
        # no longer the one being sent.
        hooks = HookBus()
        measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=hooks)

        async def shrink(ctx: Any) -> None:
            ctx["messages"][:] = ctx["messages"][:1]

        hooks.on("onContextMeasure", shrink)
        messages = [{"role": "user", "content": "x" * 400} for _ in range(10)]
        result = await measurer.measure_and_emit("openai", "gpt-4.1", messages)
        assert result["total"] < 400
        assert not result["abort"]

    @pytest.mark.asyncio
    async def test_an_abort_is_reported_back_to_the_caller(self) -> None:
        hooks = HookBus()
        measurer = ContextMeasurer(resolve_catalog("defaults"), hooks=hooks)

        async def refuse(ctx: Any) -> None:
            ctx["abort"] = True
            ctx["abortReason"] = "no room"

        hooks.on("onContextMeasure", refuse)
        result = await measurer.measure_and_emit(
            "openai", "gpt-4.1", [{"role": "user", "content": "hi"}]
        )
        assert result["abort"] and result["abortReason"] == "no room"
