"""The plumbing the guard hands a strategy.

The invariant that matters most here is that a compaction reaches the messages
being SENT, not only the stored transcript. A strategy that rewrote history
alone would look like it worked and send the long conversation anyway.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.history import ConversationHistory
from combycode_llm_sdk.agent.layers import LAYER_CHAT_FACTS, PRIORITY_CHAT_FACTS
from combycode_llm_sdk.catalog.catalog import resolve_catalog
from combycode_llm_sdk.context.facts import ExtractedFact
from combycode_llm_sdk.context.tools import (
    FIRST_USER_PREFIX,
    SYSTEM_APPEND,
    NoopContextTools,
    RunnerContextTools,
    StrategyToolsImpl,
)
from combycode_llm_sdk.tokens import HeuristicCounter, TiktokenCounter


class RecordingTools:
    """A ContextTools that answers predictably and remembers what it was asked."""

    def __init__(self, summary: str = "a summary", facts: list[ExtractedFact] | None = None) -> None:
        self.summary = summary
        self.facts = facts or []
        self.summarize_calls: list[tuple[str, int, str | None]] = []
        self.extract_calls: list[tuple[str, Any]] = []

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        self.summarize_calls.append((content, max_length, focus))
        return self.summary

    async def extract_facts(
        self, content: str, categories: Sequence[str] | None = None
    ) -> list[ExtractedFact]:
        self.extract_calls.append((content, categories))
        return self.facts


def _tools(
    n: int = 6, context_tools: Any = None
) -> tuple[StrategyToolsImpl, ConversationHistory, list[Any]]:
    history = ConversationHistory("conv")
    for i in range(n):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    active = history.messages()
    impl = StrategyToolsImpl(
        history=history,
        active_messages=active,
        counter=HeuristicCounter(resolve_catalog("defaults")),
        context_tools=context_tools or NoopContextTools(),
        provider="openai",
        model="gpt-4.1",
    )
    return impl, history, active


class TestSegmenting:
    def test_a_recent_count_splits_the_rest_in_half(self) -> None:
        impl, _, _ = _tools(6)
        seg = impl.segment(recent_count=2)
        assert [e.message["content"] for e in seg["recent"]] == ["m4", "m5"]
        assert [e.message["content"] for e in seg["old"]] == ["m0", "m1"]
        assert [e.message["content"] for e in seg["middle"]] == ["m2", "m3"]

    def test_asking_for_more_recent_than_exists_leaves_nothing_older(self) -> None:
        impl, _, _ = _tools(3)
        seg = impl.segment(recent_count=99)
        assert len(seg["recent"]) == 3
        assert seg["old"] == [] and seg["middle"] == []

    def test_with_no_hint_it_falls_back_to_thirds(self) -> None:
        impl, _, _ = _tools(6)
        seg = impl.segment()
        assert [len(seg[k]) for k in ("old", "middle", "recent")] == [2, 2, 2]

    def test_an_empty_history_segments_to_nothing(self) -> None:
        impl, _, _ = _tools(0)
        assert impl.segment(recent_count=3) == {"recent": [], "middle": [], "old": []}

    def test_a_time_window_sorts_by_age(self) -> None:
        impl, history, _ = _tools(3)
        now = history.at(0).timestamp  # type: ignore[union-attr]
        history.entries[0].timestamp = now - 10_000
        history.entries[1].timestamp = now - 2_000
        seg = impl.segment(time_window=1_000)
        assert len(seg["old"]) == 1 and len(seg["middle"]) == 1 and len(seg["recent"]) == 1


class TestMeasuring:
    def test_it_counts_entries_and_bare_messages_alike(self) -> None:
        impl, history, active = _tools(4)
        assert impl.measure(history.all()) == impl.measure(active)

    def test_more_text_costs_more(self) -> None:
        impl, _, _ = _tools(2)
        small = impl.measure([{"role": "user", "content": "hi"}])
        large = impl.measure([{"role": "user", "content": "hi" * 500}])
        assert large > small


class TestRewriting:
    def test_a_replacement_reaches_the_messages_being_sent(self) -> None:
        # The whole point. History alone would compact and still send the long
        # list the caller is holding.
        impl, history, active = _tools(6)
        impl.replace_range(0, 4, {"role": "user", "content": "[summary]"})
        assert history.length == 3
        assert active[0]["content"] == "[summary]"
        assert len(active) == 3

    def test_dropping_reaches_them_too(self) -> None:
        impl, history, active = _tools(6)
        impl.drop_oldest(4)
        assert history.length == 2
        assert [m["content"] for m in active] == ["m4", "m5"]

    def test_dropping_everything_leaves_an_empty_conversation(self) -> None:
        impl, history, active = _tools(3)
        impl.drop_oldest(99)
        assert history.length == 0 and active == []

    def test_dropping_nothing_is_not_an_error(self) -> None:
        impl, history, _ = _tools(3)
        impl.drop_oldest(0)
        assert history.length == 3

    def test_the_live_list_stays_the_same_object(self) -> None:
        # The caller kept a reference to it; rebinding would leave them holding
        # the pre-compaction list.
        impl, _, active = _tools(6)
        before = id(active)
        impl.drop_oldest(2)
        assert id(impl.active_messages) == before


class TestInjectingFacts:
    FACTS: ClassVar[list[ExtractedFact]] = [
        ExtractedFact(key="account", value="ACC-1", category="identifier")
    ]

    def test_system_append_writes_a_layer_the_next_request_composes(self) -> None:
        impl, history, _ = _tools(4)
        impl.inject_facts(self.FACTS, SYSTEM_APPEND)
        layer = history.registry.get(LAYER_CHAT_FACTS)
        assert layer is not None
        assert layer.priority == PRIORITY_CHAT_FACTS
        assert "ACC-1" in str(layer.content)
        assert "system" in layer.tags

    def test_it_keeps_the_structured_facts_beside_the_rendering(self) -> None:
        # So the next extraction reads data rather than re-parsing its own prose.
        impl, history, _ = _tools(4)
        impl.inject_facts(self.FACTS, SYSTEM_APPEND)
        layer = history.registry.get(LAYER_CHAT_FACTS)
        assert layer is not None and layer.metadata is not None
        assert layer.metadata["facts"][0]["key"] == "account"

    def test_the_facts_end_up_in_the_system_prompt(self) -> None:
        impl, history, _ = _tools(4)
        impl.inject_facts(self.FACTS, SYSTEM_APPEND)
        assert "ACC-1" in history.system

    def test_the_first_user_prefix_edits_the_first_user_message(self) -> None:
        impl, _, active = _tools(4)
        impl.inject_facts(self.FACTS, FIRST_USER_PREFIX)
        assert active[0]["content"].startswith("## Key facts")
        assert active[0]["content"].endswith("m0")

    def test_it_prefixes_the_text_part_of_a_multipart_message(self) -> None:
        history = ConversationHistory("c")
        history.append({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        active = history.messages()
        impl = StrategyToolsImpl(
            history=history, active_messages=active,
            counter=HeuristicCounter(resolve_catalog("defaults")),
            context_tools=NoopContextTools(), provider="openai", model="gpt-4.1",
        )
        impl.inject_facts(self.FACTS, FIRST_USER_PREFIX)
        assert active[0]["content"][0]["text"].endswith("hello")
        assert "ACC-1" in active[0]["content"][0]["text"]

    def test_nothing_to_inject_writes_no_layer(self) -> None:
        impl, history, _ = _tools(4)
        impl.inject_facts([], SYSTEM_APPEND)
        assert history.registry.get(LAYER_CHAT_FACTS) is None


class TestTheModelBackedHelpers:
    @pytest.mark.asyncio
    async def test_it_hands_the_summariser_the_flattened_range(self) -> None:
        tools = RecordingTools()
        impl, history, _ = _tools(4, tools)
        out = await impl.summarize(history.all()[:2], 300, focus="ids")
        assert out == "a summary"
        content, max_length, focus = tools.summarize_calls[0]
        assert "[user] m0" in content and "[assistant] m1" in content
        assert (max_length, focus) == (300, "ids")

    @pytest.mark.asyncio
    async def test_an_empty_range_never_reaches_the_model(self) -> None:
        tools = RecordingTools()
        impl, _, _ = _tools(4, tools)
        assert await impl.summarize([], 300) == ""
        assert tools.summarize_calls == []

    @pytest.mark.asyncio
    async def test_a_range_with_no_text_never_reaches_the_model(self) -> None:
        tools = RecordingTools()
        history = ConversationHistory("c")
        history.append({"role": "user", "content": "   "})
        impl = StrategyToolsImpl(
            history=history, active_messages=history.messages(),
            counter=HeuristicCounter(resolve_catalog("defaults")),
            context_tools=tools, provider="openai", model="gpt-4.1",
        )
        assert await impl.summarize(history.all(), 300) == ""
        assert tools.summarize_calls == []

    @pytest.mark.asyncio
    async def test_earlier_facts_are_fed_back_into_the_next_extraction(self) -> None:
        # Without this the second compaction extracts only from the range in
        # front of it, and everything the first one learned is dropped the
        # moment its messages are gone.
        tools = RecordingTools()
        impl, history, _ = _tools(4, tools)
        impl.inject_facts(
            [ExtractedFact(key="account", value="ACC-1", category="identifier")], SYSTEM_APPEND
        )
        await impl.extract_facts(history.all()[:2])
        content, _ = tools.extract_calls[0]
        assert "ACC-1" in content
        assert "[user] m0" in content

    @pytest.mark.asyncio
    async def test_nothing_at_all_does_not_call_the_model(self) -> None:
        tools = RecordingTools()
        impl, _, _ = _tools(0, tools)
        assert await impl.extract_facts([]) == []
        assert tools.extract_calls == []

    @pytest.mark.asyncio
    async def test_prior_facts_alone_are_still_worth_an_extraction(self) -> None:
        # Nothing new to read, but the carried facts still have to survive.
        tools = RecordingTools()
        impl, _, _ = _tools(0, tools)
        impl.inject_facts(
            [ExtractedFact(key="account", value="ACC-1", category="identifier")], SYSTEM_APPEND
        )
        await impl.extract_facts([])
        assert "ACC-1" in tools.extract_calls[0][0]

    @pytest.mark.asyncio
    async def test_the_category_filter_is_passed_through(self) -> None:
        tools = RecordingTools()
        impl, history, _ = _tools(4, tools)
        await impl.extract_facts(history.all(), ["identifier"])
        assert tools.extract_calls[0][1] == ["identifier"]


class TestTheDefaultTools:
    @pytest.mark.asyncio
    async def test_the_noop_returns_nothing_rather_than_failing(self) -> None:
        # Truncation must work without a model, so the default cannot raise.
        noop = NoopContextTools()
        assert await noop.summarize("anything", 100) == ""
        assert await noop.extract_facts("anything") == []


class _Registry:
    def __init__(self, has: bool) -> None:
        self._has = has

    async def get(self, tool_id: str) -> Any:
        return {"id": tool_id} if self._has else None


class _Runner:
    def __init__(self, has_tool: bool, out: Any) -> None:
        self.registry = _Registry(has_tool)
        self.out = out
        self.ran: list[tuple[str, Any]] = []

    async def run(self, tool_id: str, payload: Any) -> Any:
        self.ran.append((tool_id, payload))
        return self.out


class TestTheRunnerBackedTools:
    @pytest.mark.asyncio
    async def test_a_missing_extractor_costs_the_facts_not_the_compaction(self) -> None:
        # fact-extract ships as an extension, not in core.
        runner = _Runner(has_tool=False, out={"facts": [{"key": "a", "value": "1"}]})
        assert await RunnerContextTools(runner).extract_facts("text") == []
        assert runner.ran == []

    @pytest.mark.asyncio
    async def test_it_reads_the_facts_the_tool_returned(self) -> None:
        runner = _Runner(
            has_tool=True, out={"facts": [{"key": "a", "value": "1", "category": "name"}]}
        )
        facts = await RunnerContextTools(runner).extract_facts("text")
        assert [(f.key, f.value, f.category) for f in facts] == [("a", "1", "name")]

    @pytest.mark.asyncio
    async def test_a_tool_that_answers_with_nothing_useful_yields_no_facts(self) -> None:
        runner = _Runner(has_tool=True, out={"unexpected": True})
        assert await RunnerContextTools(runner).extract_facts("text") == []

    @pytest.mark.asyncio
    async def test_it_reads_the_summary_field(self) -> None:
        runner = _Runner(has_tool=True, out={"summary": "short"})
        assert await RunnerContextTools(runner).summarize("long text", 100) == "short"

    @pytest.mark.asyncio
    async def test_a_summariser_that_answers_with_nothing_returns_empty(self) -> None:
        runner = _Runner(has_tool=True, out={})
        assert await RunnerContextTools(runner).summarize("long text", 100) == ""

    @pytest.mark.asyncio
    async def test_the_tool_ids_are_overridable(self) -> None:
        runner = _Runner(has_tool=True, out={"summary": "s"})
        await RunnerContextTools(runner, summarize_id="mine:sum@2").summarize("t", 10)
        assert runner.ran[0][0] == "mine:sum@2"


class TestEstimatingAWholeMessage:
    CTX: ClassVar[dict[str, str]] = {"provider": "openai", "model": "gpt-4.1"}
    RICH: ClassVar[dict[str, Any]] = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "here you go"},
            {"type": "tool_call", "name": "search", "arguments": {"q": "berlin"}},
            {"type": "image", "url": "https://example/x.png"},
        ],
    }

    def test_a_picture_is_not_free(self) -> None:
        # A part with no text cannot be tokenised, and counting only the text
        # would report an image-heavy conversation as nearly free -- right up
        # to the request the provider refuses.
        counter = HeuristicCounter(resolve_catalog("defaults"))
        plain = counter.estimate_message({"role": "user", "content": "here you go"}, self.CTX)
        assert counter.estimate_message(self.RICH, self.CTX) > plain * 10

    def test_the_tokenizer_agrees_that_it_is_not_free(self) -> None:
        counter = TiktokenCounter(resolve_catalog("defaults"))
        plain = counter.estimate_message({"role": "user", "content": "here you go"}, self.CTX)
        assert counter.estimate_message(self.RICH, self.CTX) > plain * 10

    def test_an_image_costs_a_flat_figure_not_the_length_of_its_url(self) -> None:
        # Mutation-driven. Pricing the message by `str(content)` -- the repr of
        # the parts list -- survived every other test here, because a repr is
        # long and "an image is not free" passes either way. It is not the same
        # number: a repr grows with the URL, and the real cost of an image does
        # not. A library that prices attachments by how their metadata happens
        # to serialise is guessing, and the guess breaks the moment a caller
        # passes a data: URI.
        counter = HeuristicCounter(resolve_catalog("defaults"))
        short = {"role": "user", "content": [{"type": "image", "url": "u"}]}
        long_url = {"role": "user", "content": [{"type": "image", "url": "https://" + "x" * 400}]}
        assert counter.estimate_message(short, self.CTX) == counter.estimate_message(
            long_url, self.CTX
        )
        assert counter.estimate_message(short, self.CTX) > 100

    def test_a_plain_string_message_is_just_its_text(self) -> None:
        counter = TiktokenCounter(resolve_catalog("defaults"))
        message = {"role": "user", "content": "hello world"}
        assert counter.estimate_message(message, self.CTX) == counter.count(
            "hello world", "openai", "gpt-4.1"
        )

    def test_the_exact_counter_and_the_estimate_do_not_have_to_agree(self) -> None:
        # If they did, the tokenizer would be wired to the heuristic and every
        # "exact" count in this library would be an estimate wearing a label.
        rich = {"role": "user", "content": "こんにちは世界" * 8}
        heuristic = HeuristicCounter(resolve_catalog("defaults")).estimate_message(rich, self.CTX)
        exact = TiktokenCounter(resolve_catalog("defaults")).estimate_message(rich, self.CTX)
        assert heuristic != exact
