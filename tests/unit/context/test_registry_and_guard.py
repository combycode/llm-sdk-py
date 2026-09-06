"""Layers, and the guard that acts on how big they got.

The failures worth guarding are the quiet ones: an order that depends on which
code ran first (and moves the cache prefix), an estimate wearing a
measurement's label, and a compaction judged on the count that triggered it
rather than on what it left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from combycode_llm_sdk import (
    ContextMeasurer,
    ContextRegistry,
)
from combycode_llm_sdk.context import registry as registry_module

WINDOW = 1_000
CHARS_PER_TOKEN = 4
MESSAGE_CHARS = 400
TURN_TOKENS = MESSAGE_CHARS // CHARS_PER_TOKEN


@dataclass(frozen=True)
class Entry:
    context_window: int
    raw: dict[str, Any]


class StubCatalog:
    def __init__(self, strategy: str = "heuristic") -> None:
        self.strategy = strategy

    def get(self, provider: str, model: str) -> Entry:
        return Entry(
            context_window=WINDOW,
            raw={"tokenizer": {"strategy": self.strategy,
                               "charsPerTokenDefault": CHARS_PER_TOKEN}},
        )


def conversation(turns: int) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"turn {i}".ljust(MESSAGE_CHARS, ".")}
        for i in range(turns)
    ]


class _Clock:
    """A pinned clock: each call takes the next tick, then holds the last.

    It holds rather than raising because `render()` reads the clock too, and a
    test about ORDER should not depend on how many times the code asks the time.
    """

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)

    def __call__(self) -> float:
        return self._ticks.pop(0) if len(self._ticks) > 1 else self._ticks[0]


class TestPrecedence:
    def test_priority_decides_the_order_not_insertion(self) -> None:
        # Written churniest-first on purpose: if insertion order decided, the
        # facts would render in front of the persona and the cache prefix would
        # move every time either was written.
        registry = ContextRegistry(id="c")
        registry.set("chat.facts", "a fact", priority=250)
        registry.set("agentloop.context", "the run", priority=100)
        registry.set("agentloop.system", "the persona", priority=10)
        assert [p.name for p in registry.render().parts] == [
            "agentloop.system",
            "agentloop.context",
            "chat.facts",
        ]

    def test_a_rewrite_keeps_the_priority_it_was_given(self) -> None:
        # The quiet one: a writer updating CONTENT without restating the
        # priority would fall back to the default and jump the run context.
        registry = ContextRegistry()
        registry.set("chat.facts", "one", priority=250)
        assert registry.set("chat.facts", "two").priority == 250

    def test_every_mutation_bumps_the_version(self) -> None:
        registry = ContextRegistry()
        assert registry.set("a", "1").version == 1
        assert registry.set("a", "2").version == 2

    def test_rewriting_a_churny_layer_leaves_the_prefix_alone(self) -> None:
        registry = ContextRegistry()
        registry.set("agentloop.system", "PERSONA", priority=10)
        registry.set("chat.facts", "one", priority=250)
        before = registry.flat()[:7]
        registry.set("chat.facts", "one, and more")
        assert registry.flat()[:7] == before

    def test_a_tie_is_broken_by_age(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The clock is pinned, because otherwise this tests the CLOCK. Two
        # adjacent `set` calls land in the same millisecond wherever the timer
        # is coarse -- measured on a Windows CI runner, where `updated_at` tied,
        # the order fell through to name, and this assertion failed there while
        # passing on Linux and macOS.
        monkeypatch.setattr(registry_module, "_now_ms", _Clock([1000.0, 2000.0]))
        registry = ContextRegistry()
        registry.set("b", "second", priority=100)
        registry.set("a", "first", priority=100)
        assert [p.name for p in registry.render().parts] == ["b", "a"]

    def test_layers_written_in_the_same_instant_fall_back_to_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What a coarse clock actually produces, and the case the suite never
        # covered. The order still has to be FIXED: the rendered text is a cache
        # prefix, and one that varies with the host's timer resolution is a
        # different prefix on every machine.
        monkeypatch.setattr(registry_module, "_now_ms", _Clock([1000.0]))
        registry = ContextRegistry()
        registry.set("b", "second", priority=100)
        registry.set("a", "first", priority=100)
        assert [p.name for p in registry.render().parts] == ["a", "b"]


class TestInheritance:
    def build(self) -> tuple[ContextRegistry, ContextRegistry]:
        org = ContextRegistry(id="org")
        org.set("agentloop.system", "ORG PERSONA", priority=10)
        org.set("memory", "org memory", priority=200)
        chat = ContextRegistry(id="chat", parent=org)
        chat.set("agentloop.system", "CHAT PERSONA", priority=10)
        chat.set("memory", "chat memory", priority=200, merge_parent=True)
        return org, chat

    def test_a_child_layer_replaces_the_parents_by_default(self) -> None:
        _org, chat = self.build()
        composed = chat.flat()
        assert "CHAT PERSONA" in composed
        assert "ORG PERSONA" not in composed

    def test_merge_parent_keeps_the_inherited_half_first(self) -> None:
        _org, chat = self.build()
        composed = chat.flat()
        assert composed.index("org memory") < composed.index("chat memory")

    def test_merging_is_a_render_not_a_write(self) -> None:
        # `get` answering the merged value would report content this registry
        # does not hold and cannot edit.
        _org, chat = self.build()
        layer = chat.get("memory")
        assert layer is not None
        assert layer.content == "chat memory"

    def test_the_parent_can_be_skipped(self) -> None:
        _org, chat = self.build()
        assert "org memory" not in chat.flat(include_parent=False)

    def test_a_part_says_which_registry_it_came_from(self) -> None:
        # The thing you need when a cascade is not composing as expected.
        org = ContextRegistry(id="org")
        org.set("memory", "inherited")
        chat = ContextRegistry(id="chat", parent=org)
        assert [p.registry for p in chat.render().parts] == ["org"]


class TestLayersAreNotMessages:
    def test_a_layer_outlives_the_turn_that_stated_it(self) -> None:
        # A fact stated once in message 3 dies the moment message 3 is
        # summarised away. Compaction rewrites the message list and has nothing
        # here to drop.
        registry = ContextRegistry()
        registry.set("chat.facts", "the on-call engineer is Dana", priority=250)
        messages = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
        del messages[:4]
        assert "Dana" not in " ".join(str(m["content"]) for m in messages)
        assert "Dana" in registry.flat()


class TestTheMeasurer:
    def test_it_estimates_cheaply_below_the_threshold(self) -> None:
        escalations: list[str] = []

        def counter(**kwargs: Any) -> Any:
            escalations.append(str(kwargs["model"]))
            raise AssertionError("should not escalate")

        measurer = ContextMeasurer(StubCatalog(), count=counter)
        measured = measurer.measure("stub", "model", conversation(4))
        assert measured.tokens == 4 * TURN_TOKENS
        assert measured.strategy == "estimate"
        assert measured.exact is False
        # A round trip at 40% of the window buys a number nobody compares.
        assert escalations == []

    def test_it_escalates_near_the_window(self) -> None:
        @dataclass(frozen=True)
        class Counted:
            tokens: int
            strategy: str
            exact: bool

        seen: list[str] = []

        def counter(**kwargs: Any) -> Counted:
            seen.append(str(kwargs["model"]))
            return Counted(tokens=920, strategy="tiktoken", exact=True)

        measured = ContextMeasurer(StubCatalog(), count=counter).measure(
            "stub", "model", conversation(10)
        )
        assert seen == ["stub/model"]
        assert measured.strategy == "tiktoken"
        assert measured.exact is True
        # The exact number REPLACES the estimate rather than joining it.
        assert measured.tokens == 920

    def test_an_inexact_answer_is_discarded_not_promoted(self) -> None:
        @dataclass(frozen=True)
        class Counted:
            tokens: int
            strategy: str
            exact: bool

        measured = ContextMeasurer(
            StubCatalog(), count=lambda **k: Counted(1, "estimate", False)
        ).measure("stub", "model", conversation(10), exact=True)
        assert measured.strategy == "estimate"
        assert measured.exact is False
        # The estimate stands, not the bad count.
        assert measured.tokens == 10 * TURN_TOKENS

    def test_an_unknown_window_gives_no_percentage(self) -> None:
        # None is not 0 and must not be compared as if it were.
        assert ContextMeasurer(None).measure("p", "m", conversation(2)).percentage is None
