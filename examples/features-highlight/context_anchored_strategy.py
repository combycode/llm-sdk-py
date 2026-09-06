"""Compaction that summarises raw text once, instead of summarising its summaries.

`LayeredStrategy` emits a NEW summary at every compaction, so a long run
accumulates summaries-of-summaries: the oldest material is re-summarised again
and again, drifting further from what was actually said each time. The failure
is quiet -- the transcript stays fluent while becoming wrong.

`AnchoredStrategy` keeps ONE anchor entry at the head of history and merges each
compaction into it, so every fact is summarised from the raw turns exactly once.
The trade is real and worth stating: one anchor means one blast radius, where a
chain of summaries corrupts only one link. Anchored suits long-running state
("what have we established so far"); layered suits conversations where recency
matters more than a durable record.

Both are policy objects and nothing else. They do not reach into history
themselves: the guard hands them a reading of the pressure and a set of tools,
and they answer with a decision. That is what lets this file drive them with no
engine, no key and no network -- the history and the tools are real, and only
the summariser is stubbed.
"""

import asyncio
from typing import Any

from _check import check, report

from combycode_llm_sdk import (
    ANCHOR_MARKER,
    AnchoredStrategy,
    ConversationHistory,
    HeuristicCounter,
    LayeredStrategy,
    ReactContext,
    StrategyToolsImpl,
)
from combycode_llm_sdk.catalog.catalog import resolve_catalog

WINDOW = 1_000
KEEP_RECENT = 4
TURNS = 14
ANCHOR_CAP = 40
#: The rung each strategy declares for itself.
ANCHORED_LEVEL = "warn"
LAYERED_LEVEL = "urgent"


class Summariser:
    """Stands in for the model, and records what it was asked to read.

    What it was ASKED is the interesting half: it makes "was this built from the
    raw turns, or from an earlier summary?" a question this file can answer.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        self.asked.append(content)
        return f"state after {content.count('[user]')} turns"

    async def extract_facts(self, content: str, categories: Any = None) -> list[Any]:
        return []


class Silent(Summariser):
    """A summariser that fails by returning nothing at all."""

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        return "   "


def conversation(turns: int) -> ConversationHistory:
    history = ConversationHistory("demo")
    for i in range(turns):
        history.append({"role": "user", "content": f"Question {i}"})
    return history


def tools_for(history: ConversationHistory, summariser: Any) -> StrategyToolsImpl:
    return StrategyToolsImpl(
        history=history,
        active_messages=history.messages(),
        counter=HeuristicCounter(resolve_catalog("defaults")),
        context_tools=summariser,
        provider="openai",
        model="gpt-4.1",
    )


def reading(tools: Any, level: str) -> ReactContext:
    """The pressure reading the guard would hand a strategy.

    `level` is the name of the rung that fired, and the rungs belong to the
    STRATEGY rather than to the guard: anchored declares one called "warn",
    layered declares healthy/pressure/urgent/critical and escalates how much it
    rewrites as it climbs them. Handing a strategy a level from someone else's
    ladder is how you get a compaction that silently does nothing.
    """
    return ReactContext(
        level=level,
        percentage=0.9,
        current=int(0.9 * WINDOW),
        window=WINDOW,
        delta=0,
        provider="openai",
        model="gpt-4.1",
        attempt=0,
        tools=tools,
    )


async def main() -> None:
    # -- anchored: one anchor, merged into on every pass ---------------------

    summariser = Summariser()
    history = conversation(TURNS)
    tools = tools_for(history, summariser)
    anchored = AnchoredStrategy(keep_recent=KEEP_RECENT)

    first = await anchored.react(reading(tools, ANCHORED_LEVEL))
    check(first.action == "compacted", f"the first pass did not compact: {first}")

    head = history.at(0)
    check(head is not None, "the anchor is missing")
    check(head.message["role"] == "system", "the anchor must be a system turn")
    check(
        str(head.message["content"]).startswith(ANCHOR_MARKER),
        "the anchor must be MARKED, or the next pass cannot find it and writes a second one",
    )

    for i in range(8):
        history.append({"role": "user", "content": f"Later question {i}"})
    tools.active_messages[:] = history.messages()

    second = await anchored.react(reading(tools, ANCHORED_LEVEL))
    check(second.action == "compacted", f"the second pass did not compact: {second}")
    check("existing" in (second.note or ""), f"the second pass ignored the anchor: {second.note}")

    # The point of the strategy. The second summarisation read the new raw
    # turns -- not the summary the first pass produced.
    check(len(summariser.asked) == 2, f"expected one summarisation per pass, got {len(summariser.asked)}")
    check(
        ANCHOR_MARKER not in summariser.asked[1],
        "the anchor was handed back as material to re-summarise, which is the drift this "
        "strategy exists to avoid",
    )

    anchor = history.at(0)
    check(anchor is not None, "the anchor disappeared on the second pass")
    check(
        str(anchor.message["content"]).startswith(ANCHOR_MARKER),
        "the second pass must merge INTO the anchor, not bury it behind a new summary",
    )
    check(history.length < TURNS + 8, "the second pass did not shrink anything")

    # -- the bound that stops the anchor becoming the problem it solves ------

    capped = conversation(TURNS)
    await AnchoredStrategy(keep_recent=KEEP_RECENT, anchor_max_chars=ANCHOR_CAP).react(
        reading(tools_for(capped, Summariser()), ANCHORED_LEVEL)
    )
    bounded = capped.at(0)
    check(bounded is not None, "the capped anchor is missing")
    text = str(bounded.message["content"])[len(ANCHOR_MARKER) :].strip()
    check(len(text) <= ANCHOR_CAP, f"the anchor outgrew its own cap: {len(text)} chars")

    # -- a summariser that says nothing must not delete the conversation -----

    silent = conversation(TURNS)
    refused = await AnchoredStrategy(keep_recent=KEEP_RECENT).react(
        reading(tools_for(silent, Silent()), ANCHORED_LEVEL)
    )
    check(refused.action == "decline", f"a failed summariser was allowed to compact: {refused}")
    check(
        silent.length == TURNS,
        "trading a context overflow for silent data loss is strictly worse: one is a failed "
        "request, the other is a conversation that continues while missing what it was about",
    )

    # -- layered, for contrast ----------------------------------------------

    layered_history = conversation(TURNS)
    layered = await LayeredStrategy(recent_count=KEEP_RECENT).react(
        reading(tools_for(layered_history, Summariser()), LAYERED_LEVEL)
    )
    check(layered.action == "compacted", f"the layered strategy did not compact: {layered}")
    check(
        layered_history.length < TURNS,
        "both strategies must bound the window -- that is the point of compacting at all",
    )
    layered_head = layered_history.at(0)
    check(layered_head is not None, "the layered summary is missing")
    check(
        not str(layered_head.message["content"]).startswith(ANCHOR_MARKER),
        "the layered strategy writes a plain summary; only anchored keeps an anchor",
    )

    report(
        anchored_len=history.length,
        layered_len=layered_history.length,
        summarised_once_per_pass=len(summariser.asked),
        anchor_capped_to=len(text),
        empty_summary=refused.action,
    )


asyncio.run(main())
