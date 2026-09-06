"""The three compaction policies.

Transposed from `unified-library-ts/src/plugins/context-guard/strategies/`.

They differ in what they are willing to spend and what they are willing to lose:

  - `TruncateStrategy` drops the oldest turns and calls no model. Nothing to pay
    for, and everything dropped is gone.
  - `LayeredStrategy` keeps recent turns verbatim, summarises the middle, and
    reduces the oldest to facts. It emits a NEW summary every time, so a long
    run accumulates summaries-of-summaries and the oldest material drifts
    further from what was said with each pass. The failure is quiet: the
    transcript stays fluent while becoming wrong.
  - `AnchoredStrategy` keeps ONE growing anchor and merges into it, so every
    fact is summarised from raw text exactly once. The trade is real: one anchor
    means one blast radius, where a chain of summaries corrupts only one link.

Anchored suits long-running state ("what have we established"); layered suits
conversations where recency matters more than a durable record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .strategy_types import ReactContext, StrategyDecision, TriggerLevel
from .tools import SYSTEM_APPEND


def _share_left(ctx: Any) -> float:
    """The share of the window AFTER this strategy did its work.

    Re-measured rather than read off `ctx.current`, which still holds the count
    that TRIGGERED the compaction. Judging on that number declines work that
    succeeded: a truncation taking a conversation from 100% of the window to 20%
    was refused for being "still above 95%" -- after it had already destroyed
    the messages. The caller lost the history and the call.
    """
    if not ctx.window:
        return 0.0
    return float(ctx.tools.measure_current()) / float(ctx.window)


# -- TruncateStrategy --------------------------------------------------------

TRUNCATE_DEFAULT_TRIGGERS = [TriggerLevel("urgent", 0.85)]
TRUNCATE_DEFAULT_KEEP_RECENT = 20
DEFAULT_DECLINE_CEILING = 0.95


class TruncateStrategy:
    """Drop the oldest turns. The cheapest compaction, and it calls no model.

    For prototypes, smoke tests, and conversations where losing old context is
    genuinely acceptable. It is not a summariser: what it drops is gone.
    """

    name = "truncate"

    def __init__(
        self,
        *,
        keep_recent: int = TRUNCATE_DEFAULT_KEEP_RECENT,
        decline_ceiling: float = DEFAULT_DECLINE_CEILING,
        triggers: Sequence[TriggerLevel] | None = None,
    ) -> None:
        self.keep_recent = keep_recent
        self.decline_ceiling = decline_ceiling
        self.triggers = list(triggers) if triggers else list(TRUNCATE_DEFAULT_TRIGGERS)

    async def react(self, ctx: ReactContext) -> StrategyDecision:
        total = ctx.tools.history_length
        if total <= self.keep_recent:
            return StrategyDecision.none()

        drop = total - self.keep_recent
        ctx.tools.drop_oldest(drop)

        used = _share_left(ctx)
        if used >= self.decline_ceiling:
            return StrategyDecision.decline(
                f"still above {round(self.decline_ceiling * 100)}% after dropping {drop} entries"
            )
        return StrategyDecision.compacted(f"dropped {drop} oldest entries")


# -- LayeredStrategy ---------------------------------------------------------

LAYERED_DEFAULT_TRIGGERS = [
    TriggerLevel("healthy", 0.5),
    TriggerLevel("pressure", 0.7),
    TriggerLevel("urgent", 0.85),
    TriggerLevel("critical", 0.95),
]
LAYERED_DEFAULTS = {
    "recent_count": 6,
    "middle_summary_chars": 300,
    "old_summary_chars": 400,
    "jump_escalate_delta": 0.3,
    "decline_ceiling": 0.9,
}
#: At `critical`, how many turns survive untouched.
CRITICAL_KEEP_LAST = 2


class LayeredStrategy:
    """Three zones -- recent verbatim, middle summarised, old reduced to facts.

    Which zones are touched escalates with the trigger level, so ordinary
    pressure costs one summary and only a genuine emergency rewrites the recent
    turns as well.
    """

    name = "layered"

    def __init__(
        self,
        *,
        recent_count: int | None = None,
        middle_summary_chars: int | None = None,
        old_summary_chars: int | None = None,
        jump_escalate_delta: float | None = None,
        decline_ceiling: float | None = None,
        triggers: Sequence[TriggerLevel] | None = None,
    ) -> None:
        d = LAYERED_DEFAULTS
        self.recent_count = int(recent_count if recent_count is not None else d["recent_count"])
        self.middle_summary_chars = int(
            middle_summary_chars
            if middle_summary_chars is not None
            else d["middle_summary_chars"]
        )
        self.old_summary_chars = int(
            old_summary_chars if old_summary_chars is not None else d["old_summary_chars"]
        )
        self.jump_escalate_delta = float(
            jump_escalate_delta
            if jump_escalate_delta is not None
            else d["jump_escalate_delta"]
        )
        self.decline_ceiling = float(
            decline_ceiling if decline_ceiling is not None else d["decline_ceiling"]
        )
        self.triggers = list(triggers) if triggers else list(LAYERED_DEFAULT_TRIGGERS)

    async def react(self, ctx: ReactContext) -> StrategyDecision:
        level = self._escalate_on_a_jump(ctx)

        # Only after an attempt has already been made. Declining on the first
        # look would refuse work that compaction was about to make possible.
        if ctx.percentage >= self.decline_ceiling and ctx.attempt >= 1:
            return StrategyDecision.decline(
                f"Context at {ctx.percentage * 100:.1f}% after {ctx.attempt + 1} compaction "
                "attempt(s); unable to fit safely."
            )

        if level == "healthy":
            return await self._compact_old(ctx)
        if level == "pressure":
            return await self._compact_old_and_middle(ctx)
        if level == "urgent":
            return await self._compact_all(ctx)
        if level == "critical":
            return await self._compact_aggressively(ctx)
        return StrategyDecision.none()

    async def _compact_old(self, ctx: ReactContext) -> StrategyDecision:
        old = ctx.tools.segment(recent_count=self.recent_count)["old"]
        if not old:
            return StrategyDecision.none()

        facts = await ctx.tools.extract_facts(old)
        summary = await ctx.tools.summarize(old, self.old_summary_chars)
        ctx.tools.replace_range(0, len(old), _summary_message(summary, "Earlier conversation"))
        if facts:
            ctx.tools.inject_facts(facts, SYSTEM_APPEND)
        return StrategyDecision.compacted(
            f"compacted {len(old)} old entries into one summary; {len(facts)} facts preserved"
        )

    async def _compact_old_and_middle(self, ctx: ReactContext) -> StrategyDecision:
        seg = ctx.tools.segment(recent_count=self.recent_count)
        old, middle = seg["old"], seg["middle"]

        replaced_old = 0
        if old:
            facts = await ctx.tools.extract_facts(old)
            summary = await ctx.tools.summarize(old, self.old_summary_chars)
            ctx.tools.replace_range(0, len(old), _summary_message(summary, "Earlier conversation"))
            if facts:
                ctx.tools.inject_facts(facts, SYSTEM_APPEND)
            replaced_old = 1

        if middle:
            facts = await ctx.tools.extract_facts(middle)
            summary = await ctx.tools.summarize(middle, self.middle_summary_chars)
            # Offset by the entry the old range collapsed into: the middle has
            # shifted down by however many entries went before it.
            ctx.tools.replace_range(
                replaced_old,
                replaced_old + len(middle),
                _summary_message(summary, "Prior discussion"),
            )
            if facts:
                ctx.tools.inject_facts(facts, SYSTEM_APPEND)

        return StrategyDecision.compacted("compacted old + middle layers")

    async def _compact_all(self, ctx: ReactContext) -> StrategyDecision:
        await self._compact_old_and_middle(ctx)

        half = max(2, self.recent_count // 2)
        total = ctx.tools.history_length
        seg = ctx.tools.segment(recent_count=half)
        merged = [*seg["old"], *seg["middle"]]
        if merged:
            facts = await ctx.tools.extract_facts(merged)
            summary = await ctx.tools.summarize(merged, self.old_summary_chars)
            ctx.tools.replace_range(
                0, total - half, _summary_message(summary, "Compacted prior context")
            )
            if facts:
                ctx.tools.inject_facts(facts, SYSTEM_APPEND)
        return StrategyDecision.compacted(
            f"urgent: compacted old+middle and shrunk recent to last {half}"
        )

    async def _compact_aggressively(self, ctx: ReactContext) -> StrategyDecision:
        total = ctx.tools.history_length
        if total <= CRITICAL_KEEP_LAST:
            return StrategyDecision.decline(
                f"Context at {ctx.percentage * 100:.1f}% with only {total} entries -- the new "
                "content alone exceeds what compaction can free."
            )

        seg = ctx.tools.segment(recent_count=CRITICAL_KEEP_LAST)
        to_compact = [*seg["old"], *seg["middle"]]
        if not to_compact:
            return StrategyDecision.decline(
                "Nothing left to compact but still above critical threshold."
            )

        facts = await ctx.tools.extract_facts(to_compact)
        summary = await ctx.tools.summarize(to_compact, self.old_summary_chars)
        ctx.tools.replace_range(
            0, len(to_compact), _summary_message(summary, "Conversation so far - compacted")
        )
        if facts:
            ctx.tools.inject_facts(facts, SYSTEM_APPEND)
        return StrategyDecision.compacted(
            f"critical: kept last {CRITICAL_KEEP_LAST}, compacted {len(to_compact)}, "
            f"{len(facts)} facts preserved"
        )

    def _escalate_on_a_jump(self, ctx: ReactContext) -> str:
        """Move up a rung when the growth since last time was itself large.

        Arriving at 70% gradually and arriving at 70% from 30% in one turn are
        not the same situation: the second says the next turn may overshoot the
        window entirely, and reacting at the gentler level would waste the one
        chance to get ahead of it.
        """
        if not ctx.window:
            return ctx.level
        if ctx.delta / ctx.window < self.jump_escalate_delta:
            return ctx.level
        names = [t.level for t in self.triggers]
        if ctx.level not in names:
            return ctx.level
        idx = names.index(ctx.level)
        return ctx.level if idx >= len(names) - 1 else names[idx + 1]


def _summary_message(summary: str, label: str) -> dict[str, Any]:
    text = f"[{label} summary]\n{summary}" if summary else f"[{label} omitted]"
    return {"role": "user", "content": text}


# -- AnchoredStrategy --------------------------------------------------------

ANCHORED_DEFAULT_TRIGGERS = [TriggerLevel("warn", 0.7)]
ANCHORED_DEFAULTS = {"keep_recent": 12, "anchor_max_chars": 4000, "decline_ceiling": 0.95}

#: Marks the anchor so the next compaction can find it. Kept in the TEXT rather
#: than in metadata because the anchor has to survive an export/import of
#: history, and metadata on a synthetic entry does not.
ANCHOR_MARKER = "[context-anchor]"


def calculate_retain_start_index(entries: Sequence[Any], keep_recent: int) -> int:
    """Where the retained tail starts, refusing to split a call from its result.

    Cutting between them leaves a `tool_result` whose call is gone, which some
    providers reject outright and the rest silently misread. Walking the
    boundary backwards keeps the pair together.
    """
    start = max(0, len(entries) - keep_recent)
    while start > 0:
        if _has_part(entries[start], "tool_result") and _has_part(entries[start - 1], "tool_call"):
            start -= 1
        else:
            break
    return start


def _parts(entry: Any) -> list[Any]:
    message = getattr(entry, "message", None)
    content = message.get("content") if isinstance(message, Mapping) else None
    return list(content) if isinstance(content, list) else []


def _has_part(entry: Any, kind: str) -> bool:
    return any(isinstance(p, Mapping) and p.get("type") == kind for p in _parts(entry))


def _is_anchor(entry: Any) -> bool:
    message = getattr(entry, "message", None)
    if not isinstance(message, Mapping) or message.get("role") != "system":
        return False
    content = message.get("content")
    return isinstance(content, str) and content.startswith(ANCHOR_MARKER)


def _anchor_text(entry: Any) -> str:
    message = getattr(entry, "message", None)
    content = message.get("content") if isinstance(message, Mapping) else None
    return content[len(ANCHOR_MARKER) :].strip() if isinstance(content, str) else ""


def merge_anchor(previous: str, addition: str, max_chars: int) -> str:
    """Fold a summary into the anchor, bounded so it cannot become the problem.

    When it must be trimmed the NEWER text is kept: it already subsumes the
    older state, so the end is the half worth having.
    """
    merged = f"{previous}\n{addition}" if previous else addition
    return merged if len(merged) <= max_chars else merged[len(merged) - max_chars :]


class AnchoredStrategy:
    """One growing scratchpad instead of a chain of summaries.

    Ported from google-adk-ts `AnchoredContextCompactor` (adk 1.5), including
    its refusal to split a tool call from its result.
    """

    name = "anchored"

    def __init__(
        self,
        *,
        keep_recent: int | None = None,
        anchor_max_chars: int | None = None,
        decline_ceiling: float | None = None,
        triggers: Sequence[TriggerLevel] | None = None,
    ) -> None:
        d = ANCHORED_DEFAULTS
        self.keep_recent = int(keep_recent if keep_recent is not None else d["keep_recent"])
        self.anchor_max_chars = int(
            anchor_max_chars if anchor_max_chars is not None else d["anchor_max_chars"]
        )
        self.decline_ceiling = float(
            decline_ceiling if decline_ceiling is not None else d["decline_ceiling"]
        )
        self.triggers = list(triggers) if triggers else list(ANCHORED_DEFAULT_TRIGGERS)

    async def react(self, ctx: ReactContext) -> StrategyDecision:
        # `segment` is the tools' view of history; re-joining it gives the
        # ordered entry list the retain boundary needs, which works in absolute
        # indices.
        seg = ctx.tools.segment(recent_count=self.keep_recent)
        entries = [*seg["old"], *seg["middle"], *seg["recent"]]
        if ctx.tools.history_length <= self.keep_recent + 1:
            return StrategyDecision.none()

        retain_start = calculate_retain_start_index(entries, self.keep_recent)
        anchored = bool(entries) and _is_anchor(entries[0])
        if retain_start <= (1 if anchored else 0):
            return StrategyDecision.none()

        to_merge = entries[1 if anchored else 0 : retain_start]
        if not to_merge:
            return StrategyDecision.none()

        previous = _anchor_text(entries[0]) if anchored else ""
        summary = await ctx.tools.summarize(
            to_merge,
            self.anchor_max_chars,
            "Merge these events into the running state summary; keep established facts, drop "
            "superseded ones."
            if previous
            else "Summarise these events as a running state summary.",
        )

        # A summariser that returns nothing must not be allowed to erase
        # history: that trades a context overflow for silent data loss, which is
        # strictly worse -- one is a failed request, the other is a conversation
        # that continues while quietly missing what it was about.
        if not summary.strip():
            return StrategyDecision.decline(
                "summarizer returned nothing; refusing to drop entries"
            )

        merged = merge_anchor(previous, summary, self.anchor_max_chars)
        ctx.tools.replace_range(
            0, retain_start, {"role": "system", "content": f"{ANCHOR_MARKER}\n{merged}"}
        )

        used = _share_left(ctx)
        if used >= self.decline_ceiling:
            return StrategyDecision.decline(
                f"still above {round(self.decline_ceiling * 100)}% after merging "
                f"{len(to_merge)} entries into the anchor"
            )
        return StrategyDecision.compacted(
            f"merged {len(to_merge)} entries into the "
            f"{'existing' if previous else 'new'} anchor"
        )


__all__ = [
    "ANCHOR_MARKER",
    "CRITICAL_KEEP_LAST",
    "DEFAULT_DECLINE_CEILING",
    "AnchoredStrategy",
    "LayeredStrategy",
    "TruncateStrategy",
    "calculate_retain_start_index",
    "merge_anchor",
]
