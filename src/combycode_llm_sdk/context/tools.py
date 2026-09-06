"""What the guard hands a strategy, so policy code stays short.

Transposed from `unified-library-ts/src/plugins/context-guard/tools.ts` and the
strategy-facing half of its `types.ts`.

A compaction strategy decides WHAT to drop. Everything else -- segmenting the
transcript, measuring it, rewriting history, calling a model to summarise, and
putting the surviving facts somewhere the next request will actually read them
-- is the same every time, and is here.

The one invariant worth stating up front: every mutation reaches BOTH the stored
history and the live message list the request is being built from. A compaction
that rewrote only history would send the un-compacted messages anyway, and the
symptom -- an over-window request after a successful compaction -- points at the
strategy rather than at the plumbing that ignored it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..agent.layers import LAYER_CHAT_FACTS, PRIORITY_CHAT_FACTS
from ..wire.interpreter import js_json
from .facts import (
    ExtractedFact,
    parse_facts_block,
    read_facts_layer,
    render_facts_block,
    render_facts_layer,
    render_prior_facts_for_extraction,
)

#: Where injected facts go. `system-append` writes a registry layer the next
#: request composes; `first-user-prefix` edits the first user message in place,
#: for providers or setups where nothing composes a system prompt.
FactInjectionSite = str

SYSTEM_APPEND = "system-append"
FIRST_USER_PREFIX = "first-user-prefix"


class ContextTools(Protocol):
    """The model-backed helpers a strategy may need.

    Pluggable because summarising and fact extraction cost a model call, and the
    cheapest strategy must not require one. Truncation works with the no-op.
    """

    async def summarize(self, content: str, max_length: int, focus: str | None = ...) -> str: ...

    async def extract_facts(
        self, content: str, categories: Sequence[str] | None = ...
    ) -> list[ExtractedFact]: ...


class NoopContextTools:
    """No summary, no facts. Enough for truncation and for tests."""

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        return ""

    async def extract_facts(
        self, content: str, categories: Sequence[str] | None = None
    ) -> list[ExtractedFact]:
        return []


class RunnerContextTools:
    """Delegates to internal tools run through an `InternalToolRunner`.

    `extract_facts` checks the registry before running: fact extraction ships as
    an extension rather than in core, and a missing tool must cost the facts,
    not the compaction.
    """

    DEFAULT_SUMMARIZE_ID = "orxa:summarize@1.0.0"
    DEFAULT_FACT_EXTRACT_ID = "orxa:fact-extract@1.0.0"

    def __init__(
        self,
        runner: Any,
        *,
        summarize_id: str | None = None,
        fact_extract_id: str | None = None,
    ) -> None:
        self.runner = runner
        self.summarize_id = summarize_id or self.DEFAULT_SUMMARIZE_ID
        self.fact_extract_id = fact_extract_id or self.DEFAULT_FACT_EXTRACT_ID

    async def summarize(self, content: str, max_length: int, focus: str | None = None) -> str:
        out = await self.runner.run(
            self.summarize_id, {"content": content, "maxLength": max_length, "focus": focus}
        )
        summary = out.get("summary") if isinstance(out, Mapping) else None
        return str(summary) if isinstance(summary, str) else ""

    async def extract_facts(
        self, content: str, categories: Sequence[str] | None = None
    ) -> list[ExtractedFact]:
        tool = await self.runner.registry.get(self.fact_extract_id)
        if not tool:
            return []
        out = await self.runner.run(
            self.fact_extract_id,
            {"content": content, "categories": list(categories) if categories else None},
        )
        raw = out.get("facts") if isinstance(out, Mapping) else None
        if not isinstance(raw, Sequence):
            return []
        return [ExtractedFact.of(f) for f in raw if isinstance(f, Mapping)]


def _content_to_plain_text(content: Any) -> str:
    """Flatten a message's content into something a summariser can read."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append(str(part.get("text") or ""))
        elif kind == "tool_call":
            parts.append(f"[tool_call {part.get('name')}]({js_json(part.get('arguments'))})")
        elif kind == "tool_result":
            body = part.get("content")
            parts.append(f"[tool_result] {body if isinstance(body, str) else js_json(body)}")
    return "\n".join(parts)


class StrategyToolsImpl:
    """The plumbing every strategy is handed."""

    def __init__(
        self,
        *,
        history: Any,
        active_messages: list[Any],
        counter: Any,
        context_tools: ContextTools,
        provider: str,
        model: str,
    ) -> None:
        self.history = history
        self.active_messages = active_messages
        self.counter = counter
        self.context_tools = context_tools
        self.provider = provider
        self.model = model

    @property
    def history_length(self) -> int:
        return len(self.history.all())

    # -- reading -------------------------------------------------------------

    def segment(
        self, *, recent_count: int | None = None, time_window: float | None = None
    ) -> dict[str, list[Any]]:
        """Split the transcript into old / middle / recent.

        Three ways to cut it, in priority order: a count of recent turns, a time
        window, or equal thirds. The count is first because it is the only one
        that behaves the same whether the conversation happened over a minute or
        a week.
        """
        entries = list(self.history.all())
        if not entries:
            return {"recent": [], "middle": [], "old": []}

        if recent_count is not None and recent_count > 0:
            start = max(0, len(entries) - recent_count)
            recent = entries[start:]
            remainder = entries[:start]
            mid = len(remainder) // 2
            return {"old": remainder[:mid], "middle": remainder[mid:], "recent": recent}

        if time_window is not None and time_window > 0:
            now = _now_ms()
            cutoff_recent = now - time_window
            cutoff_old = now - time_window * 3
            old, middle, recent = [], [], []
            for entry in entries:
                if entry.timestamp < cutoff_old:
                    old.append(entry)
                elif entry.timestamp < cutoff_recent:
                    middle.append(entry)
                else:
                    recent.append(entry)
            return {"old": old, "middle": middle, "recent": recent}

        third = -(-len(entries) // 3)
        return {
            "old": entries[:third],
            "middle": entries[third : 2 * third],
            "recent": entries[2 * third :],
        }

    def measure_current(self) -> int:
        """What the request weighs RIGHT NOW, this strategy's work included.

        `ReactContext.current` is the count that TRIGGERED the compaction, and
        no mutation updates it, so it is the wrong number to judge a compaction
        by. A strategy asking "did that help?" asks here.
        """
        return self.measure(self.active_messages)

    def measure(self, items: Sequence[Any]) -> int:
        """Token count for entries or bare messages, whichever was passed."""
        ctx = {"provider": self.provider, "model": self.model}
        total = 0
        for item in items:
            message = item.message if hasattr(item, "message") else item
            total += self.counter.estimate_message(message, ctx)
        return total

    # -- model-backed --------------------------------------------------------

    async def extract_facts(
        self, entries: Sequence[Any], categories: Sequence[str] | None = None
    ) -> list[ExtractedFact]:
        """Pull facts out of a range, WITH what earlier compactions already found.

        Feeding the prior facts back in is the point. Without it the second
        compaction extracts only from the range in front of it, and everything
        the first one learned is dropped the moment its messages are gone --
        quietly, because the summary still reads fine.
        """
        prior = read_facts_layer(self.history.registry)
        if prior is None:
            prior = parse_facts_block(self.history.system or "")

        from_entries = self._concat_content(entries) if len(entries) else ""
        if not from_entries.strip() and not prior:
            return []

        prior_block = render_prior_facts_for_extraction(prior) if prior else ""
        if prior_block and from_entries:
            content = f"{prior_block}\n\n---\n\n{from_entries}"
        else:
            content = prior_block or from_entries
        return await self.context_tools.extract_facts(content, categories)

    async def summarize(
        self, entries: Sequence[Any], max_length: int, focus: str | None = None
    ) -> str:
        if not entries:
            return ""
        content = self._concat_content(entries)
        if not content.strip():
            return ""
        return await self.context_tools.summarize(content, max_length, focus)

    # -- writing -------------------------------------------------------------

    def replace_range(self, start: int, stop: int, replacement: Mapping[str, Any]) -> None:
        self.history.splice_range(start, stop, replacement)
        self._resync()

    def drop_oldest(self, n: int) -> None:
        if n <= 0:
            return
        total = self.history.length
        if n >= total:
            self.history.clear()
        else:
            self.history.truncate(total - n)
        self._resync()

    def inject_facts(self, facts: Sequence[ExtractedFact], site: str = SYSTEM_APPEND) -> None:
        """Put the surviving facts where the next request will read them."""
        if not facts:
            return

        if site == SYSTEM_APPEND:
            # The raw facts ride along in metadata as well as the rendering,
            # so the next extraction reads structured data rather than
            # re-parsing its own prose.
            self.history.registry.set(
                LAYER_CHAT_FACTS,
                render_facts_layer(facts),
                priority=PRIORITY_CHAT_FACTS,
                tags=["system"],
                owner="context-guard",
                merge_parent=True,
                metadata={"facts": [f.as_row() for f in facts]},
            )
            return

        block = render_facts_block(facts, bare_block=True)
        index = next(
            (
                i
                for i, m in enumerate(self.active_messages)
                if isinstance(m, Mapping) and m.get("role") == "user"
            ),
            -1,
        )
        if index < 0:
            return
        message = dict(self.active_messages[index])
        message["content"] = _prepend_text(message.get("content"), f"{block}\n\n")
        self.active_messages[index] = message
        for entry in self.history.all():
            if entry.message.get("role") == "user":
                entry.message = message
                break

    # -- internals -----------------------------------------------------------

    def _resync(self) -> None:
        """Rebuild the live message list from history, in place."""
        rebuilt = self.history.messages()
        self.active_messages[:] = rebuilt

    def _concat_content(self, entries: Sequence[Any]) -> str:
        parts: list[str] = []
        for entry in entries:
            message = entry.message if hasattr(entry, "message") else entry
            text = _content_to_plain_text(message.get("content"))
            if not text.strip():
                continue
            parts.append(f"[{message.get('role')}] {text}")
        return "\n\n".join(parts)


def _prepend_text(content: Any, prefix: str) -> Any:
    if isinstance(content, str):
        return prefix + content
    if not isinstance(content, Sequence):
        return prefix
    parts = [dict(p) if isinstance(p, Mapping) else p for p in content]
    for i, part in enumerate(parts):
        if isinstance(part, Mapping) and part.get("type") == "text":
            parts[i] = {**part, "text": prefix + str(part.get("text") or "")}
            return parts
    return [{"type": "text", "text": prefix}, *parts]


def _now_ms() -> float:
    import time

    return time.time() * 1000


__all__ = [
    "FIRST_USER_PREFIX",
    "SYSTEM_APPEND",
    "ContextTools",
    "NoopContextTools",
    "RunnerContextTools",
    "StrategyToolsImpl",
]
