"""The transcript an agent keeps between calls.

Transposed from `unified-library-ts/src/agent/history.ts`.

This is the whole difference between an `Agent` and a `complete()` call: the
second turn of a conversation can only refer to the first if something kept it.

It carries a `ContextRegistry` because more than one writer contributes to a
conversation's system prompt -- the persona, the run scenario, memory, and the
facts a compaction preserved -- and a single string cannot say who wrote what.
`system` stays a plain string over the top of it, so callers written before the
registry keep working.

The mutation surface (`truncate`, `splice_range`) exists for compaction. Both
reindex in place rather than returning a copy, because the caller is holding
this object: handing back a shorter history would leave them sending the long
one they still have.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..context.registry import ContextRegistry
from ..results import Usage
from ..wire.interpreter import js_json
from .layers import LAYER_LEGACY_SYSTEM, LEGACY_SYSTEM_WRITE_PRIORITY


def _now_ms() -> float:
    return time.time() * 1000


@dataclass
class HistoryEntry:
    """One message, plus what it cost to produce."""

    index: int
    message: dict[str, Any]
    timestamp: float
    model: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    token_estimate: int | None = None


class ConversationHistory:
    """An ordered list of messages with an identity.

    The identity matters as much as the list: it is the agent's id, it is the
    `conversationId` stamped on every LLM call the agent makes, and it is how a
    hook subscriber tells one agent's traffic from another's when both share an
    engine.
    """

    def __init__(
        self,
        id: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        counter: Any = None,
        strategy: str | bool | None = None,
    ) -> None:
        self.id = id or f"agent_{uuid.uuid4().hex[:12]}"
        self.provider = provider
        self.model = model
        self.counter = counter
        self.entries: list[HistoryEntry] = []
        self.metadata: dict[str, Any] = {}
        self.created_at = _now_ms()
        self.updated_at = self.created_at
        self.context = ""
        #: Where every system-prompt contributor writes. `system` below is a
        #: view over the `system`-tagged layers in here.
        self.registry = ContextRegistry(
            f"history-{self.id[:8]}", counter=counter, default_owner="history"
        )
        #: The provider's own input-token count for the last request, and the
        #: entry it covered. Everything after that index is estimated, so a long
        #: conversation is measured mostly by the provider rather than by us.
        self._last_actual_total = 0
        self._last_actual_index = -1
        if strategy is not None:
            self.metadata["contextStrategy"] = strategy

    # -- the system prompt ---------------------------------------------------

    @property
    def system(self) -> str:
        """Every `system`-tagged layer, composed in render order.

        A property rather than a field: once the guard and the memory writer can
        also contribute, the string a caller set is no longer the whole system
        prompt, and returning only their half would quietly under-report what
        the model is about to read.
        """
        return self.registry.flat(tag="system", include_parent=False)

    @system.setter
    def system(self, value: str | None) -> None:
        if not value:
            self.registry.remove(LAYER_LEGACY_SYSTEM)
        else:
            self.registry.set(
                LAYER_LEGACY_SYSTEM,
                value,
                priority=LEGACY_SYSTEM_WRITE_PRIORITY,
                tags=["system"],
                owner="history.system-setter",
            )
        self.updated_at = _now_ms()

    def append_system(self, text: str) -> None:
        """Add to the legacy layer instead of replacing it."""
        current = self.registry.get(LAYER_LEGACY_SYSTEM)
        prior = current.content if current and isinstance(current.content, str) else ""
        self.registry.set(
            LAYER_LEGACY_SYSTEM,
            f"{prior}\n\n{text}" if prior else text,
            priority=LEGACY_SYSTEM_WRITE_PRIORITY,
            tags=["system"],
            owner="history.append_system",
        )
        self.updated_at = _now_ms()

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
        self.updated_at = _now_ms()

    # -- the provider's own count --------------------------------------------

    @property
    def last_actual_total(self) -> int:
        """The provider's input-token count for the most recent request."""
        return self._last_actual_total

    def record_actual_usage(self, input_tokens: int) -> None:
        """Anchor the estimate on a number the provider reported.

        Call BEFORE appending the reply: the input tokens describe the history
        that was SENT, which is what is here before the answer is added.
        """
        self._last_actual_total = int(input_tokens)
        self._last_actual_index = len(self.entries) - 1
        self.updated_at = _now_ms()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def length(self) -> int:
        return len(self.entries)

    def append(
        self,
        message: Mapping[str, Any],
        *,
        model: str | None = None,
        usage: Usage | None = None,
        latency_ms: float | None = None,
    ) -> HistoryEntry:
        # An assistant reply carrying the provider's own input count tells us
        # exactly what the request before it weighed -- which is everything in
        # history right now, since this reply is not in it yet.
        if message.get("role") == "assistant" and usage and (usage.input_tokens or 0) > 0:
            self._last_actual_total = int(usage.input_tokens)
            self._last_actual_index = len(self.entries) - 1

        if message.get("role") == "assistant" and usage and (usage.output_tokens or 0) > 0:
            token_estimate = int(usage.output_tokens)
        else:
            token_estimate = estimate_tokens(message)

        entry = HistoryEntry(
            index=len(self.entries),
            message=dict(message),
            timestamp=_now_ms(),
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            token_estimate=token_estimate,
        )
        self.entries.append(entry)
        self.updated_at = entry.timestamp
        return entry

    def messages(self) -> list[dict[str, Any]]:
        """The transcript, as the client wants it."""
        return [dict(e.message) for e in self.entries]

    # -- reading -------------------------------------------------------------

    def all(self) -> list[HistoryEntry]:
        """The entries themselves, with what each one cost."""
        return self.entries

    def at(self, index: int) -> HistoryEntry | None:
        """One entry, counting from the end when the index is negative."""
        i = index + len(self.entries) if index < 0 else index
        return self.entries[i] if 0 <= i < len(self.entries) else None

    def last(self, n: int) -> list[HistoryEntry]:
        return self.entries[-n:] if n > 0 else []

    def last_messages(self, n: int) -> list[dict[str, Any]]:
        return [dict(e.message) for e in self.last(n)]

    def filter(self, fn: Callable[[HistoryEntry], bool]) -> list[HistoryEntry]:
        return [e for e in self.entries if fn(e)]

    def by_role(self, role: str) -> list[HistoryEntry]:
        return [e for e in self.entries if e.message.get("role") == role]

    def __iter__(self) -> Iterator[HistoryEntry]:
        return iter(self.entries)

    def total_usage(self) -> Usage:
        """What the whole conversation has cost so far.

        Summed into locals and built once, because `Usage` is frozen -- and it
        is frozen on purpose: a per-call record that a later reader can add to
        is a record that stops matching what the provider actually billed.

        The audio counts and the tier are deliberately not summed. `None` there
        means "no audio was sent", which is not the same as zero, and a tier
        describes one call rather than a conversation that may have spanned
        several.
        """
        used = [e.usage for e in self.entries if e.usage is not None]
        return Usage(
            input_tokens=sum(u.input_tokens or 0 for u in used),
            output_tokens=sum(u.output_tokens or 0 for u in used),
            total_tokens=sum(u.total_tokens or 0 for u in used),
            cached_tokens=sum(u.cached_tokens or 0 for u in used),
            cache_write_tokens=sum(u.cache_write_tokens or 0 for u in used),
            reasoning_tokens=sum(u.reasoning_tokens or 0 for u in used),
        )

    def estimated_tokens(self) -> int:
        """What the next request will weigh.

        Exact for everything the provider has already counted, estimated only
        for what has been added since. Estimating the whole transcript every
        time compounds the heuristic's error over the part that is no longer in
        doubt.
        """
        if 0 <= self._last_actual_index < len(self.entries):
            total = self._last_actual_total
            for entry in self.entries[self._last_actual_index + 1 :]:
                total += entry.token_estimate or estimate_tokens(entry.message)
            return total

        total = estimate_tokens(self.system) if self.system else 0
        return total + sum(
            e.token_estimate or estimate_tokens(e.message) for e in self.entries
        )

    # -- compaction ----------------------------------------------------------

    def truncate(self, keep_last: int) -> list[HistoryEntry]:
        """Keep the most recent `keep_last`. Returns what was dropped."""
        if keep_last >= len(self.entries):
            return []
        cut = len(self.entries) - keep_last
        removed = self.entries[:cut]
        self.entries = self.entries[cut:]
        self._reindex()
        self.updated_at = _now_ms()
        return removed

    def splice_range(
        self, start: int, stop: int, replacement: Mapping[str, Any]
    ) -> list[HistoryEntry]:
        """Replace `[start, stop)` with one synthetic message.

        The replacement inherits the LATEST timestamp of the range it stands
        for, not the current time: a summary of yesterday's turns that claims to
        have been said just now sorts itself in front of the messages it
        summarises.
        """
        if start < 0 or stop > len(self.entries) or start >= stop:
            return []
        removed = self.entries[start:stop]
        timestamp = max((e.timestamp for e in removed), default=0) or _now_ms()
        self.entries[start:stop] = [
            HistoryEntry(index=start, message=dict(replacement), timestamp=timestamp)
        ]
        self._reindex()
        # The provider's count described a history that no longer exists, so it
        # is dropped rather than carried forward against different messages.
        if self._last_actual_index >= start:
            self._last_actual_total = 0
            self._last_actual_index = -1
        self.updated_at = _now_ms()
        return removed

    def _reindex(self) -> None:
        for i, entry in enumerate(self.entries):
            entry.index = i

    def fork(self, new_id: str | None = None) -> ConversationHistory:
        """A deep copy, registry included, under a new identity."""
        forked = ConversationHistory(new_id, provider=self.provider, model=self.model)
        forked.metadata = dict(self.metadata)
        forked.context = self.context
        forked.entries = [
            HistoryEntry(
                index=e.index,
                message=dict(e.message),
                timestamp=e.timestamp,
                model=e.model,
                # Shared, not copied: `Usage` is frozen, so the fork cannot
                # change what the original recorded.
                usage=e.usage,
                latency_ms=e.latency_ms,
                token_estimate=e.token_estimate,
            )
            for e in self.entries
        ]
        for layer in self.registry.list():
            forked.registry.set(
                layer.name,
                layer.content,
                priority=layer.priority,
                tags=list(layer.tags),
                owner=layer.owner,
                merge_parent=layer.merge_parent,
                metadata=dict(layer.metadata) if layer.metadata else None,
            )
        return forked

    def clear(self) -> None:
        """Forget the conversation, keep the identity.

        The id survives on purpose: hooks, telemetry and any observer are bound
        to it, and a clear that also re-identified the agent would silently
        detach every one of them.
        """
        self.entries = []
        self._last_actual_total = 0
        self._last_actual_index = -1
        self.updated_at = _now_ms()

    def composed_system(self) -> str | None:
        """The system prompt to send, or None when there is nothing to say.

        `context` -- background for the current task -- is appended after the
        persona rather than merged into it, so a caller reading the request can
        still tell which half came from where.
        """
        parts = [p for p in (self.system, self.context) if p]
        return "\n\n".join(parts) if parts else None

    # -- serialisation -------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        """A snapshot that survives a process restart."""
        return {
            "id": self.id,
            "entries": [
                {
                    "index": e.index,
                    "message": e.message,
                    "timestamp": e.timestamp,
                    "model": e.model,
                    "usage": e.usage.__dict__ if e.usage else None,
                    "latencyMs": e.latency_ms,
                    "tokenEstimate": e.token_estimate,
                }
                for e in self.entries
            ],
            "system": self.system,
            "context": self.context,
            "registry": self.registry.dump(),
            "metadata": dict(self.metadata),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    @staticmethod
    def restore(snapshot: Mapping[str, Any]) -> ConversationHistory:
        """Rebuild from `dump()`.

        Usage is restored as a `Usage`, not as the raw dict it was stored as --
        a restored history whose entries carry a different type from a live
        one's is a bug that only appears after a restart.
        """
        history = ConversationHistory(id=str(snapshot.get("id") or "") or None)
        for raw in snapshot.get("entries") or []:
            usage = raw.get("usage")
            history.entries.append(
                HistoryEntry(
                    index=int(raw.get("index", len(history.entries))),
                    message=dict(raw.get("message") or {}),
                    timestamp=float(raw.get("timestamp") or _now_ms()),
                    model=raw.get("model"),
                    usage=Usage(**usage) if isinstance(usage, Mapping) else None,
                    latency_ms=raw.get("latencyMs"),
                    token_estimate=raw.get("tokenEstimate"),
                )
            )
        # The registry wins when both are present: `system` is the flattened
        # view of it, so restoring the string over restored layers would write a
        # copy of every layer back in as one more layer.
        registry = snapshot.get("registry")
        if isinstance(registry, Mapping):
            history.registry = ContextRegistry.restore(registry)
        else:
            history.system = str(snapshot.get("system") or "")
        history.context = str(snapshot.get("context") or "")
        history.metadata = dict(snapshot.get("metadata") or {})
        history.created_at = float(snapshot.get("createdAt") or _now_ms())
        history.updated_at = float(snapshot.get("updatedAt") or history.created_at)
        return history


def estimate_tokens(message: Mapping[str, Any] | str) -> int:
    """The length/4 heuristic, over whatever shape the message is.

    An ESTIMATE, and named one: it exists so a caller can see a conversation
    growing before the provider bills for it. `count_tokens` is the exact
    answer and costs a request.
    """
    if isinstance(message, str):
        return -(-len(message) // 4)
    content: Any = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence):
        text = js_json(list(content))
    else:
        text = js_json(content)
    return -(-len(text) // 4)


__all__ = ["ConversationHistory", "HistoryEntry", "estimate_tokens"]
