"""`CostCollector` -- what the run actually cost, and what it could not price.

Transposed from `unified-library-ts/src/plugins/cost-collector/`. Ported here:
the ledger, the summary, and the unpriced warning. Budgets and tag/scope filters
are NOT -- their example (`cost_estimate_and_budget.py`) is blocked on the
estimator, so porting them now would ship a second thing nothing can check.

The reason this exists rather than summing `result.cost` at the call site: a
model the catalog does not know still CALLS perfectly well -- a fine-tune, a
private deployment, a model released this morning -- and what it cannot do is be
priced. Summed as zero like any other entry, the report reads $0.00 and looks
like a cheap run rather than an unpriced one, and a budget built on that total
never fires. A live benchmark once reported $0.00000 per task across every
Anthropic run and looked like the cheapest arm in the table until someone
checked.

So the summary COUNTS what it could not price and NAMES the models. Genuinely
free calls are not counted here: they are priced, at zero, which is a different
thing and stays quiet.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .catalog.catalog import ModelCatalog
from .cost import Cost, compute_cost


@dataclass(frozen=True)
class CostEntry:
    """One priced call.

    `cost` is None when the model could not be priced -- never a `Cost` whose
    total is 0.0. That is the whole point: in Python the absence is a different
    type from the number, so a reader cannot mistake one for the other.
    """

    id: str
    timestamp: float
    provider: str
    model: str
    tokens: Mapping[str, int]
    cost: Cost | None
    service_tier: str | None = None
    tags: Mapping[str, Any] = field(default_factory=dict)

    @property
    def unpriced(self) -> bool:
        return self.cost is None


@dataclass(frozen=True)
class TokenTotals:
    input: int = 0
    output: int = 0
    cached: int = 0
    cache_write: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class CostSummary:
    """Several calls, added up -- and how much of the sum is trustworthy."""

    total: float = 0.0
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    reasoning: float = 0.0
    tokens: TokenTotals = field(default_factory=TokenTotals)
    #: How many calls are in this summary.
    entries: int = 0
    #: How many of them could not be priced. They contribute 0 to every figure
    #: above, so a summary without this number cannot tell "this run was cheap"
    #: from "this run was never priced".
    unpriced: int = 0
    #: The distinct `provider/model` values behind `unpriced`, so the gap is
    #: actionable rather than merely visible.
    unpriced_models: Sequence[str] = ()


def summarize(entries: Sequence[CostEntry]) -> CostSummary:
    """Add up a ledger, keeping the unpriced ones countable."""
    totals = {
        "total": 0.0,
        "input": 0.0,
        "output": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
        "reasoning": 0.0,
    }
    tokens = {"input": 0, "output": 0, "cached": 0, "cache_write": 0, "reasoning": 0}
    unpriced = 0
    unpriced_models: list[str] = []

    for entry in entries:
        if entry.cost is None:
            unpriced += 1
            key = f"{entry.provider}/{entry.model}"
            if key not in unpriced_models:
                unpriced_models.append(key)
        else:
            totals["total"] += entry.cost.total
            totals["input"] += entry.cost.input
            totals["output"] += entry.cost.output
            totals["cache_read"] += entry.cost.cache_read
            totals["cache_write"] += entry.cost.cache_write
            totals["reasoning"] += entry.cost.reasoning
        for key_name in tokens:
            tokens[key_name] += int(entry.tokens.get(key_name, 0))

    return CostSummary(
        **totals,
        tokens=TokenTotals(**tokens),
        entries=len(entries),
        unpriced=unpriced,
        unpriced_models=tuple(unpriced_models),
    )


class CostCollector:
    """Listens to `onCompletion`, prices each call, and keeps the ledger."""

    def __init__(
        self,
        hooks: Any,
        catalog: ModelCatalog,
        *,
        session_id: str | None = None,
        default_tags: Mapping[str, Any] | None = None,
    ) -> None:
        self.hooks = hooks
        self.catalog = catalog
        self.session_id = session_id
        self.default_tags = dict(default_tags or {})
        self._ledger: list[CostEntry] = []
        self._running_total = 0.0
        #: `provider/model` values already warned about. An unpriced model is a
        #: configuration fact, not a per-request event: warning on every call
        #: trains the reader to ignore the warning.
        self._warned: set[str] = set()
        self._unsubscribe: Callable[[], None] | None = hooks.on(
            "onCompletion", self._on_completion
        )

    def destroy(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # -- queries -------------------------------------------------------------

    def total(self) -> CostSummary:
        """Everything recorded so far."""
        return summarize(self._ledger)

    def by_model(self) -> dict[str, CostSummary]:
        groups: dict[str, list[CostEntry]] = {}
        for entry in self._ledger:
            groups.setdefault(f"{entry.provider}/{entry.model}", []).append(entry)
        return {key: summarize(group) for key, group in groups.items()}

    def by_provider(self) -> dict[str, CostSummary]:
        groups: dict[str, list[CostEntry]] = {}
        for entry in self._ledger:
            groups.setdefault(entry.provider, []).append(entry)
        return {key: summarize(group) for key, group in groups.items()}

    def entries(self) -> Sequence[CostEntry]:
        return tuple(self._ledger)

    @property
    def entry_count(self) -> int:
        return len(self._ledger)

    @property
    def running_total(self) -> float:
        return self._running_total

    def clear(self) -> None:
        """Forget the ledger.

        The warned-about models are forgotten too: after a clear the next call
        is the first evidence again, and staying silent about it would leave a
        fresh ledger with an unexplained unpriced count.
        """
        self._ledger.clear()
        self._running_total = 0.0
        self._warned.clear()

    # -- recording -----------------------------------------------------------

    def _on_completion(self, ctx: Any) -> None:
        provider = str(ctx.get("provider") or "")
        model = str(ctx.get("model") or "")
        response: Mapping[str, Any] = ctx.get("response") or {}
        request: Mapping[str, Any] = ctx.get("request") or {}
        usage: Mapping[str, Any] = response.get("usage") or {}

        tokens = {
            # The estimate is the fallback, not the preference: a provider that
            # reported nothing still sent something, and counting it as zero
            # would under-report the run rather than merely approximate it.
            "input": int(usage.get("inputTokens") or request.get("estimatedInputTokens") or 0),
            "output": int(usage.get("outputTokens") or 0),
            "cached": int(usage.get("cachedTokens") or 0),
            "cache_write": int(usage.get("cacheWriteTokens") or 0),
            "reasoning": int(usage.get("reasoningTokens") or 0),
        }

        raw = response.get("raw")
        cost = compute_cost(
            self.catalog,
            provider,
            model,
            usage,
            provider_evidence=raw if isinstance(raw, Mapping) else {},
            tier=usage.get("pricingTier"),
        )

        inner: Mapping[str, Any] = ctx.get("ctx") or {}
        entry = CostEntry(
            id=f"cost_{uuid.uuid4().hex[:12]}",
            timestamp=time.time() * 1000,
            provider=provider,
            model=model,
            tokens=tokens,
            cost=cost,
            service_tier=usage.get("serviceTier"),
            tags={
                **self.default_tags,
                "provider": provider,
                "model": model,
                "sessionId": self.session_id,
                "runId": inner.get("requestId"),
                "conversationId": inner.get("conversationId"),
            },
        )

        self._ledger.append(entry)
        if cost is not None:
            self._running_total += cost.total
        self._note_if_unpriced(entry)
        # The ENTRY itself, not a `{entry, runningTotal}` wrapper. A
        # subscriber wanting the total reads `engine.cost.running_total`,
        # which is authoritative rather than a snapshot that can drift from
        # the ledger it was copied out of.
        self.hooks.emit_sync("onCostEntry", entry)

    def _note_if_unpriced(self, entry: CostEntry) -> None:
        """Say so once, the first time a model turns out to have no price."""
        if not entry.unpriced:
            return
        key = f"{entry.provider}/{entry.model}"
        if key in self._warned:
            return
        self._warned.add(key)
        self.hooks.emit_sync(
            "onWarning",
            {
                "source": "cost",
                "code": "unpriced_model",
                "message": (
                    f"No catalog pricing for {key} -- its cost is reported as unknown, "
                    f"which is not the same as free. Check the model id against the "
                    f"catalog, or add pricing for it."
                ),
                "details": {"provider": entry.provider, "model": entry.model},
            },
        )


__all__ = ["CostCollector", "CostEntry", "CostSummary", "TokenTotals", "summarize"]
