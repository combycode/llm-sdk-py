"""What a compaction strategy is handed, and what it may answer.

Transposed from the strategy half of
`unified-library-ts/src/plugins/context-guard/types.ts`.

A strategy decides policy and nothing else. It is given a reading of the
pressure and a set of tools, and it answers with one of four decisions. It never
touches the transport, never measures for itself, and never decides whether the
request goes -- the guard does that with the answer.

The four decisions are deliberately distinct. `none` and `compacted` both mean
"carry on", but only one of them says the transcript changed, and the guard
re-measures after the second. `warn` and `decline` both mean "something is
wrong", but only one of them stops the request. Collapsing either pair loses the
distinction the guard needs to act correctly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

#: What to do about a conversation naming a strategy the guard does not have.
#: `skip` leaves the conversation un-guarded, `fallback-default` guards it with
#: the default, `throw` refuses. `skip` is the default because a typo in a
#: strategy name must not take down a request -- but it warns either way, so the
#: typo is not silent.
UnknownStrategyPolicy = Literal["skip", "fallback-default", "throw"]


@dataclass(frozen=True)
class TriggerLevel:
    """One step of the ladder: a name, and the share of the window it fires at."""

    level: str
    at: float


@dataclass(frozen=True)
class StrategyDecision:
    """What a strategy answered.

    Built through the four constructors below rather than by hand, so an
    `action` that no branch of the guard handles cannot be spelled.
    """

    action: Literal["none", "compacted", "warn", "decline"]
    note: str | None = None
    message: str | None = None
    reason: str | None = None

    @staticmethod
    def none() -> StrategyDecision:
        """Nothing to do. The guard leaves the request alone."""
        return StrategyDecision(action="none")

    @staticmethod
    def compacted(note: str | None = None) -> StrategyDecision:
        """The transcript changed. The guard re-measures what is LEFT."""
        return StrategyDecision(action="compacted", note=note)

    @staticmethod
    def warn(message: str) -> StrategyDecision:
        """Say something, send anyway."""
        return StrategyDecision(action="warn", message=message)

    @staticmethod
    def decline(reason: str) -> StrategyDecision:
        """Stop the request. Compaction will not save it."""
        return StrategyDecision(action="decline", reason=reason)


@dataclass
class ReactContext:
    """The reading a strategy acts on.

    Mutable because the guard updates `percentage` and `current` between
    compaction attempts: a strategy asked to try again is told what its last
    attempt achieved, not what the situation was before it ran.
    """

    #: The trigger that fired, by name.
    level: str
    percentage: float
    current: int
    window: int | None
    #: Growth since the last measurement, in tokens. A strategy may escalate on
    #: a jump: arriving at 70% is not the same as arriving at 70% from 30%.
    delta: int
    provider: str
    model: str
    #: 0 on the first call, incremented each time the guard asks again.
    attempt: int
    tools: Any
    #: Scratch space that survives between turns of ONE conversation, per
    #: strategy. The guard persists it; the strategy decides what it means.
    state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextStrategy(Protocol):
    """Policy: when to act, and what to do."""

    triggers: list[TriggerLevel]

    async def react(self, ctx: ReactContext) -> StrategyDecision: ...


@dataclass
class GuardConversationState:
    """What the guard remembers about one conversation between requests.

    Versioned because it is written into `history.metadata` and therefore
    survives an export/import: a snapshot taken by an older build must be
    recognisable as one, rather than read as the current shape and misused.
    """

    v: int = 1
    #: Index into the sorted trigger ladder, or -1 for "below every trigger".
    last_level_idx: int = -1
    last_current: int = 0
    strategy_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def of(raw: Mapping[str, Any] | None) -> GuardConversationState:
        if not isinstance(raw, Mapping) or raw.get("v") != 1:
            return GuardConversationState()
        state = raw.get("strategyState")
        return GuardConversationState(
            v=1,
            last_level_idx=int(raw.get("lastLevelIdx", -1)),
            last_current=int(raw.get("lastCurrent", 0)),
            strategy_state=dict(state) if isinstance(state, Mapping) else {},
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "lastLevelIdx": self.last_level_idx,
            "lastCurrent": self.last_current,
            "strategyState": self.strategy_state,
        }


def highest_crossed_level(triggers: list[TriggerLevel], percentage: float) -> int:
    """The furthest rung this share reaches, as an index, or -1.

    Stops at the first rung NOT reached rather than scanning them all, which is
    what makes the ladder a ladder: the levels are ordered, so once one is out
    of reach every higher one is too.
    """
    idx = -1
    for i, trigger in enumerate(triggers):
        if percentage >= trigger.at:
            idx = i
        else:
            break
    return idx


__all__ = [
    "ContextStrategy",
    "GuardConversationState",
    "ReactContext",
    "StrategyDecision",
    "TriggerLevel",
    "UnknownStrategyPolicy",
    "highest_crossed_level",
]
