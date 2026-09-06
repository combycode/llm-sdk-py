"""Routing context pressure to a per-conversation strategy.

Transposed from `unified-library-ts/src/plugins/context-guard/guard.ts`.

The guard itself holds no per-conversation state. Everything it remembers lives
in `history.metadata`, so a conversation that is exported, restored, or handed
to a different process keeps its place on the trigger ladder -- and two
conversations sharing one guard cannot read each other's.

Two things it is careful about, both of which look like details and are not:

  - It reacts to a NEW crossing, or to continued growth at the level it is
    already on. Reacting to every measurement would summarise the same range
    repeatedly while the conversation sat still at 72%.
  - It judges a strategy on what compaction LEFT, re-measuring in between. The
    alternative is declining work that succeeded: a truncation that took a
    conversation from 100% of the window to 20% was refused for being "still
    above 95%", after it had already destroyed eight messages. The caller lost
    both the history and the call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from .strategy_types import (
    ContextStrategy,
    GuardConversationState,
    ReactContext,
    StrategyDecision,
    TriggerLevel,
    UnknownStrategyPolicy,
    highest_crossed_level,
)
from .tools import ContextTools, NoopContextTools, StrategyToolsImpl

#: How many times the guard will ask a strategy to try again before giving up.
DEFAULT_MAX_COMPACT_RETRIES = 2

#: Above this share AFTER compacting, keep going rather than accepting it.
DEFAULT_CRITICAL_FLOOR = 0.95

#: Where guard state lives inside `history.metadata`. Namespaced so it cannot
#: collide with a caller's own metadata keys.
STATE_KEY = "__orxa"
GUARD_STATE_SUBKEY = "contextGuard"


class ContextGuard:
    """Compact before the provider refuses, then judge what is left."""

    def __init__(
        self,
        *,
        hooks: Any,
        measurer: Any,
        strategies: Mapping[str, ContextStrategy],
        default_strategy: str,
        context_tools: ContextTools | None = None,
        on_unknown_strategy: UnknownStrategyPolicy = "skip",
        max_compact_retries: int = DEFAULT_MAX_COMPACT_RETRIES,
        critical_floor: float = DEFAULT_CRITICAL_FLOOR,
    ) -> None:
        if default_strategy not in strategies:
            raise ValueError(
                f'ContextGuard: default_strategy "{default_strategy}" is not in the strategies '
                f"map (keys: [{', '.join(strategies)}])"
            )
        self.hooks = hooks
        self.measurer = measurer
        self.strategies = dict(strategies)
        self.default_strategy = default_strategy
        self.context_tools: ContextTools = context_tools or NoopContextTools()
        self.on_unknown_strategy = on_unknown_strategy
        self.max_retries = max_compact_retries
        self.critical_floor = critical_floor

        self._trigger_cache: dict[int, list[TriggerLevel]] = {}
        self._warned_unknown: set[str] = set()
        self._unsubscribe: Callable[[], None] | None = hooks.on(
            "onContextMeasure", self._handle_measure
        )

    def destroy(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    # -- the hook ------------------------------------------------------------

    async def _handle_measure(self, ctx: MutableMapping[str, Any]) -> None:
        history = ctx.get("history")
        if history is None:
            return
        window, percentage = ctx.get("window"), ctx.get("percentage")
        if window is None or percentage is None:
            return

        strategy = self._resolve_strategy(history)
        if strategy is None:
            return

        triggers = self._sorted_triggers(strategy)
        state = self._read_state(history)
        crossed = highest_crossed_level(triggers, float(percentage))
        delta = int(ctx.get("current") or 0) - state.last_current

        # A new rung, or still climbing on the rung we are already on. Anything
        # else is the same pressure re-reported, and acting on it would
        # re-summarise the same range for no gain.
        is_new_crossing = crossed > state.last_level_idx
        is_climbing = crossed >= 0 and crossed == state.last_level_idx and delta > 0
        if not is_new_crossing and not is_climbing:
            state.last_current = int(ctx.get("current") or 0)
            state.last_level_idx = max(state.last_level_idx, crossed)
            self._write_state(history, state)
            return

        state.last_level_idx = crossed
        state.last_current = int(ctx.get("current") or 0)
        self._write_state(history, state)

        name = self._resolve_strategy_name(history)
        strategy_state = state.strategy_state.setdefault(name, {})

        messages = ctx.get("messages")
        tools = StrategyToolsImpl(
            history=history,
            active_messages=messages if isinstance(messages, list) else [],
            counter=self.measurer.counter,
            context_tools=self.context_tools,
            provider=str(ctx.get("provider") or ""),
            model=str(ctx.get("model") or ""),
        )

        attempt = 0
        current = int(ctx.get("current") or 0)
        share = float(percentage)

        while attempt <= self.max_retries:
            react_ctx = ReactContext(
                level=triggers[crossed].level if 0 <= crossed < len(triggers) else "",
                percentage=share,
                current=current,
                window=int(window),
                delta=delta,
                provider=str(ctx.get("provider") or ""),
                model=str(ctx.get("model") or ""),
                attempt=attempt,
                tools=tools,
                state=strategy_state,
            )
            decision = await strategy.react(react_ctx)
            self._write_strategy_state(history, name, strategy_state)

            if decision.action == "none":
                return

            if decision.action == "warn":
                await self._warn(
                    "context_pressure",
                    decision.message or "",
                    history,
                    {"level": react_ctx.level, "percentage": share, "current": current},
                )
                return

            if decision.action == "decline":
                ctx["abort"] = True
                ctx["abortReason"] = decision.reason
                await self._warn(
                    "context_declined",
                    f"Context declined: {decision.reason}",
                    history,
                    {"percentage": share, "current": current},
                )
                return

            # compacted: re-measure what is LEFT, and report it back so the
            # caller's own view of the request agrees with the transcript.
            current = tools.measure(tools.active_messages)
            share = current / int(window) if window else 0.0
            ctx["current"] = current
            ctx["percentage"] = share

            if share < self.critical_floor and share < triggers[crossed].at:
                return
            attempt += 1

        ctx["abort"] = True
        ctx["abortReason"] = (
            f"Context still at {share * 100:.1f}% after {self.max_retries} compaction attempts; "
            "unable to fit request."
        )
        await self._warn(
            "context_exhausted", str(ctx["abortReason"]), history,
            {"percentage": share, "current": current, "attempts": attempt},
        )

    # -- strategy resolution -------------------------------------------------

    def _resolve_strategy(self, history: Any) -> ContextStrategy | None:
        raw = history.metadata.get("contextStrategy")
        # `False` opts this conversation out entirely, which is not the same as
        # not having said anything.
        if raw is False:
            return None
        name = raw if isinstance(raw, str) and raw else self.default_strategy

        found = self.strategies.get(name)
        if found is not None:
            return found

        # Warned ONCE per unknown name: this runs on every measurement, and a
        # per-request warning about one typo would bury everything else.
        if name not in self._warned_unknown:
            self._warned_unknown.add(name)
            self.hooks.emit_sync(
                "onWarning",
                {
                    "source": "context",
                    "code": "context_unknown_strategy",
                    "message": f'ContextGuard: strategy "{name}" is not registered',
                    "details": {
                        "conversationId": history.id,
                        "strategyName": name,
                        "available": list(self.strategies),
                        "resolution": self.on_unknown_strategy,
                    },
                },
            )

        if self.on_unknown_strategy == "skip":
            return None
        if self.on_unknown_strategy == "fallback-default":
            return self.strategies[self.default_strategy]
        raise ValueError(
            f'ContextGuard: unknown strategy "{name}" on conversation {history.id}. '
            f"Available: [{', '.join(self.strategies)}]"
        )

    def _resolve_strategy_name(self, history: Any) -> str:
        raw = history.metadata.get("contextStrategy")
        if isinstance(raw, str) and raw and raw in self.strategies:
            return raw
        return self.default_strategy

    def _sorted_triggers(self, strategy: ContextStrategy) -> list[TriggerLevel]:
        """The ladder, lowest first, cached per strategy object.

        Sorted rather than trusted: `highest_crossed_level` stops at the first
        rung it cannot reach, so a ladder given out of order would silently
        stop looking partway up.
        """
        key = id(strategy)
        cached = self._trigger_cache.get(key)
        if cached is None:
            cached = sorted(strategy.triggers, key=lambda t: t.at)
            self._trigger_cache[key] = cached
        return cached

    # -- per-conversation state ---------------------------------------------

    def _read_state(self, history: Any) -> GuardConversationState:
        orxa = history.metadata.get(STATE_KEY)
        raw = orxa.get(GUARD_STATE_SUBKEY) if isinstance(orxa, Mapping) else None
        return GuardConversationState.of(raw)

    def _write_state(self, history: Any, state: GuardConversationState) -> None:
        orxa = history.metadata.get(STATE_KEY)
        if not isinstance(orxa, dict):
            orxa = {}
        orxa[GUARD_STATE_SUBKEY] = state.as_row()
        history.metadata[STATE_KEY] = orxa

    def _write_strategy_state(
        self, history: Any, name: str, strategy_state: dict[str, Any]
    ) -> None:
        current = self._read_state(history)
        current.strategy_state[name] = strategy_state
        self._write_state(history, current)

    async def _warn(
        self, code: str, message: str, history: Any, details: Mapping[str, Any]
    ) -> None:
        await self.hooks.emit(
            "onWarning",
            {
                "source": "context",
                "code": code,
                "message": message,
                "details": {"conversationId": history.id, **details},
            },
        )


__all__ = [
    "DEFAULT_CRITICAL_FLOOR",
    "DEFAULT_MAX_COMPACT_RETRIES",
    "GUARD_STATE_SUBKEY",
    "STATE_KEY",
    "ContextGuard",
    "StrategyDecision",
]
