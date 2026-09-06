"""Fitting the next request into the window.

Transposed from `unified-library-ts/src/plugins/context-measurer/measurer.ts`.
The strategies live in `strategies.py` and the guard that routes to them in
`context_guard.py`, mirroring the TypeScript's own split.

`ContextMeasurer` answers two questions at once: how many tokens this is, and
how sure it is. The second half is the point. An exact count costs a tokenizer
load or an HTTP round trip, so it estimates cheaply and escalates only near the
window -- and whatever it does, the number arrives labelled. A count that cannot
say how it was reached lets an estimate stand in for a measurement, which is the
one substitution a context check does not survive.

It publishes what it finds on `onContextMeasure` rather than acting on it, so a
guard, a telemetry sink and a caller that only wants a warning can all read the
same measurement without knowing about each other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .strategy_types import TriggerLevel

#: Share of the window at which to warn, and at which an exact count is worth
#: paying for. Below the second, a round trip buys a number nobody compares.
DEFAULT_WARN_AT = 0.80
DEFAULT_EXACT_AT = 0.90

#: Characters per token when the catalog names none.
FALLBACK_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class Measured:
    """A token count, and what it is worth."""

    tokens: int
    strategy: str
    exact: bool
    window: int | None = None
    #: Share of the window, or None when the window is unknown -- which is not
    #: the same as 0 and must not be compared as if it were.
    percentage: float | None = None


def _entry_for(catalog: Any, provider: str, model: str) -> Any:
    if catalog is None:
        return None
    getter = getattr(catalog, "get", None)
    return getter(provider, model) if callable(getter) else None


def _window_of(entry: Any) -> int | None:
    if entry is None:
        return None
    window = getattr(entry, "context_window", None)
    if isinstance(window, (int, float)) and window > 0:
        return int(window)
    return None


def _tokenizer_of(entry: Any) -> Mapping[str, Any]:
    """The tokenizer block, from either shape a catalog entry comes in.

    A `ModelInfo` is a mapping over the stored entry; a caller's own catalog may
    hand back an object with a `raw` dict. Reading only one of them would make
    the measurer work against the bundled catalog and silently fall back to
    defaults against anyone else's.
    """
    if entry is None:
        return {}
    raw = getattr(entry, "raw", None)
    if isinstance(raw, Mapping):
        return raw.get("tokenizer") or {}
    if isinstance(entry, Mapping):
        return entry.get("tokenizer") or {}
    return {}


def _text_of(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, Mapping):
        return str(message)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(
            str(p.get("text") or "") for p in content if isinstance(p, Mapping)
        )
    return str(content or "")


class ContextMeasurer:
    """How big this is, and how sure we are."""

    def __init__(
        self,
        catalog: Any = None,
        *,
        count: Callable[..., Any] | None = None,
        counter: Any = None,
        hooks: Any = None,
        warn_at: float = DEFAULT_WARN_AT,
        exact_at: float = DEFAULT_EXACT_AT,
    ) -> None:
        from ..tokens import HybridCounter

        self.catalog = catalog
        #: The exact counter to escalate to. `count_tokens` fits this shape.
        self.count = count
        #: Per-message counting, for anything that measures a transcript rather
        #: than a blob of text. The guard's tools use this one.
        self.counter = counter or HybridCounter(catalog)
        self.hooks = hooks
        self.warn_at = warn_at
        self.exact_at = exact_at
        self._unsubscribe: Callable[[], None] | None = None
        if hooks is not None:
            self._unsubscribe = hooks.on("onMessageResolve", self._on_message_resolve)

    def destroy(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    async def _on_message_resolve(self, ctx: Any) -> None:
        """Measure the request being built, and let listeners stop it."""
        messages = ctx.get("messages") if isinstance(ctx, Mapping) else None
        if not isinstance(messages, list):
            return
        result = await self.measure_and_emit(
            str(ctx.get("provider") or ""),
            str(ctx.get("model") or ""),
            messages,
            history=ctx.get("history"),
            system=ctx.get("system"),
        )
        if result.get("abort"):
            ctx["abort"] = True
            if result.get("abortReason") is not None:
                ctx["abortReason"] = result["abortReason"]

    async def measure_and_emit(
        self,
        provider: str,
        model: str,
        messages: list[Any],
        *,
        history: Any = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Measure, publish the reading, and report what listeners decided.

        This is the seam the guard hangs off. It is a hook rather than a direct
        call because more than one listener may care about context pressure --
        a guard that compacts, a telemetry sink that records, a caller that just
        wants to warn -- and none of them should have to know about each other.

        The total is re-derived after the emit when nothing aborted, because a
        listener may have compacted the transcript: reporting the pre-compaction
        number would describe a request that is no longer the one being sent.
        """
        total = self._sum(provider, model, messages, system)
        window = self.window(provider, model)
        percentage = (total / window) if window else None

        ctx: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "current": total,
            "window": window,
            "percentage": percentage,
            "messages": messages,
            "system": system,
            "history": history,
            "abort": None,
            "abortReason": None,
        }
        if self.hooks is not None:
            await self.hooks.emit("onContextMeasure", ctx)

        final = int(ctx.get("current") or 0)
        if not ctx.get("abort"):
            final = self._sum(provider, model, messages, system)
        return {
            "total": final,
            "abort": bool(ctx.get("abort")),
            "abortReason": ctx.get("abortReason"),
        }

    def _sum(
        self, provider: str, model: str, messages: list[Any], system: str | None
    ) -> int:
        ctx = {"provider": provider, "model": model}
        total = self.counter.estimate_message(system, ctx) if system else 0
        for message in messages:
            total += self.counter.estimate_message(message, ctx)
        return int(total)

    def chars_per_token(self, provider: str, model: str) -> float:
        tokenizer = _tokenizer_of(_entry_for(self.catalog, provider, model))
        rate = tokenizer.get("charsPerTokenDefault")
        return float(rate) if isinstance(rate, (int, float)) and rate > 0 else (
            FALLBACK_CHARS_PER_TOKEN
        )

    def declared_strategy(self, provider: str, model: str) -> str:
        """What the catalog says this model SHOULD be counted with."""
        tokenizer = _tokenizer_of(_entry_for(self.catalog, provider, model))
        named = tokenizer.get("strategy")
        return str(named) if isinstance(named, str) else "heuristic"

    def window(self, provider: str, model: str) -> int | None:
        return _window_of(_entry_for(self.catalog, provider, model))

    def estimate(self, provider: str, model: str, messages: Sequence[Any]) -> int:
        rate = self.chars_per_token(provider, model)
        chars = sum(len(_text_of(m)) for m in messages)
        return -(-chars // int(rate)) if float(rate).is_integer() else int(chars / rate + 0.999)

    def measure(
        self,
        provider: str,
        model: str,
        messages: Sequence[Any],
        *,
        exact: bool | None = None,
    ) -> Measured:
        """The count, escalating only when the answer is worth paying for."""
        window = self.window(provider, model)
        tokens = self.estimate(provider, model, messages)
        percentage = (tokens / window) if window else None

        wants_exact = exact if exact is not None else (
            percentage is not None and percentage >= self.exact_at
        )
        if wants_exact and self.count is not None:
            counted = self.count(
                model=f"{provider}/{model}",
                provider=provider,
                input=list(messages),
            )
            # An exact count is an IMPROVEMENT on the answer, never a
            # precondition for having one -- and an answer that came back
            # inexact is discarded rather than promoted, so `exact` still means
            # what it says.
            if getattr(counted, "exact", False):
                tokens = int(counted.tokens)
                return Measured(
                    tokens=tokens,
                    strategy=str(getattr(counted, "strategy", "exact")),
                    exact=True,
                    window=window,
                    percentage=(tokens / window) if window else None,
                )

        return Measured(
            tokens=tokens,
            strategy="estimate",
            exact=False,
            window=window,
            percentage=percentage,
        )


__all__ = [
    "DEFAULT_EXACT_AT",
    "DEFAULT_WARN_AT",
    "FALLBACK_CHARS_PER_TOKEN",
    "ContextMeasurer",
    "Measured",
    "TriggerLevel",
]
