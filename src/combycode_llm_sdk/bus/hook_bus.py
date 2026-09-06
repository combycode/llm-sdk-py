"""HookBus -- pub/sub for SDK instrumentation.

Transposed from `unified-library-ts/src/bus/hook-bus.ts`.

Subsystems (network engine, llm client, agent loop, server, plugins) emit hook
events; consumers (logger, cache, cost, context guard, ...) subscribe.

Distinct from AgentBus: HookBus carries SDK-internal instrumentation, keyed per
event name. AgentBus carries cross-module business events with pattern matching
and correlation.

**Failure mode: handlers raising inside `emit` propagate.** We do NOT swallow --
emitters need to know if a critical handler (like a ContextGuard abort) failed.
Plugins that should never break the request must catch their own errors.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Callable
from typing import Any

from .context import as_context
from .events import build_event, to_snake
from .hook_map import HOOK_NAMES, HookContext, HookEvent, HookName

#: A handler for one hook. May be a plain function or a coroutine function; the
#: TypeScript signature is `(ctx) => void | Promise<void>` and both halves are
#: used, so both are accepted here.
HookHandler = Callable[[HookContext], Any]

#: Catch-all handler: receives EVERY emit as one tagged event.
#:
#: It used to receive `(name, ctx: unknown)`, which pushed the type back onto the
#: subscriber -- and the SDK's own telemetry adapter, the only consumer, answered
#: the way anyone would: a cast, then a cast per field, 26 of them in one method.
#: Renaming a context field left those reads compiling and silently undefined,
#: which for the token and cost fields means a metric that quietly goes to zero.
AnyHookHandler = Callable[[HookEvent], Any]


#: Every accepted spelling of every hook, mapped to the canonical camelCase one.
#: A dict rather than a scan of `HOOK_NAMES`: `emit_sync` runs once per stream
#: chunk, and a linear search over fifty names on that path would be paid for by
#: every token of every response.
_CANONICAL: dict[str, HookName] = {
    **{name: name for name in HOOK_NAMES},
    **{to_snake(name): name for name in HOOK_NAMES},
}


def _hook_event(name: HookName, ctx: HookContext) -> HookEvent:
    """Pair a name with its context as one matchable event value."""
    return build_event(name, ctx)


class HookBus:
    """`class HookBus` (hook-bus.ts:37)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookHandler]] = {}
        self._any_handlers: list[AnyHookHandler] = []

    @staticmethod
    def _resolve(name: HookName) -> HookName:
        """The canonical name, and a refusal for one no hook is emitted under.

        TypeScript gets this from `K extends HookName` at compile time. Python
        has to ask at runtime, and asking is worth it: a subscription to
        `'onComplete'` (the real name is `onCompletion`) is silence, and silence
        from instrumentation looks exactly like nothing happening.

        Both spellings are accepted and normalised to the camelCase one the
        catalog and the emitters use. `bus/context.py` already delivers the
        PAYLOAD under either spelling, and `bus/events.py` names each event in
        snake so the name off one can be pasted back into a subscription -- a
        subscriber who writes `ctx.tool_name` and then has to write
        `'onInternalToolCallStart'` to reach it has been handed the boundary
        rather than spared it.
        """
        canonical = _CANONICAL.get(name)
        if canonical is None:
            raise ValueError(f'unknown hook "{name}"')
        return canonical

    def on_any(self, handler: AnyHookHandler) -> Callable[[], None]:
        """Register a catch-all handler invoked for every event (telemetry tap).

        Returns an unsubscribe function.
        """
        self._any_handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._any_handlers:
                self._any_handlers.remove(handler)

        return unsubscribe

    def on(self, name: HookName, handler: HookHandler) -> Callable[[], None]:
        """Register a handler. Returns an unsubscribe function."""
        name = self._resolve(name)
        self._handlers.setdefault(name, []).append(handler)

        def unsubscribe() -> None:
            arr = self._handlers.get(name)
            if arr is None:
                return
            if handler in arr:
                arr.remove(handler)
            if not arr:
                del self._handlers[name]

        return unsubscribe

    def once(self, name: HookName, handler: HookHandler) -> Callable[[], None]:
        """Register a one-time handler."""
        holder: list[Callable[[], None]] = []

        def wrapper(ctx: HookContext) -> Any:
            holder[0]()
            return handler(ctx)

        holder.append(self.on(name, wrapper))
        return holder[0]

    def off(self, name: HookName | None = None) -> None:
        """Remove all handlers for a hook, or all hooks (incl. catch-all)."""
        if name:
            self._handlers.pop(self._resolve(name), None)
        else:
            self._handlers.clear()
            self._any_handlers = []

    async def emit(self, name: HookName, ctx: HookContext) -> None:
        """Emit asynchronously.

        Handlers run in registration order and are awaited sequentially, so a
        handler that mutates the context is seen by the next one -- ordering the
        ContextGuard depends on. Any handler error propagates to the caller.

        Iterates over a COPY of the list: a handler that unsubscribes itself (or
        another) mutates the list mid-loop, which in Python skips the next
        handler rather than raising. `once` does exactly that on every fire.
        """
        # Wrapped ONCE for the whole emit, not per handler: the view is what
        # makes mutation visible to the next handler, and a fresh one each time
        # would still work but allocate per subscriber for no gain.
        name = _CANONICAL.get(name, name)
        ctx = as_context(ctx)
        for handler in list(self._handlers.get(name, ())):
            result = handler(ctx)
            if inspect.isawaitable(result):
                await result
        if not self._any_handlers:
            return
        event = _hook_event(name, ctx)
        for any_handler in list(self._any_handlers):
            result = any_handler(event)
            if inspect.isawaitable(result):
                await result

    def emit_sync(self, name: HookName, ctx: HookContext) -> None:
        """Emit synchronously. For hot paths (per-chunk stream events).

        Async handlers START but are NOT awaited; their resolution timing is
        undefined. Use only when the emitter cannot block.

        "Starting" a coroutine needs a loop to start it on, which JavaScript
        always has and Python does not. With a running loop the coroutine is
        scheduled as a task, which is what the TypeScript does. With none, it
        cannot run at all -- and dropping it silently would make a subscribed
        handler that never fires indistinguishable from one that did, so it warns
        instead.
        """
        name = _CANONICAL.get(name, name)
        if self._handlers.get(name) or self._any_handlers:
            ctx = as_context(ctx)
        for handler in list(self._handlers.get(name, ())):
            self._start(handler(ctx), name)
        # Nothing is allocated when no catch-all is subscribed, which is the
        # common case on the hot paths that use emit_sync (one call per stream
        # chunk).
        if not self._any_handlers:
            return
        event = _hook_event(name, ctx)
        for any_handler in list(self._any_handlers):
            self._start(any_handler(event), name)

    @staticmethod
    def _start(result: Any, name: HookName) -> None:
        if not inspect.isawaitable(result):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            close = getattr(result, "close", None)
            if close is not None:
                close()  # never-awaited coroutine, closed rather than left to warn twice
            warnings.warn(
                f'HookBus.emit_sync("{name}"): an async handler was not run -- '
                "emit_sync can only start one while an event loop is running.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        # `ensure_future`, not `create_task`: a handler may return any awaitable,
        # not only a coroutine, and only this one wraps both.
        # The task handle is deliberately dropped: emit_sync's contract is
        # fire-and-forget, and holding a reference would only defer the question
        # of who awaits it to a caller that cannot.
        asyncio.ensure_future(result)

    def has(self, name: HookName) -> bool:
        """Whether any handlers are registered for a hook name."""
        return bool(self._handlers.get(_CANONICAL.get(name, name)))

    @property
    def handler_count(self) -> int:
        """Diagnostic: total handlers across all names.

        Useful for leak detection in tests -- `engine.destroy()` should bring
        this back to 0 if no external subscribers remain.
        """
        return len(self._any_handlers) + sum(len(v) for v in self._handlers.values())


__all__ = ["AnyHookHandler", "HookBus", "HookHandler"]
