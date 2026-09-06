"""`Observer` -- watch ONE agent, on a bus several agents share.

Transposed from `unified-library-ts/src/helpers/observer.ts`.

    with Observer(expert, "on_completion", seen.append):
        boss.run("ask the expert what 6*7 is")

The whole difficulty is attribution. A specialist called as a tool runs INSIDE
its caller's run, on the same engine and usually the same model, so anything
that scopes by run -- or guesses by model -- files the specialist's tokens under
the caller. Nothing downstream can tell: every number stays plausible, the
per-agent cost is simply wrong, and the specialist looks free.

What separates them is `conversationId`, which `AgentLoop` stamps on every LLM
call it makes and which equals the agent's own id. That is the filter here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..results import Completion

#: `on_completion` -> `onCompletion`, and the rest of the agent-scoped events.
#: Spelled out rather than derived by camel-casing, so a name that does not
#: exist is an error here instead of a subscription that never fires.
EVENTS: dict[str, str] = {
    "on_agent_create": "onAgentCreate",
    "on_agent_destroy": "onAgentDestroy",
    "on_run_start": "onRunStart",
    "on_run_complete": "onRunComplete",
    "on_run_error": "onRunError",
    "on_step_start": "onStepStart",
    "on_step_complete": "onStepComplete",
    "on_tool_call_start": "onToolCallStart",
    "on_tool_call_complete": "onToolCallComplete",
    "on_tool_call_error": "onToolCallError",
    "on_tool_search": "onToolSearch",
    "on_completion": "onCompletion",
    "on_warning": "onWarning",
}


class Observer:
    """A scoped subscription, usable as a context manager.

    Entering returns the AGENT, not the observer, so the watched agent is what
    the `as` clause binds and the run reads naturally inside the block.
    """

    def __init__(self, agent: Any, event: str, reactor: Callable[[Any], Any]) -> None:
        name = EVENTS.get(event, event)
        if name not in set(EVENTS.values()):
            raise ValueError(
                f"Observer: {event!r} is not an observable event. One of: "
                f"{', '.join(sorted(EVENTS))}."
            )
        self.agent = agent
        self.event = name
        self._reactor = reactor
        self._unsubscribe: Callable[[], None] | None = None
        self.subscribe()

    def subscribe(self) -> Observer:
        if self._unsubscribe is None:
            self._unsubscribe = self.agent.hooks.on(self.event, self._deliver)
        return self

    def close(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def __enter__(self) -> Any:
        return self.agent

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _deliver(self, ctx: Any) -> None:
        if not self._is_ours(ctx):
            return
        self._reactor(self._view(ctx))

    def _is_ours(self, ctx: Any) -> bool:
        """Whether this event belongs to the agent being watched.

        `agentId` when the agent layer emitted it; `ctx.conversationId` when the
        LLM client did, which is the case for `onCompletion` and the one that
        matters -- it is the only place a sub-agent's turn can be told from its
        caller's.
        """
        if not isinstance(ctx, Mapping):
            return bool(getattr(ctx, "agent_id", None) == self.agent.id)
        if ctx.get("agentId") is not None:
            return bool(ctx.get("agentId") == self.agent.id)
        inner = ctx.get("ctx")
        if isinstance(inner, Mapping):
            return bool(inner.get("conversationId") == self.agent.id)

        # No identity on the event at all. Refusing it is the safe direction: a
        # watcher that reports another agent's work as this one's is worse than
        # one that reports nothing, because only the first is believed.
        return False

    def _view(self, ctx: Any) -> Any:
        """What the reactor is handed.

        For `onCompletion` that is a `Completion` -- the same view the caller of
        `complete()` gets, so `seen[0].usage` reads the same as `answer.usage`
        and nobody has to learn a second shape for one response.
        """
        if self.event != "onCompletion" or not isinstance(ctx, Mapping):
            return ctx
        response = ctx.get("response")
        if not isinstance(response, Mapping):
            return ctx
        return Completion.of(
            response,
            provider=str(ctx.get("provider") or ""),
            model=str(ctx.get("model") or ""),
            api=str((ctx.get("ctx") or {}).get("api") or ""),
            cost=None,
        )


__all__ = ["EVENTS", "Observer"]
