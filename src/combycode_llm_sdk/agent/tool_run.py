"""What a tool can learn about the call it is serving.

Transposed from the `ToolExecutionContext` half of
`unified-library-ts/src/agent/loop-internals.ts`, but through a ContextVar rather
than an argument, because a tool here is a PLAIN FUNCTION: its signature is the
model's parameter list and nothing else. Threading a context argument would mean
every tool declaring a parameter the model must not see.

Two things live here, and a nested agent run needs both:

* the run's TRACE, so a sub-agent started inside a tool joins the trace of the
  run that started it instead of rooting one of its own -- correlation being the
  entire point of a trace id;
* whether a STOP has been asked for. Not a signal that fires: `AgentLoop.stop()`
  on this side is cooperative on purpose (killing a worker mid-write is how a
  half-applied side effect happens), so this is something a long-running tool
  CHECKS. A sub-agent given it stops at its next step boundary, which is the
  same guarantee the parent gives itself.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolRunContext:
    """The call a tool is currently serving."""

    tool_name: str
    call_id: str = ""
    step: int = 0
    #: The run's trace: sessionId, requestId, and traceparent when the caller
    #: gave one. Handed to a nested run whole -- see `nested_run_options`.
    trace: Mapping[str, Any] = field(default_factory=dict)
    #: Reads the live flag; it can flip WHILE the tool runs, which is the only
    #: reason it is a callable rather than a bool copied at entry.
    is_stopping: Callable[[], bool] = field(default=lambda: False)


_current: contextvars.ContextVar[ToolRunContext | None] = contextvars.ContextVar(
    "combycode_llm_sdk_tool_run", default=None
)


def current_tool_run() -> ToolRunContext | None:
    """The call being served, or None outside a tool.

    None is a normal answer: a tool is directly callable -- `ask(task="...")` --
    and testing one without a loop around it must not require a fake context.
    """
    return _current.get()


@contextmanager
def tool_run(ctx: ToolRunContext) -> Iterator[None]:
    """Mark the enclosing block as serving `ctx`."""
    token = _current.set(ctx)
    try:
        yield
    finally:
        _current.reset(token)


def nested_run_options(ctx: ToolRunContext | None = None) -> dict[str, Any]:
    """The options a run started INSIDE a tool should be given.

    Defaults to the call in progress, so a delegating tool needs no argument.

    Passes the WHOLE trace, not a field or two of it: `sessionId` and
    `requestId` are resolved together when a run begins, and supplying one
    without the other is how a single request ends up in two traces. `callId`
    names the call that delegated, so the nested run hangs off it rather than
    floating beside the parent.

    Outside a tool it returns `{}` -- nothing to inherit, and a caller that
    wanted its own ctx can still pass one.
    """
    ctx = ctx or current_tool_run()
    if ctx is None:
        return {}
    trace = dict(ctx.trace)
    # `conversationId` is NOT inherited. It is how an Observer tells one agent's
    # completions from another's on a shared engine, and a specialist has its own
    # history -- handing it the parent's would file the child's turns under the
    # parent's conversation. This trace carries it because `_begin_run` puts it
    # there; the TypeScript's run trace does not, which is why only this side
    # needs the exclusion.
    trace.pop("conversationId", None)
    # And the caller's stop. A sub-agent whose caller gave up is answering a
    # question nobody will read, and the parent cannot reach into the child's
    # flag -- so the child is given the question to ask, and stops at its next
    # step boundary. Checked rather than fired: cancellation on this side is
    # cooperative by design.
    return {"ctx": {**trace, "callId": ctx.call_id}, "stop_when": ctx.is_stopping}


__all__ = ["ToolRunContext", "current_tool_run", "nested_run_options", "tool_run"]
