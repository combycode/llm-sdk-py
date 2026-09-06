"""The event stream as one value per hook.

`on_any` could have handed over `(name, ctx)` and let every consumer read fields
off a bag. That works forever -- until a field is renamed, when the exporter
keeps running and starts reading `None`. For a token count or a cost that is a
metric which quietly goes to zero and a dashboard that looks calm.

So a catch-all handler receives a `HookEvent`: one class for the hooks worth
matching by name, and the base class for everything else, so

    match event:
        case CompletionEvent(ctx=ctx): ...
        case WarningEvent(ctx=ctx): ...
        case _: ...

reads structurally and a mistyped field is an error rather than a silent None.

`type` is the SNAKE name -- `on_cost_entry`, not `onCostEntry` -- because that
is the name a Python caller subscribes with (`engine.on_cost_entry`), and an
event whose `type` cannot be pasted back into a subscription is a second
vocabulary for one thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CAMEL_PART = re.compile(r"(?<!^)(?=[A-Z])")


def to_snake(name: str) -> str:
    """`onCostEntry` -> `on_cost_entry`."""
    return _CAMEL_PART.sub("_", name).lower()


@dataclass(frozen=True)
class HookEvent:
    """One emitted event: its name, and the context that came with it.

    The base class every hook produces. A hook with no named subclass still
    arrives as one of these, still carries its `type`, and still matches
    `case _` -- so a logger never has to enumerate the catalog.
    """

    type: str
    ctx: Any


# Each named class defaults its own `type`, so one can be BUILT from a context
# alone -- `RetryEvent(ctx=...)`. An event is a value: it can be stored, queued,
# replayed or sent over a wire because the name and the payload are one object,
# and making the caller restate the name it already chose by picking the class
# would be the one part of that they could get wrong.


@dataclass(frozen=True)
class CompletionEvent(HookEvent):
    """`on_completion` -- one LLM call finished."""

    type: str = "on_completion"
    ctx: Any = None


@dataclass(frozen=True)
class WarningEvent(HookEvent):
    """`on_warning` -- something was adjusted, dropped, or is not what it seems."""

    type: str = "on_warning"
    ctx: Any = None


@dataclass(frozen=True)
class ToolCallStartEvent(HookEvent):
    """`on_tool_call_start` -- an agent is about to run a tool.

    NOT `combycode_llm_sdk.events.ToolCallStartEvent`, which is the STREAM event
    of the same name: that one carries an id and a name off the wire, this one
    carries a hook context. They are different events about different things,
    and only this one is what `on_any` delivers.
    """

    type: str = "on_tool_call_start"
    ctx: Any = None


@dataclass(frozen=True)
class RetryEvent(HookEvent):
    """`on_retry` -- a request is being attempted again."""

    type: str = "on_retry"
    ctx: Any = None


@dataclass(frozen=True)
class RetryContext:
    """The payload of an `on_retry`, as a value you can build.

    A live emit delivers these same fields as a `HookContext` view over the
    bus's dict; this is for a caller CONSTRUCTING one -- a replay, a test, a
    queued event. Both answer `.attempt` and `.reason`, which is what a reader
    of either needs.
    """

    provider: str
    model: str
    attempt: int
    reason: str
    delay_ms: float = 0.0


#: Which hooks get a named class. Deliberately a short list: a class per hook
#: for all ninety would be ninety names in the public surface to make four of
#: them matchable, and the base class already carries the rest faithfully.
EVENT_CLASSES: dict[str, type[HookEvent]] = {
    "onCompletion": CompletionEvent,
    "onWarning": WarningEvent,
    "onToolCallStart": ToolCallStartEvent,
    "onRetry": RetryEvent,
}


def build_event(name: str, ctx: Any) -> HookEvent:
    """Pair a hook name with its context as one matchable value."""
    return EVENT_CLASSES.get(name, HookEvent)(type=to_snake(name), ctx=ctx)


__all__ = [
    "EVENT_CLASSES",
    "CompletionEvent",
    "HookEvent",
    "RetryContext",
    "RetryEvent",
    "ToolCallStartEvent",
    "WarningEvent",
    "build_event",
    "to_snake",
]
