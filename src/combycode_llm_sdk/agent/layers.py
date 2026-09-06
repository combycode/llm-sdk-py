"""The well-known context layers, and the order they render in.

Transposed from `unified-library-ts/src/agent/context-registry/layers.ts`.

Several writers contribute to one conversation's system prompt -- the agent's
persona, the run's scenario, memory, the facts the guard preserved across a
compaction. They do not know about each other, so the only thing deciding what
the model actually reads is the priority each one writes at. Naming those
numbers here is what stops two subsystems from silently swapping places.

**Lower renders earlier.** The order is not cosmetic: a provider's prompt cache
matches on a PREFIX, so the layers that never change are deliberately first and
the ones rewritten every turn are last. Putting a per-turn fact layer in front
of a stable persona invalidates the cached prefix on every single request, and
nothing about the output looks wrong when it happens -- only the bill.
"""

from __future__ import annotations

from typing import Any

# -- Layer names -------------------------------------------------------------

#: Role / persona / behaviour. Stable per conversation, so it renders first and
#: the cache prefix starts here.
LAYER_AGENTLOOP_SYSTEM = "agentloop.system"

#: The scenario the loop was constructed with. Stable per run.
LAYER_AGENTLOOP_CONTEXT = "agentloop.context"

#: What `history.system = ...` writes. Kept for callers that predate the
#: registry.
LAYER_LEGACY_SYSTEM = "_legacy_system"

#: Free-form notes. Long-lived, low churn.
LAYER_MEMORY = "memory"

#: Facts carried across a compaction. Rewritten turn to turn, so it lands late.
LAYER_CHAT_FACTS = "chat.facts"

#: Worker-side: examples distilled from earlier tool calls.
LAYER_EXECUTOR_TOOL_EXAMPLES = "executor.tool-examples"

#: The summary a compaction leaves behind in place of the range it replaced.
LAYER_CONTEXT_GUARD_SUMMARY = "context-guard.summary"

#: How to reach tools that are registered but not declared. Present only while
#: at least one lazy tool exists.
LAYER_LAZY_TOOLS = "agentloop.lazy-tools"

# -- Priorities (lower renders earlier) --------------------------------------

PRIORITY_AGENTLOOP_SYSTEM = 10
#: Immediately after the persona: a model told its tools are incomplete needs to
#: hear that before anything else it is told about them.
PRIORITY_LAZY_TOOLS = 20
#: Declared by the TypeScript and exported from its index, but NOT what the
#: legacy setter writes -- both writers of `_legacy_system` there hardcode 200,
#: which is `PRIORITY_MEMORY`. The port keeps the behaviour rather than the
#: constant, because changing where that layer renders would move the cache
#: prefix for every existing caller. Read `LEGACY_SYSTEM_WRITE_PRIORITY` for
#: what actually happens.
PRIORITY_LEGACY_SYSTEM = 50
PRIORITY_AGENTLOOP_CONTEXT = 100
PRIORITY_MEMORY = 200
PRIORITY_CHAT_FACTS = 250
PRIORITY_EXECUTOR_TOOL_EXAMPLES = 280
PRIORITY_CONTEXT_GUARD_SUMMARY = 300

#: What `history.system` and `history.append_system` actually write at.
LEGACY_SYSTEM_WRITE_PRIORITY = 200

#: The protocol text for lazily-exposed tools. A few lines on purpose, not a
#: catalog: a catalog is paid for on every turn of the whole conversation, which
#: is most of what makes eager exposure expensive to begin with.
LAZY_TOOLS_PROTOCOL = (
    "Not all of your tools are listed. Use `tool_search` to find the ones you "
    "need -- it returns their exact names and full argument schemas -- then run "
    "them with `call_tool`. Search for every capability the request needs in ONE "
    "call, passing several queries. If the result reports a query as unmatched, "
    "search again with different words before answering; never answer as if a "
    "capability you could not find does not matter."
)


def write_agentloop_system(registry: Any, text: str | None, owner: str) -> None:
    """Set the persona layer, or remove it when there is no persona."""
    if not text:
        registry.remove(LAYER_AGENTLOOP_SYSTEM)
        return
    registry.set(
        LAYER_AGENTLOOP_SYSTEM,
        text,
        priority=PRIORITY_AGENTLOOP_SYSTEM,
        tags=["system"],
        owner=owner,
    )


def write_agentloop_context(registry: Any, text: str | None, owner: str) -> None:
    """Set the run-scenario layer, or remove it when there is no scenario."""
    if not text:
        registry.remove(LAYER_AGENTLOOP_CONTEXT)
        return
    registry.set(
        LAYER_AGENTLOOP_CONTEXT,
        text,
        priority=PRIORITY_AGENTLOOP_CONTEXT,
        tags=["system"],
        owner=owner,
    )


def write_lazy_tools_protocol(registry: Any, active: bool, owner: str) -> None:
    """Tell the model its tool list is partial, and how to reach the rest.

    Stated rather than implied because a model has no reason to suspect a tool
    exists that it cannot see. The TypeScript records the measurement that put
    this here: over 24 live runs against 308 lazy tools the model scored 8/12
    and 9/12 without the protocol -- sometimes never searching, more often
    searching once for a request needing two capabilities and answering from the
    one tool it found. With the protocol, the same tasks and ranker scored 18/18.
    """
    if not active:
        registry.remove(LAYER_LAZY_TOOLS)
        return
    registry.set(
        LAYER_LAZY_TOOLS,
        LAZY_TOOLS_PROTOCOL,
        priority=PRIORITY_LAZY_TOOLS,
        tags=["system"],
        owner=owner,
    )


__all__ = [
    "LAYER_AGENTLOOP_CONTEXT",
    "LAYER_AGENTLOOP_SYSTEM",
    "LAYER_CHAT_FACTS",
    "LAYER_CONTEXT_GUARD_SUMMARY",
    "LAYER_EXECUTOR_TOOL_EXAMPLES",
    "LAYER_LAZY_TOOLS",
    "LAYER_LEGACY_SYSTEM",
    "LAYER_MEMORY",
    "LAZY_TOOLS_PROTOCOL",
    "LEGACY_SYSTEM_WRITE_PRIORITY",
    "PRIORITY_AGENTLOOP_CONTEXT",
    "PRIORITY_AGENTLOOP_SYSTEM",
    "PRIORITY_CHAT_FACTS",
    "PRIORITY_CONTEXT_GUARD_SUMMARY",
    "PRIORITY_EXECUTOR_TOOL_EXAMPLES",
    "PRIORITY_LAZY_TOOLS",
    "PRIORITY_LEGACY_SYSTEM",
    "PRIORITY_MEMORY",
    "write_agentloop_context",
    "write_agentloop_system",
    "write_lazy_tools_protocol",
]
