"""The catalog of hook names, and the event shape they travel in.

PARTIAL transposition of `unified-library-ts/src/bus/hook-map.ts`. That file is
742 lines of which every one is a type declaration: 51 context interfaces and the
`HookMap` that pairs each with its name. TypeScript needs them because `HookMap`
is what makes `on('onCompletion', h)` know its own `ctx` without a cast --
`hook-bus.ts` records that the alternative cost the telemetry adapter 26 casts in
one method, and a renamed field then read `undefined` while still compiling.

Python has no equivalent inference to buy, so what is ported is what carries
meaning at runtime: the NAMES. The contexts stay camelCase dicts, per
`llm/types/messages.py`.

`HOOK_NAMES` is the catalog and it is exhaustive -- a hook emitted under a name
not in this tuple is a typo that reaches no subscriber, which is precisely the
silence the typed map exists to prevent, so `HookBus` refuses it.

An event, as a value, is `{'type': <name>, 'ctx': <context>}`. The payload stays
nested under `ctx` rather than spread onto the event: spreading would collide
with the contexts that already carry their own `type` field, and would copy an
object on every emit -- including the per-chunk ones.
"""

from __future__ import annotations

from typing import Any

from .events import HookEvent

#: Every hook the SDK emits, grouped by the layer that owns it. The order is the
#: TypeScript's, so the two catalogs can be diffed line for line.
HOOK_NAMES: tuple[str, ...] = (
    # Cross-cutting
    "onWarning",
    "onInternalError",
    # Network layer
    "onEnqueue",
    "onDequeue",
    "onQueueTimeout",
    "onRateLimitUpdate",
    "onRequestStart",
    "onRequestComplete",
    "onModelError",
    "onRateLimitHit",
    "onRetry",
    "onStreamChunk",
    "onRealtimeOpen",
    "onRealtimeFrame",
    "onRealtimeClose",
    "onRealtimeError",
    # LLM layer
    "onClientCreate",
    "onClientDestroy",
    "onMessageResolve",
    "onBeforeSubmit",
    "onCompletion",
    # Agent layer
    "onAgentCreate",
    "onAgentDestroy",
    "onRunStart",
    "onStepStart",
    "onStepComplete",
    "onToolCallStart",
    "onToolCallComplete",
    "onToolCallError",
    "onToolSearch",
    "onRunComplete",
    "onRunError",
    "onGuardrailTriggered",
    "onApprovalRequested",
    "onApprovalResolved",
    # Server layer
    "onServerRequest",
    "onServerResponse",
    "onAuthFail",
    # Cost layer (plugin)
    "onCostEntry",
    "onBudgetWarning",
    "onBudgetExceeded",
    # Context layer (plugin)
    "onContextMeasure",
    # Media layer (plugin)
    "onMediaGenerated",
    "onMediaError",
    "onMediaProgress",
    # Internal tools (plugin)
    "onInternalToolCallStart",
    "onInternalToolCallComplete",
    "onInternalToolCallError",
    # MCP (plugin)
    "onMcpConnect",
    "onMcpToolCall",
    "onMcpError",
)

#: `type HookName = keyof HookMap` (hook-map.ts:726). A `str` rather than a
#: `Literal` of all 51: the membership check is `HOOK_NAMES`, at runtime, where a
#: plugin registering a hook from a name it computed is also checked.
HookName = str

#: One hook's context payload -- the 51 shapes `HookMap` names.
HookContext = dict[str, Any]

__all__ = ["HOOK_NAMES", "HookContext", "HookEvent", "HookName"]
