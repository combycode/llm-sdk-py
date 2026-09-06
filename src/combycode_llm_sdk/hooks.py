"""The hook bus, under the name that reads like what it is.

`from combycode_llm_sdk.hooks import HookBus` is what the reviewed examples
write, and it is the better address: `bus/` is where the machinery lives, and
naming an internal package in an import is how a layout becomes an API.

One re-export, not a second class. A wrapper here would be a second place that
has to agree about hook names, and the whole point of the catalog in
`bus/hook_map.py` is that there is exactly one.
"""

from __future__ import annotations

from .bus.context import HookContext
from .bus.events import HookEvent
from .bus.hook_bus import AnyHookHandler, HookBus, HookHandler
from .bus.hook_map import HOOK_NAMES, HookName

__all__ = [
    "HOOK_NAMES",
    "AnyHookHandler",
    "HookBus",
    "HookContext",
    "HookEvent",
    "HookHandler",
    "HookName",
]
