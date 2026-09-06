"""What a hook handler receives.

The bus carries wire-shaped keys -- `toolName`, `queueLength`, `agentId` --
because they are the names the specs, the corpus and the TypeScript all use, and
renaming them at the source would put a translation layer between the library
and its own recorded fixtures. But a Python subscriber writes
`ctx.tool_name`, and the reviewed examples do exactly that.

So the translation happens once, here, at the point of delivery: the same
camelCase-on-the-wire / snake_case-in-the-API boundary the rest of this port
draws, applied to hooks.

Both spellings work, and each has a job. Attribute access is what a subscriber
writes. Mapping access is what the library's own handlers use, and it stays
because the field set is OPEN -- a provider or a plugin adding a key must not
break a reader, which a Mapping expresses and a fixed set of attributes does
not.

Mutation passes straight through to the underlying dict. Several hooks are
in/out parameters -- `onMessageResolve` lets a plugin rewrite the messages,
`onBeforeSubmit` lets a cache short-circuit the call -- so a read-only view here
would quietly break them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, MutableMapping
from typing import Any

_SNAKE_PART = re.compile(r"_([a-z0-9])")


def to_camel(name: str) -> str:
    """`tool_name` -> `toolName`. A name with no underscore is returned as-is."""
    return _SNAKE_PART.sub(lambda m: m.group(1).upper(), name)


class HookContext(MutableMapping[str, Any]):
    """One hook payload, readable as attributes or as a mapping.

    A VIEW, not a copy: it wraps the dict the emitter built, so a handler that
    writes through it is seen by the next handler and by the emitter -- which is
    what the in/out hooks rely on.
    """

    __slots__ = ("_data",)

    def __init__(self, data: MutableMapping[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    # -- mapping -------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attributes ----------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # `_data` itself, and any dunder a copy/pickle probe asks for, must fail
        # normally rather than be looked up as a payload key.
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        camel = to_camel(name)
        if camel in data:
            return data[camel]
        raise AttributeError(
            f"this hook context has no {name!r}. It carries: {', '.join(sorted(data))}."
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        data = object.__getattribute__(self, "_data")
        # Write to the key that already exists, under either spelling, so a
        # handler updating a field cannot silently create a second one beside it.
        camel = to_camel(name)
        data[name if name in data else camel] = value

    def __repr__(self) -> str:
        return f"HookContext({self._data!r})"

    def raw(self) -> MutableMapping[str, Any]:
        """The underlying dict, for a caller that wants the wire spelling."""
        data: MutableMapping[str, Any] = object.__getattribute__(self, "_data")
        return data


def as_context(ctx: Any) -> Any:
    """Wrap a payload for delivery, leaving anything else alone.

    Already-wrapped contexts pass through rather than nesting: hooks are
    re-emitted (an engine forwards an agent's, a plugin re-raises one), and a
    view of a view would resolve names one layer deeper on every hop.
    """
    if isinstance(ctx, HookContext) or not isinstance(ctx, MutableMapping):
        return ctx
    return HookContext(ctx)


__all__ = ["HookContext", "as_context", "to_camel"]
