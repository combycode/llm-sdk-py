"""Named settings bundles, with inheritance.

A queue's rate limit, a cache's TTL, a retry policy: each is configuration that
belongs to a NAME rather than to a call, and several names usually differ from
one base by two fields. `extend("prod", "prod-batch", {...})` stores only what
differs and `get` merges the chain, so changing the base changes everything
derived from it.

The registry is opaque about content: it stores whatever a consumer puts in and
that consumer reads its own slice back. Making it typed would mean this module
knowing about every plugin that might ever use it.

**Snapshot semantics.** A consumer captures its slice when it is built -- a queue
created before a `set()` keeps the settings it started with. Live re-reading is
deliberately not offered: a rate limiter whose limit changes mid-flight has no
defined behaviour, and pretending otherwise would be worse than saying no.

Transposed from `unified-library-ts/src/plugins/configuration/configuration.ts`.
The TypeScript's `load`/`save` are async because its `Persistence` is; this
one's is synchronous, so these are too.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

#: Where the registry lives in a `Persistence` when one is attached.
DEFAULT_STORAGE_KEY = "__configurations"

#: The only snapshot format there has been. Checked on load rather than assumed,
#: so a file written by something newer fails with its version rather than with
#: a KeyError three frames deeper.
SNAPSHOT_VERSION = 1

Entry = Mapping[str, Any]


class ConfigurationPlugin:
    """Settings by name, with a parent chain."""

    def __init__(
        self,
        persistence: Any = None,
        storage_key: str = DEFAULT_STORAGE_KEY,
        initial: Mapping[str, Entry] | None = None,
    ) -> None:
        self._entries: dict[str, Entry] = {}
        self._parents: dict[str, str] = {}
        self._persistence = persistence
        self._storage_key = storage_key
        for name, settings in (initial or {}).items():
            self.set(name, settings)

    # -- reading and writing -------------------------------------------------

    def set(self, name: str, settings: Entry) -> None:
        """Register or replace one name's settings.

        Deep-copied on the way in and handed back read-only, so a caller can
        keep mutating the dict they passed and a consumer cannot edit the
        registry by editing what it read. Configuration that changes underneath
        the thing configured by it is the bug this prevents.
        """
        if not name:
            raise ValueError("ConfigurationPlugin.set: name must be non-empty")
        self._entries[name] = _frozen(settings)

    def get(self, name: str) -> Mapping[str, Any] | None:
        """The resolved settings for a name, or None if it is unknown.

        Walks the parent chain and merges child over parent, one level deep --
        a nested dict is replaced wholesale rather than merged, because a
        half-merged nested policy is harder to reason about than a replaced one.
        """
        if name not in self._entries and name not in self._parents:
            return None

        chain: list[Entry] = []
        current: str | None = name
        seen: set[str] = set()
        # `seen` guards a cycle: `extend` cannot make one on its own, but
        # `deserialize` accepts whatever is in the file.
        while current is not None and current not in seen:
            seen.add(current)
            own = self._entries.get(current)
            if own is not None:
                chain.insert(0, own)
            current = self._parents.get(current)

        if not chain:
            return None
        merged: dict[str, Any] = {}
        for layer in chain:
            merged.update(layer)
        return MappingProxyType(merged)

    def extend(self, base_name: str, new_name: str, overrides: Entry) -> None:
        """Derive a name from another, storing only what differs."""
        if base_name not in self._entries and base_name not in self._parents:
            raise KeyError(
                f"ConfigurationPlugin.extend: unknown base {base_name!r}. "
                f"Registered: {', '.join(sorted(self.names())) or 'none'}"
            )
        self._parents[new_name] = base_name
        self._entries[new_name] = _frozen(overrides)

    def has(self, name: str) -> bool:
        return name in self._entries or name in self._parents

    def delete(self, name: str) -> None:
        """Forget a name.

        Anything extending it survives as a name but `get` on it returns
        whatever is left of its chain -- the deletion is not cascaded, because
        deleting a base should not silently delete work derived from it.
        """
        self._entries.pop(name, None)
        self._parents.pop(name, None)

    def names(self) -> list[str]:
        return sorted({*self._entries, *self._parents})

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)

    def __len__(self) -> int:
        return len(self.names())

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    # -- persistence ---------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        return {
            "version": SNAPSHOT_VERSION,
            "entries": {name: dict(entry) for name, entry in self._entries.items()},
            "parents": dict(self._parents),
        }

    def deserialize(self, data: Mapping[str, Any]) -> None:
        version = data.get("version")
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"ConfigurationPlugin.deserialize: unsupported snapshot version "
                f"{version!r} (this build reads {SNAPSHOT_VERSION})"
            )
        self._entries.clear()
        self._parents.clear()
        for name, settings in (data.get("entries") or {}).items():
            self._entries[str(name)] = _frozen(settings)
        for child, parent in (data.get("parents") or {}).items():
            self._parents[str(child)] = str(parent)

    def load(self) -> bool:
        """Replace the registry from storage. False when there was nothing."""
        if self._persistence is None:
            return False
        data = self._persistence.get(self._storage_key)
        if not data:
            return False
        self.deserialize(data)
        return True

    def save(self) -> None:
        """Write the registry to storage.

        Never automatic: a registry that saved itself on every `set()` would
        write once per line of a startup sequence, and the caller is the only
        one who knows when it is done.
        """
        if self._persistence is None:
            raise RuntimeError(
                "ConfigurationPlugin.save: no persistence attached. "
                "Pass one to the constructor."
            )
        self._persistence.set(self._storage_key, self.serialize())

    def __repr__(self) -> str:
        return f"<ConfigurationPlugin {len(self)} name(s)>"


def _frozen(settings: Entry) -> Entry:
    """A read-only deep copy.

    Deep, because a shallow one shares the nested dicts and the caller can still
    reach in. Read-only at the top, which is where a consumer would reach.
    """
    if not isinstance(settings, Mapping):
        raise TypeError(
            f"ConfigurationPlugin: settings must be a mapping, got {type(settings).__name__}"
        )
    return MappingProxyType(copy.deepcopy(dict(settings)))


__all__ = [
    "DEFAULT_STORAGE_KEY",
    "SNAPSHOT_VERSION",
    "ConfigurationPlugin",
    "Entry",
]
