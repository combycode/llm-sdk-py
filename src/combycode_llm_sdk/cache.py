"""A cached answer belongs to a request AND to where that request was sent.

Transposed from `unified-library-ts/src/plugins/cache/`.

`Cache` is content-agnostic: it stores a body under `(cache_name, cache_key)`
with a TTL. `request_cache_key` is the half that turns a request into that key,
and it takes `provider` as a REQUIRED argument with no default.

That is the part worth reading twice. A normalized request holds wire fields
only -- by the time it exists the provider prefix has been stripped off the
model id -- so `openai/gpt-5` and `openrouter/openai/gpt-5` arrive
indistinguishable. They are different services, with different moderation and
different answers. Without the route in the key they hash the same, and one
caller is handed the other service's completion: no error, no warning, just a
wrong answer that looks exactly like a right one. `base_url` is the same
argument one level down, for the same model behind an Azure deployment.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .wire.interpreter import js_json

#: Every storage key starts with this, so `invalidate()` can tell the cache's
#: own rows from anything else sharing the store.
STORAGE_PREFIX = "cache:"

DEFAULT_CACHE_NAME = "default"
DEFAULT_TTL_MS = 5 * 60 * 1000


@dataclass
class CacheEntry:
    """One stored body, and what makes it stale."""

    body: Any
    stored_at: float = 0.0
    ttl_ms: float = DEFAULT_TTL_MS
    cache_name: str = DEFAULT_CACHE_NAME

    def expired(self, now: float | None = None) -> bool:
        """Whether this is past its TTL.

        An infinite TTL never expires, which is how a caller says "until I
        invalidate it" without picking a number that is really a guess.
        """
        if self.ttl_ms == float("inf"):
            return False
        return (now if now is not None else time.time() * 1000) - self.stored_at > self.ttl_ms


@runtime_checkable
class CacheStore(Protocol):
    """Where entries live. Structural, so an application's own store qualifies."""

    def get(self, storage_key: str) -> CacheEntry | None: ...

    def set(self, storage_key: str, entry: CacheEntry) -> None: ...

    def delete(self, storage_key: str) -> None: ...

    def keys(self, prefix: str | None = None) -> list[str]: ...

    def clear(self) -> None: ...


class MemoryCacheStore:
    """In-process entries. Nothing outlives the process."""

    def __init__(self) -> None:
        self._rows: dict[str, CacheEntry] = {}

    def get(self, storage_key: str) -> CacheEntry | None:
        return self._rows.get(storage_key)

    def set(self, storage_key: str, entry: CacheEntry) -> None:
        self._rows[storage_key] = entry

    def delete(self, storage_key: str) -> None:
        self._rows.pop(storage_key, None)

    def keys(self, prefix: str | None = None) -> list[str]:
        return [k for k in self._rows if prefix is None or k.startswith(prefix)]

    def clear(self) -> None:
        self._rows.clear()


class PersistentCacheStore:
    """Entries in any `Persistence`, so they outlive the process.

    A thin adapter rather than a second store: the two would otherwise expire,
    namespace and invalidate in two places, and only one of them would be the
    one anybody tested.
    """

    def __init__(self, store: Any) -> None:
        self.store = store

    def get(self, storage_key: str) -> CacheEntry | None:
        raw = self.store.get(storage_key)
        if not isinstance(raw, Mapping):
            return None
        return CacheEntry(
            body=raw.get("body"),
            stored_at=float(raw.get("storedAt") or 0.0),
            ttl_ms=float(raw.get("ttlMs") or DEFAULT_TTL_MS),
            cache_name=str(raw.get("cacheName") or DEFAULT_CACHE_NAME),
        )

    def set(self, storage_key: str, entry: CacheEntry) -> None:
        self.store.set(
            storage_key,
            {
                "body": entry.body,
                "storedAt": entry.stored_at,
                "ttlMs": entry.ttl_ms,
                "cacheName": entry.cache_name,
            },
        )

    def delete(self, storage_key: str) -> None:
        self.store.delete(storage_key)

    def keys(self, prefix: str | None = None) -> list[str]:
        return [k for k in self.store.list(prefix) if k.startswith(prefix or "")]

    def clear(self) -> None:
        for key in list(self.store.list()):
            self.store.delete(key)


def make_storage_key(cache_name: str, cache_key: str) -> str:
    return f"{STORAGE_PREFIX}{cache_name}:{cache_key}"


def parse_storage_key(storage_key: str) -> tuple[str, str] | None:
    if not storage_key.startswith(STORAGE_PREFIX):
        return None
    rest = storage_key[len(STORAGE_PREFIX) :]
    separator = rest.find(":")
    if separator < 0:
        return None
    return rest[:separator], rest[separator + 1 :]


def request_cache_key(
    request: Mapping[str, Any], *, provider: str, base_url: str | None = None
) -> str:
    """A stable key for one request to one service.

    `provider` has no default on purpose. A normalized request carries wire
    fields only, so two services offering the same model produce the same bytes
    -- and a key built from the bytes alone hands one caller the other service's
    answer, with nothing anywhere saying it is wrong.

    Order-independent: the same request written with its keys in another order
    is the same request, and a key that disagreed would just miss the cache
    silently.
    """
    material = js_json(
        {
            "provider": provider,
            "baseURL": base_url or "",
            "request": _canonical(request),
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """The same value with every mapping's keys sorted, recursively."""
    if isinstance(value, Mapping):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        # Lists stay ordered: the order of MESSAGES is the conversation, and
        # sorting them would make two different conversations one key.
        return [_canonical(v) for v in value]
    return value


@dataclass
class Cache:
    """Bodies under `(cache_name, cache_key)`, expired lazily on read."""

    store: Any = field(default_factory=MemoryCacheStore)
    #: The namespace used when a caller passes None.
    default_name: str = DEFAULT_CACHE_NAME
    ttl_ms: float = DEFAULT_TTL_MS

    def get(self, cache_name: str | None, cache_key: str) -> Any:
        """The stored body, or None.

        Expiry is checked HERE and the entry dropped here, so there is no
        sweeper thread and nothing has to sleep to prove expiry works.
        """
        name = cache_name or self.default_name
        storage_key = make_storage_key(name, cache_key)
        entry = self.store.get(storage_key)
        if entry is None:
            return None
        if entry.expired():
            self.store.delete(storage_key)
            return None
        return entry.body

    def set(
        self,
        cache_name: str | None,
        cache_key: str,
        body: Any,
        *,
        ttl_ms: float | None = None,
    ) -> CacheEntry:
        name = cache_name or self.default_name
        entry = CacheEntry(
            body=body,
            stored_at=time.time() * 1000,
            ttl_ms=ttl_ms if ttl_ms is not None else self.ttl_ms,
            cache_name=name,
        )
        self.store.set(make_storage_key(name, cache_key), entry)
        return entry

    def delete(self, cache_name: str | None, cache_key: str) -> None:
        self.store.delete(make_storage_key(cache_name or self.default_name, cache_key))

    def invalidate(self, *, cache_name: str | None = None) -> int:
        """Drop a namespace's entries, and answer how many there were.

        Scoped to the namespace: one tenant clearing its cache must not empty
        another's, and a count of 0 is how a caller learns its scope was empty
        rather than that the call did nothing.
        """
        prefix = (
            make_storage_key(cache_name, "") if cache_name is not None else STORAGE_PREFIX
        )
        keys = [k for k in self.store.keys(prefix) if k.startswith(prefix)]
        for key in keys:
            self.store.delete(key)
        return len(keys)

    def clear(self) -> None:
        self.store.clear()


__all__ = [
    "DEFAULT_CACHE_NAME",
    "DEFAULT_TTL_MS",
    "STORAGE_PREFIX",
    "Cache",
    "CacheEntry",
    "CacheStore",
    "MemoryCacheStore",
    "PersistentCacheStore",
    "make_storage_key",
    "parse_storage_key",
    "request_cache_key",
]
