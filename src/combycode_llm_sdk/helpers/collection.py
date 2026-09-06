"""A named corner of the engine's persistence, typed and prefix-free.

Transposed from `unified-library-ts/src/helpers/collection.ts`.

The store underneath is one flat key space shared by everything that persists:
checkpoints, calibration, batches, an application's own rows. Written against
directly, every caller has to remember to prefix its keys and to strip the
prefix back off when listing them -- and the day one forgets, it reads another
subsystem's rows and reports them as its own.

A collection is that prefix, applied once. `subagents.keys()` answers short
names; nothing above this line ever sees `subagents:`.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Collection(Generic[T]):
    """One namespace within a persistence store."""

    __slots__ = ("_engine", "_prefix", "name")

    def __init__(self, name: str, *, engine: Any = None) -> None:
        # A "/" would silently nest under whatever the file backend makes of a
        # path separator, so two collections could end up sharing a directory.
        if not name or "/" in name:
            raise ValueError(
                f"create_collection: name must be a non-empty string without '/' (got {name!r})"
            )
        self.name = name
        self._prefix = f"{name}:"
        self._engine = engine

    def _store(self) -> Any:
        """Resolved per call, not captured at construction.

        A collection built before the default engine exists -- at import time,
        in a module-level constant -- would otherwise hold the wrong store for
        the rest of the process.
        """
        from .engine import default_engine

        engine = self._engine if self._engine is not None else default_engine()
        if engine is None:
            raise ValueError(
                f"Collection {self.name!r}: no engine. Pass engine=, or create one "
                f"so the rows have somewhere to live."
            )
        return engine.persistence

    def _full(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _short(self, key: str) -> str:
        return key[len(self._prefix) :]

    def set(self, key: str, value: T) -> None:
        self._store().set(self._full(key), value)

    def get(self, key: str) -> T | None:
        value: T | None = self._store().get(self._full(key))
        return value

    def has(self, key: str) -> bool:
        return bool(self._store().has(self._full(key)))

    def delete(self, key: str) -> None:
        self._store().delete(self._full(key))

    def keys(self) -> list[str]:
        """The short names in this collection, without the prefix."""
        return [self._short(k) for k in self._store().list(self._prefix)]

    def values(self) -> list[T]:
        """Every value. A key that vanished between listing and reading is
        skipped rather than returned as `None`, which would not be a `T`."""
        store = self._store()
        found = (store.get(k) for k in store.list(self._prefix))
        return [v for v in found if v is not None]

    def items(self) -> list[tuple[str, T]]:
        """`(short_key, value)` pairs, on the same terms as `values()`."""
        store = self._store()
        pairs = ((self._short(k), store.get(k)) for k in store.list(self._prefix))
        return [(k, v) for k, v in pairs if v is not None]

    def __len__(self) -> int:
        return len(self._store().list(self._prefix))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.has(key)

    def __repr__(self) -> str:
        return f"<Collection {self.name!r}>"


def create_collection(name: str, *, engine: Any = None) -> Collection[Any]:
    """A typed handle on one namespace of `engine.persistence`."""
    return Collection(name, engine=engine)


__all__ = ["Collection", "create_collection"]
