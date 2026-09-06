"""One key-value interface; where the bytes land is a swap, not a rewrite.

Transposed from `unified-library-ts/src/plugins/persistence/`.

Checkpointing a run, a cache that outlives the process, a schedule of deferred
tasks: all three want `get`/`set`/`delete`/`list`/`has` and differ only in where
the bytes end up.

So `Persistence` is a runtime-checkable Protocol -- structural, not a base class
-- and an application's own Redis or SQL store satisfies it WITHOUT importing
anything from this library or inheriting from anything in it.

What that buys: a resume path exercised against `MemoryPersistence` in a test is
the same code that resumes from `FilePersistence` in production. Without it the
only backend the resume was ever proved against is the one it was written for,
and the first restart in production is the first real run.

Which is also why `MemoryPersistence` copies values in and out. A caller
mutating what it read would otherwise silently edit the checkpoint -- something
the file backend cannot do -- and two backends that disagree about that are not
interchangeable.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Everything outside this set is %-escaped in a filename. A colon is how keys
#: are namespaced here (`task:...`, `cache:default:...`) and is not legal in a
#: filename on Windows, so it cannot survive unescaped.
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")
_ESCAPED = re.compile(r"%([0-9a-fA-F]{2})")


@runtime_checkable
class Persistence(Protocol):
    """Where bytes live, as five methods.

    A Protocol rather than a base class: an application's own store should not
    have to import this library to be usable by it. `isinstance(store,
    Persistence)` answers structurally.
    """

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...

    def delete(self, key: str) -> None: ...

    def list(self, prefix: str | None = None) -> list[str]: ...

    def has(self, key: str) -> bool: ...


class MemoryPersistence:
    """In-process storage. For tests, and for state that need not outlive it."""

    def __init__(self) -> None:
        self._rows: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """A COPY, so a caller cannot edit the checkpoint by reading it.

        The file backend physically cannot hand back a live reference, and two
        backends that disagree about this are not interchangeable -- which is
        the whole point of there being an interface.
        """
        value = self._rows.get(key)
        return copy.deepcopy(value) if value is not None else None

    def set(self, key: str, value: Any) -> None:
        self._rows[key] = copy.deepcopy(value)

    def delete(self, key: str) -> None:
        self._rows.pop(key, None)

    def list(self, prefix: str | None = None) -> list[str]:
        return [k for k in self._rows if prefix is None or k.startswith(prefix)]

    def has(self, key: str) -> bool:
        return key in self._rows

    def clear(self) -> None:
        self._rows.clear()

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"<MemoryPersistence {len(self._rows)} key(s)>"


class FilePersistence:
    """One JSON file per key, under a directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{encode_key(key)}.json"

    def get(self, key: str) -> Any:
        path = self._path(key)
        if not path.is_file():
            # A key that was never written reads as None. Raising would make
            # "nothing here yet" -- the ordinary state of a fresh store -- an
            # error every caller has to catch before it can start.
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        # Written beside and moved into place, so a reader never sees half a
        # file and a crash mid-write leaves the previous checkpoint intact.
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
        temporary.replace(path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def list(self, prefix: str | None = None) -> list[str]:
        keys = [decode_key(p.stem) for p in self.directory.glob("*.json")]
        return [k for k in keys if prefix is None or k.startswith(prefix)]

    def has(self, key: str) -> bool:
        return self._path(key).is_file()

    def __repr__(self) -> str:
        return f"<FilePersistence {self.directory}>"


def encode_key(key: str) -> str:
    """A key as a filename. Everything unsafe becomes `%XX`."""
    return _SAFE.sub(lambda m: f"%{ord(m.group(0)):02x}", key)


def decode_key(name: str) -> str:
    """The inverse, so `list()` answers keys rather than filenames."""
    return _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), name)


__all__ = [
    "FilePersistence",
    "MemoryPersistence",
    "Persistence",
    "decode_key",
    "encode_key",
]
