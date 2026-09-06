"""Reusing a list the server said is still good.

A 2026-07-28 server can tell the client how long a list or read result stays
fresh. Without that, `list_tools()` goes to the wire on every call -- the churn
the hint exists to remove.

Deliberately conservative, and each restraint matters:

- **Off unless asked for.** Caching changes when a caller observes a server-side
  change, and that is the caller's decision to make, not this library's.
- **No hint, no caching.** Every pre-2026 server sends no `ttlMs`, so nothing is
  stored and behaviour is byte-identical to having no cache at all.
- **`ttlMs: 0` means stale NOW**, which is an instruction and not a missing
  value. It evicts. Without that eviction the hint is inert: a server that said
  "good for 60s" and then "stale now" would keep being answered from the stale
  entry for the rest of the original minute.
- **`cacheScope` is recorded and never used to widen sharing.** This cache lives
  inside one client holding one credential, so `public` buys nothing here --
  but storing it keeps the entry honest if the cache is ever shared.

Transposed from `unified-library-ts/src/plugins/mcp/result-cache.ts`.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Entry:
    value: Any
    expires_at: float
    scope: str


def _now_ms() -> float:
    return time.time() * 1000


class McpResultCache:
    """List and read results, kept only as long as the server said to."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    @staticmethod
    def key(method: str, params: Any = None) -> str:
        """The method plus its arguments.

        Two `resources/read` calls for different URIs are different entries, so
        the params have to be in the key -- otherwise one document is served for
        another, which is the worst kind of cache hit.
        """
        if params is None:
            return method
        return f"{method}:{json.dumps(params, sort_keys=True, default=str)}"

    def get(self, key: str, now: float | None = None) -> Any:
        """The stored value, or None once it has expired."""
        moment = _now_ms() if now is None else now
        hit = self._entries.get(key)
        if hit is None:
            return None
        if hit.expires_at <= moment:
            del self._entries[key]
            return None
        return hit.value

    def set(
        self, key: str, value: Any, hints: Mapping[str, Any] | None, now: float | None = None
    ) -> bool:
        """Store only when the server actually asked for it. Whether it stored."""
        moment = _now_ms() if now is None else now
        ttl = (hints or {}).get("ttlMs")
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl):
            # No instruction: leave whatever is held alone, so a pre-2026 server
            # behaves exactly as it did before this cache existed.
            return False
        if ttl <= 0:
            self._entries.pop(key, None)
            return False
        scope = "public" if (hints or {}).get("cacheScope") == "public" else "private"
        self._entries[key] = _Entry(value=value, expires_at=moment + ttl, scope=scope)
        return True

    def clear(self) -> None:
        """Drop everything -- what a `*_changed` notification means."""
        self._entries.clear()

    def clear_method(self, method: str) -> None:
        """Drop every entry for one method, leaving the others alone."""
        prefix = f"{method}:"
        for key in [k for k in self._entries if k == method or k.startswith(prefix)]:
            del self._entries[key]

    @property
    def size(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<McpResultCache {len(self._entries)} entr(ies)>"


__all__ = ["McpResultCache"]
