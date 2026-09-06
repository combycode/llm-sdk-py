"""Conversations the server can be asked to continue.

A stateless HTTP API and a stateful conversation are not the same shape, and
this is the seam. A client sends a response id back; the server looks up the
history that id names and carries on from it, instead of asking the caller to
resend a transcript that grows every turn.

Keyed by `(user_id, response_id)`, never by id alone when a user is known:
response ids are guessable enough that one tenant reading another's
conversation must be impossible by construction rather than by convention.
`AuthPlugin` sets the user; without one every entry is unowned and the key is
the id.

Two layers, deliberately. The in-memory cache is an LRU so a busy server does
not grow without bound, and the `Persistence` behind it is what survives a
restart -- a store with only the cache would lose every conversation on deploy,
which is exactly when a user is most likely to be mid-conversation.

Transposed from `unified-library-ts/src/server/response-store.ts`. That one's
`get`/`put`/`delete` are async because its `Persistence` is; this one's is
synchronous, so these are too.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

#: Where entries live in a `Persistence`, so they cannot collide with anything
#: else the same store holds.
DEFAULT_KEY_PREFIX = "response:"

#: How many entries the in-memory half keeps. The cache is a convenience over
#: the persistence, not the record: evicting one costs a read, not a
#: conversation.
DEFAULT_MEMORY_CAPACITY = 10_000


def now_ms() -> float:
    """Epoch milliseconds -- the unit the TypeScript snapshot stores."""
    return time.time() * 1000


@dataclass(frozen=True)
class ResponseTarget:
    """Which registered model a conversation belongs to."""

    #: `direct` is a registered LLMClient, which the server wraps in an agent
    #: loop. The field exists so a second kind can arrive without changing the
    #: shape of everything that reads one.
    kind: str = "direct"
    #: What the client ASKED for, before resolution -- the name it will send
    #: again next turn.
    model: str = ""
    #: The routing tag: `provider/model` for a direct entry.
    id: str = ""

    def as_row(self) -> dict[str, Any]:
        return {"kind": self.kind, "model": self.model, "id": self.id}

    @staticmethod
    def of(raw: Mapping[str, Any] | None) -> ResponseTarget:
        raw = raw or {}
        return ResponseTarget(
            kind=str(raw.get("kind") or "direct"),
            model=str(raw.get("model") or ""),
            id=str(raw.get("id") or ""),
        )


@dataclass
class ResponseEntry:
    """One continuable conversation."""

    local_response_id: str
    #: Owner, when an `AuthPlugin` is attached. None when unauthenticated --
    #: and None is a real value here, not a missing one: it is the key under
    #: which every anonymous entry lives.
    user_id: str | None = None
    target: ResponseTarget = field(default_factory=ResponseTarget)
    #: The transcript. Held as whatever the caller put in -- a
    #: `ConversationHistory`, or its snapshot after a reload.
    history: Any = None
    #: The PROVIDER's own response id, for providers that keep server-side
    #: state (OpenAI, xAI). Chaining to it is what makes a follow-up cheap.
    provider_response_id: str | None = None
    #: When that provider-side state stops being usable. None means the
    #: provider never said, which is not the same as "it never expires" --
    #: `has_fresh_provider_state` treats it as unusable for that reason.
    provider_state_expires_at: float | None = None
    created_at: float = field(default_factory=now_ms)
    updated_at: float = field(default_factory=now_ms)

    def as_row(self) -> dict[str, Any]:
        """The camelCase snapshot, matching what the TypeScript writes."""
        # `dump()` here, not the TypeScript's `export()`: this port named
        # the pair `dump`/`restore`, and the SNAPSHOT is identical either way.
        history = self.history
        exported = history.dump() if hasattr(history, "dump") else history
        return {
            "meta": {
                "localResponseId": self.local_response_id,
                "userId": self.user_id,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "target": self.target.as_row(),
                "providerResponseId": self.provider_response_id,
                "providerStateExpiresAt": self.provider_state_expires_at,
            },
            "history": exported,
        }

    @staticmethod
    def of(row: Mapping[str, Any], history: Any = None) -> ResponseEntry:
        meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        meta = meta or {}
        return ResponseEntry(
            local_response_id=str(meta.get("localResponseId") or ""),
            user_id=meta.get("userId"),
            target=ResponseTarget.of(meta.get("target")),
            history=history if history is not None else row.get("history"),
            provider_response_id=meta.get("providerResponseId"),
            provider_state_expires_at=meta.get("providerStateExpiresAt"),
            created_at=float(meta.get("createdAt") or 0.0),
            updated_at=float(meta.get("updatedAt") or 0.0),
        )


def new_response_id() -> str:
    """A fresh id, in the shape a client will send back."""
    return f"resp_{uuid.uuid4().hex[:24]}"


def has_fresh_provider_state(entry: ResponseEntry, now: float | None = None) -> bool:
    """Whether the PROVIDER's own state can still be chained to.

    An unknown expiry counts as stale. Chaining to state that turns out to be
    gone fails the whole call, where re-sending the transcript merely costs
    tokens -- so the cheap wrong answer is the safe one here.
    """
    if not entry.provider_response_id or entry.provider_state_expires_at is None:
        return False
    return entry.provider_state_expires_at > (now_ms() if now is None else now)


class ResponseStore:
    """Conversations, by id, with an LRU in front of a store."""

    def __init__(
        self,
        persistence: Any = None,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        memory_capacity: int = DEFAULT_MEMORY_CAPACITY,
    ) -> None:
        self._persistence = persistence
        self._key_prefix = key_prefix
        self._capacity = max(1, memory_capacity)
        self._cache: OrderedDict[str, ResponseEntry] = OrderedDict()

    # -- reading and writing -------------------------------------------------

    def get(self, response_id: str, user_id: str | None = None) -> ResponseEntry | None:
        """One conversation, from the cache or from the store behind it."""
        key = self._cache_key(response_id, user_id)
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            return entry
        if self._persistence is None:
            return None
        row = self._persistence.get(self._persist_key(response_id, user_id))
        if not isinstance(row, Mapping):
            return None
        restored = ResponseEntry.of(row, self._history_of(row))
        self._remember(restored)
        return restored

    def put(self, entry: ResponseEntry) -> ResponseEntry:
        """Record a conversation, and write it through if there is a store."""
        entry.updated_at = now_ms()
        self._remember(entry)
        if self._persistence is not None:
            self._persistence.set(
                self._persist_key(entry.local_response_id, entry.user_id), entry.as_row()
            )
        return entry

    def delete(self, response_id: str, user_id: str | None = None) -> None:
        self._cache.pop(self._cache_key(response_id, user_id), None)
        if self._persistence is not None:
            self._persistence.delete(self._persist_key(response_id, user_id))

    def list(self, user_id: str | None = None) -> list[str]:
        """Which conversations this user has.

        Read from the STORE when there is one: the cache holds only what is hot,
        and a listing that quietly omitted everything evicted would be worse
        than no listing at all.
        """
        if self._persistence is not None:
            prefix = self._persist_key_prefix(user_id)
            return [str(key)[len(prefix) :] for key in self._persistence.list(prefix)]
        return [
            entry.local_response_id
            for entry in self._cache.values()
            if entry.user_id == user_id
        ]

    # -- keys ----------------------------------------------------------------

    def _cache_key(self, response_id: str, user_id: str | None) -> str:
        return f"{user_id}:{response_id}" if user_id else response_id

    def _persist_key(self, response_id: str, user_id: str | None) -> str:
        return f"{self._persist_key_prefix(user_id)}{response_id}"

    def _persist_key_prefix(self, user_id: str | None) -> str:
        # The user id is percent-encoded because it comes from an auth plugin
        # and can be anything -- an email, a subject claim. A `:` in one would
        # otherwise let a user address another user's namespace.
        return f"{self._key_prefix}{quote(user_id, safe='')}:" if user_id else self._key_prefix

    def _history_of(self, row: Mapping[str, Any]) -> Any:
        """The transcript, rebuilt when the class that owns it can rebuild it."""
        snapshot = row.get("history")
        if not isinstance(snapshot, Mapping):
            return snapshot
        try:
            from ..agent.history import ConversationHistory
        except ImportError:  # pragma: no cover -- the agent layer is always here
            return snapshot
        return ConversationHistory.restore(snapshot)

    def _remember(self, entry: ResponseEntry) -> None:
        key = self._cache_key(entry.local_response_id, entry.user_id)
        self._cache.pop(key, None)
        self._cache[key] = entry
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        backing = "persisted" if self._persistence is not None else "memory-only"
        return f"<ResponseStore {len(self._cache)} cached, {backing}>"


__all__ = [
    "DEFAULT_KEY_PREFIX",
    "DEFAULT_MEMORY_CAPACITY",
    "ResponseEntry",
    "ResponseStore",
    "ResponseTarget",
    "has_fresh_provider_state",
    "new_response_id",
    "now_ms",
]
