"""`ClientPool` -- one engine, many clients, no duplicates.

The public sibling of `helpers/client_pool.py`, which keys `LLMClient`s for the
resolver. This one hands out `LLM` helpers to a fleet of agents.

    pool = ClientPool(engine=shared)
    haiku = pool.get("anthropic", "claude-haiku-4.5", api_key=key)

One engine for the whole fleet is the point: two engines against one provider
means two rate limiters, and the concurrency configured for it silently
doubles.
"""

from __future__ import annotations

from typing import Any


class ClientPool:
    """Clients keyed by provider AND model.

    Not by provider alone: `LLM` binds its model at construction, so a
    provider-keyed pool would hand back a client pinned to whichever model asked
    for it first, and the second model's calls would quietly go to the first.
    """

    def __init__(self, engine: Any = None, **defaults: Any) -> None:
        self.engine = engine
        self._defaults = defaults
        self._clients: dict[tuple[str, str], Any] = {}

    @property
    def size(self) -> int:
        return len(self._clients)

    def get(self, provider: str, model: str, **options: Any) -> Any:
        """The client for this provider and model, built once."""
        from .llm import LLM

        key = (provider, model)
        existing = self._clients.get(key)
        if existing is not None:
            return existing

        settings: dict[str, Any] = {**self._defaults, **options}
        if self.engine is not None and "engine" not in settings:
            settings["engine"] = self.engine
        client = LLM(provider=provider, model=model, **settings)
        self._clients[key] = client
        return client

    def keys(self) -> list[tuple[str, str]]:
        return list(self._clients)

    def clear(self) -> None:
        """Destroy every client, then forget them.

        Destroyed rather than dropped: a client that is merely unreferenced
        keeps its hook subscriptions, and a pool cleared in a long-lived process
        would leak one set per clear.
        """
        for client in self._clients.values():
            destroy = getattr(client, "destroy", None)
            if callable(destroy):
                destroy()
        self._clients.clear()

    def __len__(self) -> int:
        return len(self._clients)

    def __repr__(self) -> str:
        return f"<ClientPool {self.size} client(s)>"


__all__ = ["ClientPool"]
