"""ClientPool -- keys LLMClients by provider, or by provider+model when the
catalog marks a model as `requiresDedicatedClient`.

Transposed from `unified-library-ts/src/helpers/client-pool.ts`.

Used by `ClientResolver` so one keying policy holds across the SDK.
"""

from __future__ import annotations

from typing import Any

from ..catalog.catalog import ModelCatalog
from ..llm.client import LLMClient
from ..llm.client_base import BaseLLMClient


class ClientPool:
    """`class ClientPool` (client-pool.ts:8)."""

    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        client_cls: type[BaseLLMClient] = LLMClient,
    ) -> None:
        """`client_cls` decides which CORE this pool hands out.

        A pool is not a place to mix them: a caller awaiting what it was given
        and a caller not awaiting it cannot share one, so the choice is made once
        per pool rather than per `get`. Defaults to the synchronous client, which
        is what the plain name means throughout this library.
        """
        self._catalog = catalog
        self._client_cls = client_cls
        self._clients: dict[str, BaseLLMClient] = {}

    def get(self, provider: str, model: str, config: dict[str, Any]) -> BaseLLMClient:
        """A pooled client for this provider/model, built on first ask.

        The config is only consulted when a client is actually CONSTRUCTED. Two
        calls that share a key share a client, so the second call's config is
        ignored -- which is the point of a pool, and the reason the key includes
        the model for anything the catalog marks as needing its own.
        """
        key = self._key_for(provider, model)
        client = self._clients.get(key)
        if client is None:
            client = self._client_cls({**config, "model": model})
            self._clients[key] = client
        return client

    def destroy(self) -> None:
        """Tear down every pooled client.

        Synchronous, unlike the TypeScript's `async destroy()`: it awaits nothing
        there either -- `client.destroy()` emits a hook and returns -- so an
        `async def` here would only oblige every caller to await something that
        never suspends.
        """
        for client in self._clients.values():
            client.destroy()
        self._clients.clear()

    @property
    def size(self) -> int:
        return len(self._clients)

    def _key_for(self, provider: str, model: str) -> str:
        info = self._catalog.get(provider, model) if self._catalog else None
        if info and info.get("requiresDedicatedClient"):
            return f"{provider}/{model}"
        return provider


__all__ = ["ClientPool"]
