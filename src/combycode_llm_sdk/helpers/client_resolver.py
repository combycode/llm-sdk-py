"""ClientResolver -- turns a `"provider/model"` string into a pooled LLMClient.

Transposed from `unified-library-ts/src/helpers/client-resolver.ts`.

Lives on the orchestrator so every module registered with it can share one pool
of clients (one per provider, keyed per-model only when the catalog marks
`requiresDedicatedClient`). Wraps the same `ClientPool` the internal-tool runners
use, so there is one keying policy across the SDK.

The four module-level functions here are the model-id grammar, and they are used
far beyond this class -- `create_llm`, the estimator and the cost paths all parse
ids through them rather than each splitting on a slash.
"""

from __future__ import annotations

from typing import Any

from ..bus.hook_bus import HookBus
from ..catalog.catalog import ModelCatalog
from ..llm.async_client import AsyncLLMClient
from ..llm.client import LLMClient
from ..llm.client_base import BaseLLMClient
from ..network.types import EngineFetch, EngineFetchStream
from .client_pool import ClientPool

#: `interface ClientResolverConfig` (client-resolver.ts:36) --
#: `{apiKeys, hooks, fetch?, fetchStream?, catalog?, clientOptions?}`.
ClientResolverConfig = dict[str, Any]

#: `interface ResolvedClient` (client-resolver.ts:63) -- `{client, provider, model}`.
ResolvedClient = dict[str, Any]

#: Recognized tier suffixes for the `model:tier` selector sugar.
#:
#: Deliberately an ALLOWLIST -- OpenRouter ids legitimately end in `:free` and
#: `:online`, which must NOT be mistaken for a service tier.
_TIER_SUFFIXES = frozenset({"auto", "standard", "priority", "flex", "scale"})


def parse_model_id(model_id: str) -> tuple[str, str]:
    """`"anthropic/claude-haiku-4-5"` -> `("anthropic", "claude-haiku-4-5")`.

    The model half may itself contain slashes -- every OpenRouter id does -- so
    only the FIRST slash separates.
    """
    slash = model_id.find("/")
    if slash <= 0 or slash == len(model_id) - 1:
        raise ValueError(f'Invalid model id "{model_id}" -- expected format "provider/model"')
    return model_id[:slash], model_id[slash + 1 :]


def is_namespaced_model_id(model_id: str) -> bool:
    slash = model_id.find("/")
    return 0 < slash < len(model_id) - 1


def resolve_model(model: str, provider: str | None, label: str) -> dict[str, str]:
    """Resolve a model plus optional provider to a concrete `{provider, model}`.

    An EXPLICIT provider always wins. It used to lose to the model's prefix, and
    that is not a preference -- every OpenRouter model id is `vendor/model`, so
    `provider='openrouter', model='openai/gpt-5.4-nano'` resolved to the provider
    `openai` and sent the **OpenRouter key to api.openai.com**. Ids whose vendor
    is not one of our five (`qwen/qwen3`) fared differently and no better: the
    prefix became a provider name and failed later as "no default adapter for
    provider 'qwen'".

    With a provider given, a leading `<provider>/` on the model is redundant and
    is stripped -- `openrouter` + `openrouter/openai/gpt-5.4-nano` is the
    catalog's own slug form and means the OpenRouter model `openai/gpt-5.4-nano`.

    Without one, the `provider/model` prefix is still the documented sugar, and
    it stays permissive about the prefix on purpose: the pricing paths resolve
    models catalogued under a provider nobody can CALL -- a private deployment, a
    test fixture -- and rejecting those would break costing a model you never
    send. A prefix that is not callable fails where it matters, in the adapter
    factory, naming the provider it could not build.

    `label` names the caller in the error.
    """
    if provider:
        prefix = f"{provider}/"
        stripped = model.removeprefix(prefix)
        return {"provider": provider, "model": stripped}
    if is_namespaced_model_id(model):
        parsed_provider, parsed_model = parse_model_id(model)
        return {"provider": parsed_provider, "model": parsed_model}
    raise ValueError(
        f'{label}: bare model "{model}" requires a provider -- pass it explicitly '
        'or use "provider/model".'
    )


def parse_model_tier(model_id: str) -> dict[str, Any]:
    """Split a trailing `:tier` off a model id, but only for a KNOWN tier.

    `"anthropic/claude-opus-4.8:priority"` -> `{modelId, serviceTier}`;
    `".../qwen3-coder:free"` -> unchanged, because `:free` is an OpenRouter
    variant rather than a tier.
    """
    colon = model_id.rfind(":")
    if colon <= 0:
        return {"modelId": model_id}
    suffix = model_id[colon + 1 :]
    if suffix not in _TIER_SUFFIXES:
        return {"modelId": model_id}
    return {"modelId": model_id[:colon], "serviceTier": suffix}


def direct_fetch() -> Any:
    """Direct-HTTP fetch, SYNCHRONOUS -- bypasses the NetworkEngine queue and retry.

    A zero-config fallback for when `ClientResolver` is not given an explicit
    fetch (tests, a one-off script). **Production callers should pass
    `engine.fetch`**: everything the engine owns -- queueing, rate-limit
    backoff, retry, cost accounting, the network hooks -- is absent from this
    path, and a request that goes through it is invisible to all of it.

    Synchronous because the pool it feeds defaults to the synchronous client. A
    fetch of the wrong colour is not a type error at construction -- it surfaces
    as a coroutine where a response was expected, one call later -- so the two
    defaults are chosen together, here.

    Imported inside the function rather than at module scope so that importing
    the resolver does not import httpx2 -- a caller who passes `engine.fetch`
    never needs it.
    """
    from ..transport import as_fetch, http_transport

    return as_fetch(http_transport())


def adirect_fetch() -> EngineFetch:
    """The async twin of `direct_fetch`, for a resolver pooling async clients."""
    from ..transport import ahttp_transport, as_async_fetch

    return as_async_fetch(ahttp_transport())


class ClientResolver:
    """`class ClientResolver` (client-resolver.ts:68)."""

    def __init__(
        self,
        config: ClientResolverConfig,
        client_cls: type[BaseLLMClient] = LLMClient,
    ) -> None:
        self._config = config
        catalog: ModelCatalog | None = config.get("catalog")
        self._client_cls = client_cls
        self._pool = ClientPool(catalog, client_cls)

    def resolve(self, model_id: str) -> ResolvedClient:
        """`"anthropic/claude-haiku-4-5"` -> a resolved, pooled client."""
        provider, model = parse_model_id(model_id)
        api_keys: dict[str, str] = self._config.get("apiKeys") or {}
        api_key = api_keys.get(provider)
        if not api_key:
            known = [k for k, v in api_keys.items() if v]
            raise ValueError(
                f'ClientResolver: no API key for provider "{provider}". '
                f"Configured providers: [{', '.join(known) or 'none'}]"
            )
        fetch = self._config.get("fetch") or self._default_fetch()
        fetch_stream: EngineFetchStream | None = self._config.get("fetchStream")
        hooks: HookBus = self._config["hooks"]
        client = self._pool.get(
            provider,
            model,
            {
                "provider": provider,
                "apiKey": api_key,
                "hooks": hooks,
                "model": model,
                "fetch": fetch,
                "fetchStream": fetch_stream,
                **(self._config.get("clientOptions") or {}),
            },
        )
        return {"client": client, "provider": provider, "model": model}

    def _default_fetch(self) -> Any:
        """Whichever colour of direct fetch matches this resolver's client."""
        return adirect_fetch() if issubclass(self._client_cls, AsyncLLMClient) else direct_fetch()

    def available_providers(self) -> list[str]:
        api_keys: dict[str, str] = self._config.get("apiKeys") or {}
        return [p for p, key in api_keys.items() if key]

    def destroy(self) -> None:
        self._pool.destroy()

    @property
    def size(self) -> int:
        return self._pool.size


__all__ = [
    "ClientResolver",
    "ClientResolverConfig",
    "ResolvedClient",
    "adirect_fetch",
    "direct_fetch",
    "is_namespaced_model_id",
    "parse_model_id",
    "parse_model_tier",
    "resolve_model",
]
