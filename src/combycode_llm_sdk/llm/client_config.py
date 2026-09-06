"""LLMClient configuration types.

Transposed from `unified-library-ts/src/llm/client-config.ts`.

`LLMClientConfig` is a camelCase dict, as everywhere else in this port:

`{provider, model, apiKey, system?, baseURL?, sessionId?, hooks?, fetch?,
  fetchStream?, adapter?, queueName?, configName?, cacheName?, cacheKeyFn?,
  api?, mode?, batchable?, priority?, catalog?, checkResponseShapes?}`

Required and immutable: `provider`, `model`, `apiKey`.

- `sessionId` is the trace session id. `create_llm` passes `engine.session_id`; a
  standalone client mints its own. It flows onto every RequestContext this client
  builds.
- `adapter` is a pre-built `ProviderAdapter` OR an `AdapterFactory`. `create_llm`
  supplies a default; passing one directly is how tests inject a stub.
- `queueName` / `configName` / `cacheName` are routing identifiers. Formulas are
  not supported -- strings only.
- `cacheKeyFn` is `(req, ctx) -> str`.
- `api` is an `ApiType` or `'auto'`; `mode` is `'foreground'` or `'background'`.
- `catalog` is the model catalog -- the source of truth for server-state
  retention and model binding, and for the wire-spec pin. `create_llm` supplies
  `engine.catalog`. Optional, because an empty catalog still yields correct
  provider-level defaults.
- `checkResponseShapes` warns when a provider's response stops looking like the
  one we learned to read -- a field we have never seen, a field that was always
  there and is now absent, or a discriminator carrying a value nothing branches
  on. OFF by default and it never changes what is parsed: it only emits
  `onWarning`. See `response_shape.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .types.provider import ProviderAdapter

#: `type AdapterFactory` (client-config.ts:11) -- builds a ProviderAdapter for a
#: `(provider, apiKey, api, baseURL)`.
AdapterFactory = Callable[..., ProviderAdapter]

#: `interface LLMClientConfig` (client-config.ts:18).
LLMClientConfig = dict[str, Any]

__all__ = ["AdapterFactory", "LLMClientConfig"]
