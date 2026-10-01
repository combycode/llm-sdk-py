"""LLMClient configuration types.

Transposed from `unified-library-ts/src/llm/client-config.ts`.

`LLMClientConfig` is a camelCase dict, as everywhere else in this port:

`{provider, model, apiKey, system?, baseURL?, dataResidency?, sessionId?, hooks?, fetch?,
  fetchStream?, adapter?, queueName?, configName?, cacheName?, cacheKeyFn?,
  api?, mode?, batchable?, priority?, catalog?, checkResponseShapes?}`

Required and immutable: `provider`, `model`, `apiKey`.

- `dataResidency` is OpenAI-only: `"global" | "us" | "eu" | "ae"`, resolving to
  `api.openai.com` and `{region}.api.openai.com`, so a caller names a region
  instead of writing a host by hand. A project provisioned for one region must
  use that region's host, and the wrong choice fails loudly rather than leaking:
  measured 2026-10-01 from an unrestricted project, `us.` answers "Attempted to
  access resource with incorrect regional hostname" and `eu.` answers "This
  endpoint is only accessible by projects with geography restrictions enabled".
  **Mutually exclusive with `baseURL`** -- setting both RAISES rather than picking
  a winner, because they are two different answers to "which host" and honouring
  one would silently discard a configuration the caller wrote. Setting it on any
  other provider also raises: none of them has regional hosts, and ignoring it
  would let a caller believe their data was pinned to a region when the option
  did nothing at all. See `llm/providers/openai/data_residency.py`.
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
