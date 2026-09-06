"""Hosted server-side tools each provider's chat models support.

Transposed from `unified-library-ts/src/catalog/builtin-tools.ts`.

This is ADAPTER-SOURCED: it reflects what the provider adapters actually map the
unified `{'type': 'web_search' | 'code_interpreter'}` builtins onto (each to the
provider's native shape), verified by the live tools-parity test. It is applied
PROVIDER-LEVEL to tool-capable (chat-family) models at catalog load -- a
deliberate first pass; a reliable per-model source can refine it later.

Coverage (verified live 2026-07):

- `web_search`       -> anthropic, openai, google, xai, openrouter
- `code_interpreter` -> anthropic, openai, google, xai. **Not** openrouter: it
  proxies function tools and its own plugins, but does not route hosted code
  execution -- confirmed against the API.
- `web_fetch`        -> anthropic (`web_fetch_20260318`), google (`urlContext`).
  OpenAI has no separate fetch tool (its `web_search` does page-open); xAI and
  openrouter expose none.
"""

from __future__ import annotations

PROVIDER_BUILTIN_TOOLS: dict[str, tuple[str, ...]] = {
    "anthropic": ("web_search", "web_fetch", "code_interpreter"),
    "openai": ("web_search", "code_interpreter"),
    "google": ("web_search", "web_fetch", "code_interpreter"),
    "xai": ("web_search", "code_interpreter"),
    "openrouter": ("web_search",),
}

__all__ = ["PROVIDER_BUILTIN_TOOLS"]
