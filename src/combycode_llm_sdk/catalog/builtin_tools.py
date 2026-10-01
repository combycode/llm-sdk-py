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
- `image_generation` -> openai, xai (measured live 2026-10-01, THROUGH the
  library: `{'type': 'image_generation'}` on openai/gpt-5.6-sol,
  openai/gpt-5.4-nano, xai/grok-4.6, xai/grok-4.5 and xai/grok-4.3 all returned
  an image on `response.media`). It was absent for BOTH providers, including
  OpenAI where the tool has worked all along -- so
  `supports_builtin_tool(..., "image_generation")` answered `False` about a tool
  that works, and a caller gating on the catalog refused itself. xAI additionally
  takes `action: 'auto' | 'generate' | 'edit'`, validated rather than inert
  (`action: 'paint'` is a 400 naming the three). The output needs no new parsing:
  xAI's REST item is OpenAI's `image_generation_call` with the same keys, not the
  `{'__type': 'image_generation_result'}` envelope its gRPC surface uses.
"""

from __future__ import annotations

PROVIDER_BUILTIN_TOOLS: dict[str, tuple[str, ...]] = {
    "anthropic": ("web_search", "web_fetch", "code_interpreter"),
    "openai": ("web_search", "code_interpreter", "image_generation"),
    "google": ("web_search", "web_fetch", "code_interpreter"),
    "xai": ("web_search", "code_interpreter", "image_generation"),
    "openrouter": ("web_search",),
}

__all__ = ["PROVIDER_BUILTIN_TOOLS"]
