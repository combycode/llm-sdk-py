"""OpenRouter Responses API adapter.

Transposed from `unified-library-ts/src/llm/providers/openrouter/responses.ts`.

A drop-in replacement for the OpenAI Responses API at
`openrouter.ai/api/v1/responses`. Stateless: no `previous_response_id` support
(a beta limitation), which is why the catalog reports openrouter as not
supporting server-state at all.

Everything this class used to do to `super().build_request()` -- the max_tokens
rename, the reasoning strip, the tier remap, the routing passthrough -- is the
`openrouter` overlay in the shared spec. Naming the flavor IS the override.
"""

from __future__ import annotations

from typing import Any

from ..openai.responses import OpenAIResponsesAdapter

#: `interface OpenRouterResponsesAdapterConfig` (responses.ts:8).
OpenRouterResponsesAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://openrouter.ai"


class OpenRouterResponsesAdapter(OpenAIResponsesAdapter):
    """`class OpenRouterResponsesAdapter extends OpenAIResponsesAdapter` (responses.ts:13)."""

    name = "openrouter"
    wire_flavor = "openrouter"

    def __init__(self, config: OpenRouterResponsesAdapterConfig) -> None:
        super().__init__(
            {"apiKey": config["apiKey"], "baseURL": config.get("baseURL") or _DEFAULT_BASE_URL}
        )

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/api/v1/responses"


__all__ = ["OpenRouterResponsesAdapter", "OpenRouterResponsesAdapterConfig"]
