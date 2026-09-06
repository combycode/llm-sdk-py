"""OpenRouter provider adapter -- OpenAI-compatible with extensions.

Transposed from
`unified-library-ts/src/llm/providers/openrouter/completions.ts`.

Everything this class used to do to `super().build_request()` -- the max_tokens
rename, the reasoning strip, the tier remap, the routing passthrough -- is the
`openrouter` overlay in the shared spec. Naming the flavor IS the override now,
and the same holds for the two response paths: the `:online` web-search rule is
the `openrouter` delta of the shared response and stream specs.

`:online` web search surfaces as `url_citation` annotations on the message or
delta -- there is no discrete tool-call item -- so their presence is the signal
that web search ran, and that is what the delta maps to a unified `web_search`
builtin-tool call.
"""

from __future__ import annotations

from typing import Any

from ....wire.interpreter import Registry
from ..openai.completions import OpenAIAdapter
from .response_registry import OPENROUTER_RESPONSE_REGISTRY
from .stream_registry import OPENROUTER_STREAM_REGISTRY

#: `interface OpenRouterAdapterConfig` (completions.ts:9) -- `{apiKey, baseURL?}`.
OpenRouterAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://openrouter.ai"


class OpenRouterAdapter(OpenAIAdapter):
    """`class OpenRouterAdapter extends OpenAIAdapter` (completions.ts:24)."""

    name = "openrouter"
    wire_flavor = "openrouter"

    def __init__(self, config: OpenRouterAdapterConfig) -> None:
        super().__init__(
            {"apiKey": config["apiKey"], "baseURL": config.get("baseURL") or _DEFAULT_BASE_URL}
        )

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def completion_path(self) -> str:
        return "/api/v1/chat/completions"

    def response_spec_id(self) -> str:
        return "openrouter/completions.response"

    def response_registry(self) -> Registry:
        return OPENROUTER_RESPONSE_REGISTRY

    def stream_spec_id(self) -> str:
        return "openrouter/completions.stream"

    def stream_registry(self) -> Registry:
        return OPENROUTER_STREAM_REGISTRY


__all__ = ["OpenRouterAdapter", "OpenRouterAdapterConfig"]
