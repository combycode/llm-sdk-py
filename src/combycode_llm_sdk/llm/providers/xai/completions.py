"""xAI (Grok) provider adapter -- OpenAI-compatible Chat Completions.

Transposed from `unified-library-ts/src/llm/providers/xai/completions.ts`.

Key differences from OpenAI:

- uses `max_tokens`, not `max_completion_tokens`;
- reasoning comes from the model variant (`grok-*-reasoning`), not a parameter;
- `reasoning_content` is returned in the message as plain text, where OpenAI
  hides it.

The first two are the `xai` overlay in the shared spec -- naming the flavor IS
that override. The third is the one thing left in code here, because the shared
Chat Completions response spec has no field for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..openai.completions import OpenAIAdapter

#: `interface XAIAdapterConfig` (completions.ts:12) -- `{apiKey, baseURL?}`.
XAIAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.x.ai"


class XAIAdapter(OpenAIAdapter):
    """`class XAIAdapter extends OpenAIAdapter` (completions.ts:18)."""

    name = "xai"
    wire_flavor = "xai"

    def __init__(self, config: XAIAdapterConfig) -> None:
        super().__init__(
            {"apiKey": config["apiKey"], "baseURL": config.get("baseURL") or _DEFAULT_BASE_URL}
        )

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        result = super().parse_response(raw, latency_ms)

        # xAI returns reasoning_content as plain text in Chat Completions.
        # Assigned only when non-empty, so a model that returns none leaves
        # whatever the shared spec produced rather than blanking it.
        message: Mapping[str, Any] = {}
        if isinstance(raw, Mapping):
            choices = raw.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                first = choices[0].get("message")
                message = first if isinstance(first, Mapping) else {}
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            result["thinking"] = reasoning_content

        return result


__all__ = ["XAIAdapter", "XAIAdapterConfig"]
