"""xAI Responses API adapter.

Transposed from `unified-library-ts/src/llm/providers/xai/responses.ts`.

Mirrors the OpenAI Responses API at `api.x.ai/v1/responses`. Differences:

- the system prompt goes in as `role: system` in `input`, not `instructions`;
- reasoning is automatic for reasoning models (no effort param needed);
- encrypted reasoning via `include: ['reasoning.encrypted_content']`.

All three are the `xai` overlay in the shared spec. What is left in code is file
extraction: xAI embeds code-execution files inline in the `logs` payload, so
`files_from_output_item` extends the base (OpenAI-style annotations and image
URLs) with the xAI shape.

`xai_code_exec_files` lives in `parse_helpers.py` in this port, where the
response registry needed it first; it is re-exported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....wire.interpreter import Registry
from ..openai.responses import OpenAIResponsesAdapter
from .parse_helpers import xai_code_exec_files
from .responses_registry import XAI_RESPONSES_REGISTRY
from .stream_registry import XAI_STREAM_REGISTRY

#: `interface XAIResponsesAdapterConfig` (responses.ts:19).
XAIResponsesAdapterConfig = dict[str, Any]

_DEFAULT_BASE_URL = "https://api.x.ai"


class XAIResponsesAdapter(OpenAIResponsesAdapter):
    """`class XAIResponsesAdapter extends OpenAIResponsesAdapter` (responses.ts:50)."""

    name = "xai"
    wire_flavor = "xai"

    def __init__(self, config: XAIResponsesAdapterConfig) -> None:
        super().__init__(
            {"apiKey": config["apiKey"], "baseURL": config.get("baseURL") or _DEFAULT_BASE_URL}
        )

    def base_url(self) -> str:
        return self._base_url or _DEFAULT_BASE_URL

    def response_spec_id(self) -> str:
        """Identical to OpenAI's by inheritance, but addressed by its own id so
        the target has a spec of its own rather than a special case in the
        lookup."""
        return "xai/responses.response"

    def response_registry(self) -> Registry:
        return XAI_RESPONSES_REGISTRY

    def stream_spec_id(self) -> str:
        return "xai/responses.stream"

    def stream_registry(self) -> Registry:
        return XAI_STREAM_REGISTRY

    def files_from_output_item(self, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [*super().files_from_output_item(item), *xai_code_exec_files(item)]


__all__ = ["XAIResponsesAdapter", "XAIResponsesAdapterConfig", "xai_code_exec_files"]
