"""ProviderAdapter -- each provider implements this.

Transposed from `unified-library-ts/src/llm/types/provider.ts`.

The LLMClient calls `build_request(NormalizedRequest)` -> ProviderHttpRequest,
sends via the injected fetch fn, then `parse_response(raw, latency_ms)` ->
CompletionResponse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

from ...wire.interpreter import BuiltRequest

#: `type ProviderName` (provider.ts:9).
ProviderName = Literal["anthropic", "openai", "google", "xai", "openrouter"]

#: The same five names at runtime. A `"vendor/model"` prefix can only be read as
#: a provider if it IS one -- OpenRouter's own ids are all `vendor/model`, so
#: without this check `openai/gpt-5.4-nano` on OpenRouter parses as the provider
#: `openai`, and `qwen/qwen3` parses as a provider named `qwen`.
PROVIDER_NAMES: tuple[str, ...] = ("anthropic", "openai", "google", "xai", "openrouter")


def is_provider_name(value: str) -> bool:
    """`isProviderName` (provider.ts:17). The runtime guard PROVIDER_NAMES exists for."""
    return value in PROVIDER_NAMES


#: `type ApiType` (provider.ts:21).
ApiType = Literal["completions", "responses", "messages", "interactions", "generate"]

#: `interface ProviderConfig` (provider.ts:23) -- `{provider, apiKey, baseURL?}`.
ProviderConfig = dict[str, Any]

#: `interface ProviderHttpRequest` (provider.ts:29):
#: `{body, headers?, path?, notes?}`.
#:
#: `path` overrides the default completion path, for providers that route
#: per-API or per-modality. `notes` is what the build deliberately left out, and
#: why -- a hosted tool this provider refuses to run beside the attached content,
#: for instance. The client emits each as `onWarning`, because dropping a
#: capability the caller asked for and saying nothing is how a missing feature
#: gets mistaken for a working one.
#: It IS `BuiltRequest`, not a dict shaped like it. The TypeScript writes
#: `buildFromSpec(...) as ProviderHttpRequest` -- a structural cast between two
#: names for one object. Restating it as a separate dict here would add a
#: conversion at every adapter, and a conversion is where a field silently stops
#: being carried. `BuiltRequest` is the superset (it also carries `url`,
#: `method`, `multipart`, `raw_body`, `form_body`, `no_body` for the non-chat
#: specs), which is exactly what the cast asserts.
ProviderHttpRequest = BuiltRequest


class ProviderAdapter(Protocol):
    """`interface ProviderAdapter` (provider.ts:43).

    A Protocol rather than an ABC: the TypeScript is an `interface`, satisfied
    structurally, and adapters form their own inheritance chain on top of it
    (`XAIResponsesAdapter` extends `OpenAIResponsesAdapter`).

    `enable_streaming` is OPTIONAL and therefore not declared here -- a Protocol
    member is required by definition. The client probes for it exactly as the
    TypeScript writes `adapter.enableStreaming?.(...)`::

        enable = getattr(adapter, "enable_streaming", None)
        if enable is not None:
            enable(provider_req, req)
    """

    @property
    def name(self) -> ProviderName:
        """The provider this adapter speaks for."""
        ...

    def build_request(self, req: dict[str, Any]) -> ProviderHttpRequest:
        """Convert a universal NormalizedRequest to a provider HTTP body."""
        ...

    def parse_response(self, raw: Any, latency_ms: float) -> dict[str, Any]:
        """Parse the provider's raw HTTP response body to a CompletionResponse."""
        ...

    def parse_stream_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """One SSE event -> zero or more StreamEvents.

        Stateless per-event primitive -- see `create_stream_parser` for the
        per-stream entry point the client actually calls.
        """
        ...

    def create_stream_parser(self) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        """Create a per-stream parser.

        Returns a function converting each SSE event into zero or more
        StreamEvents, holding any per-stream state (e.g. whether the turn has
        begun hosted code execution) in its closure. The client calls this once
        per `stream()` so each stream gets isolated state. Stateless adapters
        return `parse_stream_event` bound to themselves.
        """
        ...

    def auth_headers(self) -> dict[str, str]:
        """Auth headers (Bearer / x-api-key / etc.)."""
        ...

    def base_url(self) -> str:
        """Provider's domain root."""
        ...

    def completion_path(self) -> str:
        """Path appended to `base_url` for the completion endpoint."""
        ...


__all__ = [
    "PROVIDER_NAMES",
    "ApiType",
    "ProviderAdapter",
    "ProviderConfig",
    "ProviderHttpRequest",
    "ProviderName",
    "is_provider_name",
]
