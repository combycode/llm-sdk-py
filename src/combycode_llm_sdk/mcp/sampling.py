"""When the server asks US to run a completion.

`sampling/createMessage` inverts the usual direction: the MCP server wants an
LLM answer and has no model of its own, so it asks its client for one. We
fulfil it with our own engine -- which means the server borrows a multi-provider
brain it never had to integrate.

The caller either supplies a handler outright, or names a model and lets this
wire one up.

On the modern wire the same question arrives as an `input_required` result
rather than a pushed request, and it lands on this same handler. That is
deliberate: sampling is configured once and works on either wire.

The completion function is a PARAMETER rather than an import. `mcp` is a plugin
and `helpers` already imports most of the plugins, so importing back would close
a cycle. The public `sampling_handler` lives in `helpers/mcp.py` and supplies
`complete`; the mapping between MCP's message shape and ours stays here, where
it belongs.

Transposed from `unified-library-ts/src/plugins/mcp/sampling.ts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Answers one `sampling/createMessage`. Takes the raw wire params, returns the
#: raw wire result -- this is a protocol boundary, so it speaks the protocol.
McpSamplingHandler = Callable[[Mapping[str, Any]], dict[str, Any]]

#: Runs one completion. The single thing this module needs from the ergonomic
#: layer, taken as a parameter to avoid an import cycle.
McpCompleteFn = Callable[..., Any]


@dataclass(frozen=True)
class McpSamplingViaLLM:
    """Auto-wire sampling to one of our models."""

    model: str
    provider: str | None = None
    engine: Any = None


#: Either a handler, or a model to wire one up around.
McpSamplingConfig = McpSamplingHandler | McpSamplingViaLLM


def _content_to_internal(block: Any) -> Any:
    """One MCP content block as our own content.

    Text collapses to a bare string, which is what our message shape uses for
    the common case; media becomes a one-element part list.
    """
    if not isinstance(block, Mapping):
        return ""
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if kind in ("image", "audio"):
        return [
            {
                "type": kind,
                "source": {
                    "type": "base64",
                    "mimeType": block.get("mimeType"),
                    "data": block.get("data"),
                },
            }
        ]
    # An unknown block kind is dropped rather than passed through: our adapters
    # would reject it further down, where the error names our own message shape
    # and not the MCP block that actually caused it.
    return ""


def to_internal_messages(messages: Any) -> list[dict[str, Any]]:
    """MCP sampling messages as ours."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        out.append(
            {
                "role": message.get("role", "user"),
                "content": _content_to_internal(message.get("content")),
            }
        )
    return out


def to_stop_reason(finish_reason: str) -> str:
    """Our finish reason as MCP's `stopReason`.

    Only the two that have different names are translated. Anything else is
    passed through unchanged rather than mapped to a default -- a reason we do
    not recognise is still information, and flattening it to `endTurn` would
    tell the server the model stopped cleanly when it may not have.
    """
    if finish_reason == "length":
        return "maxTokens"
    if finish_reason == "stop":
        return "endTurn"
    return finish_reason


def sampling_handler_with(
    complete: McpCompleteFn, config: McpSamplingConfig
) -> McpSamplingHandler:
    """Build a sampling handler: pass a function through, or wire up a model."""
    if not isinstance(config, McpSamplingViaLLM):
        return config

    def handler(params: Mapping[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "model": config.model,
            "input": to_internal_messages(params.get("messages")),
        }
        if config.provider:
            options["provider"] = config.provider
        if config.engine is not None:
            options["engine"] = config.engine
        # Only forwarded when the server actually asked. Passing `None` through
        # would override a model's own default with "no value", which is not the
        # same as leaving it alone.
        if isinstance(params.get("systemPrompt"), str):
            options["system"] = params["systemPrompt"]
        if isinstance(params.get("maxTokens"), int):
            options["max_tokens"] = params["maxTokens"]
        if isinstance(params.get("temperature"), (int, float)):
            options["temperature"] = params["temperature"]

        answer = complete(**options)
        return {
            "role": "assistant",
            "content": {"type": "text", "text": answer.text},
            "model": answer.model,
            "stopReason": to_stop_reason(answer.finish_reason),
        }

    return handler


__all__ = [
    "McpCompleteFn",
    "McpSamplingConfig",
    "McpSamplingHandler",
    "McpSamplingViaLLM",
    "sampling_handler_with",
    "to_internal_messages",
    "to_stop_reason",
]
