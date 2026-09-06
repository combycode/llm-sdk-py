"""Universal completion response.

Transposed from `unified-library-ts/src/llm/types/response.ts`.

Shapes as plain camelCase dicts, per `messages.py`. `Usage` in particular is
read by the cost layer and written by every parse registry under those exact
names, and the 47-cell response corpus is frozen against them.

- `CompletionResponse` `{id, model, content, finishReason, usage, text,
  toolCalls, thinking, media, files?, builtinToolCalls?, citations?,
  moderation?, error?, latencyMs, raw}`
- `FileOutput`      `{id?, name?, mimeType?, data?, url?, source?, ref?}`
- `Citation`        `{url, title?, text?}`
- `BuiltinToolCall` `{tool, id?, code?, output?, query?, url?}`
- `Usage`           `{inputTokens, outputTokens, totalTokens, cachedTokens,
  cacheWriteTokens, reasoningTokens, audioInputTokens?, audioOutputTokens?,
  serviceTier?, pricingTier?}`

Three of those fields are OPTIONAL and must stay optional (CONSTITUTION.md R3, a
response type grows by optional fields only): `files`, `builtinToolCalls` and
`citations` are ABSENT when empty, not `[]`. Read them as
`response.get("citations") or []`.
"""

from __future__ import annotations

from typing import Any, Literal

#: `interface CompletionResponse` (response.ts:6).
CompletionResponse = dict[str, Any]

#: `interface FileOutput` (response.ts:70).
FileOutput = dict[str, Any]

#: `interface Citation` (response.ts:96).
Citation = dict[str, Any]

#: `interface BuiltinToolCall` (response.ts:107).
BuiltinToolCall = dict[str, Any]

#: `interface Usage` (response.ts:151).
Usage = dict[str, Any]

#: `type KnownFinishReason` (response.ts:126) -- the reasons this SDK documents
#: and maps deliberately.
KnownFinishReason = Literal[
    "stop",
    "tool_use",
    "length",
    "content_filter",
    "error",
    "pending",
    # The model tried to call a tool and produced something unusable -- malformed
    # arguments, a hallucinated tool name, a truncated call. Distinct from `error`
    # (the request itself failed) and from `tool_use` (a call we can execute):
    # this turn is *recoverable* by telling the model what went wrong and letting
    # it try again -- see `reflect_and_retry` on `AgentLoop`.
    "malformed_tool_call",
]

#: `type FinishReason = KnownFinishReason | (string & {})` (response.ts:148).
#:
#: **This union is OPEN by design** (CONSTITUTION.md R1). Providers keep
#: inventing terminal states -- four of them did so in a single upstream cycle --
#: and against a closed union every one of those is a breaking change for every
#: consumer, including consumers of providers that changed nothing. Always write
#: a default branch; use `KnownFinishReason` where you want the documented set.
#:
#: `pending` is NOT terminal: the provider accepted the request but has not
#: produced a completion yet, so the response carries no content. Several
#: providers can return a non-terminal status on an otherwise successful call --
#: Google Interactions `queued` (google 2.13) and OpenAI Responses `queued` /
#: `in_progress` (background mode). Those used to fall through to `stop`, which
#: claimed a clean finish for an empty response. Treat `pending` as "poll/retry",
#: never as a result.
FinishReason = str


def empty_usage() -> Usage:
    """A zeroed usage record.

    The six counters are REQUIRED and always present; the audio counters and the
    two tier fields are optional and absent here, exactly as in the TypeScript --
    a zero `audioInputTokens` would price silence as if audio had been sent.
    """
    return {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "cachedTokens": 0,
        "cacheWriteTokens": 0,
        "reasoningTokens": 0,
    }


__all__ = [
    "BuiltinToolCall",
    "Citation",
    "CompletionResponse",
    "FileOutput",
    "FinishReason",
    "KnownFinishReason",
    "Usage",
    "empty_usage",
]
