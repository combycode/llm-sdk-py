"""ExecuteOptions -- per-call overrides for `client.complete()` / `.stream()`.

Transposed from `unified-library-ts/src/llm/types/options.ts`.

Every field is optional. It is very nearly `NormalizedRequest` minus `model` and
`messages`, plus routing overrides -- and that is the point: the client merges
these over its construction-time defaults to produce the normalized request.

`ExecuteOptions`:
  `{system?, history?, maxTokens?, temperature?, topP?, topK?, seed?,
    presencePenalty?, frequencyPenalty?, stop?, tools?, toolChoice?,
    structured?, audio?, outputModalities?, thinking?, cache?, serviceTier?,
    moderation?, providerOptions?, previousResponseId?, stateful?, signal?,
    timeout?, cacheKey?, cacheName?, configName?, ctx?}`

Notes that are not derivable from the field names:

- `system` is a per-call system prompt, STACKED with `LLMClient.system` and any
  `role='system'` messages from the input, in that priority order. When AgentLoop
  calls the client it passes its composed registry-system here, so layered
  prompts (role / context / facts / chat.facts / context-guard.summary) reach the
  request without depending on the immutable `LLMClient.system`.
- `history` is a conversation reference, propagated into `onMessageResolve` so
  listeners (ContextGuard, FilesRegistry) can route per-conversation.
- `tools` here is SCHEMA-ONLY -- the caller dispatches. AgentLoop's executable
  tools come from its constructor and are merged in at the agent layer.
- `topK`: **accepted is not honoured.** A behavioural test on 2026-07-29 (top_k=1
  must force greedy decoding) found only **Anthropic** actually applies it: six
  samples collapsed to a single output. Google (gemini-2.5-flash, 3.6-flash) and
  xAI (grok-4.20) returned 200 but showed no greedy effect -- accepted and inert
  on those models. It is still sent (harmless, and may apply elsewhere) but do
  not rely on it outside Anthropic.
- `seed` is best-effort: the same seed and params SHOULD return the same result.
  Determinism is never guaranteed.
- `presencePenalty` / `frequencyPenalty` are `[-2, 2]`, honoured by OpenAI/xAI
  chat-completions, OpenRouter and Google; ignored by OpenAI/xAI Responses and
  Anthropic, which do not accept them.
- `structured.repairAttempts`: if the model's final output fails to parse,
  re-prompt this many times with the parse error before raising
  `InvalidFinalOutputError`. Default 0 (raise immediately). Honoured by
  `LLMClient.structured_complete`.
- `stateful` is the server-state optimization: when the prior assistant turn
  carries a usable server id (same provider, within TTL, model ok), send the id
  plus only the new turn instead of the full transcript. Default ON; set false to
  always resend history (fully portable). Ignored if `previousResponseId` is set
  manually.
- `moderation` is report-only: it attaches results, and never blocks.
"""

from __future__ import annotations

from typing import Any

#: `interface ExecuteOptions` (options.ts:12).
ExecuteOptions = dict[str, Any]

__all__ = ["ExecuteOptions"]
