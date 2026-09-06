"""Internal normalized request -- what LLMClient hands to ProviderAdapter.

Transposed from `unified-library-ts/src/llm/types/request.ts`.

The public surface is `client.complete(input, options)`; model and system are
fixed at construction, and `LLMClient` builds this `NormalizedRequest` from
(input, options, self.model, self.system).

**This is the one shape whose key names are load-bearing.** Every request spec
addresses it by path -- `req.maxTokens`, `req.providerOptions.speechConfig`,
`req.thinking.effort` -- and the 7700-case request corpus is frozen against
those paths. camelCase here is not a style choice.

`NormalizedRequest`:
  `{model, messages, system?, maxTokens?, temperature?, topP?, topK?, seed?,
    presencePenalty?, frequencyPenalty?, stop?, tools?, toolChoice?,
    structured?, thinking?, cache?, serviceTier?, moderation?,
    providerOptions?, wireSpec?, audio?, outputModalities?,
    previousResponseId?, timeout?, signal?}`

Three fields carry hard-won provider facts, live-verified 2026-07-28:

- `topK` is emitted only where the wire accepts it -- Anthropic, Google
  generateContent AND Interactions, xAI chat, OpenRouter chat. OpenAI has no
  top-k on either surface, so it is dropped there rather than sent and rejected.
- `seed` is emitted only where the wire accepts it -- OpenAI **chat-completions**
  (the Responses API rejects it, 400 "Unknown parameter: 'seed'"), Google
  generateContent + Interactions, xAI chat AND responses, OpenRouter chat.
  Anthropic has no seed (400 "Extra inputs are not permitted") and it is dropped.
- `presencePenalty` / `frequencyPenalty` reach OpenAI/xAI chat-completions,
  OpenRouter and Google.

`structured` is `{schema, name?, strict?, repairAttempts?}`; `repairAttempts` is
the opt-in repair-retry count for `structured_complete` (default 0).

`wireSpec` is which wire spec builds this request, from the catalog's
`ModelInfo.wireSpec` and resolved by `LLMClient`. Absent when the engine runs
without a catalog or the model is not catalogued, in which case the adapter
derives the spec from the model id -- the same fallback `wire` has.
"""

from __future__ import annotations

from typing import Any, Literal

#: `interface ProviderOptions` (request.ts:27).
#:
#: Provider-specific request options that have no unified equivalent. This was
#: `Record<string, unknown>` -- the one untyped hole in the request, and
#: therefore the one place a typo produced silence rather than an error:
#: `promtCacheOptions` type-checked and was simply never sent.
#:
#: Python cannot express "these keys are known, and anything else is allowed"
#: without a TypedDict that forbids the rest, so the known keys are documented
#: here and the dict stays open. Sending a key to a provider that does not read
#: it is ignored, not an error.
#:
#: - Anthropic: `userProfileId` -- forwarded as the `anthropic-user-profile-id`
#:   header; needs the account-level `user-profiles` beta.
#: - OpenAI (responses + chat-completions): `moderationPolicy` (sent alongside
#:   the `moderation` request field), `promptCacheOptions`
#:   (`prompt_cache_options`), `reasoningMode` (`'standard'` or `'pro'`, the
#:   `reasoning.mode` field on Responses).
#: - Google (generate): `responseModalities` (overrides
#:   `generationConfig.responseModalities`, winning over the modality implied by
#:   `outputModalities`), `speechConfig`, `imageConfig`, `translationConfig`,
#:   `cachedContent` (forwarded only when a non-empty string).
#: - OpenRouter: `openrouter` -- routing options merged into the request body.
ProviderOptions = dict[str, Any]

#: `interface NormalizedRequest` (request.ts:69).
NormalizedRequest = dict[str, Any]

#: `type ReasoningContext` (request.ts:154).
#:
#: OpenAI Responses-only: which of the model's prior-turn reasoning items are
#: rendered back to it on later turns of a stateful conversation (chained via
#: `previousResponseId` / server-state). `all_turns` keeps continuity at higher
#: token cost; `current_turn` drops earlier reasoning; `auto` lets OpenAI decide.
#: Omitted, the model picks: the gpt-5.6 family defaults to `all_turns`, earlier
#: models to `current_turn`. Ignored by every other provider.
ReasoningContext = Literal["auto", "current_turn", "all_turns"]

#: `type ThinkingVisibility` (request.ts:161).
#:
#: How much of the model's reasoning is returned. `full` (default) returns it as
#: fully as the provider allows; `summary` a condensed form where the provider
#: supports one (else full); `hidden` keeps reasoning internal. Best-effort per
#: provider -- Anthropic `enabled.display`, OpenAI Responses `summary`, Google
#: `includeThoughts`; providers without a control ignore it.
ThinkingVisibility = Literal["full", "summary", "hidden"]

#: `type ThinkingConfig` (request.ts:163):
#: `{mode:'auto'|'on', effort?, visibility?, context?}` or `{mode:'off'}`.
#: `effort` is one of `low`, `medium`, `high`, `max`.
ThinkingConfig = dict[str, Any]

#: `type CacheConfig` (request.ts:178): `'auto'`, `'off'`, or
#: `{system?, tools?, ttl?}`.
CacheConfig = str | dict[str, Any]

__all__ = [
    "CacheConfig",
    "NormalizedRequest",
    "ProviderOptions",
    "ReasoningContext",
    "ThinkingConfig",
    "ThinkingVisibility",
]
