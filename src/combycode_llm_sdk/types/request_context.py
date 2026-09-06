"""RequestContext -- accumulating ids and routing keys, across all four layers.

Transposed from `unified-library-ts/src/types/request-context.ts`.

Propagates across network -> llm -> agent -> server and through plugin handlers.

**Mint-if-absent rule:** each layer fills in what is missing.
  - server mints `requestId`   (or agent mints if there is no server)
  - agent mints `callId`       (or the LLM mints if there is no agent)
  - LLM mints `clientId`       (per construction, persistent)
  - LLM computes `queueName` / `cacheKey` / `configName` (formula defaults)

**Override semantics:**
  - agent MAY override `cacheKey`, `configName` (semantic)
  - agent MUST NOT override `queueName` (infrastructure)

See reports/016 RequestContext for the full lifetime table.

`RequestContext` (all optional):
  `{sessionId?, userId?, requestId?, traceparent?, conversationId?, callId?,
    clientId?, queueName?, cacheKey?, cacheName?, configName?,
    providerResponseId?, previousResponseId?}`

- `sessionId` is minted by the HOLDER (engine/server/orchestrator) once and
  lives for its lifetime (a CLI process, a browser page), shared by every
  request under it. A bare standalone client mints its own.
  `sessionId:requestId` forms the OTel trace id.
- `userId` is set by AuthPlugin on an authenticated request; it identifies the
  user across requests and scopes ResponseStore + ConversationLoader.
- `traceparent` is the W3C trace context of the span this work runs UNDER --
  `00-<32 hex trace>-<16 hex span>-<flags>`, exactly the header shape. Pass it
  and the SDK stops rooting its own trace: its spans join that trace and hang
  under that span. Without it the library cannot know it is inside an
  application's request, so the business chain and the model calls reach the
  backend as two unrelated traces. Sources: the inbound `traceparent` header, or
  an active span from an OTel SDK the app already runs.
- `conversationId` equals `history.id` and is stable for the conversation's life.
- `callId` is unique per `.complete()` / `.stream()` call.
- `clientId` is one uuid per LLMClient instance, set in the constructor.
- `queueName` is the routing key for the network engine's queues; default
  formula `"$provider/$model"`. NOT overridable from the agent.
- `cacheKey` defaults to a content hash; the agent MAY override it.
- `cacheName` is the cache namespace, default `"default"`.
- `configName` is the settings-registry key for ConfigurationPlugin; default
  formula `"$provider/$model"`. The agent MAY override it.
"""

from __future__ import annotations

from typing import Any

#: `interface RequestContext` (request-context.ts:16).
RequestContext = dict[str, Any]

__all__ = ["RequestContext"]
