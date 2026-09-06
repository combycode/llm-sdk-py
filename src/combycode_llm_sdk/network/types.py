"""Network layer shared types.

Transposed from `unified-library-ts/src/network/types.ts`.

The engine itself is not in this batch; what is here is the shape LLMClient
consumes -- the request it hands over, the response and SSE frames it gets back,
and the two function types it is injected with.

Shapes, as camelCase dicts:

- `HttpRequest`  `{url, method?, headers, body, timeout?, signal?, stream?,
  provider, model, responseType?, rawBody?, trace?, retry?}`. `provider` and
  `model` are set by the semantic layer purely for hook observability -- the
  routing key is `queueName`, set by LLMClient via formula or RequestContext
  override. `responseType` defaults to `'json'`; `'arraybuffer'` is for binary
  downloads (TTS audio bytes, video file bytes), `'text'` for plain text, and
  `'stream'` returns the body un-buffered for large downloads that pipe straight
  to a sink. `rawBody` tells the queue NOT to serialise a body that is already
  bytes (a multipart upload).
- `RequestRetryOverride` `{maxRetries?, totalTimeoutMs?, attemptTimeoutMs?,
  maxRetryAfterMs?, backoff?}` -- the retry knobs a SINGLE request may override,
  applied over the queue's policy for that request only. Deliberately a subset:
  `perKind` stays queue-level, because one request cannot sensibly redefine which
  error classes are retryable for the queue it shares with everyone else.
- `HttpResponse` `{status, headers, body}` -- raw, post-fetch and
  pre-provider-parse.
- `SSEEvent`     `{event?, data, id?}` -- `id` is the SSE `id:` field, used for
  resumption (Last-Event-ID).
- `TraceContext` `{sessionId?, requestId?, callId?, traceparent?}`.
  `sessionId:requestId` is the OTel trace id. `traceparent` rides with the ids
  because every span we emit needs it, not only the ones built straight from a
  caller's RequestContext -- an agent run reaches its tool calls and its nested
  agents through this object, and those were left rooting traces of their own
  while the LLM spans joined the caller's.
- `QueueSnapshot` `{queueName, depth, inFlight, waiting, rateLimitWaitMs,
  running, processed, peakDepth}` -- point-in-time numeric state of one queue.
  `processed` is a LIFETIME counter, so an idle queue still shows evidence of
  past activity where `depth`/`inFlight` read 0 between bursts.
- `FetchOptionsLite` `{queueName?, priority?, estimatedTokens?, ctx?}`.

The realtime socket types (`WsRequest`, `RealtimeSocket`, `ConnectFn`,
`RealtimeFrame`, `RealtimeConnection`, `EngineConnect`) land with the realtime
area, together with the adapters that consume them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

#: `interface HttpRequest` (types.ts:8).
HttpRequest = dict[str, Any]

#: `interface RequestRetryOverride` (types.ts:43).
RequestRetryOverride = dict[str, Any]

#: `interface HttpResponse` (types.ts:52).
HttpResponse = dict[str, Any]

#: `interface SSEEvent` (types.ts:59).
SSEEvent = dict[str, Any]

#: `interface TraceContext` (types.ts:68).
TraceContext = dict[str, Any]

#: `interface QueueSnapshot` (types.ts:81).
QueueSnapshot = dict[str, Any]

#: `interface FetchOptionsLite` (types.ts:105).
#:
#: Re-declared here (rather than imported from the engine) to avoid a cycle when
#: consumers -- LLMClient -- only need the function shape.
FetchOptionsLite = dict[str, Any]

#: `type EngineFetch` (types.ts:114). The function shape LLMClient is injected
#: with; the engine's own `fetch` satisfies it, and tests may provide simpler
#: stubs.
#:
#: Awaitable, because the TypeScript returns a Promise and this port keeps ONE
#: core: `LLMClient.complete` is a coroutine, and the sync `LLM` facade drives it.
#: A synchronous transport still satisfies this by wrapping its result.
EngineFetch = Callable[..., Awaitable[HttpResponse]]

#: `type EngineFetchStream` (types.ts:117). The streaming variant: an async
#: iterator of SSE frames.
EngineFetchStream = Callable[..., AsyncIterator[SSEEvent]]

__all__ = [
    "EngineFetch",
    "EngineFetchStream",
    "FetchOptionsLite",
    "HttpRequest",
    "HttpResponse",
    "QueueSnapshot",
    "RequestRetryOverride",
    "SSEEvent",
    "TraceContext",
]
