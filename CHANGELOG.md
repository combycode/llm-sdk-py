# Changelog

All notable changes to `combycode-llm-sdk` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Per-call tool guardrails**, both halves. `tool_input_guardrails` validates a call's arguments
  before it runs -- ahead of the permission gate, so a call refused on its arguments never reaches
  a person to be approved -- and a trip denies just that call as a tool RESULT, leaving the run
  going. `tool_output_guardrails` inspects what the tool returned: by then it has ALREADY run, so a
  trip does not halt anything, it WITHHOLDS the output and puts a placeholder everywhere it would
  have been kept. Both fail closed: a guardrail that raises counts as tripped, and a
  `tool_output_blocked_message` formatter that raises falls back to the default sentence rather
  than to the output it was deciding about. A tool may also carry its own output guardrails, which
  is how a rule about one MCP server travels with that server's tools.
- **A stored approval answer is matched canonically.** `PendingToolCall.matches` already compared
  the call id, the tool name AND the arguments -- the binding the TypeScript side was missing --
  but compared the arguments as serialised, so the same call with its keys in another order read as
  a different one. Resuming re-runs the model step, which gives no guarantee of key order.
- **Stored MCP OAuth credentials are bound to the authorization server that issued them**
  (SEP-2352), and a token refresh names its resource (RFC 8707). A client registration bound
  elsewhere raises; tokens bound elsewhere read as absent. Unstamped credentials are used as-is and
  stamped on their next save. The binding key is the URL discovery used, not the metadata
  document's `issuer`, which the server controls.
- **`thinking={"mode": "between_tools"}`** -- Anthropic's reason-between-tool-calls mode, and a
  gate in front of it. Measured 2026-09-29 across every active Anthropic chat model: exactly one
  accepts it (`claude-sonnet-5.5`) and the other twelve answer 400 by name. Asking for it elsewhere
  drops the mode, sends a request that works, and reports `request_adjusted` naming the model that
  takes it. Sent only on an explicit `betweenTools: True` in the catalog -- the mirror of
  `canDisable`, because the support distribution is the mirror too.
- **Google Interactions hands its thought signature back.** A turn returns a `thought` step
  carrying nothing but a `signature`, and this library dropped it on both paths. Measured
  2026-09-29 on `gemini-3.1-flash-lite`: echoing the step on the next turn is accepted, and
  echoing it with the signature corrupted is refused `400 Corrupted thought signature`, so the
  server reads it rather than tolerating it. It now rides on `response["signatures"]`, is stamped
  onto `message["origin"]["signatures"]` by `assistant_message()`, and is sent back in the position
  it arrived in -- only by the provider that minted it. A streamed turn keeps it too: the signature
  reaches the client only as its own delta, so it is rebuilt there and carried out on `done`. Kept
  for ANY signed step rather than a list of types, since `processing_*` and `retrieval_*` declare
  one too and are accepted as input.
- **A failed interaction says why.** `Interaction.errors[]` is lifted onto `response["error"]`,
  where a failure used to arrive as `finishReason: "error"` and nothing else. Every recorded
  message is joined, not just the first. On a *completed* interaction the field stays on
  `response["raw"]`: Google documents it as diagnostics rather than as a cause.
- `Completion` exposes `error` and `signatures`. Both were parsed and neither was reachable without
  touching the wire dict, so an in-band failure read as a successful empty answer.
- **`cacheDiagnostics` asks WHY the prompt cache missed**, and `response["cacheDiagnostics"]`
  carries the answer. `usage.cached_tokens` says how much was reused; on a long system prompt the
  useful question is what broke the prefix, and Anthropic and OpenAI both answer it under
  different names. Every shape was measured on 2026-09-29 rather than read out of the SDK types,
  which would have got the central case wrong: **Anthropic has no cache-hit variant** -- a request
  whose prefix WAS reused returns `diagnostics: null`, the same body an undiagnosed request
  returns, so nothing here turns that silence into `status: "hit"`. Three more measured facts:
  an unknown comparison id is a 200 on both providers (`comparison_not_found`, not an error);
  OpenAI gates the feature to `gpt-5.6` and later, so every earlier model answers `unavailable`;
  and Anthropic keeps the fingerprint only for requests that themselves opted in, so a chain must
  pass the option on every call. `status` and `reason` are open unions (R1) and each provider's
  own reason word is kept rather than translated. Requesting it where no field exists is reported
  as `request_adjusted`. `stream()` reports it too, as a `CacheDiagnosticsEvent` and on the
  streamed final response. The vendored response corpus gained a cell per branch, so the
  Python parse is checked against the same frozen oracle as the TypeScript one.
- **`catalog.refuse_call(provider, model)`** turns a measured `unavailable` into the refusal
  itself: the sentence to fail with, or `None` to go ahead. The media helpers call it before
  `generate_image`, `edit_image`, `generate_audio` and `generate_video`, so a dead endpoint costs
  an explanation rather than a provider 404. Naming the provider's own id instead of this
  library's slug (`imagen-4.0-generate-001`, not `imagen-4`) is a force mode: one account was
  measured, and an Enterprise-only endpoint answers for somebody. `unavailable_reason()` still
  reports the measurement for both spellings; only the refusal is lifted.
- **`FilesRegistry(upload_lifetime_seconds=...)`** asks the provider to delete an uploaded file
  for you. Files do not clean themselves up -- OpenAI persists everything but `purpose=batch`
  until something deletes it -- so an agent attaching a document per turn grew an unbounded pile
  on the customer's account. Off by default. The three field shapes were measured, not read:
  Anthropic takes a plain `expires_in_seconds`, OpenAI needs BRACKET fields (a JSON string is
  refused 400), and xAI requires its field BEFORE the file part. Google cannot take one --
  `expiration_time` is "Output only" -- and says so through `on_warning` with code
  `request_adjusted` rather than dropping the request quietly. `FileUploadOptions` is exported
  for anyone implementing `FileProviderAdapter` themselves.

### Fixed

- **A request that continues server-side state is no longer replayed by the retry layer.** A body
  carrying `previous_response_id` or `previous_interaction_id` has the provider append the turn to
  a conversation it holds, so a retry after a failure that reached it appends a second one into a
  transcript read back later. Opt back in per request with `approve_unsafe_replay`. A stateless
  request is unaffected, timeouts included.
- **A human's approval was discarded when the model re-emitted the same arguments in another key
  order.** `PendingToolCall.matches` compared the arguments as serialised, and resuming re-runs the
  model step, which gives no guarantee of key order. The comparison is canonical now (array order
  still counts, because that one is meaningful). The failure was safe -- nothing unapproved ran --
  but it asked a person again for a decision they had already given.
- **A tool call whose arguments did not parse ran with `{}`.** That is a valid call, so a stream
  cut at `{"path": "/et` reached the executor as `delete_files()`. Malformed calls are now marked
  and never executed, still answered so the history stays valid, and reported as
  `malformed_tool_call` on every provider instead of only Google.
- **An interrupted turn left a tool call nobody answered**, which Anthropic and OpenAI both
  reject on the next request, so the run died a turn later naming neither cause. Repaired in
  `begin_run`, the one gate every run passes through.
- **The retry layer re-sent requests it should not have**: one the caller had cancelled, a POST
  the server had already answered 200 when only the body failed to parse, and every unmapped 4xx.
  `x-should-retry: false` is now a veto.
- **A 429 or 503 arriving before a single byte of a stream ended the request outright.** The
  connect phase now retries on the same policy; nothing after the first event does, since the
  caller already holds part of the answer. A connection dropping mid-stream is carried as a
  non-retryable network `LLMError` instead of escaping outside the taxonomy.
- **The SSE parser re-split the whole accumulated buffer on every chunk** -- quadratic in the
  size of one event, which is the shape of a base64 partial image. A mixed CR/LF terminator was
  not treated as a block boundary, so two events arrived as one, and `data:` values were stripped
  of every leading space where the spec strips exactly one.
- **`Retry-After` is parsed as a float**, and an unrepresentable wait becomes infinity rather
  than nothing -- the difference between "longer than we will wait" and "the server said
  nothing", which left the request retrying on the short exponential backoff.
- **A boolean subschema crashed the MCP output validator**, so a spec-valid server took the
  caller down with it. `items: False` and `$ref` were accepted without being read.
- **A service tier the surface will not take became `auto`, silently.** It still falls back, but
  now records the substitution and raises it as `request_adjusted`.
- **Google's streamed path carried its own terminal-reason table holding one entry**, so the same
  response finished differently depending on how it was fetched: a streamed SAFETY block read as
  a clean stop with no content.
- **OpenAI's `incomplete_details.reason` has four values and only one means what `length` means.**
- **The default Google image model answered 404.** `imagen-4.0-generate-001:predict` is
  Enterprise-only, so the default Google image path was broken for anyone who did not name a
  model; it is now `gemini-3.1-flash-image`.
- **The Sora video API shut down on 2026-09-24** and `/v1/videos` answers 404, but `submit_video`
  raised nothing and returned an empty id. It now raises a typed `unsupported` error naming the
  date.
- **`select()` no longer offers models nobody can call** -- measured `unavailable`, or past an
  announced shutdown date, which is checked when you query since a catalog exported yesterday
  cannot know a date passed overnight. A merely deprecated model is still offered.

### Changed

- **Anthropic's Files API is GA and the `files-api-2025-04-14` beta header is gone.** It was not
  redundant: it selected which response shape `list` returned, rather than gating access.
- The xAI catalog no longer claims grok-4.5 and grok-4.6 support no reasoning and no effort
  control, and `quality` is mapped on xAI image generation for `grok-imagine-image-2.0`.

## [0.1.1] - 2026-09-08

### Fixed

- **`gpt-audio` could not return audio.** The request guard fired on audio INPUT alone, so
  `output_modalities=["text", "audio"]` never reached the wire and OpenAI refused the call with
  *"This model requires that either input content or output modality contain audio"*. The parser
  had always been able to build an audio part from the response, so both ends of the path existed
  and only the guard kept them apart. Fixed in the TypeScript library before 0.1.0 shipped and
  missed on the way across; the shared wire spec and the catalog entries now match again.
- **xAI Imagine video renders were billed at up to a twenty-fifth of their real price.** The
  pricing page's first price column is `Media Input`, and it was read as the first resolution's
  rate, shifting every per-resolution price one slot along. `grok-imagine-video` charged
  $0.002/sec for 480p where the page says $0.05 -- an 8-second render estimated at $0.016 instead
  of $0.40 -- and its flat `perSecond` fallback held the input rate, a 5x under-bill whenever no
  resolution was named. `grok-imagine-video-1.5` and `grok-imagine-image-2.0` were shifted the
  same way. `perUnit`/`perImage`/`perSecond` are what a model EMITS; the doc comment now says so.

### Changed

- Model catalog refreshed: 475 models, 20 added (`claude-fable-5.1`, `gpt-6-astra`,
  `gemini-3.8-flash`, `gemini-3.5-transcribe`, `gemini-omni-1.1-flash`, `lyria-3.5` and 13 on
  OpenRouter), 12 delisted and kept with `active: False`, none dropped. Prices, context windows
  and deprecation dates updated across all five providers.
- `outputModalities` now comes from what a source publishes rather than being inferred from the
  model's type, which could not tell a model that emits audio from one that only accepts it.
  23 models gained a second modality. Descriptive metadata -- no request is built differently.

## [0.1.0] - 2026-09-07

Initial release: a Python port of `@combycode/llm-sdk`, sharing its declarative wire specs
byte-for-byte.

- One API across Anthropic, OpenAI, Google, xAI and OpenRouter, with sync and async cores that
  are siblings rather than wrappers.
- Spec-driven requests and responses, the model catalog, cost tracking and budgets, the agent
  loop, MCP client, media, retrieval, batching and an OpenAI-compatible server surface.
- Typed and `py.typed`; no required runtime dependencies beyond `httpx2`.
