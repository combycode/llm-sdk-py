# Changelog

All notable changes to `combycode-llm-sdk` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Choose how Google reads a video.** New per-video `providerOptions.processing` --
  `'agentic'` (the model navigates), `'static'` (fixed frame rate, every frame in context), or
  `{ type: 'static', fps, startOffset, endOffset }`. On a long video this is cost, not style: the
  offsets are how a question about 30 seconds of a two-hour recording costs what 30 seconds should.
  Mapped per surface, because the two Google surfaces are NOT the same shape: `generateContent`
  takes `Part.mediaProcessing`, a two-value screaming-snake enum, so the object form is honoured on
  Interactions (`start_offset`/`end_offset`) and reduced to plain `STATIC` there. Interactions also
  takes a per-video `name`. Live-measured 2026-09-30, which caught a defect in the first cut:
  `mediaProcessing` is refused unless the SAME part carries a video mime type, and our `url` and
  `file` sources carry none -- so a video that asks for processing now gets one, and only then,
  leaving every request that did not ask byte-identical. `'agentic'` is gated per model by the
  provider (`400 Agentic video processing is not enabled for this model` on
  `gemini-3.1-flash-lite`); that gate is deliberately not encoded here, since a hard-coded model
  list goes stale and the provider's message is already precise.

### Added

- **Refuse an oversized image instead of letting Anthropic shrink it silently.** An image over the
  model's maximum is downsized by default and nothing says so -- the model reasons over dimensions
  the caller never chose. New per-image `providerOptions.transformations.oversized_image`
  (`'downsize' | 'error'`) on an image content part. Measured 2026-09-30: a 4000x4000 image sent
  with `'error'` is refused with *"image dimensions 4000x4000 exceed the maximum image size of a
  model named on this request and would be downsized to 1092x1092"* -- 7% of the pixels sent --
  while the same image without the field is accepted and shrunk. Per-IMAGE deliberately, via the
  new `ImagePartProviderOptions`: one oversized screenshot should not change how every other image
  in the conversation is handled. Omitted when unset, and an empty object is not sent, matching
  Anthropic's documented "equivalent to omitting the field".

### Added

- **`promptCacheOptions.prewarm`: write the prompt cache without generating anything.** Typed as
  the new `PromptCacheOptions` (`prewarm`, `mode`, `ttl`, `comparison_response_id`), forwarded
  verbatim as before. Live-measured 2026-09-30 on `gpt-5.6-terra`: the prewarm call returns 0
  output items and 0 cached tokens, and the next call on the same 4177-token prompt reads 4174 of
  them from cache -- the prompt carried a per-run nonce, so that hit can only have come from the
  prewarm. Worth paying for only when the prefix will be reused: the prewarm is billed for the
  input it writes. A prewarm response has an EMPTY `output[]`, and this library reports it as an
  ordinary empty result (`finishReason: 'stop'`, no content, no `error`) rather than a failure --
  now pinned by a test, since an empty output is exactly the shape a finish-reason extractor gets
  wrong. Also measured: `prompt_cache_options` as a whole is refused on a pre-5.6 model with
  `400 prompt_cache_options is not supported on this model`, so gpt-5.6+ is a hard requirement and
  not advice. `ttl` stays an open union -- OpenAI's own wording is that `30m` is *currently* the
  only supported value.

### Added

- **Web search image results are asked for, and kept.** `web_search_call.results` is not returned
  unless the request carries `include: ["web_search_call.results"]` -- so setting
  `search_content_types: ['image']` on its own produced a search that found images and a response
  that contained none, which reads as "no images found" rather than as a missing parameter.
  Measured 2026-09-30 by sending the same request both ways: with the include, results; without it,
  none. The adapter now derives the include from the tool, and
  `response.builtinToolCalls[].results` carries them -- the four documented image fields renamed to
  camelCase, every other key (a real result also carries `type`) passed through rather than
  dropped. `builtinToolCalls[].sources` carries the URLs a search drew on, which were parsed off
  the wire and discarded before.

- **Typed shapes for the hosted search/fetch params**, exported for editor help over the existing
  verbatim passthrough, the way `McpToolParams` already worked: `WebSearchToolParams`
  (`external_web_access` for cache-only, `search_content_types`, `image_settings`,
  `search_context_size`, `filters`, `user_location`) and `WebFetchToolParams` /
  `WebFetchUrlSources` (Anthropic `url_sources`: which of user input, your tools' results and the
  provider's own results contribute fetchable URLs, each `all` / `none` / `only` / `except`).

### Added

- **Video: ask for silence, and choose the voices.** xAI's video models generate an audio track by
  default, so the useful thing to say is "don't" -- and `false` is exactly the value a truthy
  presence gate throws away, which is why `params.generateAudio` maps with `presence: 'defined'`.
  `params.referenceAudios` (`[{ voiceId: 'ara' }]`, at most three) conditions the generated speech
  on xAI voice-catalog presets; an entry with no voice id is dropped rather than sent, because the
  wire shape's `source` is a protobuf oneof and an empty entry is a request the server must reject.
  Both live-measured against `grok-imagine-video-1.5` on 2026-09-30. `generateAudio` is a unified
  param rather than an xAI escape hatch because Veo has the same concept -- though it is NOT sent
  to Google, which accepts it only in Vertex / Gemini Enterprise mode and refuses it on the
  Developer API this library speaks. `last_frame` and `keyframes` are deliberately held: they are
  in xAI's proto but not its changelog.

### Added

- **Anthropic Workspaces are selectable.** A credential that can act on more than one Workspace has
  to name the one it means. Omitting `anthropic-workspace-id` does not fail -- it charges the
  DEFAULT Workspace, and Workspace is where spend, rate limits and retention are accounted, so the
  mistake is first visible on a bill. New `providerOptions["workspaceId"]` for completions, plus a
  `workspaceId` in `AnthropicAdapter`'s config as a client-wide default that a request overrides.
  Sent on every Anthropic request, not only completions: `AnthropicFileAdapter`,
  `AnthropicBatchAdapter`, the token counter and `list_models_live` each take a `workspace_id`, and
  retrieving a file a turn produced picks up the client's, since the file lives in the Workspace
  that turn was billed to. (`files.content` needed the header spelled out: unlike the other file
  calls it does not extend `files.base`.) Omitted from the request entirely when unset.

- **A safety block that explains itself is no longer flattened to its message.** OpenAI's
  `misalignment_policy_violation` (2026-09) arrives with `error.misalignment`: `detailedExplanation`
  says what about this turn looked wrong, `errorType` classifies it, and `steer.message` is a
  continuation the provider itself suggests sending instead. Only `code` and `message` were read,
  so an agent learned nothing but that it had been stopped. `errorType` stays an open string --
  the provider documents four values and says clients must accept more.

### Fixed

- **A numeric error code is no longer dropped.** `error.code` was read only when it was already a
  `str`, so a code OpenAI sent as a number became an error with NO code. It is now `str(code)`,
  matching openai-py 3.14. `bool` is excluded deliberately: it is an `int` in Python, and `"True"`
  is not a code -- a guard the TypeScript does not need, because `typeof true` is not `'number'`.

### Added

- **A 403 `insufficient_scope` now re-authorizes, asking for the union of scopes** (SEP-2350). Same
  as the TypeScript: only a 401 was handled, so a "valid but too narrow" token could never be
  widened. A step-up skips the refresh, which would mint the scope just refused, and asks for the
  union of the configured, the granted and the challenged scopes -- the granted one read from the
  token, because after a restart it is the only record of what was consented to. Any other 403 stays
  an error. Retried once. New `parse_bearer_challenge` and `union_scopes` in `mcp.oauth`.
  An existing `on_unauthorized` handler keeps working untouched: its arity is inspected, because a
  handler written before step-up takes no argument and Python -- unlike JavaScript, where a
  surplus argument is ignored -- raises TypeError when handed one.

### Fixed

- **A session the server has forgotten is now rebuilt instead of ending the connection.** A stateful
  MCP server answers `404` to a session id it no longer holds -- it restarted, evicted the session,
  or let it expire. That became `CONNECTION_CLOSED`, and because the id is held for the life of the
  transport, EVERY later request failed the same way: one server restart permanently broke a
  connected client. A 404 while a session id is held now drops that id, re-runs the handshake once
  and replays the request -- only the handshake, whose version question is already settled with
  this server, and only in the handshake era, since the modern wire has no session to rebuild. A
  404 with no session id held is left alone, being an ordinary wrong URL, and a recovery that itself
  404s does not start another. When recovery fails the original 404 is what surfaces.

- **The MCP client told every server it was version `"0"`.** Hard-coded in `DEFAULT_CLIENT_INFO`
  since the first release, so server-side logs and compatibility shims attributed our traffic to a
  client that does not exist. It now reports the real version. The literal moved to a new
  `combycode_llm_sdk.version` module -- re-exported as `__version__` and `SDK_VERSION`, and what
  hatchling now builds the wheel's version from -- because code inside the package needed to read
  it and importing the package root from there would be a cycle. Still exactly one literal.

- **Two streamed tool calls no longer merge into one.** Same fault as the TypeScript, in the same
  two places: the accumulator routed an unmatched delta to the FIRST call in flight, and Google's
  stream registry emitted `id: ""` on every delta and end while holding `functionCall.id`. With two
  parallel function calls the second call's arguments were appended to the first, and the second was
  left with an empty arguments string -- which is deliberately read as a genuine no-argument call, so
  it executed with `{}` instead of being refused. An unmatched event now resolves to the most
  recently started call, and an end de-dupes on the entry's own id so a repeated one cannot run the
  tool twice.

### Security

- **An MCP redirect is followed only within the endpoint's own origin.** Same rule as the
  TypeScript, fixing the OPPOSITE symptom: httpx does not follow redirects by default, so this side
  never had the credential leak -- but it also never made the PERMITTED follow, so a legitimate
  same-origin `307` (a trailing-slash normalisation) simply failed. The MCP transport and its OAuth
  flow now follow a redirect when the method survives (`307`/`308`, or any redirect of a `GET`), the
  origin does not change (or upgrades `http` to `https` on default ports), and the target introduces
  no userinfo of its own; at most three hops. The new `TransportRequest.redirect` defaults to
  `"follow"`, so provider calls are untouched.

### Fixed

- **Reasoning effort now reaches the wire on OpenAI and xAI, and `max` stops failing.** Three
  measured faults behind one field, all three shared with the TypeScript side. `effort: "max"` was
  passed through raw on both OpenAI surfaces and neither provider has it -- measured 2026-09-30, both
  answer 400, OpenAI naming the value -- so the documented way to ask for maximum thinking was a
  guaranteed failed request; it now maps to the top rung of each ladder (`xhigh` here, as Google's
  table has always mapped it to `high`, its own ladder ending there). On chat-completions the spec
  built `reasoning: {effort}`, the RESPONSES shape, which OpenAI rejects by name
  (`400 Unknown parameter: 'reasoning'`); it now sends `reasoning_effort`, the top-level string that
  API takes, while OpenRouter keeps the object form it documents. And the xAI overlay dropped
  `reasoning` for every model whose id lacked `multi-agent`, although the catalog advertised
  `effortControl` with `xhigh` for grok-4.5/4.6. Measured per model: grok-4.6 x6.8, grok-4.5 x36.6,
  grok-4.3 x7.4 on Responses and grok-4.6 x10 on chat-completions, ranges disjoint in each; the whole
  grok-4.20 line answers `400 "does not support parameter reasoningEffort"` and is still omitted,
  which is why this is an explicit table and not a version comparison -- 4.20 refuses the field while
  the numerically lower 4.3 honours it.
- **Catalog: `grok-4.3` and `grok-4.7` were recorded as having no reasoning support at all**, while
  the older 4.5 and 4.6 were recorded as having effort control. Measured 2026-09-30: both accept
  `low|medium|high|xhigh`, and 4.3 honours the difference x7.4. Corrected.

- **A cost calculated at a service tier the catalog does not price now says so** (`onWarning`,
  `code: "unpriced_tier"`). An unpriced model already reported unknown; an unpriced TIER fell back
  to the flat rate and returned a confident number computed at the wrong one. A latency tier is
  bought because it costs more, so the error always ran the same way -- too low, on exactly the
  requests someone chose to pay extra for. Fires once per model and tier, and only for a model that
  declares tier pricing at all.

- **A nested agent run now belongs to the run above it.** `delegate()` and `handoff()` started the
  specialist with nothing, so its spans rooted a trace of their own and the two halves of one
  request could not be joined -- correlation being the entire point of a trace id. Worse in one
  specific way: `_step_options` already let a caller's `ctx` reach the LLM calls, so those joined
  the caller's trace while the RUN that made them (`onRunStart`, every tool call) sat in a second,
  unrelated one. A caller's trace is now resolved in `_begin_run` too, with `sessionId` and
  `requestId` taken together because supplying one without the other is how one run is reported as
  two, and `traceparent` carried when given. `conversationId` is deliberately not inherited -- it
  is how an Observer tells one agent's completions from another's, and a specialist has its own
  history.
- **A tool can see the call it is serving**, through the new `current_tool_run()`: the tool name,
  call id, step, the run's trace, and whether a stop has been asked for. A ContextVar rather than an
  argument, because a tool here is a plain function whose signature IS the model's parameter list --
  and copied across the worker thread the timeout runs it on, which a pool otherwise starts with an
  empty context. `nested_run_options()` turns it into the options a nested run wants; it is exported
  because a hand-written agent-as-tool wrapper is the common case and one written without it keeps
  the old behaviour in silence.
  Cancellation stays COOPERATIVE here, unlike the TypeScript's abort signal: Python cannot kill a
  running worker, and pretending otherwise would leak a half-applied side effect while reporting a
  clean stop. So a long tool ASKS (`current_tool_run().is_stopping()`), and a sub-agent given the
  same stops at its next step boundary -- the guarantee the parent already gives itself.
- **A tool that returns media now sends media.** A tool has always been allowed to return content
  parts and the loop always carried them into the tool result -- then every adapter serialised the
  list into the provider's text slot. So the documented way to return a screenshot worked in the
  sense that the request succeeded: the model received a wall of base64 as prose, was billed for it
  as prose, and could not see the picture. Measured 2026-09-30 on the TypeScript side with a tool
  returning a solid-colour square and the model asked to name the colour: 0 of 6 targets right
  before, 6 of 6 after, and two of the six did not say they could not see it, they named a confident
  wrong colour. Each API has its own slot and they disagree about where, so the result is split into
  its text half and its media half: Anthropic takes blocks inside `tool_result.content`, OpenAI
  Responses items inside `function_call_output.output`, Google
  `functionResponse.parts[].inlineData`; chat-completions and Google Interactions have no slot at
  all, so the media follows in its own user turn after every tool result -- after, because those
  APIs reject a request where a call is still unanswered. A string result builds exactly the body it
  did before. Where a part cannot travel it says so (`[unsupported: audio]` on Anthropic,
  `[image omitted: ...]` for a URL source on Google, whose function response takes inline bytes
  only) rather than dropping it.
- **An output guardrail now sees a tool result that is content parts**, serialised, the way the
  TypeScript has always shown it. It used to skip the check entirely for a non-string result -- and
  a tool returning media is precisely the case a rule wants to look at. A trip withholds the whole
  result, media included.

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
