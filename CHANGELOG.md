# Changelog

All notable changes to `combycode-llm-sdk` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
