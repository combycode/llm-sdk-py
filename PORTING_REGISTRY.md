# Porting registry

Generated from the TypeScript tree. Do not hand-edit: re-run the generator.
`status` is derived from what is on disk, never from a claim.

Rules: [PORTING.md](PORTING.md). The column that matters is not `status`
— a row can say `ported` and still be a paraphrase — it is **`PY cases` must
not be less than `TS cases`**.

## Where it stands

| | TS | PY |
|---|---|---|
| source files | 260 | 18 |
| test files | 249 | 9 |
| test cases | 3806 | 119 |
| golden fixtures | 9 | 9 |
| examples | 58 | 74 |

## Etalon — never edited

Vendored byte-identical from the TypeScript tree. If one of these disagrees
with the port, the port is wrong.

| what | count | verified |
|---|---|---|
| golden fixtures | 9 | identical |
| wire specs | 150 | identical |
| catalog data | 5 | identical |

## Areas — one owner each, boundaries taken from the TypeScript test tree

| area | TS files | TS cases | PY cases | status |
|---|---|---|---|---|
| `tests/integration` | 8 | 10 | 0 | pending |
| `tests/live` | 1 | 7 | 0 | pending |
| `tests/unit/agent` | 20 | 311 | 0 | pending |
| `tests/unit/architecture` | 1 | 2 | 0 | pending |
| `tests/unit/bus` | 4 | 61 | 0 | pending |
| `tests/unit/catalog` | 3 | 27 | 0 | pending |
| `tests/unit/helpers` | 48 | 678 | 0 | pending |
| `tests/unit/llm` | 19 | 252 | 58 | started |
| `tests/unit/llm/audio` | 1 | 4 | 4 | done |
| `tests/unit/llm/files` | 2 | 47 | 0 | pending |
| `tests/unit/llm/providers` | 20 | 428 | 0 | pending |
| `tests/unit/network` | 9 | 111 | 0 | pending |
| `tests/unit/plugins` | 1 | 14 | 0 | pending |
| `tests/unit/plugins/batch` | 2 | 49 | 0 | pending |
| `tests/unit/plugins/cache` | 2 | 33 | 0 | pending |
| `tests/unit/plugins/configuration` | 1 | 21 | 0 | pending |
| `tests/unit/plugins/context-guard` | 4 | 108 | 0 | pending |
| `tests/unit/plugins/context-measurer` | 7 | 102 | 0 | pending |
| `tests/unit/plugins/cost-collector` | 4 | 66 | 0 | pending |
| `tests/unit/plugins/files` | 4 | 64 | 0 | pending |
| `tests/unit/plugins/internal-tools` | 9 | 318 | 0 | pending |
| `tests/unit/plugins/logger` | 2 | 39 | 0 | pending |
| `tests/unit/plugins/mcp` | 20 | 355 | 0 | pending |
| `tests/unit/plugins/media` | 6 | 91 | 0 | pending |
| `tests/unit/plugins/model-catalog` | 1 | 10 | 0 | pending |
| `tests/unit/plugins/permissions` | 2 | 40 | 0 | pending |
| `tests/unit/plugins/persistence` | 2 | 23 | 0 | pending |
| `tests/unit/plugins/retrieval` | 6 | 142 | 0 | pending |
| `tests/unit/plugins/scheduler` | 2 | 24 | 0 | pending |
| `tests/unit/plugins/telemetry` | 6 | 93 | 0 | pending |
| `tests/unit/plugins/tool-catalog` | 2 | 39 | 0 | pending |
| `tests/unit/runtime` | 1 | 8 | 0 | pending |
| `tests/unit/server` | 7 | 85 | 0 | pending |
| `tests/unit/util` | 7 | 64 | 36 | started |
| `tests/unit/wire` | 15 | 80 | 21 | started |

## Test files

| TS test | TS cases | PY test | PY cases | status |
|---|---|---|---|---|
| `tests/integration/mcp-http.test.ts` | 2 | `tests/integration/test_mcp_http.py` | 0 | pending |
| `tests/integration/mcp-oauth.test.ts` | 1 | `tests/integration/test_mcp_oauth.py` | 0 | pending |
| `tests/integration/mcp-resources.test.ts` | 1 | `tests/integration/test_mcp_resources.py` | 0 | pending |
| `tests/integration/mcp-sampling.test.ts` | 1 | `tests/integration/test_mcp_sampling.py` | 0 | pending |
| `tests/integration/mcp-stdio.test.ts` | 2 | `tests/integration/test_mcp_stdio.py` | 0 | pending |
| `tests/integration/mcp-tasks.test.ts` | 1 | `tests/integration/test_mcp_tasks.py` | 0 | pending |
| `tests/integration/mcp-telemetry.test.ts` | 1 | `tests/integration/test_mcp_telemetry.py` | 0 | pending |
| `tests/integration/mcp-ws.test.ts` | 1 | `tests/integration/test_mcp_ws.py` | 0 | pending |
| `tests/live/moderation.live.test.ts` | 7 | `tests/live/test_moderation_live.py` | 0 | pending |
| `tests/unit/agent/agent-identity.test.ts` | 6 | `tests/unit/agent/test_agent_identity.py` | 0 | pending |
| `tests/unit/agent/approval.test.ts` | 15 | `tests/unit/agent/test_approval.py` | 0 | pending |
| `tests/unit/agent/citations-accumulate.test.ts` | 5 | `tests/unit/agent/test_citations_accumulate.py` | 0 | pending |
| `tests/unit/agent/context-registry-subscriptions.test.ts` | 14 | `tests/unit/agent/test_context_registry_subscriptions.py` | 0 | pending |
| `tests/unit/agent/context-registry.test.ts` | 54 | `tests/unit/agent/test_context_registry.py` | 0 | pending |
| `tests/unit/agent/custom-data.test.ts` | 3 | `tests/unit/agent/test_custom_data.py` | 0 | pending |
| `tests/unit/agent/final-answer-phase.test.ts` | 7 | `tests/unit/agent/test_final_answer_phase.py` | 0 | pending |
| `tests/unit/agent/guardrails.test.ts` | 16 | `tests/unit/agent/test_guardrails.py` | 0 | pending |
| `tests/unit/agent/history-registry.test.ts` | 18 | `tests/unit/agent/test_history_registry.py` | 0 | pending |
| `tests/unit/agent/history-tokens.test.ts` | 8 | `tests/unit/agent/test_history_tokens.py` | 0 | pending |
| `tests/unit/agent/history.test.ts` | 22 | `tests/unit/agent/test_history.py` | 0 | pending |
| `tests/unit/agent/lazy-tools.test.ts` | 24 | `tests/unit/agent/test_lazy_tools.py` | 0 | pending |
| `tests/unit/agent/loop.test.ts` | 49 | `tests/unit/agent/test_loop.py` | 0 | pending |
| `tests/unit/agent/reflect-retry.test.ts` | 18 | `tests/unit/agent/test_reflect_retry.py` | 0 | pending |
| `tests/unit/agent/response-passthrough.test.ts` | 4 | `tests/unit/agent/test_response_passthrough.py` | 0 | pending |
| `tests/unit/agent/run-trace.test.ts` | 12 | `tests/unit/agent/test_run_trace.py` | 0 | pending |
| `tests/unit/agent/stream-accumulation.test.ts` | 21 | `tests/unit/agent/test_stream_accumulation.py` | 0 | pending |
| `tests/unit/agent/tool-collision.test.ts` | 9 | `tests/unit/agent/test_tool_collision.py` | 0 | pending |
| `tests/unit/agent/tool-input-guardrail.test.ts` | 2 | `tests/unit/agent/test_tool_input_guardrail.py` | 0 | pending |
| `tests/unit/agent/tool-trace.test.ts` | 4 | `tests/unit/agent/test_tool_trace.py` | 0 | pending |
| `tests/unit/architecture/layers.test.ts` | 2 | `tests/unit/architecture/test_layers.py` | 0 | pending |
| `tests/unit/bus/agent-bus.test.ts` | 32 | `tests/unit/bus/test_agent_bus.py` | 0 | pending |
| `tests/unit/bus/async-context-browser.test.ts` | 7 | `tests/unit/bus/test_async_context_browser.py` | 0 | pending |
| `tests/unit/bus/hook-bus.test.ts` | 19 | `tests/unit/bus/test_hook_bus.py` | 0 | pending |
| `tests/unit/bus/hook-event.test.ts` | 3 | `tests/unit/bus/test_hook_event.py` | 0 | pending |
| `tests/unit/catalog/catalog-required.test.ts` | 10 | `tests/unit/catalog/test_catalog_required.py` | 0 | pending |
| `tests/unit/catalog/model-id-variants.test.ts` | 10 | `tests/unit/catalog/test_model_id_variants.py` | 0 | pending |
| `tests/unit/catalog/undated-aliases.test.ts` | 7 | `tests/unit/catalog/test_undated_aliases.py` | 0 | pending |
| `tests/unit/helpers/batch.test.ts` | 6 | `tests/unit/helpers/test_batch.py` | 0 | pending |
| `tests/unit/helpers/chain.test.ts` | 9 | `tests/unit/helpers/test_chain.py` | 0 | pending |
| `tests/unit/helpers/client-pool.test.ts` | 11 | `tests/unit/helpers/test_client_pool.py` | 0 | pending |
| `tests/unit/helpers/collection.test.ts` | 7 | `tests/unit/helpers/test_collection.py` | 0 | pending |
| `tests/unit/helpers/consolidate-runtime.test.ts` | 17 | `tests/unit/helpers/test_consolidate_runtime.py` | 0 | pending |
| `tests/unit/helpers/consolidate.test.ts` | 6 | `tests/unit/helpers/test_consolidate.py` | 0 | pending |
| `tests/unit/helpers/content.test.ts` | 32 | `tests/unit/helpers/test_content.py` | 0 | pending |
| `tests/unit/helpers/conversation-export.test.ts` | 4 | `tests/unit/helpers/test_conversation_export.py` | 0 | pending |
| `tests/unit/helpers/conversation-zip.test.ts` | 4 | `tests/unit/helpers/test_conversation_zip.py` | 0 | pending |
| `tests/unit/helpers/cost-honest-zero.test.ts` | 20 | `tests/unit/helpers/test_cost_honest_zero.py` | 0 | pending |
| `tests/unit/helpers/count-tokens.test.ts` | 9 | `tests/unit/helpers/test_count_tokens.py` | 0 | pending |
| `tests/unit/helpers/define-tool.test.ts` | 16 | `tests/unit/helpers/test_define_tool.py` | 0 | pending |
| `tests/unit/helpers/delegate.test.ts` | 6 | `tests/unit/helpers/test_delegate.py` | 0 | pending |
| `tests/unit/helpers/embed.test.ts` | 8 | `tests/unit/helpers/test_embed.py` | 0 | pending |
| `tests/unit/helpers/engine.test.ts` | 13 | `tests/unit/helpers/test_engine.py` | 0 | pending |
| `tests/unit/helpers/estimate-calibration.test.ts` | 29 | `tests/unit/helpers/test_estimate_calibration.py` | 0 | pending |
| `tests/unit/helpers/estimate-media.test.ts` | 8 | `tests/unit/helpers/test_estimate_media.py` | 0 | pending |
| `tests/unit/helpers/estimate.test.ts` | 21 | `tests/unit/helpers/test_estimate.py` | 0 | pending |
| `tests/unit/helpers/explicit-provider-wins.test.ts` | 7 | `tests/unit/helpers/test_explicit_provider_wins.py` | 0 | pending |
| `tests/unit/helpers/factories.test.ts` | 7 | `tests/unit/helpers/test_factories.py` | 0 | pending |
| `tests/unit/helpers/filter-facets.test.ts` | 6 | `tests/unit/helpers/test_filter_facets.py` | 0 | pending |
| `tests/unit/helpers/handoff.test.ts` | 9 | `tests/unit/helpers/test_handoff.py` | 0 | pending |
| `tests/unit/helpers/mcp.test.ts` | 79 | `tests/unit/helpers/test_mcp.py` | 0 | pending |
| `tests/unit/helpers/media.test.ts` | 4 | `tests/unit/helpers/test_media.py` | 0 | pending |
| `tests/unit/helpers/models-live.test.ts` | 10 | `tests/unit/helpers/test_models_live.py` | 0 | pending |
| `tests/unit/helpers/moderate.test.ts` | 11 | `tests/unit/helpers/test_moderate.py` | 0 | pending |
| `tests/unit/helpers/moderation-guardrail.test.ts` | 25 | `tests/unit/helpers/test_moderation_guardrail.py` | 0 | pending |
| `tests/unit/helpers/observer.test.ts` | 7 | `tests/unit/helpers/test_observer.py` | 0 | pending |
| `tests/unit/helpers/one-shot-option-forwarding.test.ts` | 4 | `tests/unit/helpers/test_one_shot_option_forwarding.py` | 0 | pending |
| `tests/unit/helpers/one-shot.test.ts` | 12 | `tests/unit/helpers/test_one_shot.py` | 0 | pending |
| `tests/unit/helpers/parallel.test.ts` | 5 | `tests/unit/helpers/test_parallel.py` | 0 | pending |
| `tests/unit/helpers/provenance.test.ts` | 18 | `tests/unit/helpers/test_provenance.py` | 0 | pending |
| `tests/unit/helpers/realtime.test.ts` | 31 | `tests/unit/helpers/test_realtime.py` | 0 | pending |
| `tests/unit/helpers/rest-batch.test.ts` | 10 | `tests/unit/helpers/test_rest_batch.py` | 0 | pending |
| `tests/unit/helpers/rest-client-resolver.test.ts` | 17 | `tests/unit/helpers/test_rest_client_resolver.py` | 0 | pending |
| `tests/unit/helpers/rest-conversation-export.test.ts` | 19 | `tests/unit/helpers/test_rest_conversation_export.py` | 0 | pending |
| `tests/unit/helpers/rest-conversation-zip.test.ts` | 17 | `tests/unit/helpers/test_rest_conversation_zip.py` | 0 | pending |
| `tests/unit/helpers/rest-media.test.ts` | 18 | `tests/unit/helpers/test_rest_media.py` | 0 | pending |
| `tests/unit/helpers/rest-moderate.test.ts` | 7 | `tests/unit/helpers/test_rest_moderate.py` | 0 | pending |
| `tests/unit/helpers/rest-observer.test.ts` | 16 | `tests/unit/helpers/test_rest_observer.py` | 0 | pending |
| `tests/unit/helpers/rest-one-shot.test.ts` | 16 | `tests/unit/helpers/test_rest_one_shot.py` | 0 | pending |
| `tests/unit/helpers/rest-server.test.ts` | 18 | `tests/unit/helpers/test_rest_server.py` | 0 | pending |
| `tests/unit/helpers/rest-small-helpers.test.ts` | 27 | `tests/unit/helpers/test_rest_small_helpers.py` | 0 | pending |
| `tests/unit/helpers/rest-transcribe.test.ts` | 11 | `tests/unit/helpers/test_rest_transcribe.py` | 0 | pending |
| `tests/unit/helpers/route.test.ts` | 4 | `tests/unit/helpers/test_route.py` | 0 | pending |
| `tests/unit/helpers/select-model.test.ts` | 15 | `tests/unit/helpers/test_select_model.py` | 0 | pending |
| `tests/unit/helpers/sends-provider-model-id.test.ts` | 2 | `tests/unit/helpers/test_sends_provider_model_id.py` | 0 | pending |
| `tests/unit/helpers/transcribe.test.ts` | 20 | `tests/unit/helpers/test_transcribe.py` | 0 | pending |
| `tests/unit/llm/audio/voices.test.ts` | 4 | `tests/unit/llm/audio/test_voices.py` | 4 | done |
| `tests/unit/llm/catalog-resolution.test.ts` | 7 | `tests/unit/llm/test_catalog_resolution.py` | 0 | pending |
| `tests/unit/llm/catalog-wire-traits.test.ts` | 12 | `tests/unit/llm/test_catalog_wire_traits.py` | 0 | pending |
| `tests/unit/llm/citations-stream.test.ts` | 9 | `tests/unit/llm/test_citations_stream.py` | 0 | pending |
| `tests/unit/llm/citations.test.ts` | 8 | `tests/unit/llm/test_citations.py` | 0 | pending |
| `tests/unit/llm/client-stream-cost.test.ts` | 8 | `tests/unit/llm/test_client_stream_cost.py` | 0 | pending |
| `tests/unit/llm/client.test.ts` | 40 | `tests/unit/llm/test_client.py` | 0 | pending |
| `tests/unit/llm/completions-parallel-tools.test.ts` | 6 | `tests/unit/llm/test_completions_parallel_tools.py` | 0 | pending |
| `tests/unit/llm/files/provider-file-adapters.test.ts` | 34 | `tests/unit/llm/files/test_provider_file_adapters.py` | 0 | pending |
| `tests/unit/llm/files/retrieve.test.ts` | 13 | `tests/unit/llm/files/test_retrieve.py` | 0 | pending |
| `tests/unit/llm/moderation.test.ts` | 28 | `tests/unit/llm/test_moderation.py` | 0 | pending |
| `tests/unit/llm/programmatic-tool-calling.test.ts` | 21 | `tests/unit/llm/test_programmatic_tool_calling.py` | 0 | pending |
| `tests/unit/llm/provider-options-typed.test.ts` | 6 | `tests/unit/llm/test_provider_options_typed.py` | 0 | pending |
| `tests/unit/llm/providers/anthropic-thinking-shape.test.ts` | 10 | `tests/unit/llm/providers/test_anthropic_thinking_shape.py` | 0 | pending |
| `tests/unit/llm/providers/anthropic.test.ts` | 71 | `tests/unit/llm/providers/test_anthropic.py` | 0 | pending |
| `tests/unit/llm/providers/batch-results.test.ts` | 16 | `tests/unit/llm/providers/test_batch_results.py` | 0 | pending |
| `tests/unit/llm/providers/embeddings.test.ts` | 5 | `tests/unit/llm/providers/test_embeddings.py` | 0 | pending |
| `tests/unit/llm/providers/google-file-name.test.ts` | 7 | `tests/unit/llm/providers/test_google_file_name.py` | 0 | pending |
| `tests/unit/llm/providers/google-generate.test.ts` | 52 | `tests/unit/llm/providers/test_google_generate.py` | 0 | pending |
| `tests/unit/llm/providers/google-interactions.test.ts` | 36 | `tests/unit/llm/providers/test_google_interactions.py` | 0 | pending |
| `tests/unit/llm/providers/google-media.test.ts` | 12 | `tests/unit/llm/providers/test_google_media.py` | 0 | pending |
| `tests/unit/llm/providers/google-tool-schema.test.ts` | 5 | `tests/unit/llm/providers/test_google_tool_schema.py` | 0 | pending |
| `tests/unit/llm/providers/media-request-builders.test.ts` | 1 | `tests/unit/llm/providers/test_media_request_builders.py` | 0 | pending |
| `tests/unit/llm/providers/openai-completions.test.ts` | 47 | `tests/unit/llm/providers/test_openai_completions.py` | 0 | pending |
| `tests/unit/llm/providers/openai-media.test.ts` | 10 | `tests/unit/llm/providers/test_openai_media.py` | 0 | pending |
| `tests/unit/llm/providers/openai-responses.test.ts` | 74 | `tests/unit/llm/providers/test_openai_responses.py` | 0 | pending |
| `tests/unit/llm/providers/openrouter-media.test.ts` | 3 | `tests/unit/llm/providers/test_openrouter_media.py` | 0 | pending |
| `tests/unit/llm/providers/openrouter.test.ts` | 16 | `tests/unit/llm/providers/test_openrouter.py` | 0 | pending |
| `tests/unit/llm/providers/responses-phase-namespace.test.ts` | 10 | `tests/unit/llm/providers/test_responses_phase_namespace.py` | 0 | pending |
| `tests/unit/llm/providers/sampling-topk-seed.test.ts` | 14 | `tests/unit/llm/providers/test_sampling_topk_seed.py` | 0 | pending |
| `tests/unit/llm/providers/xai-batch-name.test.ts` | 4 | `tests/unit/llm/providers/test_xai_batch_name.py` | 0 | pending |
| `tests/unit/llm/providers/xai-media.test.ts` | 13 | `tests/unit/llm/providers/test_xai_media.py` | 0 | pending |
| `tests/unit/llm/providers/xai.test.ts` | 22 | `tests/unit/llm/providers/test_xai.py` | 0 | pending |
| `tests/unit/llm/realtime.test.ts` | 14 | `tests/unit/llm/test_realtime.py` | 0 | pending |
| `tests/unit/llm/response-differential.test.ts` | 5 | `tests/unit/llm/test_response_differential.py` | 0 | pending |
| `tests/unit/llm/response-shape-wiring.test.ts` | 6 | `tests/unit/llm/test_response_shape_wiring.py` | 0 | pending |
| `tests/unit/llm/response-shape.test.ts` | 13 | `tests/unit/llm/test_response_shape.py` | 0 | pending |
| `tests/unit/llm/server-state.test.ts` | 12 | `tests/unit/llm/test_server_state.py` | 0 | pending |
| `tests/unit/llm/service-tier.test.ts` | 18 | `tests/unit/llm/test_service_tier.py` | 21 | done |
| `tests/unit/llm/strict-schema.test.ts` | 22 | `tests/unit/llm/test_strict_schema.py` | 22 | done |
| `tests/unit/llm/trace-passthrough.test.ts` | 2 | `tests/unit/llm/test_trace_passthrough.py` | 0 | pending |
| `tests/unit/llm/type-helpers.test.ts` | 15 | `tests/unit/llm/test_type_helpers.py` | 15 | done |
| `tests/unit/network/connect-default.test.ts` | 5 | `tests/unit/network/test_connect_default.py` | 0 | pending |
| `tests/unit/network/engine.test.ts` | 23 | `tests/unit/network/test_engine.py` | 0 | pending |
| `tests/unit/network/rate-limiter.test.ts` | 14 | `tests/unit/network/test_rate_limiter.py` | 0 | pending |
| `tests/unit/network/realtime-connection.test.ts` | 20 | `tests/unit/network/test_realtime_connection.py` | 0 | pending |
| `tests/unit/network/request-queue.test.ts` | 8 | `tests/unit/network/test_request_queue.py` | 0 | pending |
| `tests/unit/network/retry-after.test.ts` | 18 | `tests/unit/network/test_retry_after.py` | 0 | pending |
| `tests/unit/network/semaphore.test.ts` | 3 | `tests/unit/network/test_semaphore.py` | 0 | pending |
| `tests/unit/network/sse.test.ts` | 11 | `tests/unit/network/test_sse.py` | 0 | pending |
| `tests/unit/network/stream-errors.test.ts` | 9 | `tests/unit/network/test_stream_errors.py` | 0 | pending |
| `tests/unit/plugins/anchored-provenance.test.ts` | 14 | `tests/unit/plugins/test_anchored_provenance.py` | 0 | pending |
| `tests/unit/plugins/batch/batcher.test.ts` | 41 | `tests/unit/plugins/batch/test_batcher.py` | 0 | pending |
| `tests/unit/plugins/batch/strategy.test.ts` | 8 | `tests/unit/plugins/batch/test_strategy.py` | 0 | pending |
| `tests/unit/plugins/cache/cache.test.ts` | 22 | `tests/unit/plugins/cache/test_cache.py` | 0 | pending |
| `tests/unit/plugins/cache/file-store.test.ts` | 11 | `tests/unit/plugins/cache/test_file_store.py` | 0 | pending |
| `tests/unit/plugins/configuration/configuration.test.ts` | 21 | `tests/unit/plugins/configuration/test_configuration.py` | 0 | pending |
| `tests/unit/plugins/context-guard/context-tools.test.ts` | 11 | `tests/unit/plugins/context_guard/test_context_tools.py` | 0 | pending |
| `tests/unit/plugins/context-guard/guard.test.ts` | 7 | `tests/unit/plugins/context_guard/test_guard.py` | 0 | pending |
| `tests/unit/plugins/context-guard/layered-strategy.test.ts` | 35 | `tests/unit/plugins/context_guard/test_layered_strategy.py` | 0 | pending |
| `tests/unit/plugins/context-guard/strategy-tools.test.ts` | 55 | `tests/unit/plugins/context_guard/test_strategy_tools.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/count-api-counter.test.ts` | 27 | `tests/unit/plugins/context_measurer/test_count_api_counter.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/count-api-model-id.test.ts` | 2 | `tests/unit/plugins/context_measurer/test_count_api_model_id.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/counter-strategies.test.ts` | 30 | `tests/unit/plugins/context_measurer/test_counter_strategies.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/measurer-lifecycle.test.ts` | 12 | `tests/unit/plugins/context_measurer/test_measurer_lifecycle.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/measurer.test.ts` | 7 | `tests/unit/plugins/context_measurer/test_measurer.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/tiktoken-counter.test.ts` | 17 | `tests/unit/plugins/context_measurer/test_tiktoken_counter.py` | 0 | pending |
| `tests/unit/plugins/context-measurer/tiktoken-optional.test.ts` | 7 | `tests/unit/plugins/context_measurer/test_tiktoken_optional.py` | 0 | pending |
| `tests/unit/plugins/cost-collector/collector.test.ts` | 12 | `tests/unit/plugins/cost_collector/test_collector.py` | 0 | pending |
| `tests/unit/plugins/cost-collector/compute-cost.test.ts` | 16 | `tests/unit/plugins/cost_collector/test_compute_cost.py` | 0 | pending |
| `tests/unit/plugins/cost-collector/queries.test.ts` | 34 | `tests/unit/plugins/cost_collector/test_queries.py` | 0 | pending |
| `tests/unit/plugins/cost-collector/unpriced.test.ts` | 4 | `tests/unit/plugins/cost_collector/test_unpriced.py` | 0 | pending |
| `tests/unit/plugins/files/attachment-blob.test.ts` | 4 | `tests/unit/plugins/files/test_attachment_blob.py` | 0 | pending |
| `tests/unit/plugins/files/attachment-state.test.ts` | 21 | `tests/unit/plugins/files/test_attachment_state.py` | 0 | pending |
| `tests/unit/plugins/files/registry-resolve.test.ts` | 32 | `tests/unit/plugins/files/test_registry_resolve.py` | 0 | pending |
| `tests/unit/plugins/files/registry.test.ts` | 7 | `tests/unit/plugins/files/test_registry.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/builtin.test.ts` | 49 | `tests/unit/plugins/internal_tools/test_builtin.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/define.test.ts` | 57 | `tests/unit/plugins/internal_tools/test_define.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/id.test.ts` | 15 | `tests/unit/plugins/internal_tools/test_id.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/json-enforcement.test.ts` | 5 | `tests/unit/plugins/internal_tools/test_json_enforcement.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/local-backend.test.ts` | 15 | `tests/unit/plugins/internal_tools/test_local_backend.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/registry.test.ts` | 40 | `tests/unit/plugins/internal_tools/test_registry.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/runner.test.ts` | 68 | `tests/unit/plugins/internal_tools/test_runner.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/template.test.ts` | 51 | `tests/unit/plugins/internal_tools/test_template.py` | 0 | pending |
| `tests/unit/plugins/internal-tools/variants.test.ts` | 18 | `tests/unit/plugins/internal_tools/test_variants.py` | 0 | pending |
| `tests/unit/plugins/logger/logger-hooks.test.ts` | 19 | `tests/unit/plugins/logger/test_logger_hooks.py` | 0 | pending |
| `tests/unit/plugins/logger/logger.test.ts` | 20 | `tests/unit/plugins/logger/test_logger.py` | 0 | pending |
| `tests/unit/plugins/mcp/base-transport.test.ts` | 23 | `tests/unit/plugins/mcp/test_base_transport.py` | 0 | pending |
| `tests/unit/plugins/mcp/client-api.test.ts` | 29 | `tests/unit/plugins/mcp/test_client_api.py` | 0 | pending |
| `tests/unit/plugins/mcp/client.test.ts` | 7 | `tests/unit/plugins/mcp/test_client.py` | 0 | pending |
| `tests/unit/plugins/mcp/hardening.test.ts` | 12 | `tests/unit/plugins/mcp/test_hardening.py` | 0 | pending |
| `tests/unit/plugins/mcp/input-required.test.ts` | 7 | `tests/unit/plugins/mcp/test_input_required.py` | 0 | pending |
| `tests/unit/plugins/mcp/listen-http.test.ts` | 10 | `tests/unit/plugins/mcp/test_listen_http.py` | 0 | pending |
| `tests/unit/plugins/mcp/mcp-agent-trace.test.ts` | 9 | `tests/unit/plugins/mcp/test_mcp_agent_trace.py` | 0 | pending |
| `tests/unit/plugins/mcp/oauth-flow.test.ts` | 35 | `tests/unit/plugins/mcp/test_oauth_flow.py` | 0 | pending |
| `tests/unit/plugins/mcp/oauth.test.ts` | 8 | `tests/unit/plugins/mcp/test_oauth.py` | 0 | pending |
| `tests/unit/plugins/mcp/protocol-negotiation.test.ts` | 24 | `tests/unit/plugins/mcp/test_protocol_negotiation.py` | 0 | pending |
| `tests/unit/plugins/mcp/sampling.test.ts` | 10 | `tests/unit/plugins/mcp/test_sampling.py` | 0 | pending |
| `tests/unit/plugins/mcp/subscriptions-cache.test.ts` | 17 | `tests/unit/plugins/mcp/test_subscriptions_cache.py` | 0 | pending |
| `tests/unit/plugins/mcp/tools-mapping.test.ts` | 19 | `tests/unit/plugins/mcp/test_tools_mapping.py` | 0 | pending |
| `tests/unit/plugins/mcp/tools.test.ts` | 8 | `tests/unit/plugins/mcp/test_tools.py` | 0 | pending |
| `tests/unit/plugins/mcp/transport-guard.test.ts` | 1 | `tests/unit/plugins/mcp/test_transport_guard.py` | 0 | pending |
| `tests/unit/plugins/mcp/transport-http-message.test.ts` | 3 | `tests/unit/plugins/mcp/test_transport_http_message.py` | 0 | pending |
| `tests/unit/plugins/mcp/transport-stdio.test.ts` | 30 | `tests/unit/plugins/mcp/test_transport_stdio.py` | 0 | pending |
| `tests/unit/plugins/mcp/transport-ws.test.ts` | 21 | `tests/unit/plugins/mcp/test_transport_ws.py` | 0 | pending |
| `tests/unit/plugins/mcp/url-guard.test.ts` | 76 | `tests/unit/plugins/mcp/test_url_guard.py` | 0 | pending |
| `tests/unit/plugins/mcp/win-spawn.test.ts` | 6 | `tests/unit/plugins/mcp/test_win_spawn.py` | 0 | pending |
| `tests/unit/plugins/media/file-store.test.ts` | 29 | `tests/unit/plugins/media/test_file_store.py` | 0 | pending |
| `tests/unit/plugins/media/output-generation.test.ts` | 25 | `tests/unit/plugins/media/test_output_generation.py` | 0 | pending |
| `tests/unit/plugins/media/output-video.test.ts` | 17 | `tests/unit/plugins/media/test_output_video.py` | 0 | pending |
| `tests/unit/plugins/media/output.test.ts` | 6 | `tests/unit/plugins/media/test_output.py` | 0 | pending |
| `tests/unit/plugins/media/source-image.test.ts` | 10 | `tests/unit/plugins/media/test_source_image.py` | 0 | pending |
| `tests/unit/plugins/media/store.test.ts` | 4 | `tests/unit/plugins/media/test_store.py` | 0 | pending |
| `tests/unit/plugins/model-catalog/catalog.test.ts` | 10 | `tests/unit/plugins/model_catalog/test_catalog.py` | 0 | pending |
| `tests/unit/plugins/permissions/matchers.test.ts` | 26 | `tests/unit/plugins/permissions/test_matchers.py` | 0 | pending |
| `tests/unit/plugins/permissions/policy.test.ts` | 14 | `tests/unit/plugins/permissions/test_policy.py` | 0 | pending |
| `tests/unit/plugins/persistence/file.test.ts` | 11 | `tests/unit/plugins/persistence/test_file.py` | 0 | pending |
| `tests/unit/plugins/persistence/memory.test.ts` | 12 | `tests/unit/plugins/persistence/test_memory.py` | 0 | pending |
| `tests/unit/plugins/retrieval/chunker.test.ts` | 24 | `tests/unit/plugins/retrieval/test_chunker.py` | 0 | pending |
| `tests/unit/plugins/retrieval/factories.test.ts` | 9 | `tests/unit/plugins/retrieval/test_factories.py` | 0 | pending |
| `tests/unit/plugins/retrieval/hosted-google.test.ts` | 33 | `tests/unit/plugins/retrieval/test_hosted_google.py` | 0 | pending |
| `tests/unit/plugins/retrieval/hosted-openai.test.ts` | 23 | `tests/unit/plugins/retrieval/test_hosted_openai.py` | 0 | pending |
| `tests/unit/plugins/retrieval/hosted-xai.test.ts` | 37 | `tests/unit/plugins/retrieval/test_hosted_xai.py` | 0 | pending |
| `tests/unit/plugins/retrieval/local.test.ts` | 16 | `tests/unit/plugins/retrieval/test_local.py` | 0 | pending |
| `tests/unit/plugins/scheduler/scheduler-at.test.ts` | 17 | `tests/unit/plugins/scheduler/test_scheduler_at.py` | 0 | pending |
| `tests/unit/plugins/scheduler/scheduler.test.ts` | 7 | `tests/unit/plugins/scheduler/test_scheduler.py` | 0 | pending |
| `tests/unit/plugins/telemetry/gauges.test.ts` | 5 | `tests/unit/plugins/telemetry/test_gauges.py` | 0 | pending |
| `tests/unit/plugins/telemetry/otlp-conformance.test.ts` | 13 | `tests/unit/plugins/telemetry/test_otlp_conformance.py` | 0 | pending |
| `tests/unit/plugins/telemetry/parent-context.test.ts` | 9 | `tests/unit/plugins/telemetry/test_parent_context.py` | 0 | pending |
| `tests/unit/plugins/telemetry/semconv-naming.test.ts` | 5 | `tests/unit/plugins/telemetry/test_semconv_naming.py` | 0 | pending |
| `tests/unit/plugins/telemetry/telemetry.test.ts` | 46 | `tests/unit/plugins/telemetry/test_telemetry.py` | 0 | pending |
| `tests/unit/plugins/telemetry/trace-events.test.ts` | 15 | `tests/unit/plugins/telemetry/test_trace_events.py` | 0 | pending |
| `tests/unit/plugins/tool-catalog/catalog.test.ts` | 14 | `tests/unit/plugins/tool_catalog/test_catalog.py` | 0 | pending |
| `tests/unit/plugins/tool-catalog/scope-and-events.test.ts` | 25 | `tests/unit/plugins/tool_catalog/test_scope_and_events.py` | 0 | pending |
| `tests/unit/runtime/runtime.test.ts` | 8 | `tests/unit/runtime/test_runtime.py` | 0 | pending |
| `tests/unit/server/auth.test.ts` | 5 | `tests/unit/server/test_auth.py` | 0 | pending |
| `tests/unit/server/dispatch.test.ts` | 13 | `tests/unit/server/test_dispatch.py` | 0 | pending |
| `tests/unit/server/oai-adapter.test.ts` | 22 | `tests/unit/server/test_oai_adapter.py` | 0 | pending |
| `tests/unit/server/response-store.test.ts` | 16 | `tests/unit/server/test_response_store.py` | 0 | pending |
| `tests/unit/server/server-lifecycle.test.ts` | 6 | `tests/unit/server/test_server_lifecycle.py` | 0 | pending |
| `tests/unit/server/server-plugins.test.ts` | 12 | `tests/unit/server/test_server_plugins.py` | 0 | pending |
| `tests/unit/server/server.test.ts` | 11 | `tests/unit/server/test_server.py` | 0 | pending |
| `tests/unit/util/base64-http.test.ts` | 11 | `tests/unit/util/test_base64_http.py` | 6 | started |
| `tests/unit/util/duration.test.ts` | 4 | `tests/unit/util/test_duration.py` | 0 | pending |
| `tests/unit/util/hash.test.ts` | 3 | `tests/unit/util/test_hash.py` | 4 | done |
| `tests/unit/util/image-mime.test.ts` | 4 | `tests/unit/util/test_image_mime.py` | 4 | done |
| `tests/unit/util/json-schema.test.ts` | 13 | `tests/unit/util/test_json_schema.py` | 0 | pending |
| `tests/unit/util/source-image.test.ts` | 22 | `tests/unit/util/test_source_image.py` | 22 | done |
| `tests/unit/util/wav.test.ts` | 7 | `tests/unit/util/test_wav.py` | 0 | pending |
| `tests/unit/wire/catalog-pins.test.ts` | 7 | `tests/unit/wire/test_catalog_pins.py` | 0 | pending |
| `tests/unit/wire/chain-deltas.test.ts` | 6 | `tests/unit/wire/test_chain_deltas.py` | 0 | pending |
| `tests/unit/wire/every-model-reproduces-its-adapter.test.ts` | 2 | `tests/unit/wire/test_every_model_reproduces_its_adapter.py` | 0 | pending |
| `tests/unit/wire/generation-coverage.test.ts` | 5 | `tests/unit/wire/test_generation_coverage.py` | 0 | pending |
| `tests/unit/wire/golden-corpus.test.ts` | 5 | `tests/unit/wire/test_golden_corpus.py` | 0 | pending |
| `tests/unit/wire/interpreter-primitives.test.ts` | 21 | `tests/unit/wire/test_interpreter_primitives.py` | 21 | done |
| `tests/unit/wire/mcp-differential.test.ts` | 2 | `tests/unit/wire/test_mcp_differential.py` | 0 | pending |
| `tests/unit/wire/mcp-oauth-differential.test.ts` | 2 | `tests/unit/wire/test_mcp_oauth_differential.py` | 0 | pending |
| `tests/unit/wire/media-differential.test.ts` | 3 | `tests/unit/wire/test_media_differential.py` | 0 | pending |
| `tests/unit/wire/no-credentials-in-urls.test.ts` | 3 | `tests/unit/wire/test_no_credentials_in_urls.py` | 0 | pending |
| `tests/unit/wire/retrieval-differential.test.ts` | 2 | `tests/unit/wire/test_retrieval_differential.py` | 0 | pending |
| `tests/unit/wire/service-differential.test.ts` | 9 | `tests/unit/wire/test_service_differential.py` | 0 | pending |
| `tests/unit/wire/spec-differential.test.ts` | 3 | `tests/unit/wire/test_spec_differential.py` | 0 | pending |
| `tests/unit/wire/tool-constraints.test.ts` | 7 | `tests/unit/wire/test_tool_constraints.py` | 0 | pending |
| `tests/unit/wire/utility-differential.test.ts` | 3 | `tests/unit/wire/test_utility_differential.py` | 0 | pending |

## Source files

Type-only files carry no runtime behaviour; in Python their content becomes
dataclasses and Protocols wherever the module that uses them needs it, so they
are marked rather than given a target of their own.

| TS source | lines | PY target | status |
|---|---|---|---|
| `src/agent/approval-types.ts` | 53 | — | type-only |
| `src/agent/context-registry/layers.ts` | 119 | `src/combycode_llm_sdk/agent/context_registry/layers.py` | pending |
| `src/agent/context-registry/registry-internal.ts` | 61 | `src/combycode_llm_sdk/agent/context_registry/registry_internal.py` | pending |
| `src/agent/context-registry/registry.ts` | 457 | `src/combycode_llm_sdk/agent/context_registry/registry.py` | pending |
| `src/agent/context-registry/types.ts` | 120 | — | type-only |
| `src/agent/guardrail-types.ts` | 102 | — | type-only |
| `src/agent/history-types.ts` | 44 | — | type-only |
| `src/agent/history.ts` | 398 | `src/combycode_llm_sdk/agent/history.py` | pending |
| `src/agent/lazy-tools.ts` | 238 | `src/combycode_llm_sdk/agent/lazy_tools.py` | pending |
| `src/agent/loop-config.ts` | 144 | — | type-only |
| `src/agent/loop-internals.ts` | 339 | `src/combycode_llm_sdk/agent/loop_internals.py` | pending |
| `src/agent/loop-step-state.ts` | 29 | — | type-only |
| `src/agent/loop.ts` | 1661 | `src/combycode_llm_sdk/agent/loop.py` | pending |
| `src/agent/reflect-retry.ts` | 103 | `src/combycode_llm_sdk/agent/reflect_retry.py` | pending |
| `src/agent/tool-key.ts` | 16 | `src/combycode_llm_sdk/agent/tool_key.py` | pending |
| `src/agent/types.ts` | 176 | — | type-only |
| `src/bus/agent-bus.ts` | 307 | `src/combycode_llm_sdk/bus/agent_bus.py` | pending |
| `src/bus/async-context.browser.ts` | 40 | `src/combycode_llm_sdk/bus/async_context_browser.py` | pending |
| `src/bus/async-context.ts` | 14 | `src/combycode_llm_sdk/bus/async_context.py` | pending |
| `src/bus/async-context.types.ts` | 14 | — | type-only |
| `src/bus/hook-bus.ts` | 132 | `src/combycode_llm_sdk/bus/hook_bus.py` | pending |
| `src/bus/hook-map.ts` | 742 | — | type-only |
| `src/catalog/builtin-tools.ts` | 26 | `src/combycode_llm_sdk/catalog/builtin_tools.py` | pending |
| `src/catalog/catalog.ts` | 426 | `src/combycode_llm_sdk/catalog/catalog.py` | pending |
| `src/helpers/agent.ts` | 62 | `src/combycode_llm_sdk/helpers/agent.py` | pending |
| `src/helpers/batch.ts` | 334 | `src/combycode_llm_sdk/helpers/batch.py` | pending |
| `src/helpers/calibration-store.ts` | 145 | `src/combycode_llm_sdk/helpers/calibration_store.py` | pending |
| `src/helpers/calibration-types.ts` | 99 | `src/combycode_llm_sdk/helpers/calibration_types.py` | pending |
| `src/helpers/chain.ts` | 52 | `src/combycode_llm_sdk/helpers/chain.py` | pending |
| `src/helpers/client-pool.ts` | 42 | `src/combycode_llm_sdk/helpers/client_pool.py` | pending |
| `src/helpers/client-resolver.ts` | 182 | `src/combycode_llm_sdk/helpers/client_resolver.py` | pending |
| `src/helpers/collection.ts` | 71 | `src/combycode_llm_sdk/helpers/collection.py` | pending |
| `src/helpers/consolidate.ts` | 188 | `src/combycode_llm_sdk/helpers/consolidate.py` | pending |
| `src/helpers/content.ts` | 151 | `src/combycode_llm_sdk/helpers/content.py` | pending |
| `src/helpers/conversation-export.ts` | 83 | `src/combycode_llm_sdk/helpers/conversation_export.py` | pending |
| `src/helpers/conversation-zip.ts` | 267 | `src/combycode_llm_sdk/helpers/conversation_zip.py` | pending |
| `src/helpers/count-tokens.ts` | 96 | `src/combycode_llm_sdk/helpers/count_tokens.py` | pending |
| `src/helpers/define-tool.ts` | 136 | `src/combycode_llm_sdk/helpers/define_tool.py` | pending |
| `src/helpers/delegate.ts` | 16 | `src/combycode_llm_sdk/helpers/delegate.py` | pending |
| `src/helpers/embed.ts` | 95 | `src/combycode_llm_sdk/helpers/embed.py` | pending |
| `src/helpers/engine.ts` | 318 | `src/combycode_llm_sdk/helpers/engine.py` | pending |
| `src/helpers/estimate-types.ts` | 95 | `src/combycode_llm_sdk/helpers/estimate_types.py` | pending |
| `src/helpers/estimate.ts` | 264 | `src/combycode_llm_sdk/helpers/estimate.py` | pending |
| `src/helpers/estimator.ts` | 201 | `src/combycode_llm_sdk/helpers/estimator.py` | pending |
| `src/helpers/handoff-types.ts` | 21 | — | type-only |
| `src/helpers/handoff.ts` | 42 | `src/combycode_llm_sdk/helpers/handoff.py` | pending |
| `src/helpers/llm.ts` | 96 | `src/combycode_llm_sdk/helpers/llm.py` | pending |
| `src/helpers/mcp.ts` | 286 | `src/combycode_llm_sdk/helpers/mcp.py` | pending |
| `src/helpers/media.ts` | 192 | `src/combycode_llm_sdk/helpers/media.py` | pending |
| `src/helpers/models.ts` | 228 | `src/combycode_llm_sdk/helpers/models.py` | pending |
| `src/helpers/moderate-types.ts` | 85 | — | type-only |
| `src/helpers/moderate.ts` | 138 | `src/combycode_llm_sdk/helpers/moderate.py` | pending |
| `src/helpers/moderation-guardrail.ts` | 119 | `src/combycode_llm_sdk/helpers/moderation_guardrail.py` | pending |
| `src/helpers/observer.ts` | 109 | `src/combycode_llm_sdk/helpers/observer.py` | pending |
| `src/helpers/one-shot.ts` | 319 | `src/combycode_llm_sdk/helpers/one_shot.py` | pending |
| `src/helpers/parallel.ts` | 38 | `src/combycode_llm_sdk/helpers/parallel.py` | pending |
| `src/helpers/provenance-types.ts` | 75 | — | type-only |
| `src/helpers/provenance.ts` | 74 | `src/combycode_llm_sdk/helpers/provenance.py` | pending |
| `src/helpers/realtime.ts` | 112 | `src/combycode_llm_sdk/helpers/realtime.py` | pending |
| `src/helpers/route.ts` | 99 | `src/combycode_llm_sdk/helpers/route.py` | pending |
| `src/helpers/select-model.ts` | 268 | `src/combycode_llm_sdk/helpers/select_model.py` | pending |
| `src/helpers/server.ts` | 119 | `src/combycode_llm_sdk/helpers/server.py` | pending |
| `src/helpers/transcribe.ts` | 263 | `src/combycode_llm_sdk/helpers/transcribe.py` | pending |
| `src/index.ts` | 431 | — | type-only |
| `src/llm/audio/voices.ts` | 20 | `src/combycode_llm_sdk/llm/audio/voices.py` | ported |
| `src/llm/client-config.ts` | 63 | — | type-only |
| `src/llm/client-internal.ts` | 152 | `src/combycode_llm_sdk/llm/client_internal.py` | pending |
| `src/llm/client.ts` | 729 | `src/combycode_llm_sdk/llm/client.py` | pending |
| `src/llm/files/retrieve.ts` | 215 | `src/combycode_llm_sdk/llm/files/retrieve.py` | pending |
| `src/llm/moderation/native.ts` | 75 | `src/combycode_llm_sdk/llm/moderation/native.py` | ported |
| `src/llm/moderation/runner.ts` | 196 | `src/combycode_llm_sdk/llm/moderation/runner.py` | pending |
| `src/llm/moderation/types.ts` | 81 | `src/combycode_llm_sdk/llm/moderation/types.py` | ported |
| `src/llm/output-errors.ts` | 36 | `src/combycode_llm_sdk/llm/output_errors.py` | pending |
| `src/llm/providers/_shared/builtin-tools.ts` | 24 | `src/combycode_llm_sdk/llm/providers/_shared/builtin_tools.py` | pending |
| `src/llm/providers/_shared/citations.ts` | 126 | `src/combycode_llm_sdk/llm/providers/_shared/citations.py` | pending |
| `src/llm/providers/_shared/constants.ts` | 7 | `src/combycode_llm_sdk/llm/providers/_shared/constants.py` | pending |
| `src/llm/providers/_shared/response-utils.ts` | 17 | `src/combycode_llm_sdk/llm/providers/_shared/response_utils.py` | pending |
| `src/llm/providers/_shared/sse.ts` | 23 | `src/combycode_llm_sdk/llm/providers/_shared/sse.py` | pending |
| `src/llm/providers/anthropic/batch.ts` | 134 | `src/combycode_llm_sdk/llm/providers/anthropic/batch.py` | pending |
| `src/llm/providers/anthropic/constants.ts` | 11 | `src/combycode_llm_sdk/llm/providers/anthropic/constants.py` | pending |
| `src/llm/providers/anthropic/files.ts` | 122 | `src/combycode_llm_sdk/llm/providers/anthropic/files.py` | pending |
| `src/llm/providers/anthropic/messages.ts` | 479 | `src/combycode_llm_sdk/llm/providers/anthropic/messages.py` | pending |
| `src/llm/providers/google/batch.ts` | 169 | `src/combycode_llm_sdk/llm/providers/google/batch.py` | pending |
| `src/llm/providers/google/constants.ts` | 18 | `src/combycode_llm_sdk/llm/providers/google/constants.py` | pending |
| `src/llm/providers/google/embeddings.ts` | 64 | `src/combycode_llm_sdk/llm/providers/google/embeddings.py` | pending |
| `src/llm/providers/google/files.ts` | 193 | `src/combycode_llm_sdk/llm/providers/google/files.py` | pending |
| `src/llm/providers/google/generate.ts` | 495 | `src/combycode_llm_sdk/llm/providers/google/generate.py` | pending |
| `src/llm/providers/google/interactions.ts` | 377 | `src/combycode_llm_sdk/llm/providers/google/interactions.py` | pending |
| `src/llm/providers/google/media.ts` | 318 | `src/combycode_llm_sdk/llm/providers/google/media.py` | pending |
| `src/llm/providers/google/realtime.ts` | 183 | `src/combycode_llm_sdk/llm/providers/google/realtime.py` | pending |
| `src/llm/providers/google/tiers.ts` | 24 | `src/combycode_llm_sdk/llm/providers/google/tiers.py` | ported |
| `src/llm/providers/openai/batch.ts` | 183 | `src/combycode_llm_sdk/llm/providers/openai/batch.py` | pending |
| `src/llm/providers/openai/completions.ts` | 405 | `src/combycode_llm_sdk/llm/providers/openai/completions.py` | pending |
| `src/llm/providers/openai/embeddings.ts` | 79 | `src/combycode_llm_sdk/llm/providers/openai/embeddings.py` | pending |
| `src/llm/providers/openai/files.ts` | 123 | `src/combycode_llm_sdk/llm/providers/openai/files.py` | pending |
| `src/llm/providers/openai/media.ts` | 220 | `src/combycode_llm_sdk/llm/providers/openai/media.py` | pending |
| `src/llm/providers/openai/moderations.ts` | 95 | `src/combycode_llm_sdk/llm/providers/openai/moderations.py` | pending |
| `src/llm/providers/openai/provenance.ts` | 89 | `src/combycode_llm_sdk/llm/providers/openai/provenance.py` | pending |
| `src/llm/providers/openai/realtime.ts` | 199 | `src/combycode_llm_sdk/llm/providers/openai/realtime.py` | pending |
| `src/llm/providers/openai/responses.ts` | 756 | `src/combycode_llm_sdk/llm/providers/openai/responses.py` | pending |
| `src/llm/providers/openai/tiers.ts` | 37 | `src/combycode_llm_sdk/llm/providers/openai/tiers.py` | ported |
| `src/llm/providers/openai/transcription.ts` | 182 | `src/combycode_llm_sdk/llm/providers/openai/transcription.py` | pending |
| `src/llm/providers/openrouter/completions.ts` | 77 | `src/combycode_llm_sdk/llm/providers/openrouter/completions.py` | pending |
| `src/llm/providers/openrouter/embeddings.ts` | 21 | `src/combycode_llm_sdk/llm/providers/openrouter/embeddings.py` | pending |
| `src/llm/providers/openrouter/media.ts` | 152 | `src/combycode_llm_sdk/llm/providers/openrouter/media.py` | pending |
| `src/llm/providers/openrouter/responses.ts` | 32 | `src/combycode_llm_sdk/llm/providers/openrouter/responses.py` | pending |
| `src/llm/providers/xai/batch.ts` | 136 | `src/combycode_llm_sdk/llm/providers/xai/batch.py` | pending |
| `src/llm/providers/xai/completions.ts` | 69 | `src/combycode_llm_sdk/llm/providers/xai/completions.py` | pending |
| `src/llm/providers/xai/files.ts` | 126 | `src/combycode_llm_sdk/llm/providers/xai/files.py` | pending |
| `src/llm/providers/xai/media.ts` | 249 | `src/combycode_llm_sdk/llm/providers/xai/media.py` | pending |
| `src/llm/providers/xai/responses.ts` | 68 | `src/combycode_llm_sdk/llm/providers/xai/responses.py` | pending |
| `src/llm/providers/xai/tiers.ts` | 20 | `src/combycode_llm_sdk/llm/providers/xai/tiers.py` | ported |
| `src/llm/realtime/session.ts` | 80 | `src/combycode_llm_sdk/llm/realtime/session.py` | pending |
| `src/llm/realtime/types.ts` | 60 | — | type-only |
| `src/llm/response-shape.ts` | 233 | `src/combycode_llm_sdk/llm/response_shape.py` | pending |
| `src/llm/server-state.ts` | 69 | `src/combycode_llm_sdk/llm/server_state.py` | pending |
| `src/llm/types/audio.ts` | 55 | — | type-only |
| `src/llm/types/messages.ts` | 254 | `src/combycode_llm_sdk/llm/types/messages.py` | pending |
| `src/llm/types/options.ts` | 107 | — | type-only |
| `src/llm/types/provider.ts` | 76 | `src/combycode_llm_sdk/llm/types/provider.py` | pending |
| `src/llm/types/request.ts` | 178 | — | type-only |
| `src/llm/types/response.ts` | 177 | `src/combycode_llm_sdk/llm/types/response.py` | pending |
| `src/llm/types/schema-utils.ts` | 137 | `src/combycode_llm_sdk/llm/types/schema_utils.py` | ported |
| `src/llm/types/stream.ts` | 58 | — | type-only |
| `src/llm/types/tiers.ts` | 11 | — | type-only |
| `src/llm/types/tools.ts` | 66 | `src/combycode_llm_sdk/llm/types/tools.py` | pending |
| `src/llm/wire-multipart.ts` | 39 | `src/combycode_llm_sdk/llm/wire_multipart.py` | ported |
| `src/llm/wire-transforms.ts` | 305 | `src/combycode_llm_sdk/llm/wire_transforms.py` | ported |
| `src/network/engine.ts` | 257 | `src/combycode_llm_sdk/network/engine.py` | pending |
| `src/network/errors.ts` | 112 | `src/combycode_llm_sdk/network/errors.py` | pending |
| `src/network/queue-state-config.ts` | 89 | `src/combycode_llm_sdk/network/queue_state_config.py` | pending |
| `src/network/queue-state.ts` | 567 | `src/combycode_llm_sdk/network/queue_state.py` | pending |
| `src/network/rate-limiter.ts` | 152 | `src/combycode_llm_sdk/network/rate_limiter.py` | pending |
| `src/network/realtime-connection.ts` | 163 | `src/combycode_llm_sdk/network/realtime_connection.py` | pending |
| `src/network/request-queue.ts` | 161 | `src/combycode_llm_sdk/network/request_queue.py` | pending |
| `src/network/semaphore.ts` | 35 | `src/combycode_llm_sdk/network/semaphore.py` | pending |
| `src/network/sse.ts` | 70 | `src/combycode_llm_sdk/network/sse.py` | pending |
| `src/network/types.ts` | 175 | — | type-only |
| `src/plugins/batch/batcher.ts` | 297 | `src/combycode_llm_sdk/plugins/batch/batcher.py` | pending |
| `src/plugins/batch/strategy.ts` | 39 | `src/combycode_llm_sdk/plugins/batch/strategy.py` | pending |
| `src/plugins/batch/types.ts` | 61 | — | type-only |
| `src/plugins/cache/cache.ts` | 129 | `src/combycode_llm_sdk/plugins/cache/cache.py` | pending |
| `src/plugins/cache/file-store.ts` | 38 | `src/combycode_llm_sdk/plugins/cache/file_store.py` | pending |
| `src/plugins/cache/memory-store.ts` | 28 | `src/combycode_llm_sdk/plugins/cache/memory_store.py` | pending |
| `src/plugins/cache/types.ts` | 30 | — | type-only |
| `src/plugins/configuration/configuration.ts` | 178 | `src/combycode_llm_sdk/plugins/configuration/configuration.py` | pending |
| `src/plugins/context-guard/facts.ts` | 30 | `src/combycode_llm_sdk/plugins/context_guard/facts.py` | pending |
| `src/plugins/context-guard/guard.ts` | 298 | `src/combycode_llm_sdk/plugins/context_guard/guard.py` | pending |
| `src/plugins/context-guard/strategies/anchored.ts` | 158 | `src/combycode_llm_sdk/plugins/context_guard/strategies/anchored.py` | pending |
| `src/plugins/context-guard/strategies/layered.ts` | 193 | `src/combycode_llm_sdk/plugins/context_guard/strategies/layered.py` | pending |
| `src/plugins/context-guard/strategies/truncate.ts` | 57 | `src/combycode_llm_sdk/plugins/context_guard/strategies/truncate.py` | pending |
| `src/plugins/context-guard/tools.ts` | 314 | `src/combycode_llm_sdk/plugins/context_guard/tools.py` | pending |
| `src/plugins/context-guard/types.ts` | 135 | `src/combycode_llm_sdk/plugins/context_guard/types.py` | pending |
| `src/plugins/context-measurer/calibration/store.ts` | 79 | `src/combycode_llm_sdk/plugins/context_measurer/calibration/store.py` | pending |
| `src/plugins/context-measurer/counter/count-api.ts` | 201 | `src/combycode_llm_sdk/plugins/context_measurer/counter/count_api.py` | pending |
| `src/plugins/context-measurer/counter/heuristic.ts` | 135 | `src/combycode_llm_sdk/plugins/context_measurer/counter/heuristic.py` | pending |
| `src/plugins/context-measurer/counter/hybrid.ts` | 164 | `src/combycode_llm_sdk/plugins/context_measurer/counter/hybrid.py` | pending |
| `src/plugins/context-measurer/counter/tiktoken.ts` | 153 | `src/combycode_llm_sdk/plugins/context_measurer/counter/tiktoken.py` | pending |
| `src/plugins/context-measurer/measurer.ts` | 204 | `src/combycode_llm_sdk/plugins/context_measurer/measurer.py` | pending |
| `src/plugins/context-measurer/types.ts` | 48 | `src/combycode_llm_sdk/plugins/context_measurer/types.py` | pending |
| `src/plugins/cost-collector/collector.ts` | 340 | `src/combycode_llm_sdk/plugins/cost_collector/collector.py` | pending |
| `src/plugins/cost-collector/cost-collector-internal.ts` | 292 | `src/combycode_llm_sdk/plugins/cost_collector/cost_collector_internal.py` | pending |
| `src/plugins/cost-collector/cost-collector-types.ts` | 55 | — | type-only |
| `src/plugins/embeddings/types.ts` | 25 | — | type-only |
| `src/plugins/files/attachment.ts` | 171 | `src/combycode_llm_sdk/plugins/files/attachment.py` | pending |
| `src/plugins/files/provider-adapter.ts` | 48 | — | type-only |
| `src/plugins/files/registry.ts` | 297 | `src/combycode_llm_sdk/plugins/files/registry.py` | pending |
| `src/plugins/files/strategy.ts` | 65 | `src/combycode_llm_sdk/plugins/files/strategy.py` | pending |
| `src/plugins/internal-tools/backends/local.ts` | 37 | `src/combycode_llm_sdk/plugins/internal_tools/backends/local.py` | pending |
| `src/plugins/internal-tools/builtin/builtin.ts` | 40 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/builtin.py` | pending |
| `src/plugins/internal-tools/builtin/clarify.ts` | 68 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/clarify.py` | pending |
| `src/plugins/internal-tools/builtin/classify.ts` | 64 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/classify.py` | pending |
| `src/plugins/internal-tools/builtin/score.ts` | 78 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/score.py` | pending |
| `src/plugins/internal-tools/builtin/structure.ts` | 52 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/structure.py` | pending |
| `src/plugins/internal-tools/builtin/summarize.ts` | 158 | `src/combycode_llm_sdk/plugins/internal_tools/builtin/summarize.py` | pending |
| `src/plugins/internal-tools/id.ts` | 48 | `src/combycode_llm_sdk/plugins/internal_tools/id.py` | pending |
| `src/plugins/internal-tools/registry.ts` | 124 | `src/combycode_llm_sdk/plugins/internal_tools/registry.py` | pending |
| `src/plugins/internal-tools/runner/define.ts` | 169 | `src/combycode_llm_sdk/plugins/internal_tools/runner/define.py` | pending |
| `src/plugins/internal-tools/runner/json-enforcement.ts` | 35 | `src/combycode_llm_sdk/plugins/internal_tools/runner/json_enforcement.py` | pending |
| `src/plugins/internal-tools/runner/runner.ts` | 317 | `src/combycode_llm_sdk/plugins/internal_tools/runner/runner.py` | pending |
| `src/plugins/internal-tools/runner/template.ts` | 113 | `src/combycode_llm_sdk/plugins/internal_tools/runner/template.py` | pending |
| `src/plugins/internal-tools/runner/types.ts` | 72 | — | type-only |
| `src/plugins/internal-tools/runner/variants.ts` | 75 | `src/combycode_llm_sdk/plugins/internal_tools/runner/variants.py` | pending |
| `src/plugins/internal-tools/types.ts` | 100 | — | type-only |
| `src/plugins/logger/console-sink.ts` | 65 | `src/combycode_llm_sdk/plugins/logger/console_sink.py` | pending |
| `src/plugins/logger/logger.ts` | 221 | `src/combycode_llm_sdk/plugins/logger/logger.py` | pending |
| `src/plugins/logger/types.ts` | 36 | `src/combycode_llm_sdk/plugins/logger/types.py` | pending |
| `src/plugins/mcp/base-transport.ts` | 165 | `src/combycode_llm_sdk/plugins/mcp/base_transport.py` | pending |
| `src/plugins/mcp/client.ts` | 747 | `src/combycode_llm_sdk/plugins/mcp/client.py` | pending |
| `src/plugins/mcp/input-required.ts` | 100 | `src/combycode_llm_sdk/plugins/mcp/input_required.py` | pending |
| `src/plugins/mcp/jsonrpc.ts` | 36 | `src/combycode_llm_sdk/plugins/mcp/jsonrpc.py` | pending |
| `src/plugins/mcp/oauth.ts` | 420 | `src/combycode_llm_sdk/plugins/mcp/oauth.py` | pending |
| `src/plugins/mcp/protocol-version.ts` | 109 | `src/combycode_llm_sdk/plugins/mcp/protocol_version.py` | pending |
| `src/plugins/mcp/result-cache.ts` | 83 | `src/combycode_llm_sdk/plugins/mcp/result_cache.py` | pending |
| `src/plugins/mcp/sampling.ts` | 86 | `src/combycode_llm_sdk/plugins/mcp/sampling.py` | pending |
| `src/plugins/mcp/subscriptions.ts` | 137 | `src/combycode_llm_sdk/plugins/mcp/subscriptions.py` | pending |
| `src/plugins/mcp/tools.ts` | 110 | `src/combycode_llm_sdk/plugins/mcp/tools.py` | pending |
| `src/plugins/mcp/transport-http.ts` | 382 | `src/combycode_llm_sdk/plugins/mcp/transport_http.py` | pending |
| `src/plugins/mcp/transport-stdio.ts` | 196 | `src/combycode_llm_sdk/plugins/mcp/transport_stdio.py` | pending |
| `src/plugins/mcp/transport-ws.ts` | 123 | `src/combycode_llm_sdk/plugins/mcp/transport_ws.py` | pending |
| `src/plugins/mcp/transport.ts` | 46 | — | type-only |
| `src/plugins/mcp/types.ts` | 330 | `src/combycode_llm_sdk/plugins/mcp/types.py` | pending |
| `src/plugins/mcp/url-guard.ts` | 291 | `src/combycode_llm_sdk/plugins/mcp/url_guard.py` | pending |
| `src/plugins/mcp/win-spawn.ts` | 80 | `src/combycode_llm_sdk/plugins/mcp/win_spawn.py` | pending |
| `src/plugins/mcp/wire-rules.ts` | 49 | `src/combycode_llm_sdk/plugins/mcp/wire_rules.py` | pending |
| `src/plugins/media/file-store.ts` | 140 | `src/combycode_llm_sdk/plugins/media/file_store.py` | pending |
| `src/plugins/media/memory-store.ts` | 38 | `src/combycode_llm_sdk/plugins/media/memory_store.py` | pending |
| `src/plugins/media/output.ts` | 311 | `src/combycode_llm_sdk/plugins/media/output.py` | pending |
| `src/plugins/media/types.ts` | 190 | `src/combycode_llm_sdk/plugins/media/types.py` | pending |
| `src/plugins/permissions/glob.ts` | 48 | `src/combycode_llm_sdk/plugins/permissions/glob.py` | pending |
| `src/plugins/permissions/matchers.ts` | 30 | `src/combycode_llm_sdk/plugins/permissions/matchers.py` | pending |
| `src/plugins/permissions/policy.ts` | 37 | `src/combycode_llm_sdk/plugins/permissions/policy.py` | pending |
| `src/plugins/permissions/types.ts` | 26 | — | type-only |
| `src/plugins/persistence/file.ts` | 99 | `src/combycode_llm_sdk/plugins/persistence/file.py` | pending |
| `src/plugins/persistence/memory.ts` | 57 | `src/combycode_llm_sdk/plugins/persistence/memory.py` | pending |
| `src/plugins/persistence/types.ts` | 25 | — | type-only |
| `src/plugins/retrieval/chunker.ts` | 106 | `src/combycode_llm_sdk/plugins/retrieval/chunker.py` | pending |
| `src/plugins/retrieval/document-file.ts` | 24 | `src/combycode_llm_sdk/plugins/retrieval/document_file.py` | pending |
| `src/plugins/retrieval/hosted-google.ts` | 318 | `src/combycode_llm_sdk/plugins/retrieval/hosted_google.py` | pending |
| `src/plugins/retrieval/hosted-openai.ts` | 246 | `src/combycode_llm_sdk/plugins/retrieval/hosted_openai.py` | pending |
| `src/plugins/retrieval/hosted-xai.ts` | 331 | `src/combycode_llm_sdk/plugins/retrieval/hosted_xai.py` | pending |
| `src/plugins/retrieval/index.ts` | 124 | `src/combycode_llm_sdk/plugins/retrieval/index.py` | pending |
| `src/plugins/retrieval/local.ts` | 246 | `src/combycode_llm_sdk/plugins/retrieval/local.py` | pending |
| `src/plugins/retrieval/types.ts` | 206 | — | type-only |
| `src/plugins/retrieval/vector-store.ts` | 169 | `src/combycode_llm_sdk/plugins/retrieval/vector_store.py` | pending |
| `src/plugins/scheduler/scheduler.ts` | 181 | `src/combycode_llm_sdk/plugins/scheduler/scheduler.py` | pending |
| `src/plugins/telemetry/telemetry.ts` | 1128 | `src/combycode_llm_sdk/plugins/telemetry/telemetry.py` | pending |
| `src/plugins/telemetry/types.ts` | 156 | — | type-only |
| `src/plugins/tool-catalog/catalog.ts` | 245 | `src/combycode_llm_sdk/plugins/tool_catalog/catalog.py` | pending |
| `src/plugins/tool-catalog/errors.ts` | 51 | `src/combycode_llm_sdk/plugins/tool_catalog/errors.py` | pending |
| `src/plugins/tool-catalog/types.ts` | 53 | — | type-only |
| `src/runtime/runtime.ts` | 63 | `src/combycode_llm_sdk/runtime/runtime.py` | pending |
| `src/server/auth.ts` | 47 | `src/combycode_llm_sdk/server/auth.py` | pending |
| `src/server/dispatch.ts` | 126 | `src/combycode_llm_sdk/server/dispatch.py` | pending |
| `src/server/loaders.ts` | 31 | — | type-only |
| `src/server/oai-adapter.ts` | 145 | `src/combycode_llm_sdk/server/oai_adapter.py` | pending |
| `src/server/oai-types.ts` | 121 | — | type-only |
| `src/server/response-store.ts` | 181 | `src/combycode_llm_sdk/server/response_store.py` | pending |
| `src/server/router.ts` | 112 | `src/combycode_llm_sdk/server/router.py` | pending |
| `src/server/server.ts` | 330 | `src/combycode_llm_sdk/server/server.py` | pending |
| `src/types/request-context.ts` | 67 | — | type-only |
| `src/util/async.ts` | 6 | `src/combycode_llm_sdk/util/async.py` | pending |
| `src/util/base64.ts` | 23 | `src/combycode_llm_sdk/util/base64.py` | ported |
| `src/util/duration.ts` | 31 | `src/combycode_llm_sdk/util/duration.py` | pending |
| `src/util/hash.ts` | 18 | `src/combycode_llm_sdk/util/hash.py` | ported |
| `src/util/http.ts` | 57 | `src/combycode_llm_sdk/util/http.py` | pending |
| `src/util/image-mime.ts` | 34 | `src/combycode_llm_sdk/util/image_mime.py` | ported |
| `src/util/json-schema.ts` | 90 | `src/combycode_llm_sdk/util/json_schema.py` | pending |
| `src/util/source-image.ts` | 115 | `src/combycode_llm_sdk/util/source_image.py` | ported |
| `src/util/wav.ts` | 55 | `src/combycode_llm_sdk/util/wav.py` | pending |
| `src/wire/chat-specs.ts` | 72 | `src/combycode_llm_sdk/wire/chat_specs.py` | pending |
| `src/wire/inherit.ts` | 169 | `src/combycode_llm_sdk/wire/inherit.py` | ported |
| `src/wire/interpreter.ts` | 758 | `src/combycode_llm_sdk/wire/interpreter.py` | ported |
| `src/wire/mcp-specs.ts` | 58 | `src/combycode_llm_sdk/wire/mcp_specs.py` | pending |
| `src/wire/media-specs.ts` | 93 | `src/combycode_llm_sdk/wire/media_specs.py` | ported |
| `src/wire/pins.ts` | 58 | `src/combycode_llm_sdk/wire/pins.py` | pending |
| `src/wire/registry.ts` | 329 | `src/combycode_llm_sdk/wire/registry.py` | ported |
| `src/wire/retrieval-specs.ts` | 90 | `src/combycode_llm_sdk/wire/retrieval_specs.py` | pending |
| `src/wire/service-specs.ts` | 191 | `src/combycode_llm_sdk/wire/service_specs.py` | ported |
| `src/wire/utility-specs.ts` | 50 | `src/combycode_llm_sdk/wire/utility_specs.py` | pending |

## Examples — the API contract, reviewed, read-only during the port

| TS example | PY example | status |
|---|---|---|
| `browser/browser.spec.ts` | `examples/browser/browser_spec.py` | MISSING |
| `features-highlight/agent-guardrails-and-sampling.ts` | `examples/features-highlight/agent_guardrails_and_sampling.py` | present |
| `features-highlight/context-anchored-strategy.ts` | `examples/features-highlight/context_anchored_strategy.py` | present |
| `features-highlight/cost-unpriced-models.ts` | `examples/features-highlight/cost_unpriced_models.py` | present |
| `features-highlight/engine-retry-policy.ts` | `examples/features-highlight/engine_retry_policy.py` | present |
| `features-highlight/event-stream.ts` | `examples/features-highlight/event_stream.py` | present |
| `features-highlight/exact-token-count.ts` | `examples/features-highlight/exact_token_count.py` | present |
| `features-highlight/final-answer-phase.ts` | `examples/features-highlight/final_answer_phase.py` | present |
| `features-highlight/lazy-tools.ts` | `examples/features-highlight/lazy_tools.py` | present |
| `features-highlight/mcp-lazy-server.ts` | `examples/features-highlight/mcp_lazy_server.py` | present |
| `features-highlight/model-filters.ts` | `examples/features-highlight/model_filters.py` | present |
| `features-highlight/model-selector.ts` | `examples/features-highlight/model_selector.py` | present |
| `features-highlight/provenance-adapter.ts` | `examples/features-highlight/provenance_adapter.py` | present |
| `features-highlight/response-shape-check.ts` | `examples/features-highlight/response_shape_check.py` | present |
| `features-highlight/retrieve-output-file.ts` | `examples/features-highlight/retrieve_output_file.py` | present |
| `features-highlight/server-state-google-interactions.ts` | `examples/features-highlight/server_state_google_interactions.py` | present |
| `features-highlight/streamed-citations.ts` | `examples/features-highlight/streamed_citations.py` | present |
| `features-highlight/telemetry-redaction.ts` | `examples/features-highlight/telemetry_redaction.py` | present |
| `features-highlight/telemetry-traces.ts` | `examples/features-highlight/telemetry_traces.py` | present |
| `features-highlight/tool-call-attribution.ts` | `examples/features-highlight/tool_call_attribution.py` | present |
| `features-highlight/tool-optional-parameters.ts` | `examples/features-highlight/tool_optional_parameters.py` | present |
| `features-highlight/transcribe-structured.ts` | `examples/features-highlight/transcribe_structured.py` | present |
| `features-highlight/unified-names.ts` | `examples/features-highlight/unified_names.py` | present |
| `official-comparision/01-basic-completion.ts` | `examples/official-comparison/01_basic_completion.py` | present |
| `official-comparision/02-system-prompt.ts` | `examples/official-comparison/02_system_prompt.py` | present |
| `official-comparision/03-multi-turn.ts` | `examples/official-comparison/03_multi_turn.py` | present |
| `official-comparision/04-streaming.ts` | `examples/official-comparison/04_streaming.py` | present |
| `official-comparision/05-token-count.ts` | `examples/official-comparison/05_token_count.py` | present |
| `official-comparision/06-tool-call.ts` | `examples/official-comparison/06_tool_call.py` | present |
| `official-comparision/07-parallel-tools.ts` | `examples/official-comparison/07_parallel_tools.py` | present |
| `official-comparision/08-multistep-loop.ts` | `examples/official-comparison/08_multistep_loop.py` | present |
| `official-comparision/09-tool-runner.ts` | `examples/official-comparison/09_tool_runner.py` | present |
| `official-comparision/10-server-tools.ts` | `examples/official-comparison/10_server_tools.py` | present |
| `official-comparision/10b-code-exec-files.ts` | `examples/official-comparison/10b_code_exec_files.py` | present |
| `official-comparision/11-structured-json.ts` | `examples/official-comparison/11_structured_json.py` | present |
| `official-comparision/12-structured-parse.ts` | `examples/official-comparison/12_structured_parse.py` | present |
| `official-comparision/13-vision.ts` | `examples/official-comparison/13_vision.py` | present |
| `official-comparision/14-document-pdf.ts` | `examples/official-comparison/14_document_pdf.py` | present |
| `official-comparision/15-audio-in.ts` | `examples/official-comparison/15_audio_in.py` | present |
| `official-comparision/16-image-gen.ts` | `examples/official-comparison/16_image_gen.py` | present |
| `official-comparision/17-tts.ts` | `examples/official-comparison/17_tts.py` | present |
| `official-comparision/18-stt.ts` | `examples/official-comparison/18_stt.py` | MISSING |
| `official-comparision/19-reasoning.ts` | `examples/official-comparison/19_reasoning.py` | present |
| `official-comparision/20-prompt-caching.ts` | `examples/official-comparison/20_prompt_caching.py` | present |
| `official-comparision/21-files-upload.ts` | `examples/official-comparison/21_files_upload.py` | present |
| `official-comparision/22-batch.ts` | `examples/official-comparison/22_batch.py` | present |
| `official-comparision/23-embeddings.ts` | `examples/official-comparison/23_embeddings.py` | present |
| `official-comparision/24-response-state.ts` | `examples/official-comparison/24_response_state.py` | MISSING |
| `official-comparision/25-realtime-live.ts` | `examples/official-comparison/25_realtime_live.py` | present |
| `official-comparision/26-provider-routing.ts` | `examples/official-comparison/26_provider_routing.py` | present |
| `official-comparision/27-web-search.ts` | `examples/official-comparison/27_web_search.py` | present |
| `official-comparision/28-models-list.ts` | `examples/official-comparison/28_models_list.py` | present |
| `official-comparision/29-mcp-tools.ts` | `examples/official-comparison/29_mcp_tools.py` | present |
| `official-comparision/29b-mcp-protocol.ts` | `examples/official-comparison/29b_mcp_protocol.py` | present |
| `official-comparision/30-provenance.ts` | `examples/official-comparison/30_provenance.py` | present |
| `official-comparision/31-hosted-retrieval.ts` | `examples/official-comparison/31_hosted_retrieval.py` | present |
| `official-comparision/run.ts` | `examples/official-comparison/run.py` | present |
| `playwright.config.ts` | `examples/playwright_config.py` | MISSING |

