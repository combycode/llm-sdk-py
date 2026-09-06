# features-highlight

The scenarios in `../official-comparison/` show the API doing what every SDK
does. **These show what this one does differently** -- and they are the ones
whose Python shape most needs review, because there is no convention to copy.

Ported from `unified-examples-ts/features-highlight/`, one file per scenario,
same names.

## They are tests, not illustrations

Every file is **deterministic**: a stub transport, no API keys, no network. Each
asserts the behaviour it documents and exits non-zero when it stops being true,
so the gate can execute them. An example that merely *looks* right is how a
documented behaviour quietly stops happening.

```
python engine_retry_policy.py     # one line of JSON, exit 0
```

## What they pin down

| file | the decision it fixes |
|---|---|
| `engine_retry_policy` | retry configured once, three layers, narrowest wins |
| `event_stream` | `@engine.on_any` + structural matching instead of casting |
| `cost_unpriced_models` | `cost is None` when unpriced -- never `0.0` |
| `unified_names` | any spelling resolves; what is sent is always callable |
| `model_selector` / `model_filters` | `select()` plus a discoverable vocabulary |
| `exact_token_count` | a count that says whether it is exact |
| `tool_optional_parameters` | `required` derived from the signature |
| `lazy_tools` | registered without being declared |
| `tool_call_attribution` | `call_id` and opaque `signature` round-trip |
| `agent_guardrails_and_sampling` | bounded second chances; sampling that warns |
| `final_answer_phase` | `result.text` is the answer, not the narration |
| `context_anchored_strategy` | compaction that re-reads instead of re-summarising |
| `telemetry_traces` / `telemetry_redaction` | spans; free-text redaction on by default |
| `response_shape_check` | a 200 with a renamed field is still a problem |
| `transcribe_structured` | absent (None) is not empty (`[]`) |
| `retrieve_output_file` | descriptors, with an explicit `.read()` |
| `provenance_adapter` | evidence, not a boolean |
| `server_state_google_interactions` | the second turn must not resend history |
| `mcp_lazy_server` | **deferred** with the rest of MCP |
| `steps_chain_and_fanout` | one step type; a pipeline fans out without a rewrite |
| `multi_agent_delegation` | delegation returns an answer, not a transcript |
| `tool_catalog_search` | search narrows; `call()` is the boundary |
| `llm_backed_tools` | a tool declared as a prompt, checked on the wire |
| `content_moderation` | a flagged input stops before the completion is paid for |
| `openai_compatible_server` | the registered id ships, never the upstream one |
| `response_cache` | the route is part of the key |
| `checkpoint_persistence` | one pipeline, three backends, unchanged |
| `tool_permissions` | deny beats a later allow; unmentioned means denied |
| `scheduled_and_batched_work` | results correlate by id, never by position |
| `context_window_guard` | judged on what is left, not on what triggered the drop |
| `cost_estimate_and_budget` | bounds with published assumptions; refuse before spending |
| `tool_approval_gate` | `result_for`; an answer belongs to the call it was about |
| `agent_context_layers` | a fact outlives the turn that stated it |
| `context_guard_in_the_loop` | the guard shrinks the request, not a copy of it |

## The one rule that matters more than where these live

They must run against an **installed wheel**, never the source tree:

```
python -m build
python -m venv .venv-examples && .venv-examples/bin/pip install dist/*.whl
.venv-examples/bin/python examples/features-highlight/event_stream.py
```

`pip install -e .` would defeat the point. The `src/` layout makes the source
tree unimportable by accident, but only the install rule makes these a test of
the artifact we actually publish.
