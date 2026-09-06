# combycode-llm-sdk

A unified, pluggable AI SDK for accessing the LLMs of every major provider —
**Anthropic, OpenAI, Google, xAI, and OpenRouter** — through one API.

- **One API, every provider.** Switch model or provider without rewriting calls.
- **Sync and async side by side.** `LLM` and `AsyncLLM`, `complete` and
  `acomplete` — neither is a wrapper around the other, so neither pays for the
  other's machinery.
- **Pluggable.** Opt-in subsystems: a model catalog, cost tracking and budgets,
  rate-limit-aware queueing, tools and agents, retrieval, an MCP client, and an
  OpenAI-compatible server.
- **Spec-driven wire format.** Requests and responses are built and read by an
  interpreter over declarative JSON — shared byte-for-byte with the TypeScript
  [`@combycode/llm-sdk`](https://github.com/combycode/llm-sdk-ts), so the two
  libraries cannot drift apart.
- **Typed throughout**, `py.typed` included. One runtime dependency.

## Install

```sh
pip install combycode-llm-sdk
```

Requires **Python ≥ 3.11**.

For exact local OpenAI token counting, add the `tokens` extra:

```sh
pip install "combycode-llm-sdk[tokens]"
```

Without it everything still works — token counting falls back to the provider's
own count API (Anthropic, Google, xAI) or to a calibrated heuristic. It is an
extra rather than a dependency so its ~5.6 MB wasm is not installed for people
who never ask for it. MCP over WebSocket is the `ws` extra, on the same terms.

## Quickstart

### One call, no client to hold

```python
from combycode_llm_sdk import complete

result = complete(
    model="anthropic/claude-haiku-4.5",   # provider/model, sent verbatim
    api_key=key,
    prompt="Reply with exactly: OK",
    max_tokens=16,
)
print(result.text)
```

### A client, when you make more than one call

```python
from combycode_llm_sdk import LLM

llm = LLM(model="openai/gpt-5.4-nano", api_key=key)
print(llm.complete("Say hello in one word.").text)
```

`AsyncLLM` is the same surface with `await`, and `acomplete` the same for the
one-shot form.

### Tools

A plain function becomes a tool: the name comes from the function, the
description from its docstring, and the JSON Schema from its type hints — so
none of the three can drift from what the function actually does.

```python
from combycode_llm_sdk import complete, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"

result = complete(
    model="anthropic/claude-haiku-4.5",
    api_key=key,
    prompt="What is the weather in Paris?",
    tools=[get_weather],
    max_tokens=512,
)
print(result.text)
```

### Agents — a conversation that remembers

```python
from combycode_llm_sdk import Agent

agent = Agent(
    model="anthropic/claude-haiku-4.5",
    api_key=key,
    tools=[get_weather],
    system="You are terse.",
)
agent.complete("What is the weather in Paris?")
agent.complete("And is that unusual?")   # it still knows what "that" refers to
```

### Streaming

```python
llm = LLM(model="openai/gpt-5.4-nano", api_key=key)
for event in llm.stream("Count from 1 to 5."):
    if event.type == "text":
        print(event.text, end="", flush=True)
```

### Structured output

Annotate a dataclass and read it back parsed.

```python
from dataclasses import dataclass

from combycode_llm_sdk import complete

@dataclass
class Address:
    city: str
    country: str

@dataclass
class Person:
    name: str
    age: int
    address: Address

result = complete(
    model="openai/gpt-5.4-nano",
    api_key=key,
    prompt="Ada Lovelace, 36, lives in London, United Kingdom.",
    structured=Person,
    max_tokens=256,
)
print(result.parsed.name, result.parsed.address.city)
```

### Knowing the cost before you pay it

```python
from combycode_llm_sdk import estimate_cost

estimate = estimate_cost(model="anthropic/claude-haiku-4.5", prompt="...")
print(estimate.low, estimate.expected, estimate.high)
print(estimate.assumptions)   # every guess behind those numbers, written down
```

Three bounds rather than one, because a single number would have to pretend it
knows how long the answer will be. A model the catalog cannot price raises
`UnknownModelError` instead of answering `$0.00` — a budget built on a silent
zero passes every check until the invoice arrives.

## What else is in the box

| | |
|---|---|
| **Model catalog** | normalised slugs, `model:tier` selectors, capability-based `select()`, tiered pricing |
| **Cost** | per-call cost, running totals, budgets that refuse rather than warn |
| **Network** | retry policy, priority queue, rate-limit awareness, one `Engine` facade |
| **Agents** | tool loops, permissions, human approval, durable checkpoints, handoff, delegation |
| **Context** | token counting, context guards, compaction strategies, a layered prompt registry |
| **MCP** | client over stdio, Streamable HTTP and WebSocket, with OAuth 2.1, sampling and elicitation |
| **Retrieval** | local chunk-and-embed, plus hosted corpora on OpenAI, Google and xAI |
| **Media** | image generation and editing, speech, transcription, video |
| **More** | batch, embeddings, file uploads, realtime sessions, provenance, moderation, OTLP telemetry, an OpenAI-compatible server |

## Documentation

- [`examples/features-highlight/`](./examples/features-highlight) — one runnable
  file per feature, self-verifying, and needing no API key
- [`examples/official-comparison/`](./examples/official-comparison) — one
  scenario written once and run against every provider that supports it

## Development

```sh
pip install -e ".[dev]"
python -m ruff check src tests examples
python -m mypy src tests
python -m pytest -q
```

## License

MIT
