# The Python API, designed as usage first

**Read this before the library has any code in it.** These examples ARE the design.
Every file mirrors one scenario from the TypeScript corpus
(`unified-examples-ts/official-comparision/`), so the two can be compared line for
line — and any place Python needed to differ is a deliberate decision recorded below,
not an accident.

Nothing here imports anything that exists yet. That is the point: fix the shape,
review it, and only then build to it.

---

## Principles

1. **Same concepts, Python spelling.** A TypeScript user should recognise every
   example; a Python user should never feel they are reading transliterated
   JavaScript.
2. **Sync is the default path.** Most Python users start synchronous, and an SDK
   that forces `asyncio.run()` on line one is hostile. Async is a first-class
   parallel surface, not an afterthought and not a wrapper.
3. **Types do work.** Where TypeScript needs a JSON Schema literal, Python has
   dataclasses and type hints that already carry the same information. Using them
   is not sugar — it removes a whole class of hand-written schema mistakes.
4. **One obvious way.** Where TS offers a factory function, Python gets a class.
   No `create_llm()` *and* `LLM()`.

---

## The translation, decision by decision

### `createX({...})` → `X(...)`

TypeScript uses factory functions because a bare `new` is awkward there. Python has
no such problem, and a class is what a Python developer expects to construct.

| TypeScript | Python |
|---|---|
| `createLLM({ model })` | `LLM(model=...)` |
| `createEngine({ ... })` | `Engine(...)` |
| `createMediaOutput({ ... })` | `MediaOutput(...)` |
| `createRealtime({ ... })` | `Realtime(...)` |
| `createAgent({ ... })` | `Agent(...)` |

### `camelCase` → `snake_case`, everywhere

`maxTokens` → `max_tokens`, `apiKey` → `api_key`, `listModelsLive` → `list_models_live`,
`checkProvenance` → `check_provenance`, `connectMcp` → `connect_mcp`. No exceptions,
including in returned objects (`response.finish_reason`, `usage.input_tokens`).

### Sync and async are both real

```python
result = complete(model=..., prompt="hi")           # sync
result = await acomplete(model=..., prompt="hi")    # async
```

The `a`-prefix convention (`acomplete`, `aembed`, `atranscribe`) is used for
module-level functions. For classes, the async twin is a separate type:
`LLM` / `AsyncLLM`, `Agent` / `AsyncAgent` — matching what `openai` and `anthropic`
do, so the convention is already familiar.

**Neither wraps the other.** Sync is not `asyncio.run()` around async (that breaks
inside a running loop, which is exactly where notebook and web users live), and
async is not a thread pool around sync (that wastes the event loop).

### `defineTool({...})` → the `@tool` decorator

This is the biggest deliberate divergence, and the one most worth reviewing.

```typescript
// TypeScript: name, description and params are all restated by hand
const getWeather = defineTool({
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  params: { city: 'string' },
  execute: () => 'sunny',
});
```

```python
# Python: the function already says all of it
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"
```

Name comes from `__name__`, description from the docstring, the JSON Schema from
the type hints. Restating them would be three chances to disagree with the code.
Overrides stay available when the function name is not the wire name:

```python
@tool(name="get_weather", description="...")
def _weather(city: str) -> str: ...
```

An `async def` tool is detected and awaited automatically — no separate decorator.

### `structured` accepts a type, not only a schema

```typescript
const { parsed } = await complete<{ city: string; tempC: number }>({
  structured: { schema: { type: 'object', properties: { /* restated by hand */ } } },
});
```

```python
@dataclass
class Weather:
    city: str
    temp_c: float

result = complete(..., structured=Weather)
result.parsed.city        # a Weather instance, not a dict
```

The JSON-Schema dict form stays supported for schemas that come from elsewhere
(`structured={"schema": {...}}`), but the dataclass form is what the docs lead with.
`TypedDict` is accepted too, for callers who want a plain dict back.

Deriving the schema from type hints is stdlib-only (`dataclasses.fields` +
`typing.get_type_hints`) — no pydantic, consistent with the one-dependency rule.

### Streaming is a plain iterator

```python
for event in llm.stream("Count from 1 to 5."):
    if event.type == "text":
        print(event.text, end="")
```

```python
async for event in allm.stream("Count from 1 to 5."):
    ...
```

Events are frozen dataclasses with a `type` discriminator, so `event.type == "text"`
reads the same as the TypeScript. They also support structural pattern matching for
callers who prefer it:

```python
match event:
    case TextEvent(text=t): ...
    case ToolCallEvent(name=n): ...
```

### Results are dataclasses, not dicts

TypeScript destructures: `const { text } = await complete(...)`. Python returns a
frozen `Completion`:

```python
result = complete(...)
result.text          # str
result.parsed        # T | None
result.usage         # Usage(input_tokens=..., output_tokens=...)
result.cost          # Cost(total=..., input=..., output=...) — None if unpriced
```

A dict would lose autocomplete and type checking, which is most of the value of
shipping `py.typed`.

**On cost specifically:** `cost` is `None` when the model is not priced, never `0.0`.
A silent zero is exactly the bug that shipped in the TypeScript library — a client
reported 72k tokens billed at $0.00 — and the Python API should make that state
impossible to misread.

### Files and attachments take `str | Path | bytes`

```python
complete(..., attachments=["fixtures/red.png"])
complete(..., attachments=[Path("report.pdf")])
complete(..., attachments=[png_bytes])
```

`pathlib.Path` is what Python users actually hold. Accepting only strings would make
every caller write `str(path)`.

### Hooks are decorators

`engine.hooks.on('onCompletion', fn)` is a string-keyed registry — natural in
TypeScript, where the string literal is what types the handler. Python has no
such mechanism, so a string key buys nothing and costs autocomplete, so hooks
become methods:

```python
@engine.on_completion
def track(ctx: CompletionContext) -> None:
    total.append(ctx.response.usage.output_tokens)
```

**Unsubscribing** is the awkward part of decorator APIs, because the obvious
design (return the unsubscribe function) rebinds the name and throws away the
handler. So the decorator returns the function unchanged, with `.unsubscribe()`
attached to it:

```python
track.unsubscribe()      # `track` is still the function you wrote
```

**The catch-all** takes an event object rather than `(name, ctx)`, for the same
reason it does in TypeScript: name and payload as one value can be stored,
queued or replayed. Python's structural pattern matching is the direct analogue
of the TS discriminated union, and it is what makes reading the wrong field an
error rather than a silent `None`:

```python
@engine.on_any
def export(event: HookEvent) -> None:
    match event:
        case CompletionEvent(ctx=ctx):
            meter(ctx.provider, ctx.response.usage.input_tokens)
        case WarningEvent(ctx=ctx):
            log.warning("%s:%s", ctx.source, ctx.code)
        case _:
            counter[event.type] += 1
```

A string-keyed escape hatch stays for genuinely dynamic subscription
(`engine.on("on_completion", fn)`), which is what a plugin loader needs — but it
is not the path the docs lead with.

Hook names are snake_case like everything else: `onToolCallStart` →
`on_tool_call_start`.

### Errors are exceptions, with a common base

```python
try:
    complete(...)
except RateLimitError as e:
    e.retry_after
except LLMError as e:            # base class for everything we raise
    e.provider, e.model, e.status
```

Nothing returns an error-shaped result object; that is not how Python reads.

---

## Scope of this corpus

Numbering matches the TypeScript corpus exactly, so `06` is the same scenario in both.

| # | scenario | v1 |
|---|---|---|
| 01–03 | completion, system prompt, multi-turn | yes |
| 04 | streaming | yes |
| 05 | token counting | yes |
| 06–09 | tools: single, parallel, multi-step, runner | yes |
| 10, 10b | hosted server tools, code execution + files | yes |
| 11–12 | structured output | yes |
| 13–15 | vision, PDF, audio input | yes |
| 16–17 | image generation, TTS | yes |
| 18 | speech-to-text | yes |
| 19–20 | reasoning, prompt caching | yes |
| 21–22 | file upload, batch | yes |
| 23 | embeddings | yes |
| 24 | response state / conversation | yes |
| 26 | provider routing | yes |
| 27–28 | web search, model listing | yes |
| 30 | provenance | yes |
| **25** | **realtime / live** | **deferred** |
| **29, 29b** | **MCP client + protocol** | **deferred** |
| **31** | **hosted retrieval** | **deferred** |

The four deferred files are still written, and still fix their design — they are
marked `DEFERRED` at the top so nobody mistakes an absent feature for an oversight.
They are out of scope for the first Python release because they need WebSockets and
a second protocol client, not because the design is unsettled.

## Running them

Same environment contract as the TypeScript corpus, deliberately, so one runner can
drive both:

```
LLM_MODEL=anthropic/claude-haiku-4.5   LLM_API_KEY=...   python 01_basic_completion.py
```

Each prints one line of JSON: `{"result": "...", "ms": 123}`. That is what makes the
corpus checkable rather than merely readable.
