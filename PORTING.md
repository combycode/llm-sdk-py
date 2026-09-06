# Porting the TypeScript SDK to Python

**Read this before writing a line of Python.** It exists because the first
attempt failed, and it failed in a specific, repeatable way that these rules
are shaped to prevent.

## What went wrong the first time

We did not port. We paraphrased: an agent read the TypeScript, formed a mental
model, and wrote Python from the model. The model is lossy, and nothing in the
loop compared the output against the input at a fine grain. Tests were then
written by the same agent from the same model, so they passed — 1403 of them,
green, against a wheel that died on the first real request.

The evidence is a controlled experiment that already ran, same team, same week:

| half | method | result |
|---|---|---|
| request building | shared TS's 147 wire specs, replayed its 7700-case corpus | **7700/7700**, zero defects found |
| response parsing | re-derived from reading the TS | Google Interactions billing every call at **$0.00**; streamed tool calls with no arguments; a whole API surface whose streams yield nothing |

The difference was not difficulty. It was whether we used TypeScript's own
artifacts as the oracle, or re-invented from its description.

**The test of every rule below: does this step have an oracle the porting agent
cannot author, cannot argue with, and cannot quietly weaken?**

## Prime directive

> **Transpose the code. Do not reimplement the behaviour.**

If the TypeScript has a class with twelve methods, the Python has a class with
twelve methods. If a responsibility lives in the network layer there, it lives
in the network layer here. A port loses things *visibly* — a missing file. A
paraphrase loses them *invisibly*, and the only way to find out is to audit
behaviour one item at a time, which is what cost us the first attempt.

## The etalon set — never edited

Three things are the standard the port is measured against. They are read-only
for the entire effort. If one of them disagrees with our code, **our code is
wrong**.

1. **`unified-library-ts/tests/fixtures/*.json`** — 9 golden fixtures, recorded
   from live providers. Vendored byte-identical. A fixture edited to match our
   behaviour proves nothing at all.
2. **`unified-library-ts/tests/unit/**`** — 2108 test cases. Translated, not
   reinterpreted. These are the oracle for behaviour.
3. **`unified-examples-ts/**` and the reviewed Python examples** — the API
   contract. If an example fails, the library is wrong, never the example.

## Allowed and forbidden transformations

Python is not TypeScript, and honest differences are expected. The danger is
using "Python is different" as cover for redesign, so the list is closed:
anything not in the Allowed column is Forbidden.

| Allowed | Forbidden |
|---|---|
| `snake_case` naming | merging files, or splitting them differently |
| `async`/`await` for Promises | moving a responsibility between layers |
| dataclass for an interface | changing an algorithm |
| `Protocol` for a structural type | adding or dropping a parameter |
| `None` for `undefined` | "simplifying" anything |
| context managers for disposables | collapsing N implementations into one generic |
| keyword arguments for a config object | inventing a design that reads better |

Every real defect in the first attempt was a Forbidden-column move:

- the context guard became imperative where TypeScript is event-driven
  *(responsibility moved between layers)*
- six per-provider response parsers collapsed into two generic functions
  *(N implementations collapsed)*
- the cost collector became four scalar accumulators instead of a ledger
  *(algorithm changed)*
- `AnchoredStrategy` became a different algorithm entirely — raw opening
  messages held on the instance, instead of a running summary carried in the
  message list *(algorithm changed, and it leaked one conversation into another)*

## The sugar rule

A Pythonic surface is welcome. The reviewed examples already show the correct
shape, using both forms side by side:

```python
engine.on("on_cost_entry", entries.append)   # the TypeScript primitive
@engine.on_warning                            # Python sugar over it
```

That is allowed, and it is good. But:

> **Sugar must never bound the primitive.**

The primitive is ported complete from TypeScript first. Sugar is added *over*
it afterwards, covering as much as is convenient. Sugar may be partial; the
primitive may not.

This is not hypothetical. TypeScript declares 51 hook events. Someone wrote six
decorator methods on the engine, and the event table became six entries — so
`ContextGuard`, `PermissionPolicy`, `ApprovalGate` and the cost ledger were all
ported faithfully and then had nothing to attach to. The convenience layer had
been allowed to define the surface instead of covering it.

Concretely: every capability reachable in TypeScript must be reachable through
the Python primitive, whether or not sugar exists for it.

## Python-only additions

Allowed, under three conditions:

1. **Additive** — the ported path still works unchanged. A `structured=` that
   accepts a dataclass is fine *because* it still accepts a JSON schema.
2. **Declared** — recorded in the registry as `python-only`, with why.
3. **Not a substitute** — an addition never replaces the transposed behaviour.

## The porting registry

`PORTING_REGISTRY.md`, generated mechanically from the TypeScript tree so it
cannot be wrong about what exists. One row per TypeScript file:

| TS source | PY target | TS test | PY test | TS cases | PY cases | status |
|---|---|---|---|---|---|---|

`status` is one of:

- `ported` — transposed, its translated tests pass
- `transposed-only` — no TypeScript test covers it; structural transposition
  only. **This is the residual-risk set and it is listed on purpose.**
- `python-only` — an addition, per the rules above
- `not-ported` — with a stated reason, never blank

A row claiming `ported` is self-reported and catches omission, not paraphrase —
we would have written that row for the guard and still shipped a different
design. So the case counts are the real check: **if `PY cases` is less than
`TS cases`, the port is incomplete**, whatever the status column says.

## The loop, per area

The TypeScript test tree already mirrors its source tree — `tests/unit/agent`,
`tests/unit/bus`, `tests/unit/llm/providers`, `tests/unit/plugins/context-guard`.
That structure **is** the work breakdown, and each directory is one agent's
exclusive ownership. Boundaries come from TypeScript, not from us, which is how
two agents avoid wiring the same thing twice.

1. **Translate the TypeScript tests for the area.** Assertions transposed, not
   reinterpreted. Each test carries the TS `file:line` it came from.
2. **Run them. They must all fail.** A translated test that passes before the
   port exists is a fake, and it is the single easiest way to fool this process.
3. **Port the source** until they pass.
4. **Write no new tests**, except for genuinely Python-only surface.

### Skip discipline

A translated test that genuinely cannot apply is marked:

```python
@pytest.mark.skip(reason="TS-specific: relies on structural typing — messages.test.ts:214")
```

Never deleted. Never weakened. Never "adjusted until it passes." The skip list
is then the honest gap list, countable at any moment, and a rising skip count is
a visible failure rather than a quiet one.

## Phases

1. **Registry** — enumerate everything, mechanically. Nothing is ported before
   it has a row.
2. **API stub** — signatures that satisfy the reviewed examples, so the public
   surface is pinned by something already validated. Bodies raise
   `NotImplementedError`.
3. **The spec interpreter and its tests** — the foundation, and the part that
   already worked, because it had an unfoolable oracle. The golden fixtures must
   reproduce exactly.
4. **Port by area**, parallel, one owner per TypeScript test directory, each
   running the loop above.
5. **Green tests, then green examples against real providers.** Unit tests are
   synthetic and confirm the model we already had; only a real provider can
   reject the request we actually build.

## Standing gates

Before any commit:

- `ruff check src tests` clean
- the full suite green, and the **skip count reported**
- the golden fixtures reproduced exactly
- examples run against an **installed wheel**, never the source tree — a
  `pip install -e .` or a `PYTHONPATH=src` defeats the point, and hid a wheel
  that declared `httpx2` while every call site imported `httpx`

## The one-sentence test

Before writing any Python, answer: **which TypeScript file is this, and which
of its tests will prove I got it right?** If there is no answer to either half,
stop — that is the moment the first attempt went wrong.
