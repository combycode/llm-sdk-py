"""The event stream as one union -- subscribing without casting.

Two ways to watch what the SDK is doing, answering different questions:

    @engine.on_completion   ONE event you already named. The handler's context
                            is typed because you named it.
    @engine.on_any          EVERY event, name not known until runtime. What a
                            logger, a metrics exporter or an audit trail needs.

The catch-all could have handed over `(name, ctx)` and let every consumer read
fields off a bag. That works forever -- until a field is renamed, when the
exporter keeps running and starts reading `None`. For a token count or a cost,
that is a metric that quietly goes to zero and a dashboard that looks calm.

So `on_any` hands over a `HookEvent`: one class per hook, matched structurally,
so the wrong field is an error rather than a silent None.

Deterministic -- no key, no network. Events are emitted here so the output is
fixed and the assertions mean something.
"""

from _check import check, report

from combycode_llm_sdk import (
    CompletionEvent,
    Engine,
    HookEvent,
    RetryContext,
    RetryEvent,
    ToolCallStartEvent,
    WarningEvent,
)

engine = Engine(register_as_default=False)

# ── 1. one handler, every event, no casting ─────────────────────────────────
log: list[str] = []


@engine.on_any
def exporter(event: HookEvent) -> None:
    match event:
        case CompletionEvent(ctx=ctx):
            # `ctx` is a CompletionContext in this branch. Reading
            # `ctx.queue_length` here would be a type error, not a None.
            # Usage sits directly on the context: `response` carries the raw
            # provider answer, and routing a token count through it would be an
            # extra hop that can only go wrong.
            log.append(f"llm {ctx.provider}/{ctx.model} in={ctx.usage.input_tokens}")
        case ToolCallStartEvent(ctx=ctx):
            log.append(f"tool {ctx.tool_name} step={ctx.step}")
        case WarningEvent(ctx=ctx):
            log.append(f"warn {ctx.source}:{ctx.code}")
        case _:
            # Everything else still arrives, and still carries its name. A
            # consumer wanting only the three above need not enumerate the
            # other forty-eight.
            log.append(f"- {event.type}")


# ── 2. the named surface, unchanged ─────────────────────────────────────────
# Watching one event is not worth a match statement.
completions = 0


@engine.on_completion
def count_output(ctx) -> None:
    global completions
    completions += ctx.usage.output_tokens


# ── 3. drive it ─────────────────────────────────────────────────────────────
engine.emit_warning(source="cost", code="budget_near", message="over 80%")
engine.emit_tool_call_start(
    run_id="run_1", agent_id="agent_1", step=2, call_id="call_1",
    tool_name="search_orders", arguments={"q": "acme"},
)
engine.emit_completion(provider="openai", model="gpt-5.4-nano", output_tokens=34, input_tokens=120)
engine.emit_retry(provider="openai", model="gpt-5.4-nano", attempt=1, reason="rate_limit")

# FIVE lines from four emits, and the fifth is the point of subscribing to
# everything: the cost collector listens for on_completion and emits
# on_cost_entry from inside that handler. A catch-all sees events CAUSED by
# other events, and a nested emit is delivered before the emit that triggered
# it finishes -- so on_cost_entry lands AHEAD of the completion that produced
# it. Code assuming it only sees what it asked for gets this wrong.
check(len(log) == 5, f"every emit must reach the catch-all -- got {len(log)}")
check(completions == 34, "the named subscription must fire alongside the catch-all")
check(log[0].startswith("warn cost:"), "matching must reach the warning fields")
check(log[2] == "- on_cost_entry", "an event emitted inside a handler must arrive too")
check(log[3].startswith("llm openai/"), "the causing event arrives after the one it caused")
check(log[4] == "- on_retry", "an unhandled variant still arrives, carrying its name")

# ── 4. an event as a VALUE ──────────────────────────────────────────────────
# Why the shape matters beyond convenience: an event can be stored, queued,
# replayed or sent over a wire, because name and payload are ONE object. Nothing
# special is needed to make one -- it is an ordinary dataclass.
replayable = RetryEvent(
    ctx=RetryContext(provider="openai", model="gpt-5.4-nano", attempt=1, reason="rate_limit")
)
captured = [{"at": "captured", "type": replayable.type, "reason": replayable.ctx.reason}]

# ── 5. unsubscribe, or a long-lived process leaks ───────────────────────────
# The decorator returned the function unchanged, with .unsubscribe() attached --
# so `exporter` is still callable and still the handler you wrote.
exporter.unsubscribe()
engine.emit_warning(source="engine", code="after_unsub", message="ignored")
check(len(log) == 5, "unsubscribing must stop delivery")

report(log=log, completions=completions, captured=captured)
