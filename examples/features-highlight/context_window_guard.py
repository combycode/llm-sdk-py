"""Fitting the next request into the window -- and re-measuring after compacting.

`ContextMeasurer` answers two questions at once: how many tokens this is, and
how sure we are. The second half is the point. An exact count costs a tokenizer
load or an HTTP round trip, so the measurer estimates cheaply and escalates only
near the window -- and whatever it does, the number arrives labelled. A count
that cannot say how it was reached lets an estimate stand in for a measurement,
which is the one substitution a context check does not survive.

It does not act on what it finds. It PUBLISHES the reading on
`onContextMeasure`, and `ContextGuard` is one listener among however many:
telemetry may want the same number, and a caller may only want a warning. None
of them has to know about the others, which is why this is a hook rather than a
call.

**What goes wrong without the re-measurement.** A strategy is judged on what it
LEFT, never on the count that triggered it. Judging on the trigger declines work
that succeeded: a truncation that took a conversation from 100% of the window to
20% was refused for being "still above 95%" -- after it had already destroyed
eight messages. The caller lost the history and the call.

Deterministic: no provider, no key, no network. The window and the counting rate
come from a stub catalog so every percentage below is arithmetic you can check
by eye; in production both come from the real catalog, and nothing here
hardcodes a model's size.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from _check import check, report

from combycode_llm_sdk import ContextMeasurer, ConversationHistory, TruncateStrategy
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.context.context_guard import ContextGuard
from combycode_llm_sdk.context.strategy_types import TriggerLevel

WINDOW_TOKENS = 1_000
CHARS_PER_TOKEN = 4
MESSAGE_CHARS = 400
TURN_TOKENS = MESSAGE_CHARS // CHARS_PER_TOKEN
CALM_TURNS = 4
FULL_TURNS = 10
KEEP_RECENT = 2
TRIGGER_AT = 0.8
#: What the stubbed exact counter answers. Deliberately not a number the
#: heuristic could produce, so an escalation that never happened cannot pass.
EXACT_TOKENS = 920


@dataclass(frozen=True)
class Entry:
    context_window: int
    raw: dict[str, Any]


class StubCatalog:
    """The one lookup the guard and the measurer make."""

    def get(self, provider: str, model: str) -> Entry:
        return Entry(
            context_window=WINDOW_TOKENS,
            raw={"tokenizer": {"strategy": "heuristic", "charsPerTokenDefault": CHARS_PER_TOKEN}},
        )


class StubCounter:
    """Counts at the stub rate, so every number below is checkable by eye."""

    def estimate_message(self, message: Any, ctx: Any = None) -> int:
        text = message if isinstance(message, str) else str(message.get("content", ""))
        return -(-len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Counted:
    """What `count_tokens()` hands back: the number AND how it was reached."""

    tokens: int
    strategy: str
    exact: bool


escalations: list[str] = []


def exact_counter(**kwargs: Any) -> Counted:
    """Stands in for tiktoken or a provider count endpoint."""
    escalations.append(str(kwargs["model"]))
    return Counted(tokens=EXACT_TOKENS, strategy="tiktoken", exact=True)


def inexact_counter(**kwargs: Any) -> Counted:
    """The exact path answered, but only approximately."""
    return Counted(tokens=1, strategy="estimate", exact=False)


def conversation(turns: int) -> list[dict[str, Any]]:
    padded = [f"turn {i}".ljust(MESSAGE_CHARS, ".") for i in range(turns)]
    return [{"role": "user", "content": text} for text in padded]


def history_of(turns: int) -> ConversationHistory:
    """The transcript as the object the guard reasons about.

    A plain list is enough to SEND. It is not enough to guard: the guard keeps
    its place on the trigger ladder in `history.metadata` so it can tell a new
    crossing from the same pressure re-reported, and with nowhere to remember
    that it correctly does nothing.
    """
    history = ConversationHistory("demo")
    for message in conversation(turns):
        history.append(message)
    return history


def guard_over(catalog: Any, ceiling: float) -> ContextMeasurer:
    """A measurer with a guard listening to it. Nothing else connects them."""
    hooks = HookBus()
    measurer = ContextMeasurer(catalog, counter=StubCounter(), hooks=hooks)
    ContextGuard(
        hooks=hooks,
        measurer=measurer,
        strategies={
            "truncate": TruncateStrategy(
                keep_recent=KEEP_RECENT,
                decline_ceiling=ceiling,
                triggers=[TriggerLevel("urgent", TRIGGER_AT)],
            )
        },
        default_strategy="truncate",
    )
    return measurer


catalog = StubCatalog()
measurer = ContextMeasurer(catalog, count=exact_counter, counter=StubCounter())

# -- the cheap answer, and the fact that it says it is cheap -----------------

cheap = measurer.measure("stub", "model", conversation(CALM_TURNS))

check(cheap.tokens == CALM_TURNS * TURN_TOKENS, f"expected an estimate by rate, got {cheap.tokens}")
check(cheap.strategy == "estimate", f"expected an estimate, got {cheap.strategy!r}")
check(not cheap.exact, "an estimate must never report itself as exact")
check(escalations == [], "a round trip at 40% of the window buys a number nobody compares")
check(cheap.percentage == CALM_TURNS * TURN_TOKENS / WINDOW_TOKENS, "wrong share of the window")

# -- near the window, the difference is worth paying for ---------------------

full = conversation(FULL_TURNS)
measured = measurer.measure("stub", "model", full)

check(escalations == ["stub/model"], f"expected one escalation, got {escalations}")
check(measured.strategy == "tiktoken", f"expected the exact strategy, got {measured.strategy!r}")
check(measured.exact, "a tokenizer count is exact and must say so")
check(measured.tokens == EXACT_TOKENS, "the exact number must replace the estimate, not join it")

# An exact count is an improvement on the answer, never a precondition for
# having one -- and an answer that came back inexact is discarded rather than
# promoted, so `exact` still means what it says.
fallback = ContextMeasurer(catalog, count=inexact_counter, counter=StubCounter()).measure(
    "stub", "model", full, exact=True
)
check(fallback.strategy == "estimate", f"an inexact answer was adopted as {fallback.strategy!r}")
check(not fallback.exact, "an inexact answer must not be relabelled exact")
check(fallback.tokens == FULL_TURNS * TURN_TOKENS, "the estimate must stand, not the bad count")


async def guarded() -> None:
    # -- compacting, then judging what is LEFT -------------------------------

    workable = guard_over(catalog, ceiling=0.95)
    before = FULL_TURNS * TURN_TOKENS / WINDOW_TOKENS
    check(before >= TRIGGER_AT, "the setup must start above the trigger")

    history = history_of(FULL_TURNS)
    messages = history.messages()
    result = await workable.measure_and_emit("stub", "model", messages, history=history)

    check(not result["abort"], f"a truncation that worked was declined: {result['abortReason']}")
    check(len(messages) == KEEP_RECENT, f"expected {KEEP_RECENT} turns kept, got {len(messages)}")
    check(
        result["total"] == KEEP_RECENT * TURN_TOKENS,
        f"the count must describe what is LEFT to send, got {result['total']}",
    )
    check(
        messages is not history.messages(),
        "the list handed in is the one compacted, not a copy the caller does not hold",
    )

    # -- and a conversation that was never in trouble is left alone ----------

    calm = history_of(CALM_TURNS)
    calm_messages = calm.messages()
    untouched = await workable.measure_and_emit(
        "stub", "model", calm_messages, history=calm
    )
    check(
        len(calm_messages) == CALM_TURNS,
        "compacting below the trigger destroys transcript for nothing",
    )
    check(not untouched["abort"], "a conversation at 40% of the window must not be declined")

    # -- a guard that cannot make it fit refuses the call --------------------
    #
    # Nothing still holding a conversation clears a ceiling of zero, so this
    # stands in for the real case: compaction ran, it was not enough, and
    # sending anyway would buy a provider rejection.

    doomed = history_of(FULL_TURNS)
    refused = await guard_over(catalog, ceiling=0.0).measure_and_emit(
        "stub", "model", doomed.messages(), history=doomed
    )
    check(refused["abort"], "nothing clears a ceiling of zero, so this must decline")
    check(bool(refused["abortReason"]), "a refusal must say why")

    report(
        estimated=cheap.tokens,
        escalated_to=measured.strategy,
        before=round(before, 3),
        after=round(result["total"] / WINDOW_TOKENS, 3),
        kept=len(messages),
        aborted=result["abort"],
        declined_when_impossible=refused["abort"],
    )


asyncio.run(guarded())
