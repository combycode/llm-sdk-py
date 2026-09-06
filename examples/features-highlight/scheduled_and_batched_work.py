"""Deferred work with no timer, and batched work correlated by id.

Neither class owns a thread. `Scheduler` takes an injectable clock and answers a
question -- `due()` says what should have fired, `run_due()` fires it -- and
whoever owns the program's loop decides when to ask. `AutoBatcher` checks its
collection window whenever it is touched. The alternative is a background thread
this library starts and nobody can stop: not shutdownable, not testable without
sleeping, and owned by every process that imports us.

The schedule is DATA in a persistence store, not state in the scheduler, so a
process that dies with an hour left on a task picks it up on the next start.
Handlers are bound by NAME because that is all a persisted task can carry.

`AutoBatcher` collects calls that arrive one at a time from unrelated callers
and submits them as ONE provider batch job, then settles each caller's ticket
BY ID. The stub below answers in reverse on purpose: providers do (measured --
Anthropic returned a two-item batch back to front), and correlating by position
would hand each caller the other's answer and read as perfectly correct.

Deterministic: two hand-wound clocks and a stub transport. Nothing sleeps.
"""

import json

from _check import check, report

from combycode_llm_sdk import (
    AutoBatcher,
    BatchStrategy,
    LLMError,
    MemoryPersistence,
    Scheduler,
    TransportResponse,
)
from combycode_llm_sdk.scheduler import parse_duration

MODEL = "anthropic/claude-haiku-4.5"
BATCH_ID = "msgbatch_1"
BATCH_PATH = "/v1/messages/batches"
WINDOW_SECONDS = 10.0


class Clock:
    """A clock moved by hand. Sleeping would prove only that sleeping works."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, amount: float) -> None:
        self.now += amount


# -- deferred work -----------------------------------------------------------

clock = Clock(now=1_700_000_000_000.0)  # ms since the epoch, as the store holds it
tasks = MemoryPersistence()
scheduler = Scheduler(tasks, clock=clock)

fired: list[dict] = []
scheduler.register("send_report", fired.append)
scheduler.after("5m", "send_report", {"to": "ops"})

check(scheduler.due() == [], "nothing is due before its time")
check(len(scheduler.pending()) == 1, "but it is pending, and it is in the store")

clock.advance(parse_duration("5m"))
check([t.name for t in scheduler.due()] == ["send_report"], "asking is what makes it due")
check(scheduler.run_due() == 1, "and asking again is what runs it")
check(fired == [{"to": "ops"}], f"the handler gets the args it was scheduled with, got {fired}")
check(scheduler.pending() == [], "a one-shot task is gone once it has run")

# A restart: new scheduler, same store, handler re-registered by name. The
# schedule outlives the object that made it, which is the point of keeping it
# in persistence rather than in the instance.
scheduler.after("1h", "send_report", {"to": "finance"})
restarted = Scheduler(tasks, clock=clock)
resumed: list[dict] = []
restarted.register("send_report", resumed.append)

clock.advance(parse_duration("1h"))
check(restarted.run_due() == 1, "a new process picks up the schedule the old one left")
check(resumed == [{"to": "finance"}], f"with the args it was given, got {resumed}")


# -- batched work ------------------------------------------------------------

replies: dict[str, str] = {}
submissions: list[dict] = []


def _results() -> str:
    """The provider's result file -- REVERSED, because providers do that."""
    lines = [
        json.dumps(
            {"custom_id": item_id, "result": {"type": "succeeded", "message": _message(text)}}
        )
        for item_id, text in reversed(list(replies.items()))
    ]
    return "\n".join(lines) + "\n"


def _message(text: str) -> dict:
    return {
        "id": "msg_1",
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def anthropic_batches(request):
    routes = {
        f"{BATCH_PATH}/{BATCH_ID}/results": _results(),
        f"{BATCH_PATH}/{BATCH_ID}": {"processing_status": "ended"},
        BATCH_PATH: {"id": BATCH_ID},
    }
    if request.url.endswith(BATCH_PATH):
        submissions.append(request.body)
    # Longest match: the submit path is a PREFIX of the status path, so
    # first-match would answer a poll with the submit fixture and every
    # assertion after it would pass vacuously.
    route = max((r for r in routes if r in request.url), key=len)
    return TransportResponse(status=200, body=routes[route])


batch_clock = Clock()  # seconds, standing in for time.monotonic
batcher = AutoBatcher(
    model=MODEL,
    strategy=BatchStrategy(window_seconds=WINDOW_SECONDS, min_batch_size=2),
    clock=batch_clock,
    api_key="k",
    transport=anthropic_batches,
    max_tokens=16,
)

# Two unrelated callers, neither holding the other's item.
first = batcher.add({"prompt": "Summarise ticket 4711"})
check(batcher.pending == 1, "one item is not a batch: that caller wanted complete()")

second = batcher.add({"prompt": "Summarise ticket 815"})
check(batcher.pending == 2, "the window is still open, so nothing has been submitted")
check(submissions == [], "and the provider has not been touched")

replies[first.custom_id] = "4711: the printer is on fire"
replies[second.custom_id] = "815: the printer is fine"

batch_clock.advance(WINDOW_SECONDS + 1)
check(batcher.tick() is not None, "past the window, the next touch submits")
check(len(submissions) == 1, f"both items go in ONE job, saw {len(submissions)}")
check(len(submissions[0]["requests"]) == 2, "carrying both callers' items")

# Not an empty result: a lost item must not read as a model that declined.
refused = None
try:
    first.result()
except LLMError as exc:
    refused = exc
check(refused is not None, "a ticket with no answer yet must refuse rather than invent one")

batcher.wait(poll_seconds=0, timeout_seconds=5)

check(first.custom_id != second.custom_id, "every item carries an id of its own")
check(first.result().text.startswith("4711"), f"first caller got {first.result().text!r}")
check(second.result().text.startswith("815"), f"second caller got {second.result().text!r}")

report(
    fired=fired + resumed,
    submissions=len(submissions),
    first=first.result().text,
    second=second.result().text,
)
