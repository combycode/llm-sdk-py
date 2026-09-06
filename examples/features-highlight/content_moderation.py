"""Classifying content before a model reads it, and stopping a run when it is flagged.

`moderate()` is the standalone endpoint, not the `moderation=` option on a
completion: it classifies text that no model has seen and none has to see,
which is the only way to check a user's message before paying to send it. The
verdict is a `ModerationResult` -- `flagged` is the answer, the score maps are
why.

`moderation_guardrail()` is that same call wired into an `Agent`. Left to each
call site, the check is something every one of them has to remember, and the
one that forgets is the one in production. As a guardrail it runs on every
completion, and a flagged input raises BEFORE the request is built -- so the
provider never sees the text and nothing is billed for it.

The endpoint is free. The call is still REPORTED: a zero-priced entry reaches
the engine's ledger, because free is not the same as invisible and a run that
made ten thousand moderation calls should be able to say so.

Deterministic: stub transports, no keys, no network.
"""

from typing import Any

from _check import check, report

from combycode_llm_sdk import (
    Agent,
    Engine,
    ModerationResult,
    TransportResponse,
    moderate,
    moderation_guardrail,
)
from combycode_llm_sdk.agent import GuardrailError

CATEGORIES = {"harassment": False, "violence": True}
SCORES = {"harassment": 0.01, "violence": 0.98}

COMPLETION = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "done"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 3},
}


class Moderations:
    """One scripted verdict, and every request it was asked to judge."""

    def __init__(self, flagged: bool = False) -> None:
        self.flagged = flagged
        self.seen: list[Any] = []

    def __call__(self, request: Any) -> TransportResponse:
        self.seen.append(request)
        return TransportResponse(
            status=200,
            body={
                "id": "modr-1",
                "model": "omni-moderation-2024-09-26",
                "results": [
                    {
                        "flagged": self.flagged,
                        "categories": CATEGORIES,
                        "category_scores": SCORES,
                    }
                ],
            },
        )


class Provider:
    """The completion the agent would run, and whether it ever ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Any) -> TransportResponse:
        self.calls += 1
        return TransportResponse(status=200, body=COMPLETION)


verdict = moderate(input="I will find you.", api_key="k", transport=Moderations(flagged=True))
check(isinstance(verdict, ModerationResult), "one input in, one verdict out")
check(verdict.flagged, "the verdict is the answer")
# The scores are why, and they are a plain map because OpenAI keeps adding
# categories -- a field per category would drop each new one on the floor.
check(verdict.category_scores["violence"] == 0.98, "the categories say what was flagged")
check(verdict.category_applied_input_types is None,
      "a model that does not report which input triggered says None, not {}")

# Free, and still on the books.
engine = Engine(register_as_default=False)
entries: list[Any] = []
engine.on("on_cost_entry", entries.append)
moderate(input="hello", api_key="k", engine=engine, transport=Moderations())
check(len(entries) == 1, "a free call is still reported to the engine")
check(entries[0].cost.input == 0 and entries[0].cost.output == 0,
      "reported as an honest zero rather than left unpriced")

# The guardrail, in the place it is meant to live.
provider = Provider()
guarded = Agent(
    model="anthropic/claude-haiku-4.5",
    api_key="k",
    transport=provider,
    before=[moderation_guardrail(api_key="k", transport=Moderations(flagged=True))],
)

try:
    guarded.complete("I will find you.")
    stopped = False
except GuardrailError:
    stopped = True
check(stopped, "a flagged input must stop the run")
check(provider.calls == 0, "and stop it before the completion is paid for")

clean = Agent(
    model="anthropic/claude-haiku-4.5",
    api_key="k",
    transport=Provider(),
    before=[moderation_guardrail(api_key="k", transport=Moderations(flagged=False))],
    after=[moderation_guardrail(kind="output", api_key="k", transport=Moderations(flagged=False))],
)
check(clean.complete("hello").text == "done", "a clean run passes through both phases")

report(flagged=verdict.flagged, worst="violence", stopped=stopped, cost_entries=len(entries))
