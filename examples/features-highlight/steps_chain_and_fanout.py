"""One list of steps, run in a line or all at once.

`chain(steps)` prompts each step with what the last one produced. `parallel` and
`aparallel` hand every step the SAME input and collect the outputs in step order.
Both take the same step type and the same `on_step` callback.

Without that, sequential and parallel are two vocabularies, and deciding a
pipeline should fan out means rewriting every step in it -- so the decision gets
made once, early, when least is known. Worse, the rewrite is not loud: a step
written for one shape still runs in the other, it is merely prompted with the
wrong text, and the only symptom is answers that stopped depending on each other.

Deterministic: a stub transport answers FROM the prompt, so the assertions below
are about what actually went on the wire.
"""

import asyncio

from _check import check, report

from combycode_llm_sdk import ChainStepInfo, Step, TransportResponse, aparallel, chain, parallel
from combycode_llm_sdk.transport import TransportRequest

MODEL = "anthropic/claude-haiku-4.5"
INPUT = "the article"

sent: list[str] = []


def stub(request: TransportRequest) -> TransportResponse:
    # `verb(subject)`, not a fixed string: a chain and a fan-out differ only in
    # what each step is PROMPTED with, so an answer that ignored the prompt
    # would make the two indistinguishable here.
    prompt = " ".join(part["text"] for part in request.body["messages"][-1]["content"])
    sent.append(prompt)
    verb, _, subject = prompt.partition(": ")
    return TransportResponse(
        status=200,
        body={
            "content": [{"type": "text", "text": f"{verb.lower()}({subject})"}],
            "usage": {"input_tokens": 3, "output_tokens": 5},
            "stop_reason": "end_turn",
        },
    )


OPTIONS = {"api_key": "k", "transport": stub}

# Written once. Every run below is this exact list -- that is the whole point.
STEPS = [
    Step(model=MODEL, prompt=lambda text: f"Summarise: {text}", name="summarise",
         options=OPTIONS),
    Step(model=MODEL, prompt=lambda text: f"Translate: {text}", name="translate",
         options=OPTIONS),
]

# ── in a line ───────────────────────────────────────────────────────────────
progress: list[ChainStepInfo] = []
chained = chain(STEPS, on_step=progress.append)(INPUT)

check(chained == "translate(summarise(the article))", f"steps must thread -- got {chained!r}")
# The contract, seen on the wire: step two's prompt carries step ONE'S ANSWER.
# Threading the input through twice would look identical from the result alone.
check(
    sent == ["Summarise: the article", "Translate: summarise(the article)"],
    f"each step must be prompted with the previous output -- sent {sent}",
)
check(
    [(i.index, i.name, i.output) for i in progress] == [
        (0, "summarise", "summarise(the article)"),
        (1, "translate", "translate(summarise(the article))"),
    ],
    "on_step must name each step and report what it produced",
)

# ── all at once, same list ──────────────────────────────────────────────────
sent.clear()
progress.clear()
fanned = parallel(STEPS, on_step=progress.append)(INPUT)

# Position 1 is `translate(the article)`, not `translate(summarise(...))`: a
# fan-out gives every branch the original input. Same steps, other question.
check(
    fanned == ["summarise(the article)", "translate(the article)"],
    f"a fan-out must give every branch the input, in step order -- got {fanned}",
)
check(
    sorted(sent) == ["Summarise: the article", "Translate: the article"],
    f"no branch may be prompted with another branch's output -- sent {sent}",
)
# Sorted because `on_step` fires from whichever thread finished, in COMPLETION
# order -- which is why it carries the index at all. The RESULT list above is
# the one that keeps step order.
check(
    sorted((i.index, i.name) for i in progress) == [(0, "summarise"), (1, "translate")],
    "the same on_step callback must work for a fan-out, index and name intact",
)

# ── awaited, same list again ────────────────────────────────────────────────
# `asyncio.run` here because this file is a script; a library would await it.
# `parallel` spends a thread per branch and `aparallel` spends none, and that is
# the only difference a caller should be able to observe.
awaited = asyncio.run(aparallel(STEPS)(INPUT))
check(awaited == fanned, f"the async fan-out must agree with the sync one -- got {awaited}")

report(chained=chained, fanned=fanned, awaited=awaited, steps=[s.name for s in STEPS])
