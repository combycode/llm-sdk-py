"""Pricing a call BEFORE it is made, and a ceiling that refuses rather than spends.

`estimate_cost()` answers "what will this cost" with three bounds, because one
number would have to pretend it knows how long the answer will be: `low` is the
input and nothing back, `expected` adds a typical reply, `high` adds the longest
one this request could produce. Every guess behind them is published in
`assumptions` -- an estimate that cannot say what it assumed is indistinguishable
from a measurement and gets believed like one.

`Estimator` is the same thing with a memory: once real completions have been
recorded it stops guessing the reply length and uses what this model actually
replies with at this input size, which here is an order of magnitude cheaper
than the static default and changes what the run can afford.

**What goes wrong without the budget.** A limit that logs a warning and lets the
call through has not limited anything -- after the request returns, the money is
gone and the only thing left to do is report it. So `Budget.check()` raises
`BudgetExceededError` before the call leaves, and raises with the estimate, the
bound and the limit attached, because "over budget" without those is an error
nobody can act on.

Deterministic: nothing is sent and no key is needed. Prices come from the
bundled catalog, so the figures below are the real ones for this model.
"""

from _check import check, report

from combycode_llm_sdk import (
    Budget,
    BudgetExceededError,
    Estimator,
    OutputCalibrationStore,
    estimate_cost,
)

MODEL = "anthropic/claude-haiku-4.5"
PROVIDER, BARE = MODEL.split("/")
PROMPT = "Summarise the attached incident report and list every action taken."
#: What this model was seen replying with, and how often. Twenty short replies,
#: against a static default of 512 tokens that no run has ever confirmed.
REPLIES_SEEN = 20
REAL_OUTPUT_TOKENS = 40
#: Bounds the spend loop below. A budget that never fires has to fail the
#: example, not run forever.
MAX_CALLS = 50

# -- the static estimate, and what it admits to guessing ---------------------

static = estimate_cost(model=MODEL, prompt=PROMPT)
assumed = " ".join(static.assumptions)

check(static.low <= static.expected <= static.high, "the three bounds must stay ordered")
check(static.low == static.breakdown.input_usd, "`low` is the input and nothing back")
check("estimated from characters" in assumed, "an estimate must admit the count is one too")
check(str(static.est_output_tokens) in assumed, "the assumed reply length must be written down")

# -- what the model actually replies with beats what we assumed --------------

estimator = Estimator(OutputCalibrationStore())
for _ in range(REPLIES_SEEN):
    estimator.record(PROVIDER, BARE, static.input_tokens, REAL_OUTPUT_TOKENS)

calibrated = estimator.estimate(model=MODEL, prompt=PROMPT)

check(calibrated.est_output_tokens == REAL_OUTPUT_TOKENS, "recorded replies must replace the guess")
check(calibrated.expected < static.expected, "a shorter known reply must cost less than the guess")
check(calibrated.low == static.low, "the input is measured the same way either way")
check(
    f"calibrated: expected from {REPLIES_SEEN} samples" in " ".join(calibrated.assumptions),
    "a number learned from real runs must say so, and say from how many",
)

# -- the ceiling ------------------------------------------------------------

# Between the two estimates, so which guess the run is judged by decides whether
# the call happens at all -- the whole reason a static guess is worth replacing.
limit_usd = (calibrated.expected + static.expected) / 2
budget = Budget(limit_usd)

check(budget.check(calibrated) == calibrated.expected, "a call that fits returns what it will cost")

refused = None
try:
    budget.check(static)
except BudgetExceededError as exc:
    refused = exc

check(refused is not None, "a limit that lets the call through has not limited anything")
check(budget.spent == 0.0, "a refused call must not be charged for")
if refused is not None:
    check(refused.estimate is static, "'over budget' without the estimate is not actionable")
    check(refused.cost_usd == static.expected, "the refusal must name the cost it judged")
    check(refused.limit_usd == limit_usd, "...and the limit it judged that cost against")
    check(refused.bound == "expected", "and which of the three bounds it used")

# -- spending until one more call would cross it -----------------------------

made = 0
stopped = None
while made < MAX_CALLS and stopped is None:
    try:
        budget.record(budget.check(calibrated))
    except BudgetExceededError as exc:
        stopped = exc
    else:
        made += 1

check(stopped is not None, f"{MAX_CALLS} calls never reached a ${limit_usd:.6f} limit")
check(made > 0, "a budget that refuses the first affordable call is not usable")
check(budget.spent <= limit_usd, "the limit is a ceiling on what was spent, not a suggestion")
check(budget.remaining == limit_usd - budget.spent, "what is left is the limit minus the ledger")
if stopped is not None:
    check(stopped.spent_usd == budget.spent, "the refusal reports what has really been spent")
    check(
        budget.spent + stopped.cost_usd > limit_usd,
        "only the call that would cross the line may be refused",
    )

report(
    input_tokens=static.input_tokens,
    static_expected=round(static.expected, 6),
    calibrated_expected=round(calibrated.expected, 6),
    calibrated_output_tokens=calibrated.est_output_tokens,
    limit=round(limit_usd, 6),
    calls_made=made,
    spent=round(budget.spent, 6),
)
