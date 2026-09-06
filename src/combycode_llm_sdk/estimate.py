"""Pricing a call BEFORE it is made, and a ceiling that refuses rather than spends.

`estimate_cost()` answers "what will this cost" with THREE bounds, because one
number would have to pretend it knows how long the answer will be. `low` is the
input and nothing back; `expected` adds a typical reply; `high` adds the longest
one this request could produce.

Every guess behind them is published in `assumptions`. An estimate that cannot
say what it assumed is indistinguishable from a measurement and gets believed
like one -- the same reason `count_tokens` reports its strategy.

`Budget.check()` raises BEFORE the call leaves. A limit that logs a warning and
lets the call through has not limited anything: after the request returns the
money is gone and the only thing left to do is report it. And it raises WITH the
estimate, the bound and the limit attached, because "over budget" without those
is an error nobody can act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .calibration import OutputCalibrationStore

#: What a reply is assumed to be when nothing has been observed. A real number
#: from no run at all, which is exactly why `Estimator` exists to replace it.
DEFAULT_OUTPUT_TOKENS = 512

#: How much longer than expected the longest plausible reply is, when the
#: catalog names no output cap. Generous on purpose: `high` is the bound a
#: budget is checked against when the caller cannot afford a surprise.
HIGH_OUTPUT_MULTIPLIER = 4

_PER_M = 1_000_000


@dataclass(frozen=True)
class CostBreakdown:
    """Where each part of the estimate came from."""

    input_usd: float = 0.0
    expected_output_usd: float = 0.0
    high_output_usd: float = 0.0


@dataclass(frozen=True)
class CostEstimate:
    """Three bounds, and everything guessed to reach them."""

    #: Input only, nothing back. The floor: this is spent the moment the request
    #: is accepted, whatever the model decides to say.
    low: float
    #: Input plus a typical reply.
    expected: float
    #: Input plus the longest reply this request could produce.
    high: float
    input_tokens: int
    est_output_tokens: int
    breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    #: Every guess, in words. Not decoration: it is what stops the number being
    #: read as a measurement.
    assumptions: Sequence[str] = ()
    provider: str = ""
    model: str = ""


def _usd(amount: float) -> str:
    """An amount a reader can act on, however small.

    Six decimals is right for a call and useless for a limit: budgets are
    routinely set below a cent, and `$0.000000` in a refusal names neither the
    limit nor the overage. Anything that would round to zero is shown in
    scientific notation instead.
    """
    if amount and abs(amount) < 0.000_001:
        return f"${amount:.2e}"
    return f"${amount:.6f}"


class UnknownModelError(LookupError):
    """`estimate_cost` was asked to price a model the catalog does not carry.

    Raised rather than answered with zeroes, because an estimate of $0.00 is
    indistinguishable from a genuinely free call: a budget built on it passes
    every check, and the first sign that the pricing was missing is the invoice.

    A `LookupError`, not an `LLMError`: nothing was sent, and there is no
    request to retry -- the fix is to register the model or name a known one.
    """

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            f'estimate: model "{provider}/{model}" is not in the catalog -- cannot price. '
            f"Register the model via catalog.set() or use a known model string."
        )
        self.provider = provider
        self.model = model


class BudgetExceededError(RuntimeError):
    """A call was refused because it would cross the limit."""

    def __init__(
        self,
        *,
        estimate: CostEstimate,
        cost_usd: float,
        limit_usd: float,
        spent_usd: float,
        bound: str,
    ) -> None:
        # Two sentences, because there are two situations. A running ledger
        # is refused for what it has already spent; a per-call ceiling has
        # spent nothing, and saying "$0.000000 is already spent" of it reads
        # as a bug in the budget rather than a refusal of the call.
        super().__init__(
            f"this call would cost {_usd(cost_usd)} ({bound}), over the "
            f"{_usd(limit_usd)} limit for one call"
            if not spent_usd
            else f"this call would cost {_usd(cost_usd)} ({bound}), and "
            f"{_usd(spent_usd)} of the {_usd(limit_usd)} limit is already spent"
        )
        #: The estimate that was judged. Kept, not summarised: "over budget"
        #: without it is an error nobody can act on.
        self.estimate = estimate
        self.cost_usd = cost_usd
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        #: Which of the three bounds the limit was applied to.
        self.bound = bound


def _pricing(catalog: Any, provider: str, model: str) -> tuple[float, float]:
    rates = catalog.get_pricing(provider, model) or {}
    return float(rates.get("inputPerMTok") or 0.0), float(rates.get("outputPerMTok") or 0.0)


def _resolve(model: str, provider: str | None) -> tuple[str, str]:
    from .helpers.client_resolver import is_namespaced_model_id, parse_model_id

    if is_namespaced_model_id(model):
        return parse_model_id(model)
    if not provider:
        raise ValueError(
            'estimate_cost: name the provider, either as `provider=` or as "provider/model".'
        )
    return provider, model


def _build(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    est_output_tokens: int,
    high_output_tokens: int,
    catalog: Any,
    assumptions: list[str],
) -> CostEstimate:
    input_rate, output_rate = _pricing(catalog, provider, model)
    input_usd = input_tokens / _PER_M * input_rate
    expected_output_usd = est_output_tokens / _PER_M * output_rate
    high_output_usd = high_output_tokens / _PER_M * output_rate

    return CostEstimate(
        low=input_usd,
        expected=input_usd + expected_output_usd,
        high=input_usd + high_output_usd,
        input_tokens=input_tokens,
        est_output_tokens=est_output_tokens,
        breakdown=CostBreakdown(
            input_usd=input_usd,
            expected_output_usd=expected_output_usd,
            high_output_usd=high_output_usd,
        ),
        assumptions=tuple(assumptions),
        provider=provider,
        model=model,
    )


def _input_tokens(prompt: Any, provider: str, model: str, catalog: Any) -> tuple[int, str]:
    """The prompt's size, and how it was arrived at."""
    from .helpers.count_tokens import _flatten
    from .tokens import HeuristicCounter

    text, allowance = _flatten(prompt)
    counted = HeuristicCounter(catalog).count(text, provider, model) + allowance
    return counted, "input estimated from characters, not measured with a tokenizer"


def _high_output(catalog: Any, provider: str, model: str, expected: int) -> int:
    """The longest reply this request could produce."""
    info = catalog.get(provider, model)
    cap = dict(info).get("maxOutputTokens") if info else None
    if isinstance(cap, (int, float)) and cap > 0:
        return int(cap)
    return expected * HIGH_OUTPUT_MULTIPLIER


def estimate_cost(
    *,
    model: str,
    prompt: Any,
    provider: str | None = None,
    output_tokens: int | None = None,
    engine: Any = None,
) -> CostEstimate:
    """What this call will cost, with everything it assumed written down."""
    from .catalog.catalog import resolve_catalog

    provider_name, model_name = _resolve(model, provider)
    catalog = resolve_catalog(engine.catalog if engine is not None else None)
    # Before any counting: an unpriced model cannot produce a number worth
    # having, and the caller has to hear that as a refusal rather than as $0.
    if not catalog.get_pricing(provider_name, model_name):
        raise UnknownModelError(provider_name, model_name)

    input_tokens, how = _input_tokens(prompt, provider_name, model_name, catalog)
    expected_output = output_tokens if output_tokens is not None else DEFAULT_OUTPUT_TOKENS
    high_output = _high_output(catalog, provider_name, model_name, expected_output)

    assumptions = [
        how,
        f"assumed a reply of {expected_output} tokens (a static default, not observed)"
        if output_tokens is None
        else f"caller supplied a reply length of {expected_output} tokens",
        f"the longest reply is taken as {high_output} tokens",
    ]
    return _build(
        provider=provider_name,
        model=model_name,
        input_tokens=input_tokens,
        est_output_tokens=expected_output,
        high_output_tokens=high_output,
        catalog=catalog,
        assumptions=assumptions,
    )


class Estimator:
    """`estimate_cost` with a memory of what this model actually replies with."""

    def __init__(self, store: OutputCalibrationStore | None = None, *, engine: Any = None) -> None:
        self.store = store if store is not None else OutputCalibrationStore()
        self.engine = engine

    def record(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> None:
        """Fold one real completion into what is known about this model."""
        self.store.record(provider, model, input_tokens, output_tokens)

    def estimate(
        self,
        *,
        model: str,
        prompt: Any,
        provider: str | None = None,
        engine: Any = None,
    ) -> CostEstimate:
        """The estimate, using observations where there are any."""
        from .catalog.catalog import resolve_catalog

        provider_name, model_name = _resolve(model, provider)
        chosen = engine if engine is not None else self.engine
        catalog = resolve_catalog(chosen.catalog if chosen is not None else None)

        input_tokens, how = _input_tokens(prompt, provider_name, model_name, catalog)
        learned = self.store.get(provider_name, model_name, input_tokens)

        if learned is None or learned.count == 0:
            # Nothing observed for this model at this size: the static estimate
            # is the honest answer, and it already says it is a guess.
            return estimate_cost(
                model=f"{provider_name}/{model_name}", prompt=prompt, engine=chosen
            )

        expected_output = round(learned.ewma_mean)
        high_output = max(learned.quantile(), expected_output)
        assumptions = [
            how,
            (
                f"calibrated: expected from {learned.count} samples of this model "
                f"at this input size, not a default"
            ),
            f"the longest reply is the p90 of those samples, {high_output} tokens",
        ]
        return _build(
            provider=provider_name,
            model=model_name,
            input_tokens=input_tokens,
            est_output_tokens=expected_output,
            high_output_tokens=high_output,
            catalog=catalog,
            assumptions=assumptions,
        )


class Budget:
    """A ceiling that refuses, and a ledger of what has been spent."""

    def __init__(self, limit_usd: float, *, bound: str = "expected") -> None:
        self.limit_usd = limit_usd
        #: Which of the three bounds a call is judged by. `expected` by default:
        #: judging every call by `high` refuses affordable work, and judging by
        #: `low` is not a budget at all.
        self.bound = bound
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.limit_usd - self._spent

    def cost_of(self, estimate: CostEstimate) -> float:
        return float(getattr(estimate, self.bound))

    def check(self, estimate: CostEstimate) -> float:
        """What this call will cost, or raise rather than let it happen.

        Raises BEFORE the call leaves. Only the call that would CROSS the line
        is refused -- a budget that refused the first affordable call because
        the total might one day be reached is not usable.
        """
        cost = self.cost_of(estimate)
        if self._spent + cost > self.limit_usd:
            raise BudgetExceededError(
                estimate=estimate,
                cost_usd=cost,
                limit_usd=self.limit_usd,
                spent_usd=self._spent,
                bound=self.bound,
            )
        return cost

    def record(self, cost_usd: float) -> float:
        """Charge the ledger for a call that happened."""
        self._spent += cost_usd
        return self._spent

    def reset(self) -> None:
        self._spent = 0.0

    def __repr__(self) -> str:
        return f"<Budget ${self._spent:.6f} of ${self.limit_usd:.6f}>"


__all__ = [
    "DEFAULT_OUTPUT_TOKENS",
    "HIGH_OUTPUT_MULTIPLIER",
    "Budget",
    "BudgetExceededError",
    "CostBreakdown",
    "CostEstimate",
    "Estimator",
    "UnknownModelError",
    "estimate_cost",
]
