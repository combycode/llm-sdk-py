"""Several models arguing until they agree, and an outsider writing it up.

Transposed from `unified-library-ts/src/helpers/consolidate.ts`.

Each round every agent answers the task in parallel, having read the previous
round's answers. A judge then decides whether they actually agree; when it says
so the loop stops early, and either way a closing call writes the consensus plus
what each agent alone contributed.

**The judge must not be one of the agents.** Refused at the door rather than
documented, because the failure is invisible in the output: a model asked
whether the answers agree, when one of those answers is its own, is being asked
to mark its own work, and it says yes more often. The result still looks like a
verdict.

Two cores, as everywhere else here: `consolidate` runs the agents on threads and
`aconsolidate` gathers them, and neither is the other wrapped. What they share
is the part with no I/O in it -- the prompts, the validation, and reading the
judge.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..llm.client_internal import parse_structured

#: How many rounds before the loop gives up on agreement.
DEFAULT_ROUNDS = 3
#: Per-agent ceiling for a round. Small on purpose: the agents are asked for a
#: recommendation, not an essay, and a long answer is harder to converge on.
DEFAULT_MAX_TOKENS = 220
JUDGE_MAX_TOKENS = 200
SUMMARY_MAX_TOKENS = 400

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "agreed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["agreed", "reason"],
}

DEFAULT_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether multiple agents have reached "
    "substantive agreement on a task. Disagreement on framing or emphasis with the "
    "same core recommendation = agreed. You are NOT one of the agents -- your role "
    "is purely evaluative."
)

DEFAULT_SUMMARY_SYSTEM = (
    "You are an impartial editor who consolidates multi-agent debate into a clear, "
    "balanced summary. Do not express your own opinion; reflect what the agents "
    "collectively concluded."
)

_AGENT_INSTRUCTION = "Reply concisely (<=3 sentences). State your concrete recommendation."

_SUMMARY_SHAPE = (
    "\n\nWrite in markdown:\n"
    "## Consensus\n(one or two sentences capturing the shared recommendation)\n"
    "## Per-agent unique points\n(one short bullet per agent for non-shared insight)"
)


@dataclass(frozen=True)
class ConsolidateAgent:
    """One participant: who it is, which model, and the role it plays."""

    name: str
    model: str
    system: str


@dataclass(frozen=True)
class ConsolidateAnswer:
    """What one agent said in one round."""

    agent: str
    text: str


@dataclass(frozen=True)
class ConsolidateJudge:
    """The outsider. Its model must differ from every agent's."""

    model: str
    system: str | None = None


@dataclass(frozen=True)
class ConsolidateRound:
    """A finished round, and what the judge made of it."""

    round: int
    answers: Sequence[ConsolidateAnswer]
    agreed: bool
    judge_reason: str


@dataclass(frozen=True)
class ConsolidateResult:
    """The debate, and its write-up."""

    summary: str
    #: Every round's answers, in order.
    rounds: Sequence[Sequence[ConsolidateAnswer]]
    #: The 1-based round the judge called agreement in, or None if it never did.
    agreed_at: int | None = None
    #: Whether the judge's verdict was unreadable in any round. Reported rather
    #: than swallowed: "they never agreed" and "we could not tell" are different
    #: answers, and only one of them means run it again.
    judge_failures: int = 0


@dataclass
class _Plan:
    """The settled arguments, once validated."""

    agents: Sequence[ConsolidateAgent]
    task: str
    judge_model: str
    judge_system: str
    rounds: int
    max_tokens: int
    on_round: Callable[[ConsolidateRound], None] | None = None
    transcript: list[list[ConsolidateAnswer]] = field(default_factory=list)


def _plan(
    *,
    agents: Sequence[ConsolidateAgent],
    task: str,
    judge: ConsolidateJudge,
    rounds: int,
    max_tokens: int,
    on_round: Callable[[ConsolidateRound], None] | None,
) -> _Plan:
    if len(agents) < 2:
        raise ValueError("consolidate: need at least 2 agents")
    if not judge.model:
        raise ValueError("consolidate: judge.model is required")
    clash = next((a for a in agents if a.model == judge.model), None)
    if clash is not None:
        raise ValueError(
            f'consolidate: judge.model "{judge.model}" is also agent "{clash.name}" -- '
            "pick a different model for the judge to avoid self-evaluation bias"
        )
    return _Plan(
        agents=agents,
        task=task,
        judge_model=judge.model,
        judge_system=judge.system or DEFAULT_JUDGE_SYSTEM,
        rounds=rounds,
        max_tokens=max_tokens,
        on_round=on_round,
    )


def _quoted(answers: Sequence[ConsolidateAnswer]) -> str:
    return "\n".join(f"[{a.agent}]: {a.text}" for a in answers)


def _round_prompt(plan: _Plan, round_no: int) -> str:
    """What every agent is asked this round.

    The previous round is quoted back so an agent can actually move; without it
    each round is the first one again and nothing ever converges.
    """
    prior = (
        f"\n\nPrevious round answers:\n{_quoted(plan.transcript[-1])}" if plan.transcript else ""
    )
    return (
        f"{plan.task}\n\n"
        f"This is round {round_no} of {plan.rounds}. Try to converge with the other "
        f"agents where their points are valid; restate your position concisely.{prior}"
    )


def _judge_prompt(plan: _Plan, round_no: int, answers: Sequence[ConsolidateAnswer]) -> str:
    return f"Task:\n{plan.task}\n\nAgents' answers (round {round_no}):\n{_quoted(answers)}"


def _summary_prompt(plan: _Plan, answers: Sequence[ConsolidateAnswer]) -> str:
    return f"Task:\n{plan.task}\n\nFinal answers from each agent:\n{_quoted(answers)}{_SUMMARY_SHAPE}"


def _verdict_of(text: str) -> tuple[bool, str, bool]:
    """The judge's answer as `(agreed, reason, unreadable)`.

    An unreadable verdict counts as "not agreed" so the debate continues rather
    than ending on a coin toss -- but it is COUNTED, because a run that never
    agreed and a run whose judge never parsed look identical otherwise.
    """
    try:
        parsed = parse_structured(text)
    except Exception:  # noqa: BLE001 -- any malformed answer, and there are many shapes
        return False, "judge parse failed", True
    if not isinstance(parsed, Mapping):
        return False, "judge parse failed", True
    return bool(parsed.get("agreed")), str(parsed.get("reason") or ""), False


def _agent_call(plan: _Plan, agent: ConsolidateAgent, prompt: str) -> dict[str, Any]:
    return {
        "model": agent.model,
        "system": f"{agent.system} {_AGENT_INSTRUCTION}",
        "input": prompt,
        "max_tokens": plan.max_tokens,
    }


def _judge_call(plan: _Plan, prompt: str) -> dict[str, Any]:
    return {
        "model": plan.judge_model,
        "system": plan.judge_system,
        "input": prompt,
        "structured": {"schema": JUDGE_SCHEMA},
        "max_tokens": JUDGE_MAX_TOKENS,
    }


def _summary_call(plan: _Plan, prompt: str) -> dict[str, Any]:
    return {
        "model": plan.judge_model,
        "system": DEFAULT_SUMMARY_SYSTEM,
        "input": prompt,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }


def consolidate(
    *,
    agents: Sequence[ConsolidateAgent],
    task: str,
    judge: ConsolidateJudge,
    rounds: int = DEFAULT_ROUNDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_round: Callable[[ConsolidateRound], None] | None = None,
    complete: Any = None,
) -> ConsolidateResult:
    """Run the debate synchronously. The agents of one round run on threads."""
    from .one_shot import complete as default_complete

    run = complete or default_complete
    plan = _plan(
        agents=agents,
        task=task,
        judge=judge,
        rounds=rounds,
        max_tokens=max_tokens,
        on_round=on_round,
    )
    agreed_at: int | None = None
    failures = 0

    for round_no in range(1, plan.rounds + 1):
        prompt = _round_prompt(plan, round_no)
        # In parallel because they are answering independently; run in sequence
        # a five-agent round costs five round trips of wall clock for nothing.
        def ask(agent: ConsolidateAgent, prompt: str = prompt) -> str:
            # `prompt` bound as a default rather than closed over: the list()
            # below forces the map inside this iteration so it would be correct
            # either way, but a lambda that reads a loop variable is one
            # refactor away from every round asking the last round's question.
            return str(run(**_agent_call(plan, agent, prompt)).text).strip()

        with ThreadPoolExecutor(max_workers=len(plan.agents)) as pool:
            texts = list(pool.map(ask, plan.agents))
        answers = [
            ConsolidateAnswer(agent=a.name, text=t) for a, t in zip(plan.agents, texts, strict=True)
        ]
        plan.transcript.append(answers)

        verdict = run(**_judge_call(plan, _judge_prompt(plan, round_no, answers)))
        agreed, reason, unreadable = _verdict_of(verdict.text)
        failures += int(unreadable)

        if plan.on_round is not None:
            plan.on_round(
                ConsolidateRound(
                    round=round_no, answers=answers, agreed=agreed, judge_reason=reason
                )
            )
        if agreed:
            agreed_at = round_no
            break

    final = plan.transcript[-1]
    summary = run(**_summary_call(plan, _summary_prompt(plan, final)))
    return ConsolidateResult(
        summary=summary.text,
        rounds=plan.transcript,
        agreed_at=agreed_at,
        judge_failures=failures,
    )


async def aconsolidate(
    *,
    agents: Sequence[ConsolidateAgent],
    task: str,
    judge: ConsolidateJudge,
    rounds: int = DEFAULT_ROUNDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_round: Callable[[ConsolidateRound], None] | None = None,
    complete: Any = None,
) -> ConsolidateResult:
    """The twin of `consolidate`. The agents of one round are gathered."""
    from .one_shot import acomplete as default_acomplete

    run = complete or default_acomplete
    plan = _plan(
        agents=agents,
        task=task,
        judge=judge,
        rounds=rounds,
        max_tokens=max_tokens,
        on_round=on_round,
    )
    agreed_at: int | None = None
    failures = 0

    for round_no in range(1, plan.rounds + 1):
        prompt = _round_prompt(plan, round_no)
        replies = await asyncio.gather(
            *(run(**_agent_call(plan, agent, prompt)) for agent in plan.agents)
        )
        answers = [
            ConsolidateAnswer(agent=a.name, text=r.text.strip())
            for a, r in zip(plan.agents, replies, strict=True)
        ]
        plan.transcript.append(answers)

        verdict = await run(**_judge_call(plan, _judge_prompt(plan, round_no, answers)))
        agreed, reason, unreadable = _verdict_of(verdict.text)
        failures += int(unreadable)

        if plan.on_round is not None:
            plan.on_round(
                ConsolidateRound(
                    round=round_no, answers=answers, agreed=agreed, judge_reason=reason
                )
            )
        if agreed:
            agreed_at = round_no
            break

    final = plan.transcript[-1]
    summary = await run(**_summary_call(plan, _summary_prompt(plan, final)))
    return ConsolidateResult(
        summary=summary.text,
        rounds=plan.transcript,
        agreed_at=agreed_at,
        judge_failures=failures,
    )


__all__ = [
    "DEFAULT_JUDGE_SYSTEM",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_ROUNDS",
    "DEFAULT_SUMMARY_SYSTEM",
    "JUDGE_SCHEMA",
    "ConsolidateAgent",
    "ConsolidateAnswer",
    "ConsolidateJudge",
    "ConsolidateResult",
    "ConsolidateRound",
    "aconsolidate",
    "consolidate",
]
