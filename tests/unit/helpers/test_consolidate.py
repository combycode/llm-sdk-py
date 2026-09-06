"""Multi-agent debate, judged by an outsider.

Driven through a scripted `complete`, because what matters here is the sequence
and the prompts: which round quotes which answers, when the loop stops, and
whether the judge is allowed to be one of the participants.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.helpers.consolidate import (
    ConsolidateAgent,
    ConsolidateJudge,
    ConsolidateRound,
    aconsolidate,
    consolidate,
)

AGENTS = [
    ConsolidateAgent(name="Ada", model="anthropic/claude-haiku-4.5", system="You favour speed."),
    ConsolidateAgent(name="Linus", model="openai/gpt-4.1-mini", system="You favour safety."),
]
JUDGE = ConsolidateJudge(model="google/gemini-2.5-flash")
TASK = "Ship on Friday or wait?"


class Script:
    """A `complete` that answers by role and records every call."""

    def __init__(self, *, agree_on: int | None = None, judge_text: str | None = None) -> None:
        self.agree_on = agree_on
        self.judge_text = judge_text
        self.calls: list[dict[str, Any]] = []
        self.rounds_seen = 0

    def _verdict(self) -> str:
        if self.judge_text is not None:
            return self.judge_text
        agreed = self.agree_on is not None and self.rounds_seen >= self.agree_on
        return json.dumps({"agreed": agreed, "reason": "because"})

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        system = str(kwargs.get("system") or "")
        if "impartial judge" in system or kwargs.get("structured"):
            self.rounds_seen += 1
            return type("R", (), {"text": self._verdict()})()
        if "impartial editor" in system:
            return type("R", (), {"text": "## Consensus\nship"})()
        return type("R", (), {"text": f"  answer from {kwargs['model']}  "})()

    async def acall(self, **kwargs: Any) -> Any:
        return self(**kwargs)

    # -- what the calls were, by kind
    @property
    def agent_prompts(self) -> list[str]:
        return [
            str(c["input"])
            for c in self.calls
            if not c.get("structured") and "impartial" not in str(c.get("system") or "")
        ]

    @property
    def judge_prompts(self) -> list[str]:
        return [str(c["input"]) for c in self.calls if c.get("structured")]


class TestItRefusesABadSetup:
    def test_one_agent_is_not_a_debate(self) -> None:
        with pytest.raises(ValueError, match="at least 2 agents"):
            consolidate(agents=AGENTS[:1], task=TASK, judge=JUDGE, complete=Script())

    def test_a_judge_needs_a_model(self) -> None:
        with pytest.raises(ValueError, match="judge.model is required"):
            consolidate(
                agents=AGENTS, task=TASK, judge=ConsolidateJudge(model=""), complete=Script()
            )

    def test_the_judge_may_not_be_one_of_the_agents(self) -> None:
        # The failure is invisible in the output: a model asked whether the
        # answers agree, when one of them is its own, is marking its own work.
        # The result still looks like a verdict.
        with pytest.raises(ValueError, match="self-evaluation bias"):
            consolidate(
                agents=AGENTS,
                task=TASK,
                judge=ConsolidateJudge(model=AGENTS[0].model),
                complete=Script(),
            )

    def test_the_clashing_agent_is_named(self) -> None:
        with pytest.raises(ValueError, match="Ada"):
            consolidate(
                agents=AGENTS,
                task=TASK,
                judge=ConsolidateJudge(model=AGENTS[0].model),
                complete=Script(),
            )


class TestTheDebate:
    def test_it_stops_the_round_the_judge_agrees(self) -> None:
        script = Script(agree_on=2)
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=5, complete=script)
        assert result.agreed_at == 2
        assert len(result.rounds) == 2

    def test_it_runs_to_the_ceiling_when_they_never_agree(self) -> None:
        script = Script(agree_on=None)
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=3, complete=script)
        assert result.agreed_at is None
        assert len(result.rounds) == 3

    def test_every_agent_answers_every_round(self) -> None:
        script = Script(agree_on=None)
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=2, complete=script)
        assert [len(r) for r in result.rounds] == [2, 2]
        assert [a.agent for a in result.rounds[0]] == ["Ada", "Linus"]

    def test_the_answers_are_trimmed(self) -> None:
        script = Script(agree_on=1)
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, complete=script)
        assert result.rounds[0][0].text == "answer from anthropic/claude-haiku-4.5"

    def test_the_second_round_quotes_the_first(self) -> None:
        # Without it every round is the first one again and nothing converges.
        script = Script(agree_on=None)
        consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=2, complete=script)
        assert "Previous round answers:" not in script.agent_prompts[0]
        assert "Previous round answers:" in script.agent_prompts[2]
        assert "[Ada]:" in script.agent_prompts[2]

    def test_the_agents_are_told_which_round_this_is(self) -> None:
        script = Script(agree_on=None)
        consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=4, complete=script)
        assert "round 1 of 4" in script.agent_prompts[0]

    def test_the_judge_sees_this_round_and_the_task(self) -> None:
        script = Script(agree_on=1)
        consolidate(agents=AGENTS, task=TASK, judge=JUDGE, complete=script)
        assert TASK in script.judge_prompts[0]
        assert "[Linus]:" in script.judge_prompts[0]

    def test_the_summary_is_written_by_the_judges_model(self) -> None:
        script = Script(agree_on=1)
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, complete=script)
        summary_call = script.calls[-1]
        assert summary_call["model"] == JUDGE.model
        assert "impartial editor" in summary_call["system"]
        assert result.summary == "## Consensus\nship"

    def test_each_round_is_reported_as_it_finishes(self) -> None:
        seen: list[ConsolidateRound] = []
        script = Script(agree_on=2)
        consolidate(
            agents=AGENTS, task=TASK, judge=JUDGE, rounds=5, on_round=seen.append, complete=script
        )
        assert [r.round for r in seen] == [1, 2]
        assert [r.agreed for r in seen] == [False, True]
        assert seen[0].judge_reason == "because"

    def test_a_custom_judge_prompt_is_used(self) -> None:
        script = Script(agree_on=1)
        consolidate(
            agents=AGENTS,
            task=TASK,
            judge=ConsolidateJudge(model=JUDGE.model, system="Be harsh."),
            complete=script,
        )
        assert any(c.get("structured") and c["system"] == "Be harsh." for c in script.calls)


class TestAnUnreadableJudge:
    def test_the_debate_continues_rather_than_ending_on_a_coin_toss(self) -> None:
        script = Script(judge_text="not json at all")
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=3, complete=script)
        assert result.agreed_at is None
        assert len(result.rounds) == 3

    def test_but_it_is_counted_rather_than_swallowed(self) -> None:
        # "They never agreed" and "we could not tell" are different answers, and
        # only one of them means run it again.
        script = Script(judge_text="not json at all")
        result = consolidate(agents=AGENTS, task=TASK, judge=JUDGE, rounds=3, complete=script)
        assert result.judge_failures == 3

    def test_a_readable_run_reports_none(self) -> None:
        result = consolidate(
            agents=AGENTS, task=TASK, judge=JUDGE, complete=Script(agree_on=1)
        )
        assert result.judge_failures == 0


class TestTheAsyncTwin:
    @pytest.mark.asyncio
    async def test_it_reaches_the_same_result(self) -> None:
        script = Script(agree_on=2)
        result = await aconsolidate(
            agents=AGENTS, task=TASK, judge=JUDGE, rounds=5, complete=script.acall
        )
        assert result.agreed_at == 2
        assert len(result.rounds) == 2
        assert result.summary == "## Consensus\nship"

    @pytest.mark.asyncio
    async def test_it_refuses_the_same_setups(self) -> None:
        with pytest.raises(ValueError, match="self-evaluation bias"):
            await aconsolidate(
                agents=AGENTS,
                task=TASK,
                judge=ConsolidateJudge(model=AGENTS[1].model),
                complete=Script().acall,
            )

    @pytest.mark.asyncio
    async def test_an_unreadable_judge_is_counted_here_too(self) -> None:
        script = Script(judge_text="not json at all")
        result = await aconsolidate(
            agents=AGENTS, task=TASK, judge=JUDGE, rounds=2, complete=script.acall
        )
        assert result.judge_failures == 2
        assert result.agreed_at is None

    @pytest.mark.asyncio
    async def test_the_agents_of_one_round_actually_overlap(self) -> None:
        # Mutation-driven. This asserted that two agents were called and two
        # finished, which is equally true of a sequential loop -- so replacing
        # the gather with a for-loop passed it. Concurrency is only observable
        # as OVERLAP, so that is what is measured: both agents inside the call
        # at the same moment.
        inside = 0
        peak = 0

        async def slow(**kwargs: Any) -> Any:
            nonlocal inside, peak
            if kwargs.get("structured") or "impartial" in str(kwargs.get("system") or ""):
                return type("R", (), {"text": json.dumps({"agreed": True, "reason": "r"})})()
            inside += 1
            peak = max(peak, inside)
            await asyncio.sleep(0.02)
            inside -= 1
            return type("R", (), {"text": "x"})()

        await aconsolidate(agents=AGENTS, task=TASK, judge=JUDGE, complete=slow)
        assert peak == 2, f"agents ran one at a time (peak={peak})"


class TestTheSyncCoreIsParallelToo:
    def test_the_agents_of_one_round_actually_overlap(self) -> None:
        # Same claim, same measurement, on threads. A five-agent round run in
        # sequence costs five round trips of wall clock for answers that do not
        # depend on each other.
        import threading

        lock = threading.Lock()
        inside = 0
        peak = 0
        both_arrived = threading.Barrier(len(AGENTS), timeout=5)

        def slow(**kwargs: Any) -> Any:
            nonlocal inside, peak
            if kwargs.get("structured") or "impartial" in str(kwargs.get("system") or ""):
                return type("R", (), {"text": json.dumps({"agreed": True, "reason": "r"})})()
            with lock:
                inside += 1
                peak = max(peak, inside)
            # Blocks until every agent has arrived, so a sequential
            # implementation deadlocks into the timeout instead of passing.
            both_arrived.wait()
            with lock:
                inside -= 1
            return type("R", (), {"text": "x"})()

        consolidate(agents=AGENTS, task=TASK, judge=JUDGE, complete=slow)
        assert peak == len(AGENTS), f"agents ran one at a time (peak={peak})"
