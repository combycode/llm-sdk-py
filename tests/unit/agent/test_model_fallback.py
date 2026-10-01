"""Backup models for a step of an agent run.

Transposed from `unified-library-ts/tests/unit/agent/model-fallback.test.ts`.

`route()` already fell over between models, but only for a one-shot completion.
A run is where it matters more: a rate limit on step 7 of a nine-step run threw
away six steps of work and every tool call they paid for, and the caller's only
recourse was to start the run again.

Four rules, and the third is the one that cannot be got wrong:

1. each client once per step -- retrying one is the network engine's job, and
   doing it here too retries a single failure twice over;
2. only a failure another model could survive moves on;
3. **never switch once output has reached the consumer** -- a streamed turn can
   fail after several events, and a backup cannot continue someone else's
   half-rendered answer;
4. every step starts from the primary, because a rate limit is transient.

And the stamp: a step a backup served must be recorded as the backup's.
Provenance is model-bound -- a stateful continuation is only valid against the
model that issued the state -- so naming the primary on a turn it did not
produce would have the next step offer the backup's server state to the primary.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.results import Completion, Part, Usage


class Classified(Exception):
    """An error carrying a `kind`, the way a provider error does."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def answer(model: str) -> Completion:
    text = f"answer from {model}"
    return Completion(
        text=text,
        model=model,
        finish_reason="stop",
        usage=Usage(),
        parts=[Part(type="text", text=text)],
    )


class Client:
    """A client that answers or fails on command, and counts its calls."""

    provider = "mock"
    api = "completions"

    def __init__(self, model: str, *, fail: str | None = None, emit_before: int = 0) -> None:
        self.id = f"c-{model}"
        self.model = model
        self._fail = fail
        self._emit_before = emit_before
        self.completes = 0
        self.streams = 0

    def _boom(self) -> None:
        if self._fail == "plain":
            raise RuntimeError(f"{self.model} broke")
        raise Classified(f"{self.model} is busy", str(self._fail))

    def complete(self, _messages: Any, **_options: Any) -> Completion:
        self.completes += 1
        if self._fail:
            self._boom()
        return answer(self.model)

    def stream(self, _messages: Any, _options: Any = None) -> Any:
        self.streams += 1
        for index in range(self._emit_before):
            yield {"type": "text", "text": f"{self.model}-{index}"}
        if self._fail:
            self._boom()
        yield {"type": "text", "text": f"answer from {self.model}"}
        yield {"type": "done", "response": answer(self.model)}

    def destroy(self) -> None:
        pass


def agent_with(
    primary: Client, backups: list[Client], **extra: Any
) -> tuple[AgentLoop, list[dict[str, Any]]]:
    hooks = HookBus()
    warnings: list[dict[str, Any]] = []
    hooks.on("onWarning", lambda w: warnings.append(dict(w)))
    loop = AgentLoop(
        primary,
        hooks=hooks,
        system="s",
        fallback_clients=[b for b in backups],
        **extra,
    )
    return loop, warnings


class TestABufferedStepFallsOver:
    def test_it_uses_the_backup_when_the_primary_is_rate_limited(self) -> None:
        primary = Client("primary", fail="rate_limit")
        backup = Client("backup")
        loop, _ = agent_with(primary, [backup])

        assert loop.complete("go").text == "answer from backup"
        assert (primary.completes, backup.completes) == (1, 1)

    def test_it_warns_naming_both_models_and_the_reason(self) -> None:
        # A silent fallback is a performance and cost change nobody can see.
        loop, warnings = agent_with(Client("primary", fail="server_error"), [Client("backup")])
        loop.complete("go")

        warning = next(w for w in warnings if w.get("code") == "model_fallback")
        assert warning["details"] == {"from": "primary", "to": "backup", "kind": "server_error"}
        assert "is busy" in warning["message"]

    def test_it_does_not_fall_over_on_a_failure_another_model_cannot_fix(self) -> None:
        # Auth, a malformed request, a content filter: the same request fails the
        # same way everywhere, so asking every backup only multiplies the latency.
        backup = Client("backup")
        loop, _ = agent_with(Client("primary", fail="auth"), [backup])

        with pytest.raises(Classified, match="primary is busy"):
            loop.complete("go")
        assert backup.completes == 0

    def test_it_does_not_fall_over_on_an_unclassified_error(self) -> None:
        # No kind means nothing is known about whether another model would help.
        backup = Client("backup")
        loop, _ = agent_with(Client("primary", fail="plain"), [backup])

        with pytest.raises(RuntimeError, match="primary broke"):
            loop.complete("go")
        assert backup.completes == 0

    def test_it_raises_the_last_error_when_every_client_fails(self) -> None:
        # The last provider's own message is the useful half; a wrapper saying
        # "all models failed" buries it.
        loop, _ = agent_with(
            Client("primary", fail="rate_limit"),
            [Client("second", fail="rate_limit"), Client("third", fail="server_error")],
        )
        with pytest.raises(Classified, match="third is busy"):
            loop.complete("go")

    def test_it_tries_each_client_exactly_once_in_order(self) -> None:
        # Retrying one client is the network engine's concern; doing it here too
        # would retry a single failure twice over, at two layers.
        primary = Client("primary", fail="rate_limit")
        second = Client("second", fail="rate_limit")
        third = Client("third")
        loop, _ = agent_with(primary, [second, third])

        loop.complete("go")
        assert [primary.completes, second.completes, third.completes] == [1, 1, 1]

    def test_it_honours_a_caller_supplied_set_of_failure_classes(self) -> None:
        backup = Client("backup")
        loop, _ = agent_with(
            Client("primary", fail="rate_limit"), [backup], fallback_on=["server_error"]
        )
        with pytest.raises(Classified, match="primary is busy"):
            loop.complete("go")
        assert backup.completes == 0

    def test_every_step_starts_from_the_primary_again(self) -> None:
        # A rate limit is transient. A run that fell over once should not spend
        # the rest of its life on the backup.
        primary = Client("primary")
        backup = Client("backup")
        loop, _ = agent_with(primary, [backup])

        loop.complete("one")
        loop.complete("two")
        assert (primary.completes, backup.completes) == (2, 0)


class TestWhatTheStepIsRecordedAs:
    def test_it_stamps_the_model_that_actually_served(self) -> None:
        # Provenance is model-bound: a stateful continuation is only valid
        # against the model that issued the state, so recording the primary on a
        # turn the backup produced would offer the backup's state to the primary.
        loop, _ = agent_with(Client("primary", fail="rate_limit"), [Client("backup")])
        loop.complete("go")
        assert loop.history.all()[-1].model == "backup"

    def test_it_still_reports_the_primary_as_the_agents_model(self) -> None:
        # `client.model` is read before any request is made -- it cannot know who
        # will serve -- and a caller asking what model this agent is configured
        # with means the primary.
        loop, _ = agent_with(Client("primary", fail="rate_limit"), [Client("backup")])
        loop.complete("go")
        assert loop.model == "primary"


class TestAStreamedStep:
    @staticmethod
    def drain(loop: AgentLoop, text: str) -> list[str]:
        seen: list[str] = []
        for event in loop.stream(text):
            if getattr(event, "type", None) == "text" or (
                isinstance(event, dict) and event.get("type") == "text"
            ):
                seen.append(
                    event.text if hasattr(event, "text") else str(event.get("text"))  # type: ignore[union-attr]
                )
        return seen

    def test_it_falls_over_when_the_primary_fails_before_its_first_event(self) -> None:
        primary = Client("primary", fail="rate_limit", emit_before=0)
        backup = Client("backup")
        loop, _ = agent_with(primary, [backup])

        assert "answer from backup" in "".join(self.drain(loop, "go"))
        assert backup.streams == 1

    def test_it_does_not_fall_over_once_an_event_reached_the_consumer(self) -> None:
        # The rule that cannot be got wrong. The consumer has already rendered
        # part of the primary's answer; the backup would start a different one
        # mid-sentence, and the two would be spliced into a single turn.
        primary = Client("primary", fail="rate_limit", emit_before=2)
        backup = Client("backup")
        loop, _ = agent_with(primary, [backup])

        seen: list[str] = []
        with pytest.raises(Classified, match="primary is busy"):
            for event in loop.stream("go"):
                seen.append(str(event))
        assert seen  # the partial output was delivered
        assert backup.streams == 0

    def test_it_stamps_a_streamed_step_with_the_model_that_served(self) -> None:
        loop, _ = agent_with(Client("primary", fail="rate_limit"), [Client("backup")])
        self.drain(loop, "go")
        assert loop.history.all()[-1].model == "backup"


class TestTheChainItself:
    def test_it_needs_no_backups_to_work_at_all(self) -> None:
        # Every step takes the same path whether or not fallback is configured;
        # one-entry chains are how that stays true.
        loop = AgentLoop(Client("primary"), system="s")
        assert loop.complete("go").text == "answer from primary"

    def test_it_ignores_a_backup_that_is_the_primary(self) -> None:
        # Listing the primary again is the retry this layer is deliberately not
        # doing, so it would add a second attempt that looks like a fallback.
        primary = Client("primary", fail="rate_limit")
        loop, _ = agent_with(primary, [primary])
        with pytest.raises(Classified, match="primary is busy"):
            loop.complete("go")
        assert primary.completes == 1

    def test_it_carries_the_backups_through_a_restore(self) -> None:
        # A snapshot holds neither the client nor its backups -- both are live
        # objects. A restore that silently lost them would resume a run LESS
        # resilient than the one it continues.
        backup = Client("backup")
        loop, _ = agent_with(Client("primary"), [backup])
        loop.complete("first")

        restored = AgentLoop.restore(
            loop.dump(),
            client=Client("primary", fail="rate_limit"),
            tools=[],
            fallback_clients=[backup],
        )
        assert restored.complete("again").text == "answer from backup"
