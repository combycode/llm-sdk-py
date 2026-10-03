"""A streamed run the consumer walked away from is `aborted`, not an error.

`break` out of the `for` closes the generator where it stands, raising
`GeneratorExit` at the `yield`. The loop's settle path catches `BaseException` --
which `GeneratorExit` is a subclass of -- so until 2026-10-03 every walked-away
stream was recorded with `reason="error"` and fired **`onRunError`**. Alerting
hangs off that hook, so an ordinary UI that stops reading when the user navigates
away looked like a failing agent.

The TypeScript side already had `RunEndReason = … | 'aborted'` with the same
reasoning ("the run is over and is reported as over; it simply did not end on its
own terms"). This is the Python half, found by diffing the two ports' changelogs
during release readiness.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.agent.loop import AgentLoop
from combycode_llm_sdk.bus.hook_bus import HookBus
from combycode_llm_sdk.results import Completion, Usage


class Client:
    """Streams enough events that a consumer can leave part way through."""

    id = "client_1"
    provider = "anthropic"
    model = "claude-haiku-4.5"

    def complete(self, messages: Any, **options: Any) -> Completion:
        return Completion(
            text="done", model="m", finish_reason="stop", usage=Usage(), parts=[]
        )

    def stream(self, messages: Any, options: Any = None, **kw: Any) -> Any:
        for i in range(10):
            yield {"type": "text", "text": f"chunk{i} "}
        yield {"type": "done", "finishReason": "stop"}

    def destroy(self) -> None:
        pass


def run(stop_after: int | None) -> tuple[list[str], list[Any]]:
    """Hooks fired and reports recorded. `stop_after=None` consumes it all."""
    hooks = HookBus()
    fired: list[str] = []

    def record(name: str) -> Any:
        return lambda _e: fired.append(name)

    for name in ("onRunComplete", "onRunError"):
        hooks.on(name, record(name))
    loop = AgentLoop(Client(), system="s", hooks=hooks)
    for seen, _event in enumerate(loop.stream("hello"), start=1):
        if stop_after is not None and seen >= stop_after:
            break
    return fired, list(loop.reports or [])


class TestAStreamedRunTheConsumerWalkedAwayFrom:
    def test_it_settles_through_on_run_complete_not_on_run_error(self) -> None:
        # The regression that matters: `onRunError` is where alerting hangs.
        fired, _ = run(stop_after=3)
        assert fired == ["onRunComplete"]

    def test_it_is_recorded_as_aborted(self) -> None:
        # Pins the REASON, not just the hook -- a fix that routed the hook while
        # still calling it an error would pass the test above.
        _, reports = run(stop_after=3)
        assert len(reports) == 1
        assert reports[0].reason == "aborted"

    def test_it_leaves_a_trace_at_all(self) -> None:
        # The TS bug was that nothing was recorded. Worth its own assertion,
        # because it is the failure that hides every other one.
        _, reports = run(stop_after=1)
        assert len(reports) == 1

    def test_a_fully_consumed_run_is_still_done(self) -> None:
        # The control. Without it the three assertions above would also pass if
        # EVERY run were reported as aborted.
        fired, reports = run(stop_after=None)
        assert fired == ["onRunComplete"]
        assert reports[0].reason == "done"
