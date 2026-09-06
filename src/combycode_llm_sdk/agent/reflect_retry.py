"""Second chances for mistakes the MODEL made.

Transposed from `unified-library-ts/src/agent/reflect-retry.ts`.

Some turns fail in a way the model itself can fix: malformed tool arguments, a
tool name it invented, a call cut off half-written. Resending the identical
request would never fix any of them, which is exactly why the network retry
layer correctly leaves them alone -- and why the run used to just end there.

This feeds the model structured guidance naming the attempt and forbidding a
repeat of the same call, then retries within a bounded budget. Bounded on
purpose: unbounded reflection is an infinite loop with a bill attached.

The Python name for the config is `ReflectAndRetry(max_attempts=..., on=[...])`,
which is what the reviewed examples pass. The TypeScript spells the same two
fields `maxRetries` and `onFinishReasons`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: Which finish reasons are worth a second attempt, by default.
#:
#: `content_filter` is deliberately NOT here. A refusal is usually a decision
#: rather than a mistake, and retrying it spends budget to be refused again.
DEFAULT_REASONS: tuple[str, ...] = ("malformed_tool_call",)

DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class ReflectAndRetry:
    """How many second chances, and for what.

    Off unless configured: a retry is a real request against a real bill, so it
    is the caller's decision and never a default.
    """

    #: Consecutive recoverable failures to tolerate before giving up.
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    #: Finish reasons that count as recoverable.
    on: Sequence[str] = field(default_factory=lambda: list(DEFAULT_REASONS))
    #: When the budget runs out: raise (the default), or hand back the last
    #: unusable response and let the caller decide.
    raise_if_exceeded: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 0:
            raise ValueError("reflect_and_retry: max_attempts must be >= 0")

    def handles(self, finish_reason: str) -> bool:
        return finish_reason in set(self.on)


@dataclass
class _Verdict:
    retry: bool
    attempt: int
    exhausted: bool


class ReflectAndRetryPolicy:
    """The live counter behind a `ReflectAndRetry`.

    Counts CONSECUTIVE failures, and any successful turn resets it: an agent
    that recovers and then fails again much later gets a fresh budget rather
    than inheriting a spent one.
    """

    def __init__(self, config: ReflectAndRetry | None = None) -> None:
        self.config = config or ReflectAndRetry()
        self._consecutive = 0

    @property
    def max_attempts(self) -> int:
        return self.config.max_attempts

    @property
    def raise_if_exceeded(self) -> bool:
        return self.config.raise_if_exceeded

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive

    def handles(self, finish_reason: str) -> bool:
        return self.config.handles(finish_reason)

    def record_success(self) -> None:
        """A turn came back usable -- the streak is broken."""
        self._consecutive = 0

    def record_failure(self) -> _Verdict:
        """Count a recoverable failure and say what to do next.

        `attempt` counts from 1, so the guidance can say "attempt 1 of 3" the
        way a person would.
        """
        self._consecutive += 1
        retry = self._consecutive <= self.config.max_attempts
        return _Verdict(retry=retry, attempt=self._consecutive, exhausted=not retry)

    def reset(self) -> None:
        """Between runs, so one run's failures never spend another's budget."""
        self._consecutive = 0


def reflection_guidance(
    finish_reason: str,
    attempt: int,
    max_attempts: int,
    detail: str | None = None,
) -> str:
    """The corrective message fed back to the model.

    It names the attempt number and explicitly forbids repeating the identical
    call. Without that last part a model tends to re-emit the same malformed
    arguments and spend the whole budget on one mistake.
    """
    lines: Iterable[str] = (
        f'The previous turn failed ({finish_reason}) and produced no usable tool call.',
        f"Details: {detail}" if detail else "",
        "",
        "**Reflection guidance:**",
        f"- This is retry attempt {attempt} of {max_attempts}.",
        "- Analyse the arguments you produced. Do NOT repeat the same call unchanged.",
        "- If a tool name was wrong, choose one from the tools actually available to you.",
        "",
        "Form a new plan from that analysis and try a corrected approach.",
    )
    return "\n".join(line for line in lines if line != "")


__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_REASONS",
    "ReflectAndRetry",
    "ReflectAndRetryPolicy",
    "reflection_guidance",
]
