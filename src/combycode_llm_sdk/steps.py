"""One list of steps, run in a line or all at once.

`chain(steps)` prompts each step with what the last one produced. `parallel` and
`aparallel` hand every step the SAME input and collect the outputs in step
order. All three take the same `Step` and the same `on_step` callback.

Without that, sequential and parallel are two vocabularies, and deciding a
pipeline should fan out means rewriting every step in it -- so the decision gets
made once, early, when least is known.

Worse, the rewrite is not loud. A step written for one shape still RUNS in the
other; it is merely prompted with the wrong text, and the only symptom is
answers that stopped depending on each other.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

#: How many branches a fan-out runs at once. A bound rather than one thread per
#: step: a hundred-step fan-out should queue, not open a hundred sockets.
DEFAULT_MAX_CONCURRENCY = 8


@dataclass(frozen=True)
class Step:
    """One call in a pipeline.

    `prompt` is a FUNCTION of the input rather than a template string: what a
    step asks depends on what it was handed, and a step that could only
    interpolate would be unable to say "summarise this in the style of that".
    """

    model: str
    prompt: Callable[[Any], Any]
    name: str = ""
    #: Everything else `complete()` takes -- the key, the transport, tools.
    options: Mapping[str, Any] = field(default_factory=dict)

    def label(self, index: int) -> str:
        return self.name or f"step-{index}"


@dataclass(frozen=True)
class ChainStepInfo:
    """What one step did, reported as it finishes."""

    index: int
    name: str
    prompt: Any
    output: str
    #: The whole completion, for a caller that wants usage or cost rather than
    #: just the text.
    result: Any = None


def _run(step: Step, text: Any) -> Any:
    from .helpers.one_shot import complete

    return complete(model=step.model, prompt=step.prompt(text), **dict(step.options))


async def _arun(step: Step, text: Any) -> Any:
    from .helpers.one_shot import acomplete

    return await acomplete(model=step.model, prompt=step.prompt(text), **dict(step.options))


def _info(step: Step, index: int, text: Any, result: Any) -> ChainStepInfo:
    return ChainStepInfo(
        index=index,
        name=step.label(index),
        prompt=step.prompt(text),
        output=result.text,
        result=result,
    )


def chain(
    steps: Sequence[Step], *, on_step: Callable[[ChainStepInfo], Any] | None = None
) -> Callable[[Any], str]:
    """Run the steps in a line, each prompted with the last one's answer."""

    def run(text: Any) -> str:
        current: Any = text
        for index, step in enumerate(steps):
            result = _run(step, current)
            if on_step is not None:
                on_step(_info(step, index, current, result))
            current = result.text
        return str(current)

    return run


def parallel(
    steps: Sequence[Step],
    *,
    on_step: Callable[[ChainStepInfo], Any] | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Callable[[Any], list[str]]:
    """Run every step against the SAME input, answering in step order.

    `on_step` fires from whichever thread finished, in COMPLETION order -- which
    is why it carries the index. The returned list is the one that keeps step
    order, so a caller reading position 2 gets step 2 whatever finished first.
    """

    def run(text: Any) -> list[str]:
        if not steps:
            return []

        def one(pair: tuple[int, Step]) -> tuple[int, Any]:
            index, step = pair
            result = _run(step, text)
            if on_step is not None:
                on_step(_info(step, index, text, result))
            return index, result

        with ThreadPoolExecutor(max_workers=min(len(steps), max_concurrency)) as pool:
            done = list(pool.map(one, enumerate(steps)))
        return [result.text for _, result in sorted(done, key=lambda row: row[0])]

    return run


def aparallel(
    steps: Sequence[Step],
    *,
    on_step: Callable[[ChainStepInfo], Any] | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> Callable[[Any], Any]:
    """`parallel`, awaiting rather than spending a thread per branch.

    The only difference a caller should be able to observe is that one costs
    threads and the other does not -- the answers must agree.
    """

    def run(text: Any) -> Any:
        async def go() -> list[str]:
            if not steps:
                return []
            limit = asyncio.Semaphore(min(len(steps), max_concurrency))

            async def one(index: int, step: Step) -> tuple[int, Any]:
                async with limit:
                    result = await _arun(step, text)
                if on_step is not None:
                    on_step(_info(step, index, text, result))
                return index, result

            done = await asyncio.gather(
                *(one(index, step) for index, step in enumerate(steps))
            )
            return [result.text for _, result in sorted(done, key=lambda row: row[0])]

        return go()

    return run


def achain(
    steps: Sequence[Step], *, on_step: Callable[[ChainStepInfo], Any] | None = None
) -> Callable[[Any], Any]:
    """`chain`, awaited. A line cannot be parallelised, so this only avoids blocking."""

    def run(text: Any) -> Any:
        async def go() -> str:
            current: Any = text
            for index, step in enumerate(steps):
                result = await _arun(step, current)
                if on_step is not None:
                    on_step(_info(step, index, current, result))
                current = result.text
            return str(current)

        return go()

    return run


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "ChainStepInfo",
    "Step",
    "achain",
    "aparallel",
    "chain",
    "parallel",
]
