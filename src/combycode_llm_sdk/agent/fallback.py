"""Backup models for a step of an agent run.

Transposed from `unified-library-ts/src/agent/fallback.ts`.

`route()` already falls over between models, but only for a one-shot
completion. An agent run is where it matters more: a rate limit on step 7 of a
nine-step run threw away six steps of work and every tool call they paid for,
and the caller's only recourse was to start the whole run again.

Four rules, and the third is the one that cannot be got wrong:

1. **Each client is tried once per step.** Retrying one client is a different
   concern and already belongs to the network engine; doing it here too would
   retry a single failure twice over, at two layers.
2. **Only a failure another model could survive moves on.** A 429 or a 503 is
   worth another model; a bad request, an auth failure or a content filter is
   the same request failing the same way everywhere, so it propagates
   immediately rather than being asked of every backup in turn.
3. **Never switch once output has reached the caller.** A streamed turn can
   fail after several events. Falling over then splices two models into one
   turn: the consumer has already rendered half an answer and the backup starts
   a different one. So the chain is live only until the first event.
4. **Every step starts from the primary.** A rate limit is transient, and a run
   that fell over once should not spend the rest of its life on the backup.

The chain reports WHO served, because a report or a span naming the primary when
a backup answered is worse than no attribution at all.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

#: Failures worth trying another model for. The same set `route()` uses -- a
#: model swap cannot fix auth, a malformed request, a content filter, or a
#: prompt that is simply too long.
DEFAULT_AGENT_FALLBACK_KINDS: tuple[str, ...] = (
    "rate_limit",
    "server_error",
    "model_not_found",
    "timeout",
    "network",
    "quota_exceeded",
    "unsupported",
)


@dataclass(frozen=True)
class FallbackNotice:
    """What happened on the way to an answer, for the warning the loop emits."""

    #: The client that failed.
    from_model: str
    #: The client tried next.
    to_model: str
    kind: str | None
    message: str


def _kind_of(error: BaseException) -> str | None:
    """The classified failure kind, when the error carries one.

    Read by attribute rather than by `isinstance(LLMError)`: the agent package
    does not import the network package for a type test, and anything carrying a
    `kind` string is making the same claim.
    """
    kind = getattr(error, "kind", None)
    return kind if isinstance(kind, str) else None


def _retryable(error: BaseException, kinds: frozenset[str]) -> bool:
    kind = _kind_of(error)
    return kind is not None and kind in kinds


def client_chain(primary: Any, backups: Sequence[Any] | None) -> list[Any]:
    """The ordered clients a step may use.

    The primary is always first; a duplicate of it among the backups is dropped,
    because trying the same client twice is exactly the retry this layer is not
    doing.
    """
    chain = [primary]
    for client in backups or ():
        if not any(client is seen for seen in chain):
            chain.append(client)
    return chain


@dataclass
class FallbackRun:
    """One step's chain, with the warning wired to the agent's bus."""

    chain: Sequence[Any]
    kinds: Sequence[str] | None = None
    on_fallback: Callable[[FallbackNotice], None] | None = None

    def _kind_set(self) -> frozenset[str]:
        return frozenset(self.kinds if self.kinds is not None else DEFAULT_AGENT_FALLBACK_KINDS)

    def _hand_off(self, index: int, error: BaseException) -> None:
        if self.on_fallback is None:
            return
        self.on_fallback(
            FallbackNotice(
                from_model=getattr(self.chain[index], "model", "?"),
                to_model=getattr(self.chain[index + 1], "model", "?"),
                kind=_kind_of(error),
                message=str(error),
            )
        )


def complete_with_fallback(
    run: FallbackRun, messages: Any, options: dict[str, Any]
) -> tuple[Any, Any]:
    """A buffered step, with backups. Returns `(response, served_by)`."""
    kinds = run._kind_set()
    last = len(run.chain) - 1
    index = 0
    while True:
        client = run.chain[index]
        try:
            return client.complete(messages, **options), client
        except BaseException as error:
            # The last client's error is the caller's error: handing them a
            # wrapper would bury the provider's own message, the useful half.
            if index == last or not _retryable(error, kinds):
                raise
            run._hand_off(index, error)
            index += 1


def stream_with_fallback(
    run: FallbackRun, messages: Any, options: dict[str, Any], served_by: list[Any]
) -> Iterator[Any]:
    """A streamed step, with backups -- and only until the first event.

    `served_by` is a one-element list rather than a return value because a
    generator's return value is not reachable from a `for` loop: the caller needs
    to know which client answered in order to stamp the step, and would otherwise
    have to assume the primary.
    """
    kinds = run._kind_set()
    last = len(run.chain) - 1
    index = 0
    while True:
        client = run.chain[index]
        served_by[0] = client
        emitted = False
        try:
            events: Iterable[Any] = client.stream(messages, options)
            for event in events:
                emitted = True
                yield event
            return
        except BaseException as error:
            # `emitted` comes FIRST. Once part of this model's turn has reached
            # the consumer, a backup cannot continue it -- it would start a
            # second answer mid-sentence -- so the failure is theirs to handle.
            if emitted or index == last or not _retryable(error, kinds):
                raise
            run._hand_off(index, error)
            index += 1
