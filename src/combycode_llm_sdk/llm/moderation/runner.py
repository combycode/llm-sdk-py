"""Emulated-moderation runner + streaming strategy wrappers.

Transposed from `unified-library-ts/src/llm/moderation/runner.ts`.

The emulated path runs OpenAI's moderations endpoint around a call for any
provider. This module owns mode resolution (native vs emulate, by provider),
input-text extraction (the last user message), a single moderation call (errors
become an `{'error': ...}` entry, report-only), and the buffer / parallel / post
stream wrappers.

It depends only on Layer-2 primitives -- the OpenAI moderations adapter and the
injected fetch -- never on the helpers layer, so the client can use it without an
upward import.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import Any

from ...bus.hook_bus import HookBus
from ..providers.openai.moderations import OpenAIModerationAdapter
from ..types.messages import Message, content_text
from .types import MODERATION_DEFAULT_INTERVAL, MODERATION_DEFAULT_MODEL

#: `type ModerationEntry` (moderation/types.ts:20) -- a `ModerationResult`, or
#: `{'error': str}` when moderation itself failed.
ModerationEntry = dict[str, Any]

#: `interface EmulationConfig` (moderation/types.ts:71) --
#: `{apiKey, model, fetch}`.
EmulationConfig = dict[str, Any]

#: One text -> its moderation entry.
Moderate = Callable[[str], Awaitable[ModerationEntry]]


def resolve_moderation_mode(provider: str, mod: Mapping[str, Any]) -> str:
    """Native for OpenAI, emulated for everyone else -- unless explicitly forced."""
    if mod.get("mode"):
        return str(mod["mode"])
    return "native" if provider == "openai" else "emulate"


def moderation_input_text(messages: Sequence[Message]) -> str:
    """Text of the last user message -- what input moderation runs on."""
    last_user = next((m for m in reversed(list(messages)) if m.get("role") == "user"), None)
    if not last_user:
        return ""
    content = last_user.get("content")
    return content if isinstance(content, str) else content_text(content or [])


def _empty_result() -> ModerationEntry:
    return {"flagged": False, "categories": {}, "categoryScores": {}}


def run_moderation(text: str, cfg: Mapping[str, Any]) -> ModerationEntry:
    """One moderation call, synchronously.

    Empty text -> an un-flagged empty result (nothing to check). A
    moderation-infra failure -> an `{'error': ...}` entry: report-only, and it
    never raises, so a flaky moderations endpoint cannot take down the primary
    call.
    """
    if not text:
        return _empty_result()
    try:
        adapter = OpenAIModerationAdapter({"apiKey": cfg["apiKey"]})
        return _first(adapter.moderate(text, cfg["model"], cfg["fetch"]))
    except Exception as exc:  # noqa: BLE001 -- report-only by contract
        return {"error": str(exc)}


async def arun_moderation(text: str, cfg: Mapping[str, Any]) -> ModerationEntry:
    """The async twin of `run_moderation`, with the same report-only contract."""
    if not text:
        return _empty_result()
    try:
        adapter = OpenAIModerationAdapter({"apiKey": cfg["apiKey"]})
        return _first(await adapter.amoderate(text, cfg["model"], cfg["fetch"]))
    except Exception as exc:  # noqa: BLE001 -- report-only by contract
        return {"error": str(exc)}


def _first(results: list[Any]) -> ModerationEntry:
    """One input, so one result -- but an empty list is possible and must not
    raise on a report-only path."""
    return results[0] if results else _empty_result()


def moderation_model(mod: Mapping[str, Any]) -> str:
    return str(mod.get("model") or MODERATION_DEFAULT_MODEL)


def emit_moderation_zero_cost(hooks: HookBus, model: str) -> None:
    """Emit an honest-zero cost entry for one emulated moderation call.

    The moderations endpoint is free, and a free call still belongs in the
    ledger: a gap and a zero look identical in a total, but only one of them can
    be audited.
    """
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "provider": "openai",
        "model": model,
        "tokens": {"input": 0, "output": 0, "cached": 0, "cacheWrite": 0, "reasoning": 0},
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "reasoning": 0,
            "total": 0,
            "source": "calculated",
        },
        "providerEvidence": {"note": "free: moderations endpoint not billed"},
        "tags": {"provider": "openai", "model": model, "type": "moderation"},
    }
    hooks.emit_sync("onCostEntry", {"entry": entry, "runningTotal": 0})


# -- streaming strategy wrappers ---------------------------------------------


def _output_event(result: ModerationEntry) -> dict[str, Any]:
    return {"type": "moderation", "phase": "output", "result": result, "source": "emulated"}


def awrap_moderated_stream(
    raw: AsyncIterable[dict[str, Any]],
    strategy: str,
    interval: int,
    moderate: Moderate,
) -> AsyncIterator[dict[str, Any]]:
    """Dispatch to the chosen strategy. Interval defaults are applied by the caller."""
    step = interval if interval > 0 else MODERATION_DEFAULT_INTERVAL
    if strategy == "post":
        return _wrap_post(raw, moderate)
    if strategy == "parallel":
        return _wrap_parallel(raw, step, moderate)
    return _wrap_buffer(raw, step, moderate)


def wrap_moderated_stream_sync(
    raw: Iterable[dict[str, Any]],
    strategy: str,
    interval: int,
    moderate: Callable[[str], ModerationEntry],
) -> Iterator[dict[str, Any]]:
    """The synchronous dispatcher.

    `post` and `buffer` are sequential by definition and transpose directly.
    **`parallel` does not exist here**, and is not silently downgraded: its whole
    contract is that chunks keep flowing while a moderation call is in flight,
    which without an event loop means a thread -- and the API contract rejects
    threads-behind-sync as firmly as it rejects `asyncio.run` behind sync. A
    caller who asks for it is told where to get it rather than quietly given
    something slower that still claims the name.
    """
    step = interval if interval > 0 else MODERATION_DEFAULT_INTERVAL
    if strategy == "post":
        return _wrap_post_sync(raw, moderate)
    if strategy == "parallel":
        raise ValueError(
            "moderation.stream.strategy='parallel' needs an event loop and is "
            "available on AsyncLLM only. Use 'buffer' (the default) or 'post' "
            "on the synchronous client."
        )
    return _wrap_buffer_sync(raw, step, moderate)


def _wrap_post_sync(
    raw: Iterable[dict[str, Any]], moderate: Callable[[str], ModerationEntry]
) -> Iterator[dict[str, Any]]:
    """post: forward everything, then one moderation pass on the full text."""
    acc = ""
    for ev in raw:
        if ev.get("type") == "text":
            acc += ev.get("text") or ""
        yield ev
    if acc:
        yield _output_event(moderate(acc))


def _wrap_buffer_sync(
    raw: Iterable[dict[str, Any]], interval: int, moderate: Callable[[str], ModerationEntry]
) -> Iterator[dict[str, Any]]:
    """buffer: hold chunks, moderate cumulative text at each boundary, and emit
    the result BEFORE releasing the held chunks -- so the flag never trails the
    text it refers to."""
    acc = ""
    checked_at = 0
    hold: list[dict[str, Any]] = []
    for ev in raw:
        hold.append(ev)
        if ev.get("type") == "text":
            acc += ev.get("text") or ""
            if len(acc) - checked_at >= interval or "\n" in (ev.get("text") or ""):
                checked_at = len(acc)
                yield _output_event(moderate(acc))
                yield from hold
                hold = []
    # Tail: moderate any text produced since the last check, then flush the rest.
    if len(acc) > checked_at:
        yield _output_event(moderate(acc))
    yield from hold


async def _wrap_post(
    raw: AsyncIterable[dict[str, Any]], moderate: Moderate
) -> AsyncIterator[dict[str, Any]]:
    """post: forward everything, then one moderation pass on the full text."""
    acc = ""
    async for ev in raw:
        if ev.get("type") == "text":
            acc += ev.get("text") or ""
        yield ev
    if acc:
        yield _output_event(await moderate(acc))


async def _wrap_buffer(
    raw: AsyncIterable[dict[str, Any]], interval: int, moderate: Moderate
) -> AsyncIterator[dict[str, Any]]:
    """buffer: hold chunks, moderate cumulative text at each boundary, and emit
    the result BEFORE releasing the held chunks -- so the flag never trails the
    text it refers to."""
    acc = ""
    checked_at = 0
    hold: list[dict[str, Any]] = []
    async for ev in raw:
        hold.append(ev)
        if ev.get("type") == "text":
            acc += ev.get("text") or ""
            if len(acc) - checked_at >= interval or "\n" in (ev.get("text") or ""):
                checked_at = len(acc)
                yield _output_event(await moderate(acc))
                for held in hold:
                    yield held
                hold = []
    # Tail: moderate any text produced since the last check, then flush the rest.
    if len(acc) > checked_at:
        yield _output_event(await moderate(acc))
    for held in hold:
        yield held


async def _wrap_parallel(
    raw: AsyncIterable[dict[str, Any]], interval: int, moderate: Moderate
) -> AsyncIterator[dict[str, Any]]:
    """parallel: forward chunks immediately, moderate concurrently, and surface
    each result as soon as it resolves.

    The TypeScript races the source iterator against the pending moderations with
    `Promise.race`; `asyncio.wait(FIRST_COMPLETED)` is the same primitive, and
    the pending task must be REUSED across iterations rather than recreated --
    awaiting a fresh `__anext__` each time would drop the chunk the previous one
    had already pulled.
    """
    iterator = raw.__aiter__()
    pending: set[asyncio.Task[Any]] = set()
    acc = ""
    checked_at = 0
    raw_done = False
    raw_next: asyncio.Task[Any] | None = asyncio.ensure_future(iterator.__anext__())

    def schedule(text: str) -> None:
        pending.add(asyncio.ensure_future(moderate(text)))

    try:
        while not raw_done or pending:
            waiting: set[asyncio.Task[Any]] = set(pending)
            if raw_next is not None:
                waiting.add(raw_next)
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task is raw_next:
                    raw_next = None
                    try:
                        event = task.result()
                    except StopAsyncIteration:
                        raw_done = True
                        if len(acc) > checked_at:
                            schedule(acc)  # final pass on the tail
                        continue
                    yield event
                    if event.get("type") == "text":
                        acc += event.get("text") or ""
                        if len(acc) - checked_at >= interval:
                            checked_at = len(acc)
                            schedule(acc)
                    raw_next = asyncio.ensure_future(iterator.__anext__())
                else:
                    pending.discard(task)
                    yield _output_event(task.result())
    finally:
        # A consumer that stops early (a `break`) leaves both the source pull and
        # any in-flight moderation running. Neither has anywhere to deliver.
        if raw_next is not None:
            raw_next.cancel()
        for task in pending:
            task.cancel()


__all__ = [
    "EmulationConfig",
    "Moderate",
    "ModerationEntry",
    "arun_moderation",
    "awrap_moderated_stream",
    "emit_moderation_zero_cost",
    "moderation_input_text",
    "moderation_model",
    "resolve_moderation_mode",
    "run_moderation",
    "wrap_moderated_stream_sync",
]
