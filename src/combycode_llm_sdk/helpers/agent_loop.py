"""The tool loop `complete(tools=...)` owns.

Scenarios 06 through 09: one tool, two independent tools, two dependent tools,
and an async tool. All four are ONE `complete()` call with no hand-rolled
while-loop at the call site -- which is the whole claim the corpus makes.

What it is not: `unified-library-ts/src/agent/loop.ts` is 1661 lines because it
also carries guardrails, approvals, a context registry, lazy tool discovery and
reflect-retry. None of that is needed to answer a tool call, and porting it as
one piece would have made all of it unverified at once. This is the loop, alone.
`Agent` -- the durable object with history and hooks -- is the separate port.

The turn structure is the one every adapter already builds for:

    assistant  [tool_call, tool_call, ...]
    tool       [tool_result, tool_result, ...]   <- ONE message, all results

which is why parallel calls (`07`) need nothing here beyond answering every call
in the round rather than the first.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..results import Completion
from .tool import Tool
from .tool import tool as as_tool

#: How many assistant turns a single `complete()` may take before giving up.
#: A model that answers its own tool result with another call to the same tool
#: loops forever otherwise, and it does happen -- the ceiling turns an unbounded
#: spend into an error a caller can read.
DEFAULT_MAX_STEPS = 12


class ToolLoopLimit(RuntimeError):
    """The step ceiling was reached with the model still calling tools."""


def _partition(tools: Sequence[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Split what this loop runs from what the PROVIDER runs.

    A hosted retrieval corpus hands back a tool SPEC -- `{"type":
    "file_search", ...}` -- not a callable. There is nothing here to execute:
    the search happens inside the provider's own call and its results are
    already in the response. So a mapping is passed through to the wire as a
    declaration and never entered into the executable index, which is the same
    treatment `_calls_of` already gives a builtin tool call.
    """
    runnable: list[Any] = []
    hosted: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, Mapping):
            hosted.append(dict(item))
        else:
            runnable.append(item)
    return runnable, hosted


def _index(tools: Sequence[Any]) -> dict[str, Tool]:
    """Name -> tool, rejecting a duplicate name rather than shadowing one.

    Two tools with one name is not a preference to resolve: whichever wins, the
    model was told about a tool that will not run, and the loser's body is
    silently dead. The call site is the only place that knows which was meant.
    """
    out: dict[str, Tool] = {}
    for item in tools:
        resolved = item if isinstance(item, Tool) else as_tool(item)
        if resolved.name in out:
            raise ValueError(
                f"two tools are named {resolved.name!r}. Names are how the model "
                f"asks for one, so they must be unique."
            )
        out[resolved.name] = resolved
    return out


def _definitions(index: Mapping[str, Tool]) -> list[dict[str, Any]]:
    return [dict(t.definition) for t in index.values()]


def _calls_of(result: Completion) -> list[Any]:
    """The tool calls this turn asked for, ignoring the hosted ones.

    `builtin_tool_calls` ran on the provider's side and already have their
    results in the response; answering them again would be inventing output for
    a call nobody made locally.
    """
    return [p for p in result.tool_calls if p.type == "tool_call" and p.name]


def _arguments(call: Any) -> dict[str, Any]:
    """The call's arguments as a mapping.

    NOT a JSON decode. Every response registry already decodes -- OpenAI with
    `json.loads`, Google with `or {}`, and Anthropic's `input` arrives as an
    object -- so a string here would be a bug in the registry, and swallowing it
    with a second decode would hide the layer that has to fix it. The JSON string
    a stream carries is a `tool_call_delta` EVENT and never becomes a `Part`.

    A call with no arguments (`get_user_city()`) arrives as None, which is the
    one case this exists for.
    """
    raw = call.arguments
    return dict(raw) if isinstance(raw, Mapping) else {}


def _as_content(value: Any) -> Any:
    """A tool's return value, in the shape a `tool_result` part carries.

    A string passes through. A list of parts passes through -- a tool may return
    an image. Anything else is JSON, because the alternative is `str()`, and
    `str()` of a dict is Python's repr with single quotes, which is not JSON and
    which models do misread.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _failure(exc: BaseException) -> str:
    """What the model is told when a tool body raised.

    The exception goes BACK to the model rather than out to the caller: a tool
    that fails on one argument is a fact the model can act on -- it retries with
    another city, or says it could not find out. Raising instead would throw away
    a conversation that was one turn from an answer.
    """
    return f"{type(exc).__name__}: {exc}"


def _result_part(call: Any, content: Any, *, is_error: bool = False) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool_result", "id": call.id, "content": content}
    if is_error:
        part["isError"] = True
    return part


def _unknown(name: str, index: Mapping[str, Tool]) -> str:
    return (
        f"No tool named {name!r}. Available: {', '.join(sorted(index)) or '(none)'}."
    )


def _assistant_turn(result: Completion, calls: Sequence[Any]) -> dict[str, Any]:
    """The assistant turn to append before the results.

    Rebuilt from the parts rather than passed through: the provider's own message
    shape is not what the next request is built from, and the text parts have to
    travel with the calls or a model that narrated its plan loses it.
    """
    content: list[dict[str, Any]] = []
    for part in result.parts:
        if part.type == "text" and part.text:
            content.append({"type": "text", "text": part.text})
    for call in calls:
        item: dict[str, Any] = {
            "type": "tool_call",
            "id": call.id,
            "name": call.name,
            "arguments": _arguments(call),
        }
        meta = call.raw.get("_meta") if isinstance(call.raw, Mapping) else None
        if meta:
            # Google's thought signatures live here, and a turn that drops them is
            # rejected on the next request.
            item["_meta"] = meta
        content.append(item)
    return {"role": "assistant", "content": content}


def _as_messages(input_: Any) -> list[dict[str, Any]]:
    """Whatever `complete` was given, as a message list the loop can append to."""
    if isinstance(input_, str):
        return [{"role": "user", "content": [{"type": "text", "text": input_}]}]
    if isinstance(input_, list) and input_ and isinstance(input_[0], Mapping):
        if "role" in input_[0]:
            return [dict(m) for m in input_]
        return [{"role": "user", "content": list(input_)}]
    return [{"role": "user", "content": input_}]


def run_tools(
    call_model: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Completion],
    input_: Any,
    tools: Sequence[Any],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Completion:
    """Drive the loop synchronously. `call_model(messages, definitions)`.

    Async tool bodies are run here too (`09`): a sync `complete()` may take them,
    and the loop it owns is what makes that true.
    """
    runnable, hosted = _partition(tools)
    index = _index(runnable)
    definitions = [*_definitions(index), *hosted]
    messages = _as_messages(input_)

    for _ in range(max_steps):
        result = call_model(messages, definitions)
        calls = _calls_of(result)
        if not calls:
            return result
        messages.append(_assistant_turn(result, calls))
        messages.append({"role": "tool", "content": _dispatch_sync(calls, index)})

    raise ToolLoopLimit(
        f"still calling tools after {max_steps} turns. Raise max_steps= if the task "
        f"genuinely needs more, or check the tool is answering what was asked."
    )


async def arun_tools(
    call_model: Callable[
        [list[dict[str, Any]], list[dict[str, Any]]], Awaitable[Completion]
    ],
    input_: Any,
    tools: Sequence[Any],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Completion:
    """The twin of `run_tools`, awaiting the model and the tools."""
    runnable, hosted = _partition(tools)
    index = _index(runnable)
    definitions = [*_definitions(index), *hosted]
    messages = _as_messages(input_)

    for _ in range(max_steps):
        result = await call_model(messages, definitions)
        calls = _calls_of(result)
        if not calls:
            return result
        messages.append(_assistant_turn(result, calls))
        messages.append({"role": "tool", "content": await _dispatch_async(calls, index)})

    raise ToolLoopLimit(
        f"still calling tools after {max_steps} turns. Raise max_steps= if the task "
        f"genuinely needs more, or check the tool is answering what was asked."
    )


def _run_one_sync(call: Any, index: Mapping[str, Tool]) -> dict[str, Any]:
    found = index.get(str(call.name))
    if found is None:
        return _result_part(call, _unknown(str(call.name), index), is_error=True)
    try:
        value = found.func(**_arguments(call))
        if inspect.isawaitable(value):
            # An async body reached from sync `complete()`. There is no running
            # loop on this thread -- `run_tools` is only called from one that has
            # none -- so this owns one for the duration.
            value = asyncio.run(_await(value))
        return _result_part(call, _as_content(value))
    except Exception as exc:  # noqa: BLE001 -- a tool body may raise anything, and
        # by contract every failure goes back to the model as its result.
        return _result_part(call, _failure(exc), is_error=True)


async def _await(value: Any) -> Any:
    return await value


def _dispatch_sync(calls: Sequence[Any], index: Mapping[str, Tool]) -> list[dict[str, Any]]:
    """Answer every call in the round, concurrently when there is more than one.

    `07` is two independent calls in one turn, and running them in sequence turns
    two waits into one sum for no reason. Order is preserved regardless: the
    results are matched to calls by id, but a caller reading the transcript should
    see them in the order the model asked.
    """
    if len(calls) == 1:
        return [_run_one_sync(calls[0], index)]
    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        return list(pool.map(lambda c: _run_one_sync(c, index), calls))


async def _run_one_async(call: Any, index: Mapping[str, Tool]) -> dict[str, Any]:
    found = index.get(str(call.name))
    if found is None:
        return _result_part(call, _unknown(str(call.name), index), is_error=True)
    try:
        if found.is_async:
            value = await found.func(**_arguments(call))
        else:
            # A sync body inside an async loop goes to a thread: a tool that
            # blocks on a file or a socket would otherwise stall the event loop
            # and, with it, every other tool in the same round.
            value = await asyncio.to_thread(lambda: found.func(**_arguments(call)))
            if inspect.isawaitable(value):
                value = await value
        return _result_part(call, _as_content(value))
    except Exception as exc:  # noqa: BLE001 -- a tool body may raise anything, and
        # by contract every failure goes back to the model as its result.
        return _result_part(call, _failure(exc), is_error=True)


async def _dispatch_async(
    calls: Sequence[Any], index: Mapping[str, Tool]
) -> list[dict[str, Any]]:
    if len(calls) == 1:
        return [await _run_one_async(calls[0], index)]
    return list(await asyncio.gather(*(_run_one_async(c, index) for c in calls)))


__all__ = ["DEFAULT_MAX_STEPS", "ToolLoopLimit", "arun_tools", "run_tools"]
