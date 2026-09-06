"""One agent as another agent's tool.

Transposed from `unified-library-ts/src/helpers/delegate.ts` and `handoff.ts`.

`delegate` returns the specialist's answer as bare text. `handoff` is the same
tool with the specialist's name and usage attached, which is what a parent
holding three specialists needs: `delegate` alone cannot tell it which one
replied or what the reply cost.

Only the TASK crosses. The specialist starts from its own system prompt and an
empty turn -- that is the point of a specialist, and also the thing that
surprises a caller who expected the conversation to travel with the task.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..wire.interpreter import js_json
from .tool import Tool


def _tool_for(name: str, description: str, run: Callable[[str], Any]) -> Tool:
    """The shared shape: one `task: string` parameter, one string back.

    Built directly rather than through `@tool` because the name and description
    are the CALLER's, not the function's -- a parent may hold the same
    specialist under two names, and deriving them from `__name__` would give
    both the same one.
    """

    def call(task: str) -> Any:
        return run(task)

    call.__name__ = name
    call.__doc__ = description
    return Tool(
        call,
        {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task for the specialist, stated in full.",
                    }
                },
                "required": ["task"],
            },
        },
    )


def delegate(name: str, description: str, agent: Any) -> Tool:
    """Wrap an agent as a tool that returns its reply text.

    The returned tool stays callable directly -- `ask(task="...")` -- so the
    delegation can be tested without a model in the loop.
    """
    return _tool_for(name, description, lambda task: agent.complete(task).text)


def handoff(
    name: str,
    description: str,
    agent: Any,
    *,
    input_filter: Callable[[str], str] | None = None,
) -> Tool:
    """`delegate`, plus who answered and what it cost.

    The result is JSON with `text`, `agent_name` and `usage`. `usage` is None
    when the provider reported none -- a handoff that quietly substituted zeros
    would look exactly like a provider that never sent them, and the specialist
    would appear free.
    """

    def run(task: str) -> str:
        answer = agent.complete(input_filter(task) if input_filter else task)
        usage = answer.usage
        return js_json(
            {
                "text": answer.text,
                "agent_name": name,
                "usage": _usage_of(usage),
            }
        )

    return _tool_for(name, description, run)


def _usage_of(usage: Any) -> dict[str, Any] | None:
    """Usage as plain JSON, or None when the provider reported none.

    "Reported none" is every counter at zero: a real call that produced tokens
    cannot have billed for nothing, so a zeroed usage is the absence of a
    report rather than a report of absence.
    """
    if usage is None:
        return None
    fields = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_tokens": usage.cached_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
    }
    return fields if any(fields.values()) else None


__all__ = ["delegate", "handoff"]
