"""Running one chat request against whatever was registered for it.

The entry's client is an `LLM` or an `Agent`, and this deliberately does not
care which. Both answer `complete()`, so an operator can register a bare model
today and swap in an agent with tools and a system prompt tomorrow without the
server, the route, or the client's request changing.

Transposed from `unified-library-ts/src/server/dispatch.ts`, whose dispatch
always builds an `AgentLoop`. Here the entry brings its own, because the Python
`Agent` is already that object.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .oai import external_tool_declarations
from .router import ResolvedTarget


@dataclass(frozen=True)
class DispatchResult:
    """What one request produced."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    provider_response_id: str | None = None


def dispatch(
    *,
    target: ResolvedTarget,
    user_text: str,
    system_prompt: str | None = None,
    external_tools: Any = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    hooks: Any = None,
    agent: Any = None,
) -> DispatchResult:
    """Answer one request with the target's client."""
    client = agent if agent is not None else target.client

    options: dict[str, Any] = {}
    # An Agent carries its own system prompt, and overriding it with whatever
    # the client happened to send would let any caller rewrite the agent's
    # instructions -- which is most of what an agent IS.
    if system_prompt and not _has_own_system(client):
        options["system"] = system_prompt
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature

    tools = _tools_for(target, external_tools)
    if tools:
        options["tools"] = tools

    answer = client.complete(user_text, **options)
    usage = getattr(answer, "usage", None)
    return DispatchResult(
        text=answer.text,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        finish_reason=getattr(answer, "finish_reason", "") or "stop",
        provider_response_id=_response_id(answer),
    )


def _has_own_system(client: Any) -> bool:
    """Whether this client already has instructions of its own."""
    return bool(getattr(client, "system", None))


def _response_id(answer: Any) -> str | None:
    """The provider's own id for this response, when it gave one.

    Kept beside our id rather than instead of it: ours is what a client
    continues a conversation with, and the provider's is what an operator needs
    to look the call up in the provider's console.
    """
    for part in getattr(answer, "parts", ()) or ():
        candidate = getattr(part, "response_id", None)
        if isinstance(candidate, str) and candidate:
            return candidate
    candidate = getattr(answer, "response_id", None)
    return candidate if isinstance(candidate, str) and candidate else None


def _tools_for(target: ResolvedTarget, external: Any) -> list[Any]:
    """The server's tools, then the client's -- and the server's win a collision.

    Order matters and so does the precedence: a client that declares a tool
    named like one of ours must not be able to shadow it, because the model
    would then call what looks like the server's tool and reach something else.
    """
    tools: list[Any] = list(target.internal_tools)
    if not target.allow_external_tools:
        return tools
    taken = {_name_of(t) for t in tools}
    for declaration in external_tool_declarations(external):
        if declaration["name"] in taken:
            continue
        tools.append(_declared_only(declaration))
        taken.add(declaration["name"])
    return tools


def _name_of(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    definition = getattr(tool, "definition", None)
    if isinstance(definition, dict):
        return str(definition.get("name") or "")
    return ""


def _declared_only(declaration: dict[str, Any]) -> Any:
    """A tool the model can see and this process cannot run.

    The body lives in the CLIENT. Refusing loudly when the model calls one is
    the honest outcome: the alternative is a server inventing a result for a
    function it has never seen.
    """
    from ..helpers.tool import Tool

    name = declaration["name"]

    def unavailable(**_: Any) -> str:
        raise RuntimeError(
            f"tool {name!r} is declared by the client, so this server cannot run it. "
            "Answer in text instead."
        )

    unavailable.__name__ = name
    return Tool(unavailable, declaration)


def merge_tools(internal: Sequence[Any], external: Sequence[Any]) -> list[Any]:
    """Internal first, external after, first name wins."""
    merged = list(internal)
    taken = {_name_of(t) for t in merged}
    for tool in external:
        key = _name_of(tool)
        if key not in taken:
            merged.append(tool)
            taken.add(key)
    return merged


__all__ = ["DispatchResult", "dispatch", "merge_tools"]
