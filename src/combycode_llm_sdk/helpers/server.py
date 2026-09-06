"""A server from a dict of agents.

    server = create_server(agents={"assistant": Agent(model=..., system=...)})

Each key becomes a model id an ordinary OpenAI client can ask for, and the agent
behind it answers -- with its own system prompt, its own tools, its own history.
The long form (`OaiServer(entries=[ServerEntry(...)])`) is still there for a
deployment that wants to say more; this is the shape most of them need.

Transposed from `unified-library-ts/src/helpers/server.ts`, whose `createServer`
takes agent SPECS and builds a loop per request. Here an `Agent` is already that
object, so it is registered directly -- and a caller who wants one per request
passes `agent_loader=` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..server.app import OaiServer
from ..server.auth import AuthPlugin
from ..server.router import ModelCapabilities, ServerEntry


def create_server(
    *,
    agents: Mapping[str, Any] | None = None,
    models: Mapping[str, Any] | None = None,
    entries: Sequence[ServerEntry] = (),
    auth: AuthPlugin | None = None,
    hooks: Any = None,
    agent_loader: Callable[..., Any] | None = None,
    conversation_loader: Any = None,
    capabilities: Mapping[str, ModelCapabilities] | None = None,
    allow_external_tools: bool = True,
) -> OaiServer:
    """Build a server from agents, clients, or explicit entries.

    `agents=` and `models=` are the same mechanism under two names, because the
    distinction is real to a reader even though it is not to the router: an
    agent brings instructions and tools, a client is a model and nothing more.
    Both are registered as `ServerEntry` and both answer `complete()`.
    """
    registered = list(entries)
    seen = {entry.model for entry in registered}
    caps = dict(capabilities or {})

    for source in (agents or {}, models or {}):
        for model_id, client in source.items():
            if model_id in seen:
                raise ValueError(f"create_server: duplicate model id {model_id!r}")
            registered.append(
                ServerEntry(
                    model=model_id,
                    client=client,
                    allow_external_tools=allow_external_tools,
                    capabilities=caps.get(model_id) or ModelCapabilities(),
                )
            )
            seen.add(model_id)

    return OaiServer(
        entries=registered,
        auth=auth,
        hooks=hooks,
        agent_loader=agent_loader,
        conversation_loader=conversation_loader,
    )


__all__ = ["create_server"]
