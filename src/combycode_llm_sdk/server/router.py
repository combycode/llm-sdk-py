"""Which registered target a model id names.

The indirection is the feature. A client asks for `"fast"`, and what answers is
whatever `"fast"` was registered with -- an Anthropic client today, a local
model tomorrow -- with nothing in the client changing. A server that echoed the
provider's own model id back would leak that choice into every caller.

Transposed from `unified-library-ts/src/server/router.ts`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    """What an operator wants clients to know about a registered model."""

    supports_previous_response_id: bool = False
    state_retention_days: int | None = None
    tools: bool | None = None
    vision: bool | None = None
    reasoning: bool | None = None
    max_context: int | None = None


@dataclass
class ServerEntry:
    """One model id, and what answers it."""

    #: The id as it appears in requests -- the operator's name, not a provider's.
    model: str
    #: An `LLM` or an `Agent`. Both answer `complete()`, and the server does not
    #: need to know which it has.
    client: Any
    #: Tools the SERVER runs. Never sent to the client and never overridable
    #: by it.
    internal_tools: Sequence[Any] = ()
    #: Whether tools named in the request are declared to the model at all.
    allow_external_tools: bool = True
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


@dataclass(frozen=True)
class ResolvedTarget:
    """A registration, with its defaults already applied."""

    entry: ServerEntry
    client: Any
    model: str
    internal_tools: Sequence[Any]
    allow_external_tools: bool
    supports_previous_response_id: bool
    state_retention_days: int | None


class ModelNotRegistered(LookupError):
    """No entry answers that model id."""

    def __init__(self, model: str, known: Sequence[str]) -> None:
        super().__init__(
            f'model "{model}" is not registered. Known: [{", ".join(known)}]'
        )
        self.model = model
        self.known = tuple(known)


class ModelRouter:
    """The registry behind the model id."""

    def __init__(self, entries: Sequence[ServerEntry] = ()) -> None:
        self._by_model: dict[str, ServerEntry] = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: ServerEntry) -> None:
        """Add one. A duplicate id is an error, not a replacement.

        Silently replacing would mean a client's requests start reaching a
        different model with no signal anywhere -- same id in the logs, same id
        in the response.
        """
        if entry.model in self._by_model:
            raise ValueError(f'ModelRouter: duplicate model id "{entry.model}"')
        self._by_model[entry.model] = entry

    def unregister(self, model: str) -> bool:
        return self._by_model.pop(model, None) is not None

    def __len__(self) -> int:
        return len(self._by_model)

    def models(self) -> list[str]:
        return list(self._by_model)

    def resolve(self, model: str) -> ResolvedTarget:
        entry = self._by_model.get(model)
        if entry is None:
            raise ModelNotRegistered(model, self.models())
        return ResolvedTarget(
            entry=entry,
            client=entry.client,
            model=entry.model,
            internal_tools=tuple(entry.internal_tools),
            allow_external_tools=entry.allow_external_tools,
            supports_previous_response_id=entry.capabilities.supports_previous_response_id,
            state_retention_days=entry.capabilities.state_retention_days,
        )

    def listing(self) -> list[dict[str, Any]]:
        """What `/v1/models` returns: OpenAI's shape, plus what it cannot carry.

        The extra metadata lives under its own key rather than beside OpenAI's
        fields, so a client that does not know about it ignores it and one that
        does can find it without guessing.
        """
        created = int(time.time())
        rows: list[dict[str, Any]] = []
        for entry in self._by_model.values():
            caps = entry.capabilities
            rows.append(
                {
                    "id": entry.model,
                    "object": "model",
                    "created": created,
                    "owned_by": "user",
                    "orxa": {
                        "routing": "direct",
                        "apis": ["chat.completions"],
                        "supports_previous_response_id": caps.supports_previous_response_id,
                        "state_retention_days": caps.state_retention_days,
                        "capabilities": {
                            "tools": caps.tools,
                            "vision": caps.vision,
                            "reasoning": caps.reasoning,
                            "max_context": caps.max_context,
                        },
                    },
                }
            )
        return rows

    def __repr__(self) -> str:
        return f"<ModelRouter {list(self._by_model)}>"


__all__ = [
    "ModelCapabilities",
    "ModelNotRegistered",
    "ModelRouter",
    "ResolvedTarget",
    "ServerEntry",
]
