"""Where the runner looks a tool up.

A registry over backends rather than a dict, because the interesting deployments
have more than one source: the tools in this repo, the ones a team keeps in a
directory, the ones an index serves. First-added wins on an id conflict, so a
local override is a matter of ordering rather than of deleting something.

Synchronous, unlike the TypeScript. A lookup is a lookup; making it async there
was the cost of a backend that might do I/O, and a Python backend that needs to
can do the I/O in `list()` and cache it -- which is what the registry does with
the result anyway.

Transposed from `unified-library-ts/src/plugins/internal-tools/registry.ts` and
`backends/local.ts`.
"""

from __future__ import annotations

# `builtins.list[...]` throughout this module, not `list[...]`: the registry has a
# `list()` method (the backend protocol's name, and the TypeScript's), which shadows
# the builtin inside the class body and makes every annotation using it unresolvable.
import builtins
from collections.abc import Mapping, Sequence
from typing import Any

from .types import InternalTool, ToolBackend, ToolFilter

#: How many results a bare `search()` returns.
DEFAULT_SEARCH_LIMIT = 20

#: How well a model must score on a tool before a filter accepts it.
DEFAULT_MIN_SCORE = 0.8


class LocalBackend:
    """Tools written in this process. Registration is a method call, not a scan."""

    name = "local"

    def __init__(self) -> None:
        self._tools: dict[str, InternalTool] = {}

    def register(self, tool: InternalTool) -> LocalBackend:
        """Add one tool. An id already taken is an error, not a replacement.

        Replacing silently is how a caller ends up holding a tool's id and
        getting another tool's prompt -- with the same id in the logs, which
        makes it unfindable afterwards. `replace()` says it out loud.
        """
        if tool.id in self._tools:
            raise ValueError(f"tool {tool.id} is already registered (use replace() to overwrite)")
        self._tools[tool.id] = tool
        return self

    def replace(self, tool: InternalTool) -> LocalBackend:
        """Add or overwrite, deliberately."""
        self._tools[tool.id] = tool
        return self

    def unregister(self, tool_id: str) -> bool:
        return self._tools.pop(tool_id, None) is not None

    @property
    def size(self) -> int:
        return len(self._tools)

    def list(self) -> builtins.list[InternalTool]:
        return list(self._tools.values())

    def get(self, tool_id: str) -> InternalTool | None:
        return self._tools.get(tool_id)


class ToolRegistry:
    """Every backend's tools, as one namespace."""

    def __init__(self) -> None:
        self._backends: list[ToolBackend] = []
        self._cache: dict[str, InternalTool] | None = None

    # -- backends ------------------------------------------------------------

    def add_backend(self, backend: ToolBackend) -> ToolRegistry:
        """Add a source. Returns self, so a registry can be built in one expression."""
        if any(b.name == backend.name for b in self._backends):
            raise ValueError(f"backend {backend.name!r} is already registered")
        self._backends.append(backend)
        self.invalidate()
        return self

    @property
    def backends(self) -> Sequence[ToolBackend]:
        """The sources, in the order they are consulted."""
        return tuple(self._backends)

    def remove_backend(self, name: str) -> bool:
        for index, backend in enumerate(self._backends):
            if backend.name == name:
                del self._backends[index]
                self.invalidate()
                return True
        return False

    def invalidate(self) -> None:
        """Forget the merged view. A backend whose contents changed calls this."""
        self._cache = None

    # -- lookup --------------------------------------------------------------

    def get(self, tool_id: str) -> InternalTool | None:
        return self._ensure().get(tool_id)

    def list(self) -> builtins.list[InternalTool]:
        return list(self._ensure().values())

    def __len__(self) -> int:
        return len(self._ensure())

    def find(self, filter_: ToolFilter, catalog: Any = None) -> builtins.list[InternalTool]:
        """Every tool matching a filter."""
        return [t for t in self.list() if self._matches(t, filter_, catalog)]

    def search(
        self,
        query: str,
        *,
        namespace: str | None = None,
        limit: int | None = DEFAULT_SEARCH_LIMIT,
    ) -> builtins.list[InternalTool]:
        """Tools whose name, tags or description answer a query, best first."""
        tools = self.list()
        text = query.lower().strip()
        if not text:
            return tools if limit is None else tools[:limit]

        scored: builtins.list[tuple[int, int, InternalTool]] = []
        for index, tool in enumerate(tools):
            if namespace is not None and tool.namespace != namespace:
                continue
            score = self._score(tool, text)
            if score > 0:
                # The index breaks ties by registration order, so a search is
                # reproducible rather than dependent on dict iteration luck.
                scored.append((-score, index, tool))
        scored.sort()
        ranked = [tool for _, _, tool in scored]
        return ranked if limit is None else ranked[:limit]

    def models_for(
        self, tool_id: str, *, catalog: Any = None, min_score: float = DEFAULT_MIN_SCORE
    ) -> builtins.list[str]:
        """Models the catalog records as compatible with this tool.

        Empty without a catalog: no record is not the same answer as no
        compatible model, and returning a guess here would put it into a model
        chain that then looks benchmark-backed.
        """
        if catalog is None:
            return []
        matching: builtins.list[str] = []
        for info in catalog.list():
            compat = self._compat_of(info, tool_id)
            if compat is not None and compat >= min_score:
                matching.append(f"{info['provider']}/{info['model']}")
        return matching

    # -- internal ------------------------------------------------------------

    def _ensure(self) -> dict[str, InternalTool]:
        if self._cache is not None:
            return self._cache
        merged: dict[str, InternalTool] = {}
        for backend in self._backends:
            for tool in backend.list():
                merged.setdefault(tool.id, tool)
        self._cache = merged
        return merged

    def _matches(self, tool: InternalTool, filter_: ToolFilter, catalog: Any) -> bool:
        if filter_.namespace is not None and tool.namespace != filter_.namespace:
            return False
        if filter_.prefix is not None and not tool.id.startswith(filter_.prefix):
            return False
        if filter_.tag is not None and filter_.tag not in tuple(tool.tags):
            return False
        if filter_.model is not None and catalog is not None:
            info = catalog.get(filter_.model.provider, filter_.model.model)
            compat = self._compat_of(info, tool.id) if info is not None else None
            if compat is None:
                return False
            if compat < (filter_.model.min_score or DEFAULT_MIN_SCORE):
                return False
        return True

    @staticmethod
    def _compat_of(info: Any, tool_id: str) -> float | None:
        """The recorded score for one tool on one model, if the catalog has one."""
        record = dict(info).get("toolCompat") if info is not None else None
        if not isinstance(record, Mapping):
            return None
        entry = record.get(tool_id)
        if not isinstance(entry, Mapping):
            return None
        score = entry.get("score")
        return float(score) if isinstance(score, (int, float)) else None

    @staticmethod
    def _score(tool: InternalTool, query: str) -> int:
        name = tool.name.lower()
        tags = [t.lower() for t in tool.tags]
        if name == query:
            return 100
        if name.startswith(query):
            return 80
        if query in name:
            return 60
        if query in tags:
            return 50
        if any(query in tag for tag in tags):
            return 40
        if query in tool.description.lower():
            return 20
        return 0


__all__ = [
    "DEFAULT_MIN_SCORE",
    "DEFAULT_SEARCH_LIMIT",
    "LocalBackend",
    "ToolRegistry",
]
