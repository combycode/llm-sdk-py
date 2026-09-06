"""Register everything, declare nothing, and re-check on the call.

Transposed from `unified-library-ts/src/plugins/tool-catalog/`.

Every tool in `tools=` is paid for on every turn, and a model reading sixty
declarations chooses worse for having read them. A `ToolCatalog` registers
everything and declares nothing: the model searches, and only what it matched
goes on the wire.

The other half is what makes that safe. `search(agent_id=...)` filtering by
scope is a COURTESY to the model, not a boundary -- the agent id is optional, an
operator's own search passes none, and a tool name can be hallucinated without
any search at all. So `call()` re-checks the scope from scratch: a tool this
agent could never have discovered is still refused when it asks for it by name.

Two divergences from the TypeScript, both from the reviewed examples:

- `search` takes a SENTENCE and matches it word by word, sharing the ranking
  `agent/lazy_tools.py` already uses, rather than a `{name, description}`
  substring pair. A model asks in a sentence, and no description contains the
  whole phrase.
- It is CAPPED by default. A query that matched fifty-eight tools and declared
  them all would have spent more than declaring everything.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .agent.lazy_tools import rank_tools
from .permissions import PermissionPolicy, PermissionTarget

#: `AgentScope(tool_names=ALL_TOOLS)` -- every tool, present and future.
ALL_TOOLS = "*"

#: How many tools one search may declare. The cap is the other half of the
#: saving, so it is on by default and `limit=None` turns it off.
DEFAULT_SEARCH_LIMIT = 5


class ToolNotFound(LookupError):
    """No tool of that name is registered."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"tool {tool_name!r} is not registered")
        self.tool_name = tool_name


class NoToolAccess(PermissionError):
    """The caller may not call this tool at all."""

    def __init__(self, source: str, tool_name: str, detail: str) -> None:
        super().__init__(f"agent {source!r} cannot call {tool_name!r}: {detail}")
        self.source = source
        self.tool_name = tool_name
        self.detail = detail


class PermissionDenied(PermissionError):
    """The tool is in scope, but the policy refused what it would touch."""

    def __init__(
        self,
        source: str,
        target: PermissionTarget,
        action: str,
        detail: str | None = None,
    ) -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(
            f"permission denied: {source!r} -> {target.get('kind')} {action!r}{suffix}"
        )
        self.source = source
        self.target = target
        self.action = action
        self.detail = detail


class ToolRegistrationError(ValueError):
    """A tool could not be registered."""


@dataclass(frozen=True)
class TargetDeclaration:
    """What a tool says it will touch, checked BEFORE it runs.

    Declared rather than observed: a policy consulted after the write has
    already happened is an audit log, not a permission.
    """

    kind: str
    value: Mapping[str, Any] | None = None
    pattern: str | Sequence[str] | None = None

    def as_target(self) -> PermissionTarget:
        return {"kind": self.kind, **dict(self.value or {})}


@dataclass(frozen=True)
class AgentScope:
    """Which tools one agent may reach."""

    #: Names, or `ALL_TOOLS`.
    tool_names: Sequence[str] | str = ()
    #: External tools are off unless said otherwise: the ones that leave the
    #: process are exactly the ones worth naming deliberately.
    external_allowed: bool = False
    #: A policy just for this agent. Falls back to the catalog's.
    policy: PermissionPolicy | None = None

    def allows(self, name: str) -> bool:
        if self.tool_names == ALL_TOOLS:
            return True
        return name in tuple(self.tool_names)


@dataclass(frozen=True)
class ToolCallResult:
    """What one call produced."""

    call_id: str
    output: Any
    duration_ms: float


@dataclass
class CatalogedTool:
    """A registered tool, plus what it declared about itself."""

    tool: Any
    category: str = "internal"
    declared_targets: Sequence[TargetDeclaration] = ()
    declared_actions: Sequence[str] = ()

    @property
    def definition(self) -> Mapping[str, Any]:
        schema: Mapping[str, Any] = self.tool.definition
        return schema

    @property
    def name(self) -> str:
        return str(self.tool.name)


class ToolCatalog:
    """Sixty tools registered, a handful declared, the scope enforced twice."""

    def __init__(self, policy: PermissionPolicy | None = None, hooks: Any = None) -> None:
        self.policy = policy
        self.hooks = hooks
        self._tools: dict[str, CatalogedTool] = {}
        self._scopes: dict[str, AgentScope] = {}

    # -- registration --------------------------------------------------------

    def register(
        self,
        tool: Any,
        *,
        category: str = "internal",
        declared_targets: Sequence[TargetDeclaration] = (),
        declared_actions: Sequence[str] = (),
    ) -> CatalogedTool:
        """Add one tool. A name already taken is an error, not a replacement.

        Replacing silently is how a model ends up told about one tool and given
        another; `unregister` first when that is genuinely what is meant.
        """
        from .helpers.tool import Tool
        from .helpers.tool import tool as as_tool

        resolved: Tool = tool if isinstance(tool, Tool) else as_tool(tool)
        if resolved.name in self._tools:
            raise ToolRegistrationError(
                f"tool {resolved.name!r} is already registered (unregister first to replace)"
            )
        entry = CatalogedTool(
            tool=resolved,
            category=category,
            declared_targets=tuple(declared_targets),
            declared_actions=tuple(declared_actions),
        )
        self._tools[resolved.name] = entry
        return entry

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def size(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    # -- scope ---------------------------------------------------------------

    def set_agent_scope(self, agent_id: str, scope: AgentScope) -> None:
        self._scopes[agent_id] = scope

    def remove_agent_scope(self, agent_id: str) -> None:
        self._scopes.pop(agent_id, None)

    def get_agent_scope(self, agent_id: str) -> AgentScope | None:
        return self._scopes.get(agent_id)

    def _in_scope(self, scope: AgentScope, entry: CatalogedTool) -> bool:
        if not scope.allows(entry.name):
            return False
        return not (entry.category == "external" and not scope.external_allowed)

    def visible_to(self, agent_id: str) -> list[Any]:
        """Every tool this agent may reach. Empty when it has no scope."""
        scope = self._scopes.get(agent_id)
        if scope is None:
            return []
        return [e.tool for e in self._tools.values() if self._in_scope(scope, e)]

    # -- discovery -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        limit: int | None = DEFAULT_SEARCH_LIMIT,
    ) -> list[Any]:
        """The tools a sentence describes, best first.

        `agent_id` filters by scope -- a courtesy to the model, so it is never
        told about a tool it may not call. Omitted, the whole catalog is
        searched, which is what an operator's own search wants.
        """
        candidates = list(self._tools.values())
        if agent_id is not None:
            scope = self._scopes.get(agent_id)
            if scope is None:
                return []
            candidates = [e for e in candidates if self._in_scope(scope, e)]

        tools = [e.tool for e in candidates]
        # `limit=None` means uncapped, so the ranker is given a bound it cannot
        # exceed rather than a sentinel it would have to interpret.
        ranked = rank_tools(query, tools, len(tools))
        return ranked if limit is None else ranked[:limit]

    def get_definition(self, name: str, agent_id: str | None = None) -> Mapping[str, Any] | None:
        entry = self._tools.get(name)
        if entry is None:
            return None
        if agent_id is not None:
            scope = self._scopes.get(agent_id)
            if scope is None or not self._in_scope(scope, entry):
                return None
        return entry.definition

    # -- execution -----------------------------------------------------------

    def call(
        self,
        tool_name: str,
        *,
        source: str,
        arguments: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> ToolCallResult:
        """Run one tool, having checked that this caller may.

        The order matters: scope, then policy, then the body. A policy consulted
        after the tool ran would be reporting the write rather than preventing
        it.
        """
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        started = time.perf_counter() * 1000
        self._emit(
            "onInternalToolCallStart",
            {
                "callId": call_id,
                "source": source,
                "toolName": tool_name,
                "arguments": dict(arguments or {}),
                "correlationId": correlation_id,
            },
        )

        try:
            entry = self._tools.get(tool_name)
            if entry is None:
                raise ToolNotFound(tool_name)

            scope = self._scopes.get(source)
            if scope is None:
                # Default deny, and it has to be here rather than only in
                # `search`: a name can be hallucinated without any search.
                raise NoToolAccess(source, tool_name, "no scope registered for this agent")
            if not scope.allows(tool_name):
                raise NoToolAccess(source, tool_name, "tool not in agent scope")
            if entry.category == "external" and not scope.external_allowed:
                raise NoToolAccess(source, tool_name, "external tools disabled for this agent")

            self._check_targets(entry, source, scope.policy or self.policy)

            output = entry.tool.func(**dict(arguments or {}))
            duration = time.perf_counter() * 1000 - started
            self._emit(
                "onInternalToolCallComplete",
                {
                    "callId": call_id,
                    "source": source,
                    "toolName": tool_name,
                    "output": output,
                    "durationMs": duration,
                },
            )
            return ToolCallResult(call_id=call_id, output=output, duration_ms=duration)
        except Exception as exc:
            self._emit(
                "onInternalToolCallError",
                {
                    "callId": call_id,
                    "source": source,
                    "toolName": tool_name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "durationMs": time.perf_counter() * 1000 - started,
                },
            )
            raise

    def _check_targets(
        self, entry: CatalogedTool, source: str, policy: PermissionPolicy | None
    ) -> None:
        """Rule on every target the tool declared, before it runs.

        No policy means no opinion, and the scope check has already run: a
        catalog used purely to keep the tool block small should not need a rule
        list to work at all.
        """
        if policy is None or not entry.declared_targets:
            return
        actions = entry.declared_actions or ("*",)
        for declaration in entry.declared_targets:
            target = declaration.as_target()
            for action in actions:
                decision = policy.check(source, target, action)
                if not decision.allow:
                    raise PermissionDenied(source, target, action, decision.reason)

    def _emit(self, name: str, payload: Mapping[str, Any]) -> None:
        if self.hooks is not None:
            self.hooks.emit_sync(name, dict(payload))

    def __repr__(self) -> str:
        return f"<ToolCatalog {len(self._tools)} tool(s), {len(self._scopes)} scope(s)>"


#: The declaration a tool makes about itself, re-exported under the name the
#: examples import.
__all__ = [
    "ALL_TOOLS",
    "DEFAULT_SEARCH_LIMIT",
    "AgentScope",
    "CatalogedTool",
    "NoToolAccess",
    "PermissionDenied",
    "TargetDeclaration",
    "ToolCallResult",
    "ToolCatalog",
    "ToolNotFound",
    "ToolRegistrationError",
]
