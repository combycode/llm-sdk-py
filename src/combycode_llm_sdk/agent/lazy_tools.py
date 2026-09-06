"""Lazy tool loading: `tool_search` and `call_tool`.

Transposed from `unified-library-ts/src/agent/lazy-tools.ts`.

A tool registered with `lazy=True` is NOT placed in the declared `tools` array.
The model finds it by searching and calls it through `call_tool`. The point is
what does NOT move: the declared array never changes, so no discovery can
invalidate the cached prefix. Schemas travel as tool RESULTS, which land in the
history after the prefix.

Measured against declaring everything (308 tools, 6 tasks, 3 reps, both
providers): identical correctness, -72% cost on claude-haiku-4.5 and -97% on
gpt-5.4-nano, for one extra round trip per task.

Three details are load-bearing, each from a measurement that failed first:

1. `call_tool` is SINGULAR. A batching form is returned as a JSON *string*
   rather than an array by claude-haiku about half the time, because a router's
   `input` must be open and an open object cannot be strict, so no grammar holds
   the shape. Batching is not lost -- the model emits several parallel
   `call_tool` calls in one turn instead.

2. `tool_search` REPORTS QUERIES THAT MATCHED NOTHING. Merging results silently
   makes a failed lookup indistinguishable from one whose hits were folded in
   with the rest, and the model then answers confidently from what it did get.

3. Ranking is deliberately weak, and survivable only because the MODEL writes
   the query. Token overlap against raw user text scores 0-1 of 8 on colloquial
   phrasing; the model's rewriting is what carries it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..wire.interpreter import js_json

#: The two names this module owns. Registered like any other tool, so the
#: collision policy applies to them and there is no second registry.
SEARCH_TOOL = "tool_search"
CALL_TOOL = "call_tool"

DEFAULT_LIMIT = 5
#: A real bound, not a formality: returning everything re-creates the cost the
#: feature exists to avoid.
MAX_LIMIT = 20
DEFAULT_MAX_SEARCHES = 5

_STOP_WORDS = frozenset(
    ["the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "it", "that", "this", "with", "return", "returns", "my", "me", "do", "we", "i", "how", "many", "much", "what", "when", "has", "have", "need", "any", "get", "can", "you", "are", "was", "been", "does", "did", "should", "from", "by", "at", "as", "be"]
)

_WORDS = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORDS.split(text.lower()) if len(w) > 2 and w not in _STOP_WORDS]


@dataclass
class LazyToolsConfig:
    """Tuning for lazy exposure. The defaults are what the measurements used.

    There is deliberately no `threshold`: whether deferring pays depends on the
    SIZE of the tool schemas, not their count, so an automatic cutoff would be
    guessing. Mark tools lazy explicitly.
    """

    limit: int = DEFAULT_LIMIT
    max_searches: int = DEFAULT_MAX_SEARCHES


@dataclass
class LazySearchState:
    """The per-RUN search budget.

    Per run, not per agent: carrying it across runs would silently starve a long
    conversation of discovery after the fifth search of its life.
    """

    searches: int = 0


def _name_of(tool: Any) -> str:
    definition = tool.definition
    return str(definition.get("name") or "") if isinstance(definition, Mapping) else ""


def rank_tools(query: str, candidates: Sequence[Any], limit: int) -> list[Any]:
    """Rank by token overlap over name, description and parameter names.

    Name matches count double: a query naming the domain should beat a filler
    whose long description happens to share vocabulary. Local and dependency
    free by design -- no embeddings, no network call on the discovery path.
    """
    wanted = set(tokenize(query))
    if not wanted:
        return []

    scored: list[tuple[int, int, Any]] = []
    for position, tool in enumerate(candidates):
        definition = tool.definition
        if not isinstance(definition, Mapping) or not definition.get("name"):
            continue
        properties = (definition.get("parameters") or {}).get("properties") or {}
        haystack = (
            f"{definition.get('name', '')} "
            f"{definition.get('description', '')} "
            f"{' '.join(properties)}"
        )
        score = sum(1 for w in tokenize(haystack) if w in wanted)
        score += sum(2 for w in tokenize(str(definition.get("name") or "")) if w in wanted)
        if score > 0:
            # `position` breaks ties by registration order. Python's sort is
            # stable, but the key is negated for descending score, so without it
            # equal scores would come back in whatever order the negation left.
            scored.append((-score, position, tool))

    scored.sort(key=lambda row: (row[0], row[1]))
    return [tool for _, _, tool in scored[:limit]]


def unwrap_lazy_call(tool_name: str, arguments: Mapping[str, Any]) -> str | None:
    """The inner tool a `call_tool` invocation targeted, for reporting.

    Returns None for any other call. A trace that attributes every lazy tool
    call to `call_tool` is useless, and that is the one real regression this
    design causes -- so it is fixed at the source rather than documented.
    """
    if tool_name != CALL_TOOL:
        return None
    inner = arguments.get("name")
    return inner if isinstance(inner, str) and inner else None


def create_lazy_tools(
    *,
    lazy_tools: Callable[[], Sequence[Any]],
    eager_names: Callable[[], Sequence[str]],
    state: LazySearchState,
    config: LazyToolsConfig,
    on_search: Callable[[dict[str, Any]], None] | None = None,
) -> list[Any]:
    """The two built-ins, as ordinary tools.

    `lazy_tools` and `eager_names` are callables rather than lists because tools
    can be added after construction and search must see them.
    """
    from ..helpers.tool import Tool

    limit = min(config.limit, MAX_LIMIT)
    max_searches = config.max_searches

    def search(**args: Any) -> str:
        state.searches += 1
        if state.searches > max_searches:
            # Told, not killed: a model that loops on search should hear so and
            # keep the run alive with what it already found.
            return js_json(
                {
                    "error": (
                        f"Search budget exhausted ({max_searches} searches per run). "
                        f"Use the tools you already found."
                    )
                }
            )

        raw = args.get("queries")
        candidates_raw = raw if isinstance(raw, list) else [raw]
        queries = [q for q in candidates_raw if isinstance(q, str) and q.strip()]
        if not queries:
            return js_json({"tools": [], "error": "Pass at least one query string in `queries`."})

        pool = list(lazy_tools())
        hits: dict[str, Any] = {}
        unmatched: list[str] = []
        for query in queries:
            found = rank_tools(query, pool, limit)
            if not found:
                unmatched.append(query)
            for tool in found:
                hits[_name_of(tool)] = tool

        if on_search:
            on_search(
                {"queries": queries, "matched": list(hits), "unmatched": unmatched}
            )

        payload: dict[str, Any] = {"tools": [dict(t.definition) for t in hits.values()]}
        if unmatched:
            payload["unmatched"] = unmatched
            payload["hint"] = (
                "These queries matched no tool. Search again for them using different "
                "words, or tell the user the capability is unavailable."
            )
        return js_json(payload)

    def call(**args: Any) -> Any:
        name = str(args.get("name") or "")
        target = next((t for t in lazy_tools() if _name_of(t) == name), None)

        if target is None:
            if name in list(eager_names()):
                # Declared and callable directly; routing one through here is a
                # mistake worth naming precisely rather than reporting as "no
                # such tool", which sends the model looking for the wrong fix.
                return (
                    f'"{name}" is already available as a normal tool -- call it '
                    f"directly, not through {CALL_TOOL}."
                )
            return (
                f'No tool named "{name}". Call {SEARCH_TOOL} first and use a name '
                f"exactly as returned."
            )

        given = args.get("input")
        if given is not None and not isinstance(given, Mapping):
            kind = "an array" if isinstance(given, list) else type(given).__name__
            return f"`input` must be an object of {name}'s arguments, not {kind}."

        return target.func(**dict(given or {}))

    search_tool = Tool(
        search,
        {
            "name": SEARCH_TOOL,
            "description": (
                "Find the tools you need. Returns their exact names and full argument "
                "schemas. Pass every capability you need as a separate query in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One phrase per capability you need, in your own words.",
                    }
                },
                "required": ["queries"],
            },
        },
    )
    call_tool = Tool(
        call,
        {
            "name": CALL_TOOL,
            "description": (
                "Call one tool returned by tool_search. Pass its exact name and its own "
                "arguments as `input`. To use several tools, call this several times in "
                "the same turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact tool name from tool_search."},
                    "input": {
                        "type": "object",
                        "description": "That tool's own arguments, as an object.",
                        "additionalProperties": True,
                    },
                },
                "required": ["name", "input"],
            },
        },
    )
    return [search_tool, call_tool]


__all__ = [
    "CALL_TOOL",
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_SEARCHES",
    "MAX_LIMIT",
    "SEARCH_TOOL",
    "LazySearchState",
    "LazyToolsConfig",
    "create_lazy_tools",
    "rank_tools",
    "tokenize",
    "unwrap_lazy_call",
]
