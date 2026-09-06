"""`select()` -- pick a model by what it can do, not by hard-coding an id.

Transposed from `unified-library-ts/src/helpers/select-model.ts`.

    select("type:chat; cheap", engine=engine)   # -> "openai/gpt-5.4-nano"

A tiny tag DSL, `;`-separated:

    key:value      exact                type:chat, status:stable
    key            means `key:yes`      vision, tools, reasoning
    key > N        at least N           context > 200k
    key < N        at most N            price < 1

Availability-aware: only providers whose key is configured, because selecting a
model you cannot call is not a useful answer. Ranked cheapest-first, tie-broken
by the newer version.

`filter_facets()` is the other half, and it is not decoration. A compact DSL is
unguessable in practice, so the feature goes unused or gets used wrongly -- and
an unknown tag that returned "no models matched" would read like an answer
rather than a mistake. So an unknown key RAISES, and the vocabulary is published
as data derived from the same constants the parser matches on. The picker and
the parser cannot drift, the drift being the actual bug.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Named cutoffs, overridable per call.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "price.low": 1,
    "price.mid": 5,
    "context.small": 32_000,
    "context.large": 200_000,
}

#: One-word shorthands a picker can offer as presets.
DEFAULT_TAGS: dict[str, str] = {
    "cheap": "price:low",
    "free": "price:free",
    "tiny": "context:small",
    "huge": "context:large",
}

#: Query token -> the capability flag it reads.
CAP_KEYS: dict[str, str] = {
    "vision": "vision",
    "tools": "toolUse",
    "audio": "audio",
    "structured": "structuredOutput",
}

#: Query tokens that test membership of `capabilities.builtinTools` -- hosted
#: server tools. `search` is kept as an alias for `web_search`.
BUILTIN_TOOL_KEYS: dict[str, str] = {
    "search": "web_search",
    "web_search": "web_search",
    "web_fetch": "web_fetch",
    "code_interpreter": "code_interpreter",
}

KNOWN_KEYS = frozenset(
    {"price", "context", "reasoning", "type", "tier", "status", "provider", "active"}
    | set(CAP_KEYS)
    | set(BUILTIN_TOOL_KEYS)
)

_CLAUSE = re.compile(r"^([a-z][\w.]*)\s*(>=|<=|>|<|:)?\s*(.*)$", re.IGNORECASE)
_NUMBER = re.compile(r"^([\d.]+)\s*([kKmM]?)$")
_NO = re.compile(r"^(no|off|false|0)$", re.IGNORECASE)


@dataclass(frozen=True)
class FilterFacet:
    """One filter a UI can offer, and what it accepts."""

    key: str
    #: Group heading for a picker.
    category: str
    label: str
    #: The values this key accepts. Empty for a bare flag, which the parser
    #: reads as `key:yes`.
    values: Sequence[str] = ()
    #: True when the key also accepts `> N` / `< N`.
    numeric: bool = False
    #: True when a bare `key` is meaningful on its own.
    bare: bool = False


@dataclass(frozen=True)
class _Criterion:
    key: str
    op: str
    value: str


def filter_aliases() -> dict[str, str]:
    """The shorthands the parser expands before matching (`cheap` -> `price:low`)."""
    return dict(DEFAULT_TAGS)


def filter_facets(catalog: Any = None) -> list[FilterFacet]:
    """Every clause the parser understands, as data.

    Published because the alternative is a UI hand-listing the same tags: a
    second copy of this vocabulary, drifting from the parser the first time
    either moves, and drifting INVISIBLY because a wrong tag reads as "no models
    matched" rather than as an error.

    `type`, `provider`, `status` and `tier` take their values from the CATALOG,
    because those are open sets -- a provider ships a new model type and a
    hard-coded list is wrong that day. Without a catalog they come back empty
    rather than guessed: an empty list is honest, a stale one is not.
    """
    models = list(catalog.list()) if catalog is not None else []

    def distinct(key: str) -> list[str]:
        seen = {str(dict(m).get(key)) for m in models if dict(m).get(key)}
        return sorted(seen)

    facets = [
        FilterFacet("type", "what it is", "Type", distinct("type")),
        FilterFacet("provider", "what it is", "Provider", distinct("provider")),
        FilterFacet("price", "cost", "Price", ["free", "low", "mid", "high"], numeric=True),
        FilterFacet("tier", "cost", "Billing tier", _tiers(models)),
        FilterFacet("context", "inputs", "Context window", ["small", "large"], numeric=True),
        FilterFacet("reasoning", "thinking", "Reasoning", ["yes", "no"], bare=True),
        FilterFacet("status", "availability", "Status", distinct("status")),
        FilterFacet("active", "availability", "Offered now", ["yes", "no"], bare=True),
    ]
    facets += [
        FilterFacet(key, "inputs", key, ["yes", "no"], bare=True) for key in CAP_KEYS
    ]
    facets += [
        FilterFacet(key, "hosted tools", key.replace("_", " "), ["yes", "no"], bare=True)
        for key in BUILTIN_TOOL_KEYS
    ]
    return facets


def _tiers(models: Sequence[Any]) -> list[str]:
    """Billing tiers are per model, so they are collected rather than declared."""
    found: set[str] = set()
    for model in models:
        pricing = dict(model).get("pricing") or {}
        found.update((pricing.get("tiers") or {}).keys())
    return sorted(found)


def _parse_number(value: str) -> float:
    match = _NUMBER.match(value.strip())
    if not match:
        return float("nan")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "m":
        return number * 1_000_000
    if suffix == "k":
        return number * 1_000
    return number


def _parse_query(query: str | Sequence[str], tags: Mapping[str, str]) -> list[_Criterion]:
    clauses = query.split(";") if isinstance(query, str) else list(query)
    out: list[_Criterion] = []
    for raw in clauses:
        clause = raw.strip()
        if not clause:
            continue
        expansion = tags.get(clause.lower())
        if expansion:
            out.extend(_parse_query(expansion, tags))
            continue
        match = _CLAUSE.match(clause)
        if not match:
            continue
        key = match.group(1).lower()
        operator = match.group(2)
        op = ">" if operator in (">", ">=") else "<" if operator in ("<", "<=") else ":"
        value = (match.group(3) or "").strip() or "yes"
        if key not in KNOWN_KEYS:
            raise ValueError(
                f"select: unknown filter {key!r}. Known: {', '.join(sorted(KNOWN_KEYS))}"
            )
        out.append(_Criterion(key=key, op=op, value=value))
    return out


def _price_of(model: Any, tier: str | None = None) -> float | None:
    pricing = dict(model).get("pricing") or {}
    if tier and tier != "standard":
        tiered = (pricing.get("tiers") or {}).get(tier) or {}
        if tiered.get("inputPerMTok") is not None:
            return float(tiered["inputPerMTok"])
    value = pricing.get("inputPerMTok")
    return float(value) if isinstance(value, (int, float)) else None


def _matches(
    model: Any, crit: _Criterion, thresholds: Mapping[str, float], tier: str | None
) -> bool:
    entry = dict(model)
    if crit.key == "price":
        price = _price_of(model, tier)
        if price is None:
            return False
        if crit.op == "<":
            return price <= _parse_number(crit.value)
        if crit.op == ">":
            return price >= _parse_number(crit.value)
        if crit.value == "free":
            return price == 0
        if crit.value == "low":
            return price <= thresholds["price.low"]
        if crit.value == "mid":
            return price <= thresholds["price.mid"]
        if crit.value == "high":
            return price > thresholds["price.mid"]
        return False

    if crit.key == "context":
        window = entry.get("contextWindow")
        if not isinstance(window, (int, float)):
            return False
        if crit.op == "<":
            return window <= _parse_number(crit.value)
        if crit.op == ">":
            return window >= _parse_number(crit.value)
        if crit.value == "small":
            return window <= thresholds["context.small"]
        if crit.value == "large":
            return window >= thresholds["context.large"]
        return window >= _parse_number(crit.value)

    if crit.key == "reasoning":
        supported = bool((entry.get("reasoning") or {}).get("supported"))
        return not supported if _NO.match(crit.value) else supported
    if crit.key in ("type", "status", "provider"):
        return bool(entry.get(crit.key) == crit.value)
    if crit.key == "tier":
        return crit.value in ((entry.get("pricing") or {}).get("tiers") or {})
    if crit.key == "active":
        return entry.get("active") is False if _NO.match(crit.value) else entry.get("active") is not False

    capabilities = entry.get("capabilities") or {}
    tool = BUILTIN_TOOL_KEYS.get(crit.key)
    if tool:
        listed = tool in (capabilities.get("builtinTools") or [])
        # The legacy `webSearch` boolean, for catalog entries written before the
        # list existed.
        legacy = tool == "web_search" and bool(capabilities.get("webSearch"))
        has = listed or legacy
        return not has if _NO.match(crit.value) else has

    has = bool(capabilities.get(CAP_KEYS[crit.key]))
    return not has if _NO.match(crit.value) else has


def _version_key(model: Any) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", str(dict(model).get("version") or ""))]


@dataclass
class SelectPrefs:
    """Overridable cutoffs and shorthands."""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)


def select_models(
    query: str | Sequence[str],
    *,
    engine: Any = None,
    provider: str | None = None,
    tier: str | None = None,
    prefs: SelectPrefs | None = None,
) -> list[Any]:
    """Every matching model, cheapest first."""
    from ..catalog.catalog import resolve_catalog
    from .engine import default_engine

    engine = engine if engine is not None else default_engine()
    catalog = resolve_catalog(engine.catalog if engine is not None else None)

    thresholds = {**DEFAULT_THRESHOLDS, **dict((prefs.thresholds if prefs else {}) or {})}
    tags = {**DEFAULT_TAGS, **dict((prefs.tags if prefs else {}) or {})}
    criteria = _parse_query(query, tags)
    filters_on_active = any(c.key == "active" for c in criteria)

    keys = dict(getattr(engine, "api_keys", {}) or {}) if engine is not None else {}
    available = {name for name, key in keys.items() if key}

    candidates = []
    for model in catalog.list(provider):
        entry = dict(model)
        if provider and entry.get("provider") != provider:
            continue
        # Availability-aware: a model you have no key for is not an answer.
        if available and entry.get("provider") not in available:
            continue
        # Retired models are excluded unless the query asks about `active`.
        if not filters_on_active and entry.get("active") is False:
            continue
        if all(_matches(model, c, thresholds, tier) for c in criteria):
            candidates.append(model)

    return sorted(
        candidates,
        key=lambda m: (
            _price_of(m, tier) if _price_of(m, tier) is not None else float("inf"),
            # Negated so the NEWER version wins a price tie.
            [-n for n in _version_key(m)],
        ),
    )


def select(
    query: str | Sequence[str],
    *,
    engine: Any = None,
    provider: str | None = None,
    tier: str | None = None,
    prefs: SelectPrefs | None = None,
) -> str | None:
    """The single best match as `provider/slug`, ready for `complete(model=...)`."""
    best = select_models(query, engine=engine, provider=provider, tier=tier, prefs=prefs)
    if not best:
        return None
    entry = dict(best[0])
    return f"{entry.get('provider')}/{entry.get('model')}"


__all__ = [
    "BUILTIN_TOOL_KEYS",
    "CAP_KEYS",
    "DEFAULT_TAGS",
    "DEFAULT_THRESHOLDS",
    "KNOWN_KEYS",
    "FilterFacet",
    "SelectPrefs",
    "filter_aliases",
    "filter_facets",
    "select",
    "select_models",
]
