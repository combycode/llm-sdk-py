"""`select()` takes a query nobody can guess. `filter_facets()` hands over the vocabulary.

A compact DSL -- `type:chat; vision; cheap` -- is unguessable in practice, which
means the feature goes unused or gets used wrongly: an unknown tag comes back as
"no models matched", which reads like an answer rather than a mistake.

`filter_facets()` exists so a UI can OFFER the vocabulary instead of expecting
it to be typed. It is derived from the same constants the parser matches on, so
the picker and the parser cannot drift -- the drift being the actual bug, since
a stale tag fails silently.

Deterministic: catalog only, no network.
"""

from _check import check, report

from combycode_llm_sdk import Engine, filter_aliases, filter_facets, select_models

engine = Engine(catalog="defaults", api_keys={"openai": "k"}, register_as_default=False)

facets = filter_facets(engine.catalog)
check(len(facets) > 0, "expected a non-empty facet vocabulary")

# Every offered value must be a clause the parser ACCEPTS. Not "returns results"
# -- an empty result is legitimate; being rejected is what a stale value causes.
rejected: list[str] = []
for facet in facets:
    for value in facet.values:
        try:
            select_models(f"{facet.key}:{value}", engine=engine)
        except ValueError as exc:
            rejected.append(f"{facet.key}:{value} -- {exc}")
check(not rejected, f"facet values the parser rejects: {rejected}")

# Aliases are the friendly names (`cheap`) for a longer clause.
for alias, expansion in filter_aliases().items():
    try:
        select_models(alias, engine=engine)
        select_models(expansion, engine=engine)
    except ValueError as exc:
        check(False, f"alias {alias} -> {expansion} rejected: {exc}")

# Open sets come from the catalog rather than a hard-coded list, so a new model
# type appears in the picker the day it is added.
types = next(f for f in facets if f.key == "type").values
check("chat" in types, "the type facet should be catalog-derived")

# An unknown tag must RAISE, not quietly return nothing.
try:
    select_models("definitely_not_a_key:1", engine=engine)
    check(False, "an unknown filter key must raise, not return an empty list")
except ValueError:
    pass

report(facets=len(facets), types=len(types), aliases=len(filter_aliases()))
