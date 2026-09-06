"""`select(query)` picks a model from the catalog by tag, not by hard-coding an id.

Returns the single best `provider/slug`, ready to hand to `complete()`. It is
availability-aware -- only providers whose key is configured -- and ranks
cheapest-first. Needs the catalog loaded: `Engine(catalog="defaults")`.

DSL: `key:value`, bare `key` meaning `key:yes`, `key > N` / `key < N`
(inclusive), clauses separated by `;`.

Deterministic: selection reads the bundled catalog, so no network is involved.
"""

from _check import check, report

from combycode_llm_sdk import Engine, select, select_models

engine = Engine(
    catalog="defaults",
    api_keys={"anthropic": "k", "openai": "k"},
    register_as_default=False,
)

# The single best match -- a string you can pass straight to complete().
cheapest_chat = select("type:chat; cheap", engine=engine)
check("/" in cheapest_chat, f"select() returns provider/slug, got {cheapest_chat!r}")

# Only providers we actually hold a key for: selecting a model you cannot call
# is not a useful answer.
check(
    cheapest_chat.split("/")[0] in {"anthropic", "openai"},
    f"selection must respect configured keys, got {cheapest_chat}",
)

# The full ranked list, when you want to see the alternatives or pick your own.
vision_models = select_models("type:chat; vision", engine=engine)
check(len(vision_models) > 0, "expected at least one vision-capable chat model")

# Numeric comparisons, for the constraint that actually matters to you.
big_context = select_models("type:chat; context > 200000", engine=engine)
check(
    all(m.context_window >= 200_000 for m in big_context),
    "context > N must be inclusive and actually filter",
)

report(
    cheapest_chat=cheapest_chat,
    vision=len(vision_models),
    big_context=len(big_context),
)
