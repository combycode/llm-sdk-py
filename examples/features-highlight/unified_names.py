"""Our slug is never what goes on the wire -- and any spelling of it resolves.

The SDK translates a normalised slug (`claude-haiku-4.5`) into the provider's
exact callable id (`claude-haiku-4-5-20251001`). Three separate things:

    catalog key   claude-haiku-4.5             what you write
    API name      claude-haiku-4-5-20251001    what is sent
    alias         claude-haiku-4-5             what Anthropic's docs show

All three find the model, and so does the wrong case. This matters because a
spelling the catalog missed had no price and no capabilities, and was forwarded
verbatim into a 404 -- a spelling difference becoming an outage.

What is SENT is always callable: an id the provider accepts goes unchanged; one
it would reject is corrected to the canonical id rather than forwarded.

Deterministic: catalog only, no network.
"""

from _check import check, report

from combycode_llm_sdk import Engine

engine = Engine(catalog="defaults", register_as_default=False)
catalog = engine.catalog

# Every spelling reaches the same entry.
for spelling in ["claude-haiku-4.5", "claude-haiku-4-5", "claude-haiku-4-5-20251001", "CLAUDE-HAIKU-4-5"]:
    info = catalog.get("anthropic", spelling)
    check(info is not None, f"{spelling} should resolve")
    check(info.model == "claude-haiku-4.5", f"{spelling} should reach the canonical entry")
    # Pricing is the half that failed silently: None reads as free.
    check(catalog.get_pricing("anthropic", spelling) is not None, f"{spelling} should be priced")

# Our slug translates to the pinned snapshot.
check(
    catalog.resolve_model_id("anthropic", "claude-haiku-4.5") == "claude-haiku-4-5-20251001",
    "a catalog slug must translate to the provider's callable id",
)

# An id the provider itself accepts is sent unchanged -- the undated name means
# "latest 4.5", and rewriting it to a date would pin a caller who asked to float.
check(
    catalog.resolve_model_id("anthropic", "claude-haiku-4-5") == "claude-haiku-4-5",
    "a provider alias is a deliberate choice and must survive verbatim",
)

# A spelling the provider would REJECT is corrected rather than forwarded.
check(
    catalog.resolve_model_id("google", "gemini-2-5-flash") == "gemini-2.5-flash",
    "a non-callable spelling must be corrected, not forwarded into a 404",
)

# An unknown model still passes through, so a fine-tune or a same-day release works.
check(
    catalog.resolve_model_id("openai", "ft:gpt-4.1:acme::AbcXyz") == "ft:gpt-4.1:acme::AbcXyz",
    "an unknown id must pass through untouched",
)

report(checked=["slug", "alias", "dated", "wrong-case", "corrected", "unknown"])
