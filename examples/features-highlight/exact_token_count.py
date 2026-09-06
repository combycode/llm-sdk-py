"""`count_tokens()` says HOW it counted, so you know whether it is exact.

The catalog picks the strategy per model. Three exist and they are not
interchangeable:

    "api"        ask the provider -- exact, costs a round trip
    "tokenizer"  a local BPE tokenizer -- exact, needs the optional extra
    "estimate"   characters/4 -- free, approximate, and honest about it

Returning a bare integer would make an estimate indistinguishable from a
measurement, which is how a context-window check ends up wrong at exactly the
moment it matters.

`tiktoken` is an OPTIONAL extra (`pip install combycode-llm-sdk[tokenizer]`).
Without it the local strategy degrades to "estimate" and SAYS so, rather than
failing or silently lying.
"""

from _check import check, report

from combycode_llm_sdk import Engine, count_tokens

engine = Engine(catalog="defaults", api_keys={"openai": "k"}, register_as_default=False)

result = count_tokens(
    model="openai/gpt-4.1",
    input="The quick brown fox jumps over the lazy dog.",
    exact=False,
    engine=engine,
)

# The number AND its provenance.
check(result.tokens > 0, "expected a positive token count")
check(
    result.strategy in {"api", "tokenizer", "estimate"},
    f"unknown counting strategy {result.strategy!r}",
)
check(isinstance(result.exact, bool), "the caller must be able to tell exact from approximate")

# An estimate must never claim to be exact -- the whole reason the field exists.
if result.strategy == "estimate":
    check(not result.exact, "an estimate must not report itself as exact")

report(tokens=result.tokens, strategy=result.strategy, exact=result.exact)
