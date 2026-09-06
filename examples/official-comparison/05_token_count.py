"""`count_tokens()` returns the number AND how it was arrived at.

A bare integer would make an estimate indistinguishable from a measurement,
which is how a context-window check ends up wrong at exactly the moment it
matters. `.tokens` is the number; `.strategy` and `.exact` say what it is worth.

Three strategies, chosen per model by the catalog: `tiktoken` locally for
OpenAI, the provider's own endpoint for Anthropic, Google and xAI, and a
calibrated estimate for everything else -- OpenRouter has neither a counting
endpoint nor a knowable tokenizer, since the model behind a slug can change.

So the assertion is NOT "this is exact". It is that `.exact` tells the truth
about `.strategy`: a real tokenizer must not hide behind an estimate, and an
estimate must never claim to be a measurement. That holds on every provider,
which is what makes it worth asserting -- an assertion that only passes on some
of them just teaches you to ignore it.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import count_tokens

#: The strategies that produce a real count rather than an approximation.
MEASURED = {"tiktoken", "api"}


def main() -> int:
    result = count_tokens(
        model=model(),
        api_key=api_key(),
        input="The quick brown fox jumps over the lazy dog.",
    )
    assert result.tokens > 0, "a count of zero is not a count"
    assert result.exact == (result.strategy in MEASURED), (
        f"strategy {result.strategy!r} reported exact={result.exact}, which "
        f"misrepresents what this number is worth"
    )
    return result.tokens


bench(main)
