"""Caching is measured, not asserted.

The second call should report cached input tokens. `usage.cached_tokens` is the
number the caller can actually check -- the point of the scenario is that the
saving is visible, not that a flag was set.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import LLM

LONG_CONTEXT = "The quick brown fox jumps over the lazy dog. " * 400


def main() -> str:
    llm = LLM(model=model(), api_key=api_key())
    ask = "Reply with exactly: OK"

    llm.complete(f"{LONG_CONTEXT}\n\n{ask}", cache=True, max_tokens=16)
    second = llm.complete(f"{LONG_CONTEXT}\n\n{ask}", cache=True, max_tokens=16)

    return f"cached:{second.usage.cached_tokens}"


bench(main)
