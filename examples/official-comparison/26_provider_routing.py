"""Try models in order; the first that answers wins.

`route()` takes the same options as `complete()` plus the candidate list, so
adding a fallback never means restructuring the call.

`result.model` reports which one actually answered -- without it, a silent
fallback to a pricier model is invisible until the invoice arrives.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import route


def main() -> str:
    result = route(
        models=[model(), "anthropic/claude-haiku-4.5"],
        api_key=api_key(),
        prompt="Reply with exactly: OK",
        max_tokens=16,
    )
    return f"{result.text.strip()}@{result.model}"


bench(main)
