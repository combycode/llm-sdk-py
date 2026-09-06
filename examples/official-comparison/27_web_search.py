"""Hosted web search, with the citations it produced.

`result.citations` is always there. A model that searched and cited nothing is
represented as empty, not None -- callers should not need a None-check to loop.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What is the latest stable Python version? Answer with the number.",
        builtin_tools=["web_search"],
        max_tokens=256,
    )
    return f"{result.text.strip()}|cites:{len(result.citations)}"


bench(main)
