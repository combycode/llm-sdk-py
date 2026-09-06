"""An async tool is awaited automatically -- no second decorator, no flag.

`@tool` checks with `inspect.iscoroutinefunction`. Making the caller declare it
again would be a second source of truth about the same fact.

Note a sync `complete()` may still take async tools: the loop it owns runs them.
"""

import asyncio

from _bench import api_key, bench, model

from combycode_llm_sdk import complete, tool


@tool
async def fetch_price(symbol: str) -> str:
    """Look up the current price for a ticker symbol."""
    await asyncio.sleep(0)
    return f"{symbol} is 100 USD"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What is the price of ACME? Answer with the number.",
        tools=[fetch_price],
        max_tokens=512,
    )
    return result.text.strip()


bench(main)
