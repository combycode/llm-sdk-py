"""One tool, one call -- and the same file works on every provider.

The decorator is the divergence most worth reviewing. Name comes from the
function, description from the docstring, JSON Schema from the type hints. The
TypeScript corpus restates all three by hand, which is three chances to drift
from what the function actually does.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What is the weather in Paris?",
        tools=[get_weather],
        max_tokens=512,
    )
    return result.text.strip()


bench(main)
