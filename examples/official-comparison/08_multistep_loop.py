"""Two DEPENDENT tools: the second needs the first one's answer.

Still one call. No hand-rolled while-loop, no provider-specific agents runtime.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete, tool


@tool
def get_user_city() -> str:
    """Get the user's current city."""
    return "Paris"


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="What is the weather where I am?",
        tools=[get_user_city, get_weather],
        max_tokens=512,
    )
    return result.text.strip()


bench(main)
