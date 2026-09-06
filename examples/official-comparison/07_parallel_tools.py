"""Two independent tools the model may call in one turn.

`complete()` owns the loop, including dispatching both calls concurrently where
the provider returns them together.
"""

from _bench import api_key, bench, model

from combycode_llm_sdk import complete, tool


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny in {city}"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"2.1 million in {city}"


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="For Paris: what is the weather, and the population?",
        tools=[get_weather, get_population],
        max_tokens=512,
    )
    return result.text.strip()


bench(main)
