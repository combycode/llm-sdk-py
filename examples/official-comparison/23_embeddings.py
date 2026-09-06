"""One `embed()` across providers; returns vectors plus their dimensions."""

from _bench import api_key, bench, model

from combycode_llm_sdk import embed


def main() -> str:
    result = embed(
        model=model(),
        api_key=api_key(),
        input="hello",
    )
    return str(result.dimensions)


bench(main)
