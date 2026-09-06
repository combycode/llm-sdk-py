from _bench import api_key, bench, model

from combycode_llm_sdk import complete


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        system="You are a terse assistant. Answer in one word.",
        prompt="What is the capital of France?",
        max_tokens=16,
    )
    return result.text.strip()


bench(main)
