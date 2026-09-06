"""Nested and optional fields -- and what happens when parsing fails.

`result.parsed` is None when the model returned something unparseable, and the
raw text is still on `result.text` so the caller can decide what to do. It does
not raise for a merely disappointing answer; that would make a retry awkward to
write.
"""

from dataclasses import dataclass

from _bench import api_key, bench, model

from combycode_llm_sdk import complete


@dataclass
class Address:
    city: str
    country: str


@dataclass
class Person:
    name: str
    age: int
    address: Address
    nickname: str | None = None


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt="Ada Lovelace, 36, lives in London, United Kingdom.",
        structured=Person,
        max_tokens=256,
    )
    if result.parsed is None:
        return ""
    return f"{result.parsed.name}/{result.parsed.address.city}"


bench(main)
