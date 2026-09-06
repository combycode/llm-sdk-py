"""A dataclass IS the schema.

The TypeScript corpus writes the JSON Schema by hand AND repeats the shape in a
type parameter. Python already holds that shape in the dataclass, so we derive
the schema from it -- stdlib only, `dataclasses.fields` + `typing.get_type_hints`,
no pydantic -- and hand back a real instance.

`structured={"schema": {...}}` still works for schemas that arrive from
elsewhere. It is simply not what the docs lead with.
"""

from dataclasses import dataclass

from _bench import api_key, bench, model

from combycode_llm_sdk import complete


@dataclass
class Weather:
    city: str
    temp_c: float


def main() -> str:
    result = complete(
        model=model(),
        api_key=api_key(),
        prompt='Extract the city and temperature in Celsius. Text: "Paris is 20 degrees C."',
        structured=Weather,
        max_tokens=256,
    )
    return result.parsed.city if result.parsed else ""


bench(main)
