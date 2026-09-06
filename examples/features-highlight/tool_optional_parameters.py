"""Optional tool parameters come from the signature, not a second list.

Most real tools have them -- a unit, a page, a filter the caller may leave out.
In TypeScript they are declared with `optional: [...]`, a second list that can
disagree with the params it describes.

Python already encodes optionality twice over: a default value, and `| None` in
the annotation. `@tool` reads both, so `required` in the emitted JSON Schema is
derived from the function rather than restated beside it.

Deterministic: builds the schema, sends nothing.
"""

from typing import Literal

from _check import check, report

from combycode_llm_sdk import tool


@tool
def search_orders(
    query: str,
    limit: int = 10,
    unit: Literal["metric", "imperial"] | None = None,
) -> str:
    """Search orders.

    Args:
        query: What to search for.
        limit: How many results to return.
        unit: Measurement system for any quantities.
    """
    return f"{query}:{limit}:{unit}"


schema = search_orders.schema

check(schema["name"] == "search_orders", "name comes from the function")
check("Search orders." in schema["description"], "description comes from the docstring")

props = schema["parameters"]["properties"]
check(set(props) == {"query", "limit", "unit"}, f"unexpected properties: {sorted(props)}")

# Only the parameter with no default is required.
check(schema["parameters"]["required"] == ["query"], f"required should be [query], got {schema['parameters']['required']}")

# Per-argument docs come from the docstring's Args section: a model chooses
# arguments far better when each one is described, and the description already
# exists in the place a Python developer writes it.
check("What to search for" in props["query"]["description"], "arg docs come from the docstring")

# `Literal` becomes an enum -- the type already said what the legal values are.
check(props["unit"]["enum"] == ["metric", "imperial"], "Literal should emit an enum")

report(required=schema["parameters"]["required"], properties=sorted(props))
