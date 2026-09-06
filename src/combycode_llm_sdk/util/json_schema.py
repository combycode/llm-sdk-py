"""Checking a value against a JSON Schema, for the cases that actually arrive.

Transposed from `unified-library-ts/src/util/json-schema.ts`.

Covers `type`, `required`, `properties`, `items`, `enum`, `const` and
`additionalProperties`. NOT a Draft 2020-12 implementation: no `$ref`, no
`allOf`/`anyOf`/`oneOf`, no formats, no numeric bounds.

That is a deliberate stopping point rather than an unfinished one. The job here
is to check what an MCP server returned against the `outputSchema` it published,
and a schema shipped by a tool for a model to fill is made of the seven keywords
above. Going further means either a dependency -- which this library does not
have and will not add for one call site -- or a thousand lines of spec that
would then need its own test suite.

Errors come back as a list of readable strings rather than an exception: the
caller is usually handing them to a MODEL, which can act on "expected integer,
got string at $.count" and cannot act on a traceback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..wire.interpreter import js_json

#: How many errors a caller usually wants to show. More is noise -- a model
#: given twenty complaints fixes none of them.
DEFAULT_ERROR_LIMIT = 5


def _js_type(value: Any) -> str:
    """The type name JSON Schema uses, not Python's."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def _matches_type(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        # `bool` is an int subclass in Python, and `true` is not a number in
        # JSON Schema -- so it is excluded explicitly rather than by accident.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        # 3.0 IS an integer to JSON Schema; JSON has one number type, and a
        # value that arrived over the wire as `3.0` must not fail an integer
        # field it would have passed as `3`.
        return isinstance(value, float) and value.is_integer()
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))
    if expected == "null":
        return value is None
    # An unknown keyword is not a failure: the schema may be newer than we are.
    return True


def _same(left: Any, right: Any) -> bool:
    """Structural equality, by the JSON both sides serialise to."""
    return js_json(left) == js_json(right)


def validate_json_schema(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Every way `value` fails `schema`. An empty list means it passed."""
    errors: list[str] = []

    declared = schema.get("type")
    if declared is not None:
        expected = list(declared) if isinstance(declared, (list, tuple)) else [declared]
        if not any(_matches_type(str(t), value) for t in expected):
            errors.append(f"{path}: expected {'|'.join(str(t) for t in expected)}, got {_js_type(value)}")
            # Nothing below this can be meaningful once the type is wrong --
            # checking an object's properties against a string produces noise
            # that buries the one error worth reading.
            return errors

    allowed = schema.get("enum")
    if (
        isinstance(allowed, Sequence)
        and not isinstance(allowed, (str, bytes))
        and not any(_same(option, value) for option in allowed)
    ):
        errors.append(f"{path}: value not in enum")

    if "const" in schema and not _same(schema["const"], value):
        errors.append(f"{path}: value !== const")

    properties = schema.get("properties")
    if isinstance(value, Mapping) and isinstance(properties, Mapping):
        required = schema.get("required")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for name in required:
                if name not in value:
                    errors.append(f"{path}.{name}: required property missing")
        for name, sub in properties.items():
            if name in value and isinstance(sub, Mapping):
                errors.extend(validate_json_schema(sub, value[name], f"{path}.{name}"))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name}: additional property not allowed")

    items = schema.get("items")
    if isinstance(value, (list, tuple)) and isinstance(items, Mapping):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(items, item, f"{path}[{index}]"))

    return errors


__all__ = ["DEFAULT_ERROR_LIMIT", "validate_json_schema"]
