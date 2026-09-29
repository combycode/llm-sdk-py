"""Checking a value against a JSON Schema, for the cases that actually arrive.

Transposed from `unified-library-ts/src/util/json-schema.ts`.

Covers `type`, `required`, `properties`, `items`, `enum`, `const`,
`additionalProperties`, boolean schemas and local `$ref`. NOT a Draft 2020-12
implementation: no `allOf`/`anyOf`/`oneOf`, no formats, no numeric bounds.

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


def _resolve_pointer(root: Any, ref: str) -> Any:
    """A local JSON Pointer (`#`, `#/$defs/Name`) against the starting document.

    Returns `_UNRESOLVED` for anything that does not land on a schema, including
    `#anchor` forms -- those are names, not pointers, and finding one would mean
    indexing the whole document.
    """
    if not isinstance(root, Mapping):
        return _UNRESOLVED
    frag = ref[1:]
    if frag in ("", "/"):
        return root
    if not frag.startswith("/"):
        return _UNRESOLVED
    current: Any = root
    for raw in frag[1:].split("/"):
        segment = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, Mapping)):
            try:
                index = int(segment)
            except ValueError:
                return _UNRESOLVED
            if index < 0 or index >= len(current):
                return _UNRESOLVED
            current = current[index]
            continue
        if not isinstance(current, Mapping) or segment not in current:
            return _UNRESOLVED
        current = current[segment]
    return current if isinstance(current, (Mapping, bool)) else _UNRESOLVED


class _Unresolved:
    """Distinct from None, which a pointer can legitimately land on."""


_UNRESOLVED = _Unresolved()


def validate_json_schema(
    schema: Mapping[str, Any] | bool,
    value: Any,
    path: str = "$",
    _root: Any = _UNRESOLVED,
    _seen: frozenset[str] = frozenset(),
) -> list[str]:
    """Every way `value` fails `schema`. An empty list means it passed."""
    # A schema is an object OR a boolean: `true` accepts every value, `false`
    # accepts none. Both are spec-valid wherever a schema is expected, and MCP
    # servers do ship them. The recursion below used to require a Mapping, so a
    # boolean subschema was SKIPPED -- which made `false`, whose entire meaning
    # is "nothing is valid here", behave as "everything is".
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: schema is false, so no value is valid here"]
    if not isinstance(schema, Mapping):
        # Neither object nor boolean is not a schema at all. Blaming the value
        # for that would report the wrong side as broken.
        return []

    root = schema if isinstance(_root, _Unresolved) else _root
    errors: list[str] = []

    # Only local pointers can be resolved; an external one names a document we
    # were never given. We do not turn our own limitation into the caller's
    # rejection, so an unresolvable reference validates as accept. A ref already
    # on the stack is a recursive schema -- one pass checks this value, and
    # following it again would not terminate.
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in _seen:
        target = _resolve_pointer(root, ref)
        if not isinstance(target, _Unresolved):
            errors.extend(validate_json_schema(target, value, path, root, _seen | {ref}))

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
            if name in value and isinstance(sub, (Mapping, bool)):
                errors.extend(validate_json_schema(sub, value[name], f"{path}.{name}", root, _seen))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name}: additional property not allowed")

    # `items` may be a boolean too: `items: false` says the array must be empty.
    # Requiring a Mapping skipped that silently, reporting a non-empty array as
    # valid against a schema that forbids every element.
    items = schema.get("items")
    if isinstance(value, (list, tuple)) and isinstance(items, (Mapping, bool)):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(items, item, f"{path}[{index}]", root, _seen))

    return errors


__all__ = ["DEFAULT_ERROR_LIMIT", "validate_json_schema"]
