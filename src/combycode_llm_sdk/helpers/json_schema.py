"""One Python annotation -> one JSON Schema fragment.

Shared by `@tool`, which turns a signature into `parameters`, and by
`structured=`, which turns a dataclass into an output schema. Both are the same
question asked twice -- "what JSON does this annotation describe" -- and answering
it in two places is how the two drift about `str | None`.

What this does NOT do is provider strictness. `llm/types/schema_utils.py` already
owns `additionalProperties` and the per-provider strict dialects, and the wire
layer already applies it on the way out. Producing a second, differently-strict
schema here would mean two answers to one question again.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

#: The scalars a JSON Schema can name directly.
SCALARS: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_NONE = type(None)


class SchemaError(TypeError):
    """An annotation that cannot become a JSON Schema.

    Raised where the annotation is written -- at decoration, or at the first call
    that uses the dataclass -- rather than falling back to an empty `{}`. An empty
    schema is accepted by the library and then rejected by the provider with
    "Empty schema ({}) that accepts any JSON", which is a message about our bug
    delivered in their words, three layers from the line that caused it.
    """


def is_nullable(annotation: Any) -> bool:
    """Whether `None` is one of the annotation's members."""
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and _NONE in get_args(annotation)


def non_null(annotation: Any) -> Any:
    """The annotation with `None` removed, when that leaves exactly one member."""
    if not is_nullable(annotation):
        return annotation
    members = [a for a in get_args(annotation) if a is not _NONE]
    if len(members) == 1:
        return members[0]
    return functools.reduce(operator.or_, members)


def nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """The same schema, admitting null.

    Written into the `type` when there is a plain one -- `["string", "null"]` --
    because that is the form OpenAI's strict mode documents, and wrapping a
    one-member `anyOf` around every optional field instead makes the schema
    harder to read for no gain.
    """
    kind = schema.get("type")
    if isinstance(kind, str):
        return {**schema, "type": [kind, "null"]}
    if isinstance(kind, list):
        return schema if "null" in kind else {**schema, "type": [*kind, "null"]}
    return {"anyOf": [schema, {"type": "null"}]}


def schema_for(annotation: Any, *, where: str) -> dict[str, Any]:
    """The JSON Schema for one annotation."""
    if annotation in SCALARS:
        # An exact lookup rather than `issubclass`: `bool` is a subclass of `int`,
        # so a chain of issubclass checks reports every boolean as an integer.
        return {"type": SCALARS[annotation]}
    if annotation is Any:
        return {}
    if annotation is _NONE:
        return {"type": "null"}

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return object_schema(annotation, where=where)

    origin = get_origin(annotation)

    if origin is Literal:
        values = list(get_args(annotation))
        kinds = {type(v) for v in values}
        if len(kinds) != 1 or kinds.pop() not in SCALARS:
            raise SchemaError(
                f"{where}: a Literal must hold values of one JSON scalar type, got {values!r}"
            )
        return {"type": SCALARS[type(values[0])], "enum": values}

    if origin in (list, set, tuple, frozenset):
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        return {"type": "array", "items": schema_for(args[0], where=where) if args else {}}

    if origin is dict or annotation is dict:
        return {"type": "object"}

    if origin in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not _NONE]
        base = (
            schema_for(members[0], where=where)
            if len(members) == 1
            else {"anyOf": [schema_for(m, where=where) for m in members]}
        )
        return nullable(base) if _NONE in get_args(annotation) else base

    if isinstance(annotation, type) and issubclass(annotation, str):
        # A `str` subclass -- a `StrEnum`, typically. Its members are the enum.
        values = [str(v) for v in getattr(annotation, "__members__", {}).values()]
        return {"type": "string", **({"enum": values} if values else {})}

    raise SchemaError(
        f"{where}: no JSON Schema for {annotation!r}. Annotate it as a scalar, a "
        f"list, a Literal, a dataclass, or Any."
    )


def object_schema(cls: type, *, where: str = "") -> dict[str, Any]:
    """A dataclass -> an object schema.

    EVERY field is listed in `required`, with the ones that have a default made
    nullable instead. That is not a stylistic choice: OpenAI's strict mode refuses
    a schema whose `properties` are not all required, and `openaiStructuredStrict`
    turns strict OFF for a schema that cannot satisfy it -- so omitting one
    optional field from `required` silently downgrades the whole request from a
    guaranteed shape to a hopeful one. Nullable-and-required says the same thing
    and keeps the guarantee.
    """
    label = where or cls.__name__
    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:
        raise SchemaError(f"{label}: cannot resolve type hints ({exc})") from exc

    properties: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if not f.init:
            # A field the constructor does not take cannot be filled from the
            # model's answer, so describing it would invite one that is discarded.
            continue
        annotation = hints.get(f.name, Any)
        schema = schema_for(annotation, where=f"{label}.{f.name}")
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        if has_default and not is_nullable(annotation):
            schema = nullable(schema)
        properties[f.name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "SCALARS",
    "SchemaError",
    "is_nullable",
    "non_null",
    "nullable",
    "object_schema",
    "schema_for",
]
