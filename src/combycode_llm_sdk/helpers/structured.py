"""`structured=SomeDataclass` -- the class is the schema and the answer.

The Python divergence `11_structured_json.py` asks to be reviewed. The
TypeScript corpus writes the JSON Schema by hand AND repeats the same shape in a
type parameter, so the two can disagree and the compiler cannot tell. Python
already holds the shape once, in the dataclass, so the schema is derived from it
and a real instance is handed back.

Stdlib only -- `dataclasses.fields` and `typing.get_type_hints`. Requiring
pydantic for the headline feature would put a dependency in a library whose
zero-dependency footprint is the reason to choose it.

`structured={"schema": {...}}` still works, for a schema that arrives from
elsewhere. It is simply not what the docs lead with.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping, Sequence
from typing import Any, get_args, get_origin

from .json_schema import SchemaError, non_null, object_schema

#: What `complete(structured=...)` accepts.
StructuredSpec = Any


def is_schema_class(value: Any) -> bool:
    """Whether this is a dataclass TYPE rather than a schema mapping or instance.

    The type, not an instance: `structured=Weather` is the declaration, and
    `structured=Weather("Paris", 20)` would be a caller confusing an example with
    a schema -- which is worth an error rather than a guess.
    """
    return isinstance(value, type) and dataclasses.is_dataclass(value)


def to_wire(structured: StructuredSpec) -> dict[str, Any] | None:
    """The `structured` option, as the wire layer expects it.

    Returns `{"schema": ..., "name": ...}`. `None` when there is nothing to send.
    """
    if structured is None:
        return None
    if is_schema_class(structured):
        return {"schema": object_schema(structured), "name": structured.__name__}
    if isinstance(structured, Mapping):
        return dict(structured)
    if dataclasses.is_dataclass(structured):
        raise SchemaError(
            f"structured= wants the dataclass itself, not an instance of it. Pass "
            f"{type(structured).__name__} rather than {type(structured).__name__}(...)."
        )
    raise SchemaError(
        f"structured= must be a dataclass or a mapping with a 'schema' key, got "
        f"{type(structured).__name__}."
    )


def _build(annotation: Any, value: Any, *, where: str) -> Any:
    """One decoded JSON value -> the thing the annotation asked for."""
    if value is None:
        return None

    annotation = non_null(annotation)

    if is_schema_class(annotation):
        return instantiate(annotation, value, where=where)

    origin = get_origin(annotation)
    if origin in (list, set, tuple, frozenset) and isinstance(value, Sequence):
        args = [a for a in get_args(annotation) if a is not Ellipsis]
        item = args[0] if args else Any
        built = [_build(item, v, where=f"{where}[{i}]") for i, v in enumerate(value)]
        return built if origin is list else origin(built)

    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        # JSON has one number type, so a model answering `20` for a `float` field
        # is right and only its spelling is not. `int` is NOT coerced the other
        # way: silently truncating 20.7 to 20 would hide a real disagreement.
        return float(value)

    return value


def instantiate(cls: type, data: Any, *, where: str = "") -> Any:
    """Build `cls` from decoded JSON, recursively.

    Unknown keys are DROPPED rather than raising. The schema says
    `additionalProperties: false`, so an extra key means the model ignored the
    schema -- and throwing away the whole answer over one stray field, when every
    field the caller asked for is present, helps nobody.
    """
    label = where or cls.__name__
    if not isinstance(data, Mapping):
        raise SchemaError(f"{label}: expected an object, got {type(data).__name__}")

    try:
        hints = typing.get_type_hints(cls)
    except Exception as exc:
        raise SchemaError(f"{label}: cannot resolve type hints ({exc})") from exc

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        has_default = (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        )
        if f.name not in data:
            if not has_default:
                missing.append(f.name)
            continue
        value = _build(hints.get(f.name, Any), data[f.name], where=f"{label}.{f.name}")
        if value is None and has_default:
            # An explicit `null` for a field with a default means the model had
            # nothing to say. Letting the default stand is what the caller wrote
            # it for -- `nickname: str | None = None` gets None either way, and a
            # `list = field(default_factory=list)` gets its empty list rather
            # than a None the annotation never allowed.
            continue
        kwargs[f.name] = value

    if missing:
        raise SchemaError(f"{label}: the answer is missing {', '.join(missing)}")
    return cls(**kwargs)


def parse_into(structured: StructuredSpec, text: str) -> Any:
    """The model's text -> `parsed`.

    A dataclass gets an instance; a raw schema gets the decoded JSON, which is all
    the caller declared. Raises nothing of its own for a bad answer -- the caller
    of this decides, because `12_structured_parse.py` requires that
    `result.parsed` be None and `result.text` still be readable when the model
    answers something unparseable.
    """
    from ..llm.client_internal import parse_structured

    decoded = parse_structured(text)
    return instantiate(structured, decoded) if is_schema_class(structured) else decoded


__all__ = ["StructuredSpec", "instantiate", "is_schema_class", "parse_into", "to_wire"]
