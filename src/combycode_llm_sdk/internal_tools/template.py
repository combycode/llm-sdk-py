"""Rendering a prompt, and reading back what the model made of it.

Two halves of the same round trip. `render_template` is deliberately the
smallest thing that could work -- `{{var}}` and dotted paths, no conditionals,
no loops -- because a prompt template that grows a control flow has become a
program, and a program belongs in a function where it can be tested.

Strict on a missing variable, because the alternative is a prompt with a hole
in it that reads perfectly well to the model and produces a confident answer to
a question nobody asked.

Transposed from `unified-library-ts/src/plugins/internal-tools/runner/template.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")

_MISSING = object()


def _resolve_path(obj: Any, path: str) -> Any:
    """`a.b.c` through nested mappings. Anything unreachable is missing."""
    current: Any = obj
    for key in path.split("."):
        if not isinstance(current, Mapping):
            return _MISSING
        if key not in current:
            return _MISSING
        current = current[key]
    return current


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value)


def render_template(template: str, variables: Mapping[str, Any]) -> str:
    """Substitute `{{name}}`, refusing a name the caller did not supply."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = _resolve_path(variables, key)
        if value is _MISSING:
            raise KeyError(
                f"template variable not found: {key!r} "
                f"(have: {', '.join(sorted(variables)) or 'nothing'})"
            )
        return _stringify(value)

    return _VAR.sub(replace, template)


def parse_json_with_fences(text: str) -> Any:
    """The JSON a model meant, out of whatever it actually sent.

    Three attempts, cheapest first: the text as-is once a markdown fence is
    stripped, then the first balanced `{...}` or `[...]` found in it. Models
    that were told to emit raw JSON still sometimes greet you first, and the
    answer is already paid for by the time that happens.
    """
    cleaned = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except ValueError:
        pass

    block = _first_json_block(cleaned)
    if block is not None:
        return json.loads(block)

    raise ValueError(f"no valid JSON in output (first 120 chars): {cleaned[:120]}")


def _first_json_block(text: str) -> str | None:
    """The first balanced object or array, ignoring braces inside strings."""
    start = -1
    opener = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            break
    if start < 0:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def format_numbered_list(items: Sequence[Any], start: int = 1) -> str:
    """`1. a\\n2. b` -- for the templates that hand a model a list to work down."""
    return "\n".join(f"{i + start}. {item}" for i, item in enumerate(items))


def format_bulleted_list(items: Sequence[Any], bullet: str = "-") -> str:
    """`- a\\n- b`, for when the order carries no meaning."""
    return "\n".join(f"{bullet} {item}" for item in items)


__all__ = [
    "format_bulleted_list",
    "format_numbered_list",
    "parse_json_with_fences",
    "render_template",
]
