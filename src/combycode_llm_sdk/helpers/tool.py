"""`@tool` -- a callable function becomes a tool the model can call.

The Python addition to the corpus, and the divergence `06_tool_call.py` asks to
be reviewed. The TypeScript's `defineTool` restates the name, the description and
the parameter schema by hand beside the function that implements them, which is
three chances to drift from what the function actually does. Python already
carries all three::

    @tool
    def get_weather(city: str) -> str:
        '''Get the current weather for a city.'''
        return "sunny"

- the NAME is `__name__`,
- the DESCRIPTION is the docstring's summary,
- the SCHEMA is the type hints, with a parameter required exactly when it has no
  default.

Async is detected, not declared: `inspect.iscoroutinefunction` already knows, and
asking the caller to say so again would be a second source of truth about the
same fact (`09_tool_runner.py`).

The decorated object stays callable. A decorator that trades the function for a
descriptor makes the unit test of the tool body impossible to write.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Mapping
from typing import Any, overload

from .json_schema import SchemaError, schema_for

#: The decoration-time error, under the name this module has always raised.
#: One class, aliased, rather than two that callers must catch separately.
ToolSchemaError = SchemaError


def _summary(func: Callable[..., Any]) -> str:
    """The docstring's first paragraph, whitespace-folded.

    The first paragraph rather than the whole docstring: everything after it is
    written for the reader of the code -- implementation notes, the reason for a
    workaround -- and sending it spends tokens on every request telling the model
    things about a body it cannot see anyway.
    """
    doc = inspect.getdoc(func) or ""
    paragraph = doc.split("\n\n", 1)[0].strip()
    return " ".join(paragraph.split())


#: `Args:` and the names other docstring styles give the same section.
_ARG_SECTIONS = ("args:", "arguments:", "parameters:", "params:")
#: A section that ENDS the argument list.
_END_SECTIONS = ("returns:", "return:", "raises:", "yields:", "examples:", "example:", "note:")

_ARG_LINE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _arg_docs(func: Callable[..., Any]) -> dict[str, str]:
    """Per-argument descriptions, from the docstring's `Args:` section.

    A model chooses arguments far better when each one is described, and the
    description already exists in the place a Python developer writes it.
    Restating it in the decorator -- which is what the TypeScript's `params`
    object does -- is a second copy that can disagree with the first.

    Google style only, and deliberately: it is what the reviewed examples use,
    and a parser that guessed between three conventions would silently mis-file
    a description under the wrong argument rather than simply not finding one.
    """
    doc = inspect.getdoc(func) or ""
    lines = doc.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.strip().lower() in _ARG_SECTIONS), None
    )
    if start is None:
        return {}

    found: dict[str, str] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.lower() in _END_SECTIONS:
            break
        if not stripped:
            continue
        match = _ARG_LINE.match(stripped)
        if match:
            current = match.group(1).lstrip("*")
            found[current] = match.group(2).strip()
        elif current:
            # A wrapped continuation of the previous argument's description.
            found[current] = f"{found[current]} {stripped}".strip()
    return {k: v for k, v in found.items() if v}


def _parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """The signature -> the `parameters` JSON Schema."""
    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception as exc:  # a forward reference that does not resolve
        raise ToolSchemaError(f"{func.__name__}: cannot resolve type hints ({exc})") from exc

    docs = _arg_docs(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # `*args` / `**kwargs` name no argument the model could fill.
            continue
        if name not in hints:
            raise ToolSchemaError(
                f"{func.__name__}({name}): no type annotation, so there is no schema "
                f"to send. Annotate it."
            )
        described = schema_for(hints[name], where=f"{func.__name__}({name})")
        if name in docs:
            described = {**described, "description": docs[name]}
        properties[name] = described
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        # Omitted when empty rather than sent as `[]`: `08_multistep_loop.py`'s
        # `get_user_city()` takes nothing, and an empty `required` is a different
        # document from an absent one to a strict validator.
        schema["required"] = required
    return schema


class Tool:
    """A function, plus the definition the wire needs to describe it.

    Still callable: `get_weather("Paris")` does what it always did, so the body
    stays unit-testable without going through a model.
    """

    # `__doc__` is NOT here: a class with a docstring already defines it as a
    # class variable, and naming it in __slots__ is a ValueError at import.
    __slots__ = ("__name__", "__wrapped__", "definition", "func", "is_async", "lazy")

    def __init__(
        self,
        func: Callable[..., Any],
        definition: Mapping[str, Any],
        *,
        lazy: bool = False,
    ) -> None:
        self.func = func
        self.definition = dict(definition)
        self.is_async = inspect.iscoroutinefunction(func)
        #: Registered but NOT declared -- the model finds it through
        #: `tool_search` and calls it through `call_tool`. See
        #: `agent/lazy_tools.py` for what that buys and what it costs.
        self.lazy = lazy
        self.__name__ = func.__name__
        self.__wrapped__ = func

    @property
    def name(self) -> str:
        return str(self.definition["name"])

    @property
    def schema(self) -> dict[str, Any]:
        """The definition, under the name the public contract uses.

        One stored copy, two names: `schema` is what a caller reads, and
        `definition` is the wire layer's own word for the same object, used
        wherever this travels beside a provider's tool array.
        """
        return self.definition

    def to_wire(self) -> dict[str, Any]:
        """This tool as the AGENT carries it -- the declaration, plus how it is registered.

        Not what goes to a provider: `lazy` is ours, and a provider handed it
        would reject the request. What a provider gets is `definition` alone,
        which is what `AgentLoop.declared_tools()` sends.

        The distinction is the whole of lazy loading. A lazy tool is registered
        and NOT declared, so a form carrying only the declaration cannot express
        it -- and the flag has to travel WITH the tool, because a tool that came
        from an MCP server passes through several hands before it reaches the
        loop that reads it.
        """
        return {"definition": dict(self.definition), "lazy": self.lazy}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        args = ", ".join(self.definition["parameters"]["properties"])
        return f"<tool {self.name}({args})>"


def _build(func: Callable[..., Any], *, lazy: bool) -> Tool:
    if not callable(func):
        raise TypeError(f"@tool expects a function, got {type(func).__name__}")

    description = _summary(func)
    if not description:
        raise ToolSchemaError(
            f"{func.__name__}: no docstring, so the model is told nothing about what "
            f"this tool does. Give it a one-line docstring."
        )
    return Tool(
        func,
        {"name": func.__name__, "description": description, "parameters": _parameters(func)},
        lazy=lazy,
    )


@overload
def tool(func: Callable[..., Any]) -> Tool: ...


@overload
def tool(*, lazy: bool = False) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Callable[..., Any] | None = None, *, lazy: bool = False
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a `Tool`. See the module docstring.

    Both spellings work, because both read naturally::

        @tool
        def get_weather(city: str) -> str: ...

        @tool(lazy=True)
        def rebuild_search_index(namespace: str) -> str: ...

    The bare form is the common one, so it stays free of parentheses; the called
    form exists because `lazy` has to be said somewhere.
    """
    if func is None:
        return lambda f: _build(f, lazy=lazy)
    if isinstance(func, Tool):
        return func
    return _build(func, lazy=lazy)


__all__ = ["Tool", "ToolSchemaError", "tool"]
