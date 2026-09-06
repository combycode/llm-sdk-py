"""The examples are the API contract, so the surface answers to them.

The examples were reviewed and are read-only for the duration of the port. That
makes them the one part of this tree that already knows what the library should
look like, and it is worth checking from the first commit rather than the last:
a name the examples import and the package does not export is a broken contract
whether or not anything is implemented behind it.

This runs against STUBS at first, and that is the point. It proves the surface
exists before any behaviour does, so porting a module is only ever a matter of
replacing a `NotImplementedError` — never of discovering that the public name
was wrong all along.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(ROOT.glob("examples/**/*.py"))
PKG = "combycode_llm_sdk"


def imported_names() -> dict[str, set[str]]:
    """module -> the names the examples import from it."""
    found: dict[str, set[str]] = {}
    for path in EXAMPLES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PKG):
                found.setdefault(node.module or PKG, set()).update(a.name for a in node.names)
    return found


CONTRACT = imported_names()


def test_the_examples_are_actually_there() -> None:
    # Guards every test below: an empty examples tree would make them all pass
    # while checking nothing.
    assert len(EXAMPLES) > 50, f"only {len(EXAMPLES)} example files found"
    assert CONTRACT, "no imports from the package were found in the examples"


@pytest.mark.parametrize("module", sorted(CONTRACT))
def test_every_module_the_examples_import_exists(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize(
    ("module", "name"),
    sorted((m, n) for m, names in CONTRACT.items() for n in names),
)
def test_every_name_the_examples_import_exists(module: str, name: str) -> None:
    mod = importlib.import_module(module)
    assert hasattr(mod, name), (
        f"the examples import {name!r} from {module}, and it is not there. "
        f"The examples are the contract; the library answers to them."
    )
