"""Every data file the library reads at runtime ships in the wheel.

The specs, the pin tables, the catalog and the shape book are DATA, not
fixtures: the library opens them on the first request. A wheel missing one
installs cleanly and fails at that request -- the same class of failure as the
first port attempt, where 1403 tests passed against a wheel that died on the
first real call, and for the same reason: nothing compared what was declared
against what was used.

Hatchling includes `.py` files automatically and everything else only when a
glob in `[tool.hatch.build.targets.wheel] artifacts` names it. That list was
already wrong once -- `wire/pins/*.json` and `llm/response-shapes.json` arrived
with the client batch and were not added -- so it is checked here rather than
maintained by hand.
"""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "combycode_llm_sdk"


def _artifact_globs() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    globs = config["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"]
    assert isinstance(globs, list)
    return [str(g) for g in globs]


def _covered(rel_posix: str, globs: list[str]) -> bool:
    """Whether a glob in the artifacts list covers this path.

    `fnmatch` has no `**`, so `/**/` is read as exactly-one-directory. That is
    STRICTER than the gitignore-style dialect build backends generally use, where
    it also spans zero directories -- and the strictness is deliberate rather
    than an approximation waiting to be improved. Being stricter than the backend
    can only ask for a redundant glob; being looser would let this test pass over
    a file the wheel does not carry, which is the failure it exists to catch. It
    is not verified against a real build here because hatchling is not a test
    dependency, and an unverified belief about its globbing is exactly what must
    not decide whether a file ships.
    """
    return any(fnmatch.fnmatch(rel_posix, glob.replace("/**/", "/*/")) for glob in globs)


def test_every_packaged_json_is_declared_as_a_wheel_artifact() -> None:
    globs = _artifact_globs()
    found = sorted(
        p.relative_to(ROOT).as_posix()
        for p in PACKAGE.rglob("*.json")
        if "__pycache__" not in p.parts
    )
    # A filter that matched nothing would make the assertion below vacuous.
    assert len(found) > 100, f"expected the vendored specs to be found, got {len(found)}"
    undeclared = [p for p in found if not _covered(p, globs)]
    assert undeclared == [], (
        "these files are read at runtime but would not ship in a wheel: " f"{undeclared}"
    )


def test_the_globs_all_match_something() -> None:
    """A glob for a directory that was renamed is a silent hole -- it keeps the
    list looking complete while covering nothing."""
    globs = _artifact_globs()
    found = [
        p.relative_to(ROOT).as_posix()
        for p in PACKAGE.rglob("*.json")
        if "__pycache__" not in p.parts
    ]
    dead = [g for g in globs if not any(_covered(p, [g]) for p in found)]
    assert dead == []


def test_the_pep_561_marker_ships() -> None:
    """Without `py.typed`, a fully typed library is treated as untyped.

    The annotations are all inline, so nothing is shipped separately -- but a
    consumer's type checker ignores every one of them unless this marker sits
    in the installed package. Silent: their code still runs, they simply get no
    checking from a library that is mypy-clean end to end.
    """
    assert (PACKAGE / "py.typed").is_file()


def test_dunder_version_is_the_distribution_version() -> None:
    """`__version__` and the installed metadata are the same string.

    They are one literal today -- pyproject declares the version dynamic and
    hatchling reads it from `version.py`, which the package re-exports -- and
    this is what keeps that true.
    A refactor that reintroduces a second copy passes every other test in the
    suite and ships a wheel whose metadata disagrees with the module.
    """
    import importlib.metadata

    import combycode_llm_sdk

    assert combycode_llm_sdk.__version__ == importlib.metadata.version("combycode-llm-sdk")


# Every markdown file this repo tracks, and therefore every one an sdist carries.
# A file is on this list because a stranger who downloads the package should see
# it -- not because it happens to exist.
SHIPPABLE_MARKDOWN = {
    "CHANGELOG.md",
    "README.md",
    "examples/README.md",
    "examples/features-highlight/README.md",
}


def _tracked_markdown() -> set[str]:
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH; nothing to ask about what the repo tracks")
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git work tree here")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_the_package_ships_no_development_notes() -> None:
    """Only documentation a USER wants is allowed to reach PyPI.

    This exists because 0.1.0 and 0.1.1 both shipped `PORTING.md` and
    `PORTING_REGISTRY.md` inside the sdist: an instruction sheet opening "Read
    this before writing a line of Python. It exists because the first attempt
    failed", and a 656-line status table of what was still `pending`, pointing at
    a sibling TypeScript repository nobody outside this workspace has. No secret,
    no path -- just the development process, published, describing a quality
    process that had not caught it.

    Hatchling's sdist takes whatever is tracked and not ignored, so "tracked" IS
    "published" for this repo, and that is the thing worth asserting. A NEW
    markdown file fails here on purpose: adding one is a decision about what the
    package shows the world, and it should cost a line in this set rather than be
    noticed after a release.
    """
    tracked = _tracked_markdown()
    unexpected = tracked - SHIPPABLE_MARKDOWN
    assert not unexpected, (
        f"these markdown files would ship to PyPI: {sorted(unexpected)}. "
        "If a file documents the LIBRARY, add it to SHIPPABLE_MARKDOWN. If it "
        "documents how we work, untrack it (`git rm --cached`) and gitignore it "
        "-- it stays on disk either way."
    )


def test_the_shippable_list_has_no_dead_entries() -> None:
    """A name left behind after a rename would quietly widen what is allowed."""
    tracked = _tracked_markdown()
    missing = SHIPPABLE_MARKDOWN - tracked
    assert not missing, f"listed as shippable but not tracked: {sorted(missing)}"
