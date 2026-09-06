"""A pure rule evaluator for `(source, target, action)` tuples.

Transposed from `unified-library-ts/src/plugins/permissions/`.

Ordered rules, first match wins, and a triple no rule matched is DENIED.

Default-deny is the whole design. A policy that allowed what it had not thought
about would hand an agent the filesystem the first time someone added a tool the
rule list predates -- and the rules are written by hand, so that is every new
tool. The failure is silent in the direction that costs something: a rule that
wrongly refuses is reported the first time anyone runs it, a rule that wrongly
allows is found afterwards.

`ask` is a third effect, not a soft deny. It means the run stops and a human
answers, which is a different program flow and not something a boolean carries.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: What is being acted upon. `kind` says which matcher applies; the rest is
#: whatever that kind needs (`path` for fs, `command` for shell).
PermissionTarget = Mapping[str, Any]

#: `(target) -> bool`.
TargetMatcher = Callable[[PermissionTarget], bool]

_REGEX_META = re.compile(r"[.+()|\[\]{}^$\\]")


def glob_to_regex(pattern: str, *, loose: bool = False) -> re.Pattern[str]:
    """A glob as a regex.

    `loose` is for targets with no path separators to respect -- a shell command
    or a URL -- where `*` should cross anything. On a path it must not: `*`
    stopping at `/` is what makes `workspace/*` mean one level.
    """
    out = "^"
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                out += ".*"
                i += 2
                # `**/` and `**` mean the same: any number of segments,
                # including none, so the slash is consumed with them.
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            elif loose:
                out += ".*"
                i += 1
            else:
                out += "[^/]*"
                i += 1
        elif char == "?":
            out += "." if loose else "[^/]"
            i += 1
        elif _REGEX_META.match(char):
            out += "\\" + char
            i += 1
        else:
            out += char
            i += 1
    return re.compile(out + "$")


def compile_globs(patterns: Sequence[str], *, loose: bool = False) -> Callable[[str], bool]:
    """One predicate for several patterns -- true when any of them matches."""
    compiled = [glob_to_regex(p, loose=loose) for p in patterns]
    return lambda value: any(rx.match(value) for rx in compiled)


# -- built-in matchers -------------------------------------------------------


def fs_glob(*patterns: str) -> TargetMatcher:
    """Filesystem paths matching any of these globs."""
    test = compile_globs(patterns)
    return lambda target: (
        target.get("kind") == "fs"
        and isinstance(target.get("path"), str)
        and test(target["path"])
    )


def shell_glob(*patterns: str) -> TargetMatcher:
    """Shell commands matching any of these globs. Loose: a command is not a path."""
    test = compile_globs(patterns, loose=True)
    return lambda target: (
        target.get("kind") == "shell"
        and isinstance(target.get("command"), str)
        and test(target["command"])
    )


def url_pattern(*patterns: str) -> TargetMatcher:
    """URLs matching any of these globs. Loose, for the same reason."""
    test = compile_globs(patterns, loose=True)
    return lambda target: (
        target.get("kind") == "url"
        and isinstance(target.get("url"), str)
        and test(target["url"])
    )


def memory_category(*categories: str) -> TargetMatcher:
    """Memory entries in any of these categories."""
    wanted = set(categories)
    return lambda target: target.get("kind") == "memory" and target.get("category") in wanted


def any_of_kind(*kinds: str) -> TargetMatcher:
    """Any target of these kinds, whatever else it carries."""
    wanted = set(kinds)
    return lambda target: target.get("kind") in wanted


# -- the policy --------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One rule. Every selector optional; an omitted one matches anything.

    `effect` is required and has no default: a rule whose effect had to be
    guessed is a rule nobody can read.
    """

    effect: str
    source: str | Sequence[str] | None = None
    target: TargetMatcher | None = None
    action: str | Sequence[str] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PermissionDecision:
    """What the policy decided, and which rule decided it."""

    #: True only for `allow`. False for both `deny` and `ask` -- an `ask` has
    #: not been allowed yet, and code that treated it as one would run the tool
    #: while the human was still being asked.
    allow: bool
    #: True when a human must answer before anything happens.
    ask: bool = False
    reason: str | None = None
    #: The index of the rule that decided, or None when nothing matched and the
    #: default denied. The difference matters: "a rule refused this" and "no
    #: rule mentioned it" call for different fixes.
    matched_rule: int | None = None


def _matches(value: str, allowed: str | Sequence[str]) -> bool:
    candidates = [allowed] if isinstance(allowed, str) else list(allowed)
    return value in candidates or "*" in candidates


class PermissionPolicy:
    """Ordered rules, first match wins, default deny."""

    def __init__(self, rules: Sequence[Rule] = ()) -> None:
        self._rules = tuple(rules)

    @property
    def rules(self) -> Sequence[Rule]:
        return self._rules

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def size(self) -> int:
        return len(self._rules)

    def check(self, source: str, target: PermissionTarget, action: str) -> PermissionDecision:
        for index, rule in enumerate(self._rules):
            if rule.source is not None and not _matches(source, rule.source):
                continue
            if rule.action is not None and not _matches(action, rule.action):
                continue
            if rule.target is not None and not rule.target(target):
                continue
            return PermissionDecision(
                allow=rule.effect == "allow",
                ask=rule.effect == "ask",
                reason=rule.reason,
                matched_rule=index,
            )
        return PermissionDecision(allow=False, reason="no rule matched (default deny)")

    def with_additional(self, extra: Sequence[Rule]) -> PermissionPolicy:
        """A policy with more rules APPENDED.

        Appended, never prepended: a caller handed a policy cannot widen it by
        putting an allow-all in front of a deny that was already there. It can
        only rule on what no earlier rule ruled on.
        """
        return PermissionPolicy([*self._rules, *extra])

    def __repr__(self) -> str:
        return f"<PermissionPolicy {len(self._rules)} rule(s)>"


__all__ = [
    "PermissionDecision",
    "PermissionPolicy",
    "PermissionTarget",
    "Rule",
    "TargetMatcher",
    "any_of_kind",
    "compile_globs",
    "fs_glob",
    "glob_to_regex",
    "memory_category",
    "shell_glob",
    "url_pattern",
]
