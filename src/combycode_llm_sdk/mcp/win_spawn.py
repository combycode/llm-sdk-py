"""Spawning a command on Windows that is not really a program.

`npx`, `uvx`, `pnpm` -- the commands every MCP server's README tells you to run
-- are `.cmd` shims on Windows, and `CreateProcess` cannot execute one. The
result is `[WinError 2] The system cannot find the file specified` for a command
that works perfectly in the same shell, which reads as a broken installation
rather than as a missing shim lookup.

`shell=True` would fix it and open a command-injection hole: the server's own
arguments would be re-parsed by `cmd.exe`. So the resolution is explicit --
find the real file on `PATH` using `PATHEXT`, and route a `.cmd`/`.bat` through
`cmd.exe /d /s /c` with each argument quoted for `CommandLineToArgvW` and then
caret-escaped for `cmd` itself.

A no-op off Windows, where a shim is a shebang file the kernel can run.

Transposed from `unified-library-ts/src/plugins/mcp/win-spawn.ts`. The pure
functions take an `exists` probe so they are testable without the filesystem
they describe.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

#: Extensions that are batch scripts rather than executables.
CMD_EXTENSIONS = frozenset({".cmd", ".bat"})

#: What Windows uses when PATHEXT is unset.
DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD"

_NEEDS_QUOTES = re.compile(r'[\s"]')
_HAS_SEP_OR_EXT = re.compile(r"[\\/]|\.[a-z0-9]+$", re.IGNORECASE)
_EXTENSION = re.compile(r"\.[a-z0-9]+$", re.IGNORECASE)
_CMD_META = re.compile(r'[()%!^"<>&|]')


@dataclass(frozen=True)
class SpawnPlan:
    """What to actually hand to `Popen`."""

    file: str
    args: list[str]
    #: Pass the command line through untouched -- `cmd.exe` does its own
    #: parsing, and letting Python re-quote it would double every escape.
    verbatim: bool


def quote_win_arg(arg: str) -> str:
    """One argument, quoted the way `CommandLineToArgvW` reads it.

    The backslash rule is the awkward part: a run of backslashes is literal
    UNLESS it precedes a quote, in which case each one must be doubled. Getting
    this wrong turns `C:\\path\\` into an escaped quote and swallows the rest of
    the command line.
    """
    if arg and not _NEEDS_QUOTES.search(arg):
        return arg
    out = ['"']
    backslashes = 0
    for char in arg:
        if char == "\\":
            backslashes += 1
        elif char == '"':
            out.append("\\" * (backslashes * 2 + 1) + '"')
            backslashes = 0
        else:
            if backslashes:
                out.append("\\" * backslashes)
            backslashes = 0
            out.append(char)
    out.append("\\" * (backslashes * 2))
    out.append('"')
    return "".join(out)


def escape_cmd_meta(text: str) -> str:
    """Caret-escape what `cmd.exe` would otherwise interpret.

    Applied AFTER quoting, because `cmd` reads the line before the quoting rules
    apply: an unescaped `&` inside a quoted argument still ends the command.
    """
    return _CMD_META.sub(lambda m: f"^{m.group(0)}", text)


def resolve_on_path(
    command: str, env: Mapping[str, str], exists: Callable[[str], bool]
) -> str | None:
    """The real file a bare command names, or None if PATH has no such thing."""
    path_var = env.get("PATH") or env.get("Path") or ""
    extensions = [e for e in (env.get("PATHEXT") or DEFAULT_PATHEXT).split(";") if e]
    for directory in (d for d in path_var.split(";") if d):
        for extension in extensions:
            candidate = f"{directory}\\{command}{extension}"
            if exists(candidate):
                return candidate
    return None


def windows_spawn_plan(
    command: str,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    exists: Callable[[str], bool] | None = None,
) -> SpawnPlan:
    """How to spawn `command args` on Windows."""
    environment = dict(env if env is not None else os.environ)
    probe = exists if exists is not None else os.path.isfile

    named_directly = bool(_HAS_SEP_OR_EXT.search(command))
    resolved = None if named_directly else resolve_on_path(command, environment, probe)
    target = resolved or command
    match = _EXTENSION.search(target)
    extension = match.group(0).lower() if match else ""

    # Only an actual batch script goes through cmd.exe. The TypeScript also
    # sends an UNRESOLVED bare name there, as one more chance at a shim -- but
    # the lookup above already used the same PATH and PATHEXT cmd.exe would, so
    # the extra chance finds nothing and costs the error message: cmd starts
    # fine, prints "is not recognized" to stderr, and exits, which surfaces as
    # "the server exited" rather than "no such command".
    if extension in CMD_EXTENSIONS:
        comspec = environment.get("COMSPEC") or environment.get("ComSpec") or "cmd.exe"
        line = " ".join(escape_cmd_meta(quote_win_arg(a)) for a in [target, *args])
        # The outer quotes plus `/s` are a pair: `/s` tells cmd to strip exactly
        # the first and last quote and run everything between them, which is the
        # only reliable way to pass a quoted path AND quoted arguments.
        return SpawnPlan(file=comspec, args=["/d", "/s", "/c", f'"{line}"'], verbatim=True)

    return SpawnPlan(file=target, args=list(args), verbatim=False)


__all__ = [
    "CMD_EXTENSIONS",
    "DEFAULT_PATHEXT",
    "SpawnPlan",
    "escape_cmd_meta",
    "quote_win_arg",
    "resolve_on_path",
    "windows_spawn_plan",
]
