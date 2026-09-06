"""An MCP server as a child process, spoken to in newline-delimited JSON-RPC.

Three things here are not obvious and each was a real failure somewhere:

- **stderr is left alone.** A server logs there, and a server that logs to
  STDOUT breaks the protocol. Inheriting stderr means those logs reach the
  operator instead of being swallowed by a pipe nobody drains -- which is also
  how a chatty server deadlocks on a full pipe buffer.
- **The line buffer is bounded.** NDJSON has no length prefix, so a server that
  never writes a newline (a crash dump, a binary blob on the wrong stream) grows
  the buffer until the process dies. A bound turns an eventual OOM into an error
  that names the cause.
- **The child's environment is built, not inherited.** A server needs PATH to
  find its own runtime; it does not need the API keys of whatever spawned it.

Transposed from `unified-library-ts/src/plugins/mcp/transport-stdio.ts`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import McpError, McpErrorCode
from .transport import BaseJsonRpcTransport
from .win_spawn import windows_spawn_plan

DEFAULT_TIMEOUT_SECONDS = 60.0

#: Largest single unterminated line to buffer, matching mcp-ts 1.30's
#: `StdioServerParameters.maxBufferSize`.
DEFAULT_MAX_BUFFER_BYTES = 10 * 1024 * 1024

#: How long to wait for a closed child to leave before escalating.
_TERMINATE_AFTER = 0.5
_KILL_AFTER = 2.5

#: `CREATE_NO_WINDOW`. Routing through `cmd.exe` would otherwise flash a
#: console window for every server a desktop app starts.
_NO_WINDOW = 0x08000000

_SAFE_ENV_KEYS_WIN = (
    "APPDATA", "COMSPEC", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "Path",
    "PATHEXT", "PROCESSOR_ARCHITECTURE", "PROGRAMFILES", "SYSTEMDRIVE", "SYSTEMROOT",
    "TEMP", "TMP", "USERNAME", "USERPROFILE",
)
_SAFE_ENV_KEYS_POSIX = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "TMPDIR", "USER")


def safe_env() -> dict[str, str]:
    """The minimum a child needs to find its own runtime, and nothing else.

    A whitelist rather than a copy: the parent's environment is where the API
    keys live, and an MCP server is third-party code.
    """
    keys = _SAFE_ENV_KEYS_WIN if sys.platform == "win32" else _SAFE_ENV_KEYS_POSIX
    return {key: os.environ[key] for key in keys if key in os.environ}


class StdioTransport(BaseJsonRpcTransport):
    """Spawn the server and exchange NDJSON over its stdin/stdout."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    ) -> None:
        super().__init__()
        self._command = command
        self._args = list(args)
        self._env = dict(env or {})
        self._cwd = cwd
        self._timeout = timeout
        self._max_buffer_bytes = max_buffer_bytes
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        environment = {**safe_env(), **self._env}
        # `npx`, `uvx`, `pnpm` -- the commands every MCP server's README names --
        # are `.cmd` shims on Windows, and CreateProcess cannot run one. Without
        # this the error is `[WinError 2] cannot find the file specified` for a
        # command that works in the same shell, which reads as a broken install.
        plan = (
            windows_spawn_plan(self._command, self._args, environment)
            if sys.platform == "win32"
            else None
        )
        argv = [plan.file, *plan.args] if plan else [self._command, *self._args]
        extra: dict[str, Any] = (
            {"creationflags": _NO_WINDOW} if sys.platform == "win32" else {}
        )
        # `sys.platform` first, though `plan` is only ever built on Windows: a
        # type checker cannot carry that fact from one branch to another, and
        # `subprocess.STARTUPINFO` does not exist off Windows. Checked as Linux,
        # the attribute is simply absent -- which a Windows-only run never sees.
        if sys.platform == "win32" and plan is not None and plan.verbatim:
            # cmd.exe parses its own command line; letting Python re-quote it
            # would double every escape the plan just applied.
            startupinfo = subprocess.STARTUPINFO()
            extra["startupinfo"] = startupinfo
            argv = subprocess.list2cmdline([plan.file]) + " " + " ".join(plan.args)  # type: ignore[assignment]

        try:
            # The command is the caller's own: an MCP server IS a program they
            # chose to run, so there is nothing here to sanitise.
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # inherited: the server's logs are for the operator
                env=environment,
                cwd=self._cwd,
                bufsize=0,
                **extra,
            )
        except OSError as exc:
            raise McpError(
                f"MCP stdio: could not start {self._command!r}: {exc}",
                code=McpErrorCode.CONNECTION_CLOSED,
            ) from exc
        if proc.stdout is None or proc.stdin is None:
            raise McpError(
                "MCP stdio: the child has no stdin/stdout pipe",
                code=McpErrorCode.CONNECTION_CLOSED,
            )
        self._proc = proc
        # A daemon thread: a reader blocked on a child that never exits must not
        # be the reason the interpreter cannot shut down.
        self._reader = threading.Thread(target=self._read_loop, name="mcp-stdio", daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._closed = True
        self._fail_all(
            McpError("MCP transport closed", code=McpErrorCode.CONNECTION_CLOSED)
        )
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        # Closed stdin first, and only then escalate: a well-behaved server exits
        # on EOF, and killing one that was about to leave cleanly loses whatever
        # it was flushing.
        #
        # The SIGTERM step is not covered by a test and cannot be on Windows,
        # where CPython defines `Popen.kill = Popen.terminate` -- both are
        # `TerminateProcess`, so the two steps are one call and no test can tell
        # them apart. What IS covered is that a server ignoring EOF still ends
        # (`stubborn` mode); on POSIX the middle step is what gives it a chance
        # to run its own shutdown first.
        try:
            proc.wait(timeout=_TERMINATE_AFTER)
            return
        except subprocess.TimeoutExpired:
            proc.terminate()
        try:
            proc.wait(timeout=_KILL_AFTER)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_KILL_AFTER)

    @property
    def pid(self) -> int | None:
        """The child's process id, or None once it is gone."""
        proc = self._proc
        return proc.pid if proc is not None else None

    @property
    def is_running(self) -> bool:
        """Whether the server process is still alive.

        Public because 'did the child actually die' is a real question for a
        caller shutting a fleet down, and one they should not have to answer
        by reaching into a private handle.
        """
        proc = self._proc
        return proc is not None and proc.poll() is None

    def terminate_now(self) -> None:
        """Kill the child without the graceful wait. For tests of the dead-server path."""
        proc = self._proc
        if proc is not None:
            proc.kill()
            proc.wait(timeout=_KILL_AFTER)

    # -- sending -------------------------------------------------------------

    def request(self, method: str, params: Any = None) -> Any:
        request_id = self._allocate_id()
        pending = self._register(request_id)
        try:
            self._send_message(self._frame(method, params, request_id))
        except McpError:
            with self._lock:
                self._pending.pop(request_id, None)
            raise
        return self._await_response(pending, request_id, method, self._timeout)

    def notify(self, method: str, params: Any = None) -> None:
        self._send_message(self._frame(method, params))

    def _send_message(self, message: Mapping[str, Any]) -> None:
        proc = self._proc
        stdin = proc.stdin if proc is not None else None
        if stdin is None:
            raise McpError(
                "MCP stdio transport is not started (or has been closed)",
                code=McpErrorCode.CONNECTION_CLOSED,
            )
        line = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        # One writer at a time: the reader thread answers server requests while
        # the caller's thread sends its own, and two interleaved writes would
        # produce a line that is neither message.
        with self._write_lock:
            try:
                stdin.write(line)
                stdin.flush()
            except (OSError, ValueError) as exc:
                raise McpError(
                    f"MCP stdio: writing to the server failed: {exc}",
                    code=McpErrorCode.CONNECTION_CLOSED,
                ) from exc

    # -- reading -------------------------------------------------------------

    def _read_loop(self) -> None:
        proc = self._proc
        stdout = proc.stdout if proc is not None else None
        if stdout is None:
            return
        buffer = b""
        try:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                buffer = self._drain(buffer)
                if len(buffer) > self._max_buffer_bytes:
                    self._overflow(len(buffer))
                    return
        except (OSError, ValueError):
            # The pipe closed under us -- indistinguishable from the child
            # exiting, and handled the same way below.
            pass
        if not self._closed:
            self._fail_all(
                McpError(
                    "MCP stdio server exited", code=McpErrorCode.CONNECTION_CLOSED
                )
            )

    def _drain(self, buffer: bytes) -> bytes:
        """Route every complete line, and return what is left over."""
        while True:
            index = buffer.find(b"\n")
            if index < 0:
                return buffer
            line, buffer = buffer[:index], buffer[index + 1 :]
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._parse_and_route(text)

    def _parse_and_route(self, line: str) -> None:
        try:
            message = json.loads(line)
        except ValueError:
            # Not JSON. A server's own logs belong on stderr, but plenty write
            # a banner to stdout on startup, and dropping it is kinder than
            # failing the connection over a line nobody meant as a message.
            return
        if isinstance(message, Mapping):
            self._route_incoming(message)

    def _overflow(self, size: int) -> None:
        self._fail_all(
            McpError(
                f"MCP stdio server sent {size} bytes with no newline, over the "
                f"{self._max_buffer_bytes}-byte limit. It is not speaking "
                "newline-delimited JSON-RPC on stdout -- a crash dump or a log line "
                "written to the wrong stream is the usual cause. Raise "
                "max_buffer_bytes if its messages are legitimately larger.",
                code=McpErrorCode.CONNECTION_CLOSED,
            )
        )
        self.close()


__all__ = ["DEFAULT_MAX_BUFFER_BYTES", "DEFAULT_TIMEOUT_SECONDS", "StdioTransport", "safe_env"]
