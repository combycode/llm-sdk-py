"""Connect to an MCP server and get its tools.

    with connect_mcp(command="python", args=["server.py"]) as mcp:
        answer = complete("...", tools=mcp.tools())

The tools that come back are ordinary `Tool` objects, so everything that already
works for a local tool -- the loop, lazy loading, permissions, the reports --
works for a server's tools without knowing where they came from.

A CONTEXT MANAGER, unlike the TypeScript's `connectMcp` + `close()` pair. The
resource here is a child process, and a Python caller who forgets to close one
leaves it running; `with` is how Python says "this owns something".

`command=` runs a server as a child process; `url=` reaches one over Streamable
HTTP or a WebSocket. All three end at the same `McpClient`, so nothing above the
transport knows which it got.

A server can also ask things of US -- to run a completion (`sampling`), to put a
question to a person (`elicit`), or for the filesystem roots it may see
(`roots`). Each is off unless configured, and configuring one DECLARES the
matching capability: a server is entitled to assume it may use what we
advertised, so advertising something we cannot answer is worse than silence.

Transposed from `unified-library-ts/src/helpers/mcp.ts`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, Self

from ..mcp.client import McpClient, McpClientOptions
from ..mcp.errors import McpError, McpErrorCode
from ..mcp.http import HttpTransport
from ..mcp.oauth import McpOAuth, McpUnauthorizedError
from ..mcp.sampling import (
    McpSamplingConfig,
    McpSamplingHandler,
    McpSamplingViaLLM,
    sampling_handler_with,
)
from ..mcp.stdio import DEFAULT_TIMEOUT_SECONDS, StdioTransport
from ..mcp.tools import mcp_tool_to_tool, sanitize_namespace
from ..mcp.ws import WsTransport
from .tool import Tool

_SUFFIX = re.compile(r"\.[a-zA-Z0-9]+$")

#: Commands that are a runtime rather than a server. The TypeScript names a
#: connection after the command, which for these is `python` or `node` -- the
#: least distinguishing token available, and identical for every server started
#: the same way. Since the namespace exists precisely to keep two servers'
#: tools apart, an interpreter defers to the script it was asked to run.
_INTERPRETERS = frozenset(
    {"python", "python3", "py", "pythonw", "node", "bun", "deno", "ruby", "php", "sh", "bash"}
)


def sampling_handler(config: McpSamplingConfig | str) -> McpSamplingHandler:
    """A handler for the server's `sampling/createMessage`.

    Pass a model id to wire one up, an `McpSamplingViaLLM` to say more about
    which model, or your own function to keep full control.

    Lives here rather than in `mcp/sampling.py` because it needs `complete`, and
    a plugin importing the ergonomic layer would close a cycle -- `helpers`
    already imports most of the plugins. The message mapping stays down there;
    only the dependency is injected from up here.
    """
    from .one_shot import complete

    resolved = McpSamplingViaLLM(model=config) if isinstance(config, str) else config
    return sampling_handler_with(complete, resolved)


def _basename(path: str) -> str:
    return _SUFFIX.sub("", re.split(r"[\\/]", path)[-1])


def _default_url_namespace(url: str) -> str:
    """The server's own name, from its host: `mcp.deepwiki.com` -> `deepwiki`.

    The FIRST label, not the last: `com` names nothing, and `mcp` is what every
    one of these hosts is called.

    An address is not a name, so an IP or a bare `localhost` falls back to
    `mcp` -- `127.0.0.1` would otherwise namespace every local server as `127`,
    which is both meaningless and identical for all of them.
    """
    from ipaddress import ip_address
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    try:
        ip_address(host)
    except ValueError:
        pass
    else:
        return "mcp"
    labels = [part for part in host.split(".") if part and part != "mcp"]
    if not labels or labels[0] in {"localhost", "local"}:
        return "mcp"
    return labels[0]


def _http_transport(
    url: str,
    namespace: str,
    engine: Any,
    headers: Any,
    timeout: float,
    auth: Any = None,
    security: Any = None,
) -> tuple[HttpTransport, McpOAuth | None]:
    """An HTTP transport that sends through the engine, never around it.

    Returns the OAuth orchestrator beside it, because `connect_mcp` has to ask
    it whether a person must act BEFORE the handshake -- and a transport that
    hid it would leave that question unanswerable.

    A transport with its own connection would be invisible to the rate limiter
    and the retry policy -- which is the whole reason those live in one place.
    """
    from .engine import default_engine

    resolved = engine if engine is not None else default_engine()
    if resolved is None:
        from ..bus.hook_bus import HookBus
        from ..network.executor import RequestExecutor
        from ..network.retry import DEFAULT_RETRY
        from ..transport import as_fetch, as_fetch_stream, http_transport

        send = as_fetch(http_transport())
        executor = RequestExecutor(HookBus())

        def own_fetch(request: Any, options: Any = None) -> Any:
            return executor.execute(request, lambda r: send(r), DEFAULT_RETRY)

        fetch: Any = own_fetch
        stream: Any = as_fetch_stream(http_transport())
    else:
        fetch = resolved.fetch
        stream = resolved.fetch_stream

    # OAuth rides the same fetch as everything else, so a token request is
    # queued and retried like any other call rather than slipping past.
    oauth = McpOAuth(url, auth, fetch, security) if auth is not None else None
    return HttpTransport(
        url,
        fetch=fetch,
        fetch_stream=stream,
        headers=headers,
        name=namespace,
        timeout=timeout,
        queue_name=f"mcp/{namespace}",
        auth_headers=oauth.auth_header if oauth else None,
        on_unauthorized=oauth.reauthorize if oauth else None,
    ), oauth


def _default_namespace(command: str, args: Sequence[str]) -> str:
    """The server's own name, from whatever in the command line identifies it."""
    base = _basename(command)
    if base.lower() in _INTERPRETERS:
        for arg in args:
            # The script, not a flag and not a bare word: `-m pkg` names the
            # module, and a lone `-u` names nothing.
            if not arg.startswith("-") and _SUFFIX.search(arg):
                return _basename(arg)
    return base


class McpConnection:
    """One connected server: its tools, its client, and its lifetime."""

    def __init__(self, client: McpClient, namespace: str, *, lazy: bool = False) -> None:
        self._client = client
        self._namespace = namespace
        self._lazy = lazy
        self._definitions: list[dict[str, Any]] = []

    # -- state ---------------------------------------------------------------

    @property
    def client(self) -> McpClient:
        """The client underneath -- resources, prompts, and the raw request path."""
        return self._client

    @property
    def namespace(self) -> str:
        """The prefix every one of this server's tool names carries."""
        return self._namespace

    @property
    def server_info(self) -> dict[str, Any] | None:
        """What the server said about itself when it connected."""
        return self._client.info

    # -- tools ---------------------------------------------------------------

    def tools(self, *, lazy: bool | None = None) -> list[Tool]:
        """This server's tools, wrapped and namespaced.

        `lazy` overrides the connection's setting for this call. Fresh `Tool`
        objects each time, because `lazy` is a property OF the object: handing
        back a cached list would mean a per-call override silently changed the
        tools a previous caller is still holding.
        """
        defer = self._lazy if lazy is None else lazy
        return [
            mcp_tool_to_tool(self._client, definition, self._namespace, lazy=defer)
            for definition in self._definitions
        ]

    def refresh(self) -> list[Tool]:
        """Re-ask the server what it publishes, and return the new tools."""
        self._definitions = self._client.list_tools()
        return self.tools()

    def definitions(self) -> list[dict[str, Any]]:
        """The raw MCP tool definitions, as the server sent them."""
        return [dict(d) for d in self._definitions]

    # -- the protocol, for a caller who wants it directly --------------------
    #
    # Forwarded rather than left on `.client`: a caller reaching past the
    # connection for `list_tools` has to know which object owns what, and the
    # answer is uninteresting to them.

    def list_tools(self) -> list[dict[str, Any]]:
        """Re-ask the server for its raw tool definitions."""
        self._definitions = self._client.list_tools()
        return self.definitions()

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one tool by its un-namespaced server name."""
        return self._client.call_tool(name, arguments)

    def invalidate_cache(self, method: str | None = None) -> None:
        """Drop cached results, for one method or all of them."""
        self._client.invalidate_cache(method)

    # -- lifetime ------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<McpConnection {self._namespace!r}, {len(self._definitions)} tool(s)>"


def connect_mcp(
    *,
    command: str | None = None,
    args: Sequence[str] = (),
    url: str | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    name: str | None = None,
    namespace: str | None = None,
    lazy: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cache_results: bool = False,
    engine: Any = None,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] | None = None,
    connect: Any = None,
    auth: Any = None,
    security: Any = None,
    client_info: Mapping[str, str] | None = None,
    capabilities: Mapping[str, Any] | None = None,
    on_notification: Callable[[str, Any], None] | None = None,
    on_server_request: Callable[[str, Any], Any] | None = None,
    sampling: McpSamplingConfig | str | None = None,
    elicit: Callable[[Mapping[str, Any]], Any] | None = None,
    roots: Sequence[Mapping[str, Any]] | Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    input_required_max_rounds: int | None = None,
    keep_alive: float | None = None,
    hooks: Any = None,
    protocol_mode: str = "auto",
) -> McpConnection:
    """Connect to one MCP server and list its tools.

    `command=` runs a server as a child process. `url=` reaches one over the
    network: `https://` speaks Streamable HTTP, `wss://` speaks JSON-RPC over a
    WebSocket (an extra this library supports, not a spec transport).

    `lazy=True` registers every tool and declares none: the model finds what it
    needs through `tool_search`. This is the common case for a server, because a
    server is where a large tool block usually comes from -- and a large block is
    paid for on every turn, most of it for tools that will never be called.

    `keep_alive=` pings every N SECONDS so an idle connection is not reaped by
    whatever sits between here and the server. Off by default, and ignored on a
    2026-07-28 session where `ping` no longer exists.

    `sampling=`, `elicit=` and `roots=` answer the server's own requests, and
    each one declares the matching capability. They work identically on both
    wires: a pre-2026 server PUSHES the request mid-call, a 2026-07-28 server
    RETURNS it as `input_required` and we re-issue the call with the answer.
    """
    if url and command:
        raise ValueError(
            "connect_mcp: give a url OR a command, not both -- they are two "
            "different servers, and one of them would be silently ignored."
        )
    if not url and not command:
        raise ValueError("connect_mcp needs a command to run, or a url to reach")

    sampler = sampling_handler(sampling) if sampling is not None else None
    declared: dict[str, Any] = dict(capabilities or {})
    # Declared only when it can actually be answered. A capability we advertise
    # is one the server is entitled to use, so advertising an unanswerable one
    # turns our own configuration gap into the server's failed request.
    if sampler is not None:
        declared.setdefault("sampling", {})
    if elicit is not None:
        declared.setdefault("elicitation", {})
    if roots is not None:
        declared.setdefault("roots", {"listChanged": False})

    def dispatch(method: str, params: Any) -> Any:
        if method == "sampling/createMessage" and sampler is not None:
            return sampler(params if isinstance(params, Mapping) else {})
        if method == "elicitation/create" and elicit is not None:
            return elicit(params if isinstance(params, Mapping) else {})
        if method == "roots/list" and roots is not None:
            listed = roots() if callable(roots) else roots
            return {"roots": [dict(r) for r in listed]}
        # Falls through to the caller's own handler rather than replacing it.
        # The TypeScript has no `onServerRequest` on this entry point at all, so
        # it cannot combine the two; here, configuring `sampling` must not
        # silently disconnect a handler somebody already passed.
        if on_server_request is not None:
            return on_server_request(method, params)
        raise McpError(
            f"unsupported server request: {method}", code=McpErrorCode.METHOD_NOT_FOUND
        )

    handles_requests = sampler is not None or elicit is not None or roots is not None

    transport: Any
    oauth: Any = None
    if url:
        ns = sanitize_namespace(namespace or name or _default_url_namespace(url))
        if url.lower().startswith(("ws:", "wss:")):
            transport = WsTransport(
                url,
                headers=headers,
                subprotocols=subprotocols,
                name=ns,
                timeout=timeout,
                connect=connect,
            )
        else:
            transport, oauth = _http_transport(
                url, ns, engine, headers, timeout, auth=auth, security=security
            )
    else:
        ns = sanitize_namespace(namespace or name or _default_namespace(command or "", args))
        transport = StdioTransport(command or "", args, env=env, cwd=cwd, timeout=timeout)
    client = McpClient(
        transport,
        McpClientOptions(
            client_info=dict(client_info) if client_info else McpClientOptions().client_info,
            capabilities=declared,
            on_notification=on_notification,
            on_server_request=dispatch if handles_requests else on_server_request,
            hooks=hooks,
            server=ns,
            protocol_mode=protocol_mode,
            cache_results=cache_results,
            input_required_max_rounds=input_required_max_rounds,
            keep_alive=keep_alive,
        ),
    )
    # Authorized BEFORE the handshake, so `initialize` carries the bearer.
    # A server that needs a person to act says so by name rather than
    # failing the connection with a 401 nobody can act on.
    if oauth is not None and oauth.authorize() == "redirect":
        transport.close()
        raise McpUnauthorizedError(
            f"{url} requires authorization: the provider has been asked to redirect. "
            "Finish with finish_mcp_auth(), then connect again."
        )
    client.connect()
    # Only after the handshake: the GET channel needs the session the
    # handshake establishes, and opening it first is a 400.
    listen = getattr(transport, "listen", None)
    if listen is not None:
        listen()
    connection = McpConnection(client, ns, lazy=lazy)
    try:
        connection.refresh()
    except Exception:
        # A connection whose tool list failed is not usable, and leaving the
        # child running would be a process leak charged to the caller's `with`
        # that never got a chance to run.
        connection.close()
        raise
    return connection


def mcp_toolset(
    servers: Sequence[Mapping[str, Any]], **shared: Any
) -> tuple[list[Tool], list[McpConnection]]:
    """Connect to several servers and return one flat, namespaced toolset.

    The connections come back beside the tools because they have to be closed,
    and a helper that hid them would be a helper that leaks processes.
    """
    connections: list[McpConnection] = []
    try:
        for config in servers:
            connections.append(connect_mcp(**{**shared, **dict(config)}))
    except Exception:
        for open_connection in connections:
            open_connection.close()
        raise
    return [tool for c in connections for tool in c.tools()], connections


__all__ = ["McpConnection", "connect_mcp", "mcp_toolset"]
