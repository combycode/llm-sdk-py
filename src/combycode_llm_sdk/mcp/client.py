"""The MCP protocol, over whatever transport it was handed.

The client speaks METHODS -- initialize, tools/list, tools/call, resources,
prompts -- and knows nothing about pipes or sockets. That split is what lets the
same connect-and-list logic serve a child process today and an HTTP session
later without either one knowing about the other.

Two behaviours worth naming, because both look like details and are not:

- **Lists are paginated.** A server with sixty tools may answer `tools/list` in
  pages, and a client that reads only the first page silently offers the model a
  subset -- which presents as a model that "forgot" a tool rather than as a bug
  here.
- **A tool error is not an exception.** `tools/call` answering `isError: true`
  means the tool ran and failed, and the model is the right audience for that.
  Only transport and protocol failures raise.

The 2026-07-28 wire adds three subsystems that all pass through here, and each
one absorbs a difference rather than exposing it:

- **`subscriptions/listen`** replaces `resources/subscribe` and the standalone
  notification channel with one long-lived stream. `listen()` opens it; frames
  are attributed to a subscription before anything else sees them.
- **Tasks** run a tool call in the background, so `call_tool_task` returns
  immediately and `await_task` polls to a terminal state.
- **`input_required`** replaces the server->client back-channel: instead of
  pushing a request at us mid-call, the server RETURNS its questions and we
  re-issue the call with the answers. Both mechanisms are answered by the same
  handler, so sampling is configured once and works on either wire.

Transposed from `unified-library-ts/src/plugins/mcp/client.ts`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from .errors import McpError, McpErrorCode
from .input_required import (
    InputRequiredRetry,
    is_input_required,
    run_input_required_driver,
)
from .protocol import (
    MCP_CLIENT_CAPABILITIES_META_KEY,
    MCP_CLIENT_INFO_META_KEY,
    MCP_LATEST_HANDSHAKE_VERSION,
    MCP_LATEST_MODERN_VERSION,
    MCP_PROTOCOL_VERSION_META_KEY,
    MCP_SERVER_INFO_META_KEY,
    McpEra,
    is_handshake_mcp_version,
    is_modern_mcp_version,
    mcp_era_of,
    newest_mutual_modern_version,
    supported_versions_from,
)
from .result_cache import McpResultCache
from .subscriptions import (
    McpServerEvent,
    McpSubscription,
    McpSubscriptionFilter,
)
from .transport import IncomingHandlers, McpTransport

#: How this client identifies itself when a server asks.
DEFAULT_CLIENT_INFO = {"name": "combycode-llm-sdk", "version": "0"}

#: How long `close()` waits for the keep-alive thread to unwind. Short: it is
#: either sitting in an interruptible wait or inside one ping, and a caller
#: shutting down must not be held up by a server that has stopped answering.
_KEEP_ALIVE_JOIN_SECONDS = 5.0

#: Task statuses that will not change again, so polling can stop.
TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "cancelled"})

#: How long `await_task` polls before giving up, and how long it waits between
#: polls when the server suggests nothing. The server's own `pollInterval` wins
#: whenever it sends one -- it knows what the work costs.
DEFAULT_TASK_TIMEOUT_SECONDS = 120.0
DEFAULT_TASK_POLL_SECONDS = 0.5

#: A bound on pagination. A server that answers every page with the same cursor
#: would otherwise spin here forever, and "the list is enormous" and "the server
#: is broken" both end the same way.
MAX_PAGES = 1000


@dataclass
class McpClientOptions:
    """Everything the client needs beyond the transport."""

    client_info: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_CLIENT_INFO))
    #: Advertised in `initialize` -- sampling, roots, elicitation.
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    on_notification: Any = None
    on_server_request: Any = None
    hooks: Any = None
    #: The namespace label carried on this server's hook events.
    server: str = "mcp"
    #: `"auto"` probes for a modern server; `"legacy"` skips the probe; a
    #: version string adopts that revision directly.
    protocol_mode: str = "auto"
    #: Honour a server's `ttlMs` / `cacheScope` hints on list and read results.
    #: Off by default: caching changes when a caller observes a server change,
    #: which is the caller's call. A server that sends no hints caches nothing
    #: either way, so this is a no-op against every pre-2026 server.
    cache_results: bool = False
    #: Send a `ping` this often to keep the connection alive. None or 0 is
    #: off. SECONDS, where the TypeScript takes milliseconds -- the same
    #: boundary the rest of this library's public surface draws.
    #:
    #: Ignored on a modern session, where `ping` no longer exists. Ignored
    #: rather than refused: this is a hint about the connection, not a request
    #: the caller made, and failing a connect over it would be absurd.
    keep_alive: float | None = None
    #: Cap on `input_required` rounds before giving up. None takes the default
    #: every other SDK uses. A handler that never satisfies the server would
    #: otherwise loop forever, which presents as a hang rather than an error.
    input_required_max_rounds: int | None = None


class McpClient:
    """One connected MCP server."""

    def __init__(self, transport: McpTransport, options: McpClientOptions | None = None) -> None:
        self._transport = transport
        self._options = options or McpClientOptions()
        self._server_info: dict[str, Any] | None = None
        self._negotiated_version = MCP_LATEST_HANDSHAKE_VERSION
        self._discovery: dict[str, Any] | None = None
        self._cache = McpResultCache() if self._options.cache_results else None
        self._subscriptions: dict[str | int, McpSubscription] = {}
        self._keep_alive: threading.Thread | None = None
        #: Set to stop the keep-alive. An Event rather than a sleep so that
        #: closing does not have to wait out a whole interval -- a keep-alive
        #: measured in minutes would otherwise hold `close()` for minutes.
        self._keep_alive_stop = threading.Event()

    # -- state ---------------------------------------------------------------

    @property
    def info(self) -> dict[str, Any] | None:
        """The server's `initialize` result, or None before `connect()`.

        On a modern session there is no `initialize`, so this is SYNTHESISED
        from the discover result. Deliberate: what a caller reads must not
        depend on which wire was negotiated.
        """
        return self._server_info

    @property
    def protocol_version(self) -> str:
        """The revision actually negotiated."""
        return self._negotiated_version

    @property
    def era(self) -> McpEra:
        """Which wire this session speaks."""
        return mcp_era_of(self._negotiated_version)

    @property
    def discover_result(self) -> dict[str, Any] | None:
        """The raw `server/discover` result on a modern session; None otherwise."""
        return self._discovery

    @property
    def transport(self) -> McpTransport:
        """The wire underneath, for a caller that needs what this does not expose."""
        return self._transport

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """Open the transport and negotiate a protocol revision."""
        self._transport.set_handlers(
            IncomingHandlers(
                on_request=self._handle_server_request,
                on_notification=self._handle_notification,
            )
        )
        self._transport.start()

        mode = self._options.protocol_mode or "auto"
        try:
            if mode == "legacy":
                self._handshake()
            elif mode != "auto" and is_modern_mcp_version(mode):
                self._adopt_modern(self._send_discover(mode), mode)
            elif mode != "auto" and is_handshake_mcp_version(mode):
                self._handshake()
            elif mode != "auto":
                raise McpError(f"unknown MCP protocol mode {mode!r}")
            else:
                self._negotiate_auto()
        except Exception:
            # A half-open connection is worse than none: the child stays alive
            # holding a pipe nobody will ever read.
            self._transport.close()
            raise
        self._start_keep_alive()
        return self._server_info or {}

    def close(self) -> None:
        # Stopped BEFORE the transport goes: a ping racing the close would
        # write to a pipe that is being torn down, and the error it raised
        # would be reported against a connection the caller had already
        # finished with.
        self._keep_alive_stop.set()
        keep_alive = self._keep_alive
        self._keep_alive = None
        if keep_alive is not None and keep_alive is not threading.current_thread():
            keep_alive.join(timeout=_KEEP_ALIVE_JOIN_SECONDS)
        self._transport.close()

    def _start_keep_alive(self) -> None:
        """Ping on an interval, so an idle connection is not reaped.

        A daemon thread, which is the Python spelling of the TypeScript's
        `timer.unref()`: a connection nobody is using must not be the reason a
        process refuses to exit.

        Skipped entirely on a modern session -- `ping` does not exist at
        2026-07-28, so starting one there would send a method the server has
        every right to reject, on a timer, forever.
        """
        interval = self._options.keep_alive
        if not interval or interval <= 0 or self.era != "handshake":
            return
        self._keep_alive_stop.clear()
        self._keep_alive = threading.Thread(
            target=self._keep_alive_loop,
            args=(interval,),
            name="mcp-keepalive",
            daemon=True,
        )
        self._keep_alive.start()

    def _keep_alive_loop(self, interval: float) -> None:
        """One ping per interval until told to stop.

        A failed ping is swallowed. It is not the caller's problem and there is
        nobody to report it to -- the next real request will meet the same
        failure and raise it where someone is waiting, with the context of what
        they were actually trying to do.
        """
        while not self._keep_alive_stop.wait(interval):
            try:
                self._transport.request("ping")
            except Exception:  # noqa: BLE001, S110 -- see the docstring.
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- negotiation ---------------------------------------------------------

    def _handshake(self) -> None:
        """The pre-2026 path: `initialize`, then `notifications/initialized`.

        Resets the declared version first: a failed probe leaves the modern one
        stamped on the transport, and on HTTP that header is what routes the
        request -- so an `initialize` carrying `2026-07-28` would be sent to the
        very handler that just refused us.
        """
        reset = getattr(self._transport, "set_protocol_version", None)
        if reset is not None:
            reset(MCP_LATEST_HANDSHAKE_VERSION)
        result = self._transport.request(
            "initialize",
            {
                "protocolVersion": MCP_LATEST_HANDSHAKE_VERSION,
                "capabilities": dict(self._options.capabilities),
                "clientInfo": dict(self._options.client_info),
            },
        )
        self._server_info = dict(result) if isinstance(result, Mapping) else {}
        self._negotiated_version = (
            str(self._server_info.get("protocolVersion") or MCP_LATEST_HANDSHAKE_VERSION)
        )
        self._tell_transport()
        self._transport.notify("notifications/initialized")

    def _tell_transport(self) -> None:
        """Hand the negotiated revision and era down to the wire.

        Bookkeeping over stdio, which has no headers -- and load-bearing over
        HTTP, where a modern server ROUTES on `Mcp-Protocol-Version`. A
        transport that never learns it sends every request without the header
        and lands on the wrong handler, which is invisible until the wire has
        one.
        """
        setter = getattr(self._transport, "set_protocol_version", None)
        if setter is not None:
            setter(self._negotiated_version)
        era_setter = getattr(self._transport, "set_era", None)
        if era_setter is not None:
            era_setter(self.era)

    def _send_discover(self, version: str) -> dict[str, Any]:
        """One `server/discover` probe. No retry and no adoption -- that is the caller's call.

        The transport is told the version FIRST: on HTTP it is not bookkeeping,
        because a modern server routes by that header, so a probe sent without
        it lands on the legacy handler and is rejected outright instead of
        negotiating.
        """
        version_setter = getattr(self._transport, "set_protocol_version", None)
        if version_setter is not None:
            version_setter(version)
        result = self._transport.request(
            "server/discover",
            {
                "_meta": {
                    MCP_PROTOCOL_VERSION_META_KEY: version,
                    MCP_CLIENT_INFO_META_KEY: dict(self._options.client_info),
                    MCP_CLIENT_CAPABILITIES_META_KEY: dict(self._options.capabilities),
                }
            },
        )
        return dict(result) if isinstance(result, Mapping) else {}

    def _adopt_modern(self, result: Mapping[str, Any], version: str) -> None:
        """Take the modern wire, synthesising the info a handshake would have given."""
        self._discovery = dict(result)
        self._negotiated_version = version
        self._tell_transport()
        meta = result.get("_meta")
        server_info = meta.get(MCP_SERVER_INFO_META_KEY) if isinstance(meta, Mapping) else None
        self._server_info = {
            "protocolVersion": version,
            "capabilities": dict(result.get("capabilities") or {}),
            "serverInfo": dict(server_info) if isinstance(server_info, Mapping) else {},
            **({"instructions": result["instructions"]} if "instructions" in result else {}),
        }

    def _negotiate_auto(self) -> None:
        """Probe for a modern server; fall back to the handshake on anything else.

        The fallback is a DENYLIST. Every JSON-RPC error falls back except a
        `-32022` naming only modern revisions we do not share -- that one is
        positive evidence that no shared wire exists, and retrying the handshake
        against it would just fail twice. A transport error is never an era
        verdict: an outage must not silently downgrade the wire, so it is
        re-raised.
        """
        try:
            result = self._send_discover(MCP_LATEST_MODERN_VERSION)
        except McpError as exc:
            if exc.code == McpErrorCode.CONNECTION_CLOSED:
                raise
            if exc.code == McpErrorCode.UNSUPPORTED_PROTOCOL_VERSION:
                supported = supported_versions_from(exc.data)
                if supported is not None:
                    mutual = newest_mutual_modern_version(supported)
                    if mutual is not None:
                        self._adopt_modern(self._send_discover(mutual), mutual)
                        return
                    if not any(is_handshake_mcp_version(v) for v in supported):
                        raise
            self._handshake()
            return
        self._adopt_modern(result, MCP_LATEST_MODERN_VERSION)

    # -- requests ------------------------------------------------------------

    def request(self, method: str, params: Any = None) -> Any:
        """Send any request method. The escape hatch, envelope included."""
        return self._send(method, params)

    def _send(self, method: str, params: Any = None) -> Any:
        if self.era != "modern":
            return self._transport.request(method, params)
        return self._transport.request(method, self._with_envelope(params))

    def _with_envelope(self, params: Any = None) -> dict[str, Any]:
        """Stamp the modern identity envelope onto a request.

        At 2026-07-28 there is no handshake and no session id, so every request
        states its own identity. Only the discover probe used to build this,
        which made every later call on a modern session fail with a -32602
        naming the missing envelope keys.
        """
        base = dict(params) if isinstance(params, Mapping) else {}
        caller_meta = base.get("_meta")
        return {
            **base,
            "_meta": {
                MCP_PROTOCOL_VERSION_META_KEY: self._negotiated_version,
                MCP_CLIENT_CAPABILITIES_META_KEY: dict(self._options.capabilities),
                MCP_CLIENT_INFO_META_KEY: dict(self._options.client_info),
                **(dict(caller_meta) if isinstance(caller_meta, Mapping) else {}),
            },
        }

    def invalidate_cache(self, method: str | None = None) -> None:
        """Drop cached results, for one method or all of them.

        Exposed because a caller sometimes knows the server moved before the
        server says so -- and because the alternative is waiting out a TTL that
        is now wrong.
        """
        if self._cache is None:
            return
        if method:
            self._cache.clear_method(method)
        else:
            self._cache.clear()

    def _paginate(self, method: str, field_name: str) -> list[dict[str, Any]]:
        """Follow `nextCursor` to the end, collecting `field_name` from each page.

        A paginated list is only as fresh as its shortest-lived page, so the
        cached TTL is the MINIMUM across them: taking the last page's value
        would let an earlier, more volatile page go stale unnoticed.
        """
        cache_key = McpResultCache.key(method)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]

        out: list[dict[str, Any]] = []
        cursor: str | None = None
        ttl_ms: float | None = None
        scope: str | None = None
        for _ in range(MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self._send(method, params)
            page = result.get(field_name) if isinstance(result, Mapping) else None
            if isinstance(page, Sequence) and not isinstance(page, (str, bytes)):
                out.extend(dict(item) for item in page if isinstance(item, Mapping))
            if isinstance(result, Mapping):
                hint = result.get("ttlMs")
                if isinstance(hint, (int, float)) and not isinstance(hint, bool):
                    ttl_ms = float(hint) if ttl_ms is None else min(ttl_ms, float(hint))
                scope = scope or result.get("cacheScope")
            next_cursor = result.get("nextCursor") if isinstance(result, Mapping) else None
            if not next_cursor or next_cursor == cursor:
                if self._cache is not None:
                    self._cache.set(cache_key, out, {"ttlMs": ttl_ms, "cacheScope": scope})
                return out
            cursor = str(next_cursor)
        raise McpError(
            f"MCP {method} did not stop paginating after {MAX_PAGES} pages",
            code=McpErrorCode.INTERNAL_ERROR,
        )

    # -- tools ---------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Every tool the server publishes, across every page."""
        return self._paginate("tools/list", "tools")

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke one tool by its un-namespaced server name.

        A tool that ran and failed comes back as a result with `isError` set --
        not an exception. The model is the audience for that, and it can often
        recover; only the connection failing is our problem.
        """
        args = dict(arguments or {})
        started = time.perf_counter()
        try:
            first = self._send("tools/call", {"name": name, "arguments": args})
            # A modern server can answer "I need input first" instead of a
            # result. Driven to a terminal one here, so the caller only ever
            # sees a finished call. Legacy servers never set `resultType`, so
            # this costs nothing on the handshake wire.
            result = self._drive_input_required(
                first,
                lambda responses, state: self._send(
                    "tools/call",
                    self._resume({"name": name, "arguments": args}, responses, state),
                ),
            )
        except Exception as exc:
            self._emit(
                "onMcpError",
                {"server": self._options.server, "phase": "request", "error": exc},
            )
            raise
        answer = dict(result) if isinstance(result, Mapping) else {}
        self._emit(
            "onMcpToolCall",
            {
                "server": self._options.server,
                "tool": name,
                "latencyMs": (time.perf_counter() - started) * 1000,
                "isError": bool(answer.get("isError")),
            },
        )
        return answer

    # -- resources and prompts -----------------------------------------------

    def list_resources(self) -> list[dict[str, Any]]:
        return self._paginate("resources/list", "resources")

    def list_resource_templates(self) -> list[dict[str, Any]]:
        return self._paginate("resources/templates/list", "resourceTemplates")

    def read_resource(self, uri: str) -> list[dict[str, Any]]:
        """One resource's contents, by URI.

        Cached per URI when caching is on and the server sent a hint. Per URI
        and not per method: two reads of different documents are different
        entries, and sharing one would serve one document for another -- the
        worst kind of cache hit.
        """
        cache_key = McpResultCache.key("resources/read", {"uri": uri})
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return [dict(item) for item in cached]

        first = self._send("resources/read", {"uri": uri})
        result = self._drive_input_required(
            first,
            lambda responses, state: self._send(
                "resources/read", self._resume({"uri": uri}, responses, state)
            ),
        )
        contents = result.get("contents") if isinstance(result, Mapping) else None
        if not isinstance(contents, Sequence) or isinstance(contents, (str, bytes)):
            return []
        out = [dict(c) for c in contents if isinstance(c, Mapping)]
        if self._cache is not None and isinstance(result, Mapping):
            self._cache.set(cache_key, out, result)
        return out

    def subscribe_resource(self, uri: str) -> None:
        """Ask to be told when one resource changes.

        Handshake era only: 2026-07-28 replaces per-resource subscription with
        the single `subscriptions/listen` stream, which `listen()` opens.
        """
        self._require_handshake_era("resources/subscribe")
        self._send("resources/subscribe", {"uri": uri})

    def unsubscribe_resource(self, uri: str) -> None:
        self._require_handshake_era("resources/unsubscribe")
        self._send("resources/unsubscribe", {"uri": uri})

    def listen(
        self,
        notifications: McpSubscriptionFilter,
        on_event: Callable[[McpServerEvent], None],
    ) -> McpSubscription:
        """Open a `subscriptions/listen` stream -- the 2026-07-28 change channel.

        Every kind is OPT-IN: the server may not send what was not asked for,
        and it acknowledges with the subset it actually honoured, which can be
        narrower than the request. Read `subscription.honored` rather than
        assuming the ask was granted.

        Events are level triggers, so they carry nothing beyond the fact of the
        change. When caching is on, the matching entries are dropped BEFORE
        `on_event` runs, so a handler that immediately re-lists sees fresh data.

        Needs a modern session and a transport that can hold a request open.
        """
        if self.era != "modern":
            raise McpError(
                "MCP 'subscriptions/listen' requires protocol version 2026-07-28; this "
                f"session negotiated {self._negotiated_version}. Use subscribe_resource() "
                "plus the change notifications on this wire.",
                code=McpErrorCode.METHOD_NOT_FOUND,
            )
        send_stream = getattr(self._transport, "send_long_lived_request", None)
        if send_stream is None:
            raise McpError(
                "MCP 'subscriptions/listen' is not supported by this transport.",
                code=McpErrorCode.METHOD_NOT_FOUND,
            )

        # The subscription is built and registered INSIDE `on_open`, which the
        # transport runs before the request goes out. A local server
        # acknowledges faster than the send returns, and the reader thread then
        # routes that frame against a registry the subscription is not in yet --
        # so the acknowledgement is dropped and `honored` stays None forever.
        # Registering first closes the window rather than narrowing it.
        holder: dict[str, McpSubscription] = {}

        def on_open(subscription_id: int) -> None:
            def deliver(event: McpServerEvent) -> None:
                self._invalidate_for_event(event)
                on_event(event)

            def forget() -> None:
                self._subscriptions.pop(subscription_id, None)

            subscription = McpSubscription(subscription_id, notifications, deliver, forget)
            holder["subscription"] = subscription
            self._subscriptions[subscription_id] = subscription

        def on_end(error: BaseException | None) -> None:
            subscription = holder.get("subscription")
            if subscription is not None:
                subscription.mark_ended(error)
            if error is not None:
                self._emit(
                    "onMcpError",
                    {"server": self._options.server, "phase": "request", "error": error},
                )

        send_stream(
            "subscriptions/listen",
            self._with_envelope({"notifications": notifications.to_wire()}),
            on_end,
            on_open,
        )
        return holder["subscription"]

    def list_prompts(self) -> list[dict[str, Any]]:
        return self._paginate("prompts/list", "prompts")

    def get_prompt(self, name: str, arguments: Mapping[str, str] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        first = self._send("prompts/get", {"name": name, "arguments": args})
        result = self._drive_input_required(
            first,
            lambda responses, state: self._send(
                "prompts/get", self._resume({"name": name, "arguments": args}, responses, state)
            ),
        )
        return dict(result) if isinstance(result, Mapping) else {}

    def complete_argument(
        self, ref: Mapping[str, Any], argument: Mapping[str, str]
    ) -> dict[str, Any]:
        """Argument autocompletion for a prompt or a resource template.

        An answer with no `completion` becomes an empty list rather than None:
        "the server has no suggestions" and "the server did not answer the
        question" both leave nothing to offer, and a caller iterating the values
        should not have to tell them apart.
        """
        result = self._send("completion/complete", {"ref": dict(ref), "argument": dict(argument)})
        completion = result.get("completion") if isinstance(result, Mapping) else None
        return dict(completion) if isinstance(completion, Mapping) else {"values": []}

    # -- tasks ---------------------------------------------------------------

    def call_tool_task(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        ttl_ms: float | None = None,
    ) -> dict[str, Any]:
        """Start a tool call as a background task, and return it immediately.

        The point of a task is that the answer outlives the request: a tool that
        takes ten minutes cannot be held on one connection without every
        intermediary in the path deciding it has hung.
        """
        meta: dict[str, Any] = {}
        if ttl_ms is not None:
            meta["ttl"] = ttl_ms
        result = self._send(
            "tools/call", {"name": name, "arguments": dict(arguments or {}), "task": meta}
        )
        task = result.get("task") if isinstance(result, Mapping) else None
        if not isinstance(task, Mapping):
            raise McpError(
                f"MCP tools/call for {name!r} was asked to run as a task but the server "
                "answered without one, so there is nothing to poll.",
                code=McpErrorCode.INTERNAL_ERROR,
            )
        return dict(task)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Where one task has got to."""
        result = self._send("tasks/get", {"taskId": task_id})
        return dict(result) if isinstance(result, Mapping) else {}

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        """The finished task's result. Only meaningful once it is terminal."""
        result = self._send("tasks/result", {"taskId": task_id})
        return dict(result) if isinstance(result, Mapping) else {}

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._paginate("tasks/list", "tasks")

    def cancel_task(self, task_id: str) -> None:
        self._send("tasks/cancel", {"taskId": task_id})

    def await_task(
        self,
        task_id: str,
        poll_interval: float | None = None,
        timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Poll until the task stops changing, and return it.

        The SERVER's suggested `pollInterval` wins over the default, because it
        knows what the work costs; an explicit `poll_interval` wins over both,
        because the caller knows what it is willing to spend.

        Seconds here, milliseconds on the wire -- the same boundary the rest of
        this library's public surface draws.
        """
        deadline = time.monotonic() + timeout
        while True:
            task = self.get_task(task_id)
            if str(task.get("status")) in TERMINAL_TASK_STATUSES:
                return task
            if time.monotonic() > deadline:
                raise McpError(
                    f"MCP task {task_id} did not reach a terminal status within "
                    f"{timeout:g}s (last status: {task.get('status')!r}).",
                    code=McpErrorCode.REQUEST_TIMEOUT,
                )
            if poll_interval is not None:
                wait = poll_interval
            else:
                suggested = task.get("pollInterval")
                wait = (
                    float(suggested) / 1000
                    if isinstance(suggested, (int, float)) and not isinstance(suggested, bool)
                    else DEFAULT_TASK_POLL_SECONDS
                )
            # Never sleeps past the deadline: a server suggesting a five-minute
            # interval must not make a one-minute timeout mean six.
            time.sleep(max(0.0, min(wait, max(0.0, deadline - time.monotonic()))))

    def set_log_level(self, level: str) -> None:
        self._require_handshake_era("logging/setLevel")
        self._send("logging/setLevel", {"level": level})

    def ping(self) -> None:
        self._require_handshake_era("ping")
        self._send("ping")

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _resume(
        base: Mapping[str, Any], responses: dict[str, Any] | None, state: str | None
    ) -> dict[str, Any]:
        """The original call again, carrying the answers and the server's state.

        `requestState` is echoed back byte-exact and never inspected: it is the
        server's sealed continuation token, and a client that read one would be
        depending on a private format that is free to change.
        """
        params = dict(base)
        if responses is not None:
            params["inputResponses"] = responses
        if state is not None:
            params["requestState"] = state
        return params

    def _drive_input_required(self, first: Any, retry: InputRequiredRetry) -> Any:
        """Resolve an `input_required` result to a terminal one.

        The dispatcher is `_handle_server_request` -- the SAME path that serves a
        handshake-era server pushing `sampling/createMessage` at us. That is the
        entire point of routing this through here: a caller wires up sampling
        once and it works on either wire, without knowing which is in play.

        Returns instantly when `resultType` is absent or `complete`, so the
        handshake path pays nothing for it.
        """
        if not is_input_required(first):
            return first
        return run_input_required_driver(
            first,
            dispatch=lambda _key, request: self._handle_server_request(
                str(request.get("method") or ""), request.get("params")
            ),
            retry=retry,
            max_rounds=self._options.input_required_max_rounds,
        )

    def _require_handshake_era(self, method: str) -> None:
        """Refuse a method the modern revision removed, and say what replaced it.

        Naming the negotiated version matters: without it the caller sees a bare
        -32601 from the server and has no way to know the method existed until
        this session happened to negotiate modern.
        """
        if self.era == "handshake":
            return
        raise McpError(
            f"MCP {method!r} does not exist at protocol version "
            f"{self._negotiated_version}. Connect with protocol_mode='legacy' to use "
            "the pre-2026 wire, or use the 2026 replacement (listen() for resource "
            "updates; per-request _meta for log level).",
            code=McpErrorCode.METHOD_NOT_FOUND,
        )

    def _invalidate_on_change(self, method: str, params: Any = None) -> None:
        """Map a change notification onto the entries it makes wrong.

        Same vocabulary on both eras: these methods ride the listen stream at
        2026-07-28 and the back-channel before it, but "re-fetch if you care"
        means the same thing either way.
        """
        if self._cache is None:
            return
        if method == "notifications/tools/list_changed":
            self._cache.clear_method("tools/list")
        elif method == "notifications/prompts/list_changed":
            self._cache.clear_method("prompts/list")
        elif method == "notifications/resources/list_changed":
            # BOTH lists: templates are a view of the same resource set, and one
            # that survives its own list going stale is the more misleading half.
            self._cache.clear_method("resources/list")
            self._cache.clear_method("resources/templates/list")
        elif method == "notifications/resources/updated":
            uri = params.get("uri") if isinstance(params, Mapping) else None
            # Only the NAMED resource went stale. Clearing every read would
            # throw away good entries for documents the server said nothing
            # about, which turns one change into a stampede.
            if isinstance(uri, str):
                self._cache.clear_method(McpResultCache.key("resources/read", {"uri": uri}))

    def _invalidate_for_event(self, event: McpServerEvent) -> None:
        """The same invalidation, driven from a typed listen-stream event."""
        if self._cache is None:
            return
        if event.type == "tools_list_changed":
            self._cache.clear_method("tools/list")
        elif event.type == "prompts_list_changed":
            self._cache.clear_method("prompts/list")
        elif event.type == "resources_list_changed":
            self._cache.clear_method("resources/list")
            self._cache.clear_method("resources/templates/list")
        elif event.type == "resource_updated" and event.uri:
            self._cache.clear_method(McpResultCache.key("resources/read", {"uri": event.uri}))

    def _handle_notification(self, method: str, params: Any) -> None:
        """One inbound notification, to everyone entitled to it.

        Order matters twice over. A listen-stream frame belongs to its
        subscription, so that is settled first -- and the frame is still handed
        to `on_notification` afterwards, so a handler written against the
        handshake wire keeps seeing everything it used to.

        Invalidation runs BEFORE the caller's handler: a handler that re-lists
        synchronously must not be answered out of the entry the server has just
        said is stale.
        """
        for subscription in list(self._subscriptions.values()):
            if subscription.handle_frame(method, params):
                break
        self._invalidate_on_change(method, params)
        handler = self._options.on_notification
        if handler is not None:
            handler(method, params)

    def _handle_server_request(self, method: str, params: Any) -> Any:
        """Answer a request the server made of us.

        `ping` is answered here rather than passed on: it is a liveness probe
        with one correct answer, and a connection that dropped because nobody
        configured a handler for it would be a confusing way to fail.

        This is also the dispatcher for `input_required` questions on the modern
        wire -- the same path, deliberately, so a caller wires up sampling once.
        """
        if method == "ping":
            return {}
        handler = self._options.on_server_request
        if handler is None:
            raise McpError(
                f"unsupported server request: {method}", code=McpErrorCode.METHOD_NOT_FOUND
            )
        return handler(method, params)

    def _emit(self, name: str, payload: Mapping[str, Any]) -> None:
        hooks = self._options.hooks
        if hooks is not None:
            hooks.emit_sync(name, dict(payload))

    def __repr__(self) -> str:
        name = (self._server_info or {}).get("serverInfo", {}).get("name", "?")
        return f"<McpClient {name!r} at {self._negotiated_version}>"


__all__ = [
    "DEFAULT_CLIENT_INFO",
    "DEFAULT_TASK_POLL_SECONDS",
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    "MAX_PAGES",
    "TERMINAL_TASK_STATUSES",
    "McpClient",
    "McpClientOptions",
]
