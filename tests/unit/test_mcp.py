"""The MCP client against a real server in a real child process.

Nothing here mocks the transport. A mocked one would prove the client can talk
to our own idea of a server; these spawn `tests/fixtures/mcp_server.py` and run
the protocol through a pipe, which is where buffering, interleaving, partial
reads and process death actually live.

The fixture's modes are how the awkward cases are asked for rather than waited
for: a paginated list, a server that writes a banner to stdout, one that answers
too slowly, one whose tool fails, and one that speaks the 2026 wire.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from combycode_llm_sdk import connect_mcp
from combycode_llm_sdk.helpers import mcp as helpers_mcp
from combycode_llm_sdk.helpers.mcp import (
    McpConnection,
    _default_namespace,
    _default_url_namespace,
    mcp_toolset,
)
from combycode_llm_sdk.hooks import HookBus
from combycode_llm_sdk.mcp import (
    McpClient,
    McpClientOptions,
    McpError,
    McpErrorCode,
    StdioTransport,
    mcp_content_to_result,
    mcp_prompt_to_messages,
    sanitize_namespace,
)
from combycode_llm_sdk.mcp.http import (
    HttpTransport,
    _sole_error,
    media_type_essence,
    pick_response,
    sse_messages,
)
from combycode_llm_sdk.mcp.protocol import (
    mcp_era_of,
    newest_mutual_modern_version,
    supported_versions_from,
)
from combycode_llm_sdk.mcp.result_cache import McpResultCache
from combycode_llm_sdk.mcp.win_spawn import (
    escape_cmd_meta,
    quote_win_arg,
    windows_spawn_plan,
)
from combycode_llm_sdk.mcp.wire_rules import encode_mcp_header_value
from combycode_llm_sdk.mcp.ws import WsTransport, session_is_live
from combycode_llm_sdk.network.errors import LLMError

FIXTURE = str(Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py")

# The stdio fixture is SPAWNED, so it is only ever a path. The WebSocket one
# is an in-process server, so it has to be imported -- as a top-level module
# from its own directory, which is the one name the type checker also gives
# it. Importing it as `fixtures.mcp_ws_server` would be a second name for
# one file, which mypy refuses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
from mcp_ws_server import serve as serve_ws


def connect(mode: str = "normal", **kwargs: Any) -> McpConnection:
    return connect_mcp(command=sys.executable, args=[FIXTURE, mode], **kwargs)


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[StdioTransport]]:
    """Every transport `connect_mcp` creates during a test.

    How a leak is asserted on. Counting python processes on the machine cannot
    tell ours from anyone else's, so it has to be compared with `<=` and then
    passes against a transport that leaks exactly one -- which is the number a
    leak actually leaves behind.
    """
    created: list[StdioTransport] = []

    def recording(*args: Any, **kwargs: Any) -> StdioTransport:
        transport = StdioTransport(*args, **kwargs)
        created.append(transport)
        return transport

    monkeypatch.setattr(helpers_mcp, "StdioTransport", recording)
    yield created
    for transport in created:
        transport.close()


class TestConnecting:
    def test_it_speaks_the_real_protocol_to_a_real_process(self) -> None:
        with connect() as mcp:
            assert mcp.server_info is not None
            assert mcp.server_info["serverInfo"]["name"] == "fixture"
            assert mcp.client.protocol_version == "2025-11-25"
            assert mcp.client.era == "handshake"

    def test_a_server_that_refuses_discover_falls_back_to_the_handshake(self) -> None:
        # Which is every server that exists today, so this IS the common path.
        with connect() as mcp:
            assert mcp.client.era == "handshake"
            assert mcp.client.discover_result is None

    def test_a_modern_server_is_adopted_without_a_handshake(self) -> None:
        with connect("modern") as mcp:
            assert mcp.client.era == "modern"
            assert mcp.client.protocol_version == "2026-07-28"
            # Synthesised, so a caller reads the same shape either way rather
            # than having to ask which wire it got.
            assert mcp.server_info is not None
            assert mcp.server_info["serverInfo"]["name"] == "fixture"

    def test_every_modern_request_carries_its_identity_envelope(self) -> None:
        # At 2026-07-28 there is no session, so a request that omits this is
        # rejected -- and only the discover probe used to build it.
        with connect("modern") as mcp:
            sent: list[Any] = []
            original = mcp.client.transport.request

            def spy(method: str, params: Any = None) -> Any:
                sent.append((method, params))
                return original(method, params)

            mcp.client.transport.request = spy  # type: ignore[method-assign]
            mcp.client.list_tools()
            _, params = sent[0]
            meta = params["_meta"]
            assert meta["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
            assert "io.modelcontextprotocol/clientCapabilities" in meta

    def test_a_handshake_request_carries_no_envelope(self) -> None:
        # An older server has every right to reject an unexpected `_meta`.
        with connect() as mcp:
            sent: list[Any] = []
            original = mcp.client.transport.request

            def spy(method: str, params: Any = None) -> Any:
                sent.append((method, params))
                return original(method, params)

            mcp.client.transport.request = spy  # type: ignore[method-assign]
            mcp.client.list_tools()
            assert "_meta" not in (sent[0][1] or {})

    def test_a_command_that_does_not_exist_names_itself(self) -> None:
        with pytest.raises(McpError, match="could not start"):
            connect_mcp(command="definitely-not-a-real-binary-xyz")

    def test_the_scheme_chooses_the_transport(self, ws_server: Any) -> None:
        # One entry point, three wires. Nothing above the transport knows which
        # it got, so the scheme is the only place that choice is made.
        with ws_connect(ws_server()) as over_socket:
            assert isinstance(over_socket.client.transport, WsTransport)
        with http_connect(HttpServer()) as over_http:
            assert isinstance(over_http.client.transport, HttpTransport)
        with connect() as over_stdio:
            assert isinstance(over_stdio.client.transport, StdioTransport)

    def test_a_url_and_a_command_together_are_refused(self) -> None:
        # Two different servers. Silently picking one would run the wrong one.
        with pytest.raises(ValueError, match="not both"):
            connect_mcp(url="https://example.com/mcp", command="python")

    def test_neither_a_url_nor_a_command_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a command to run, or a url"):
            connect_mcp()


class TestTools:
    def test_the_servers_tools_arrive_namespaced(self) -> None:
        # Two servers may both publish `search`, and the model has to be able to
        # say which one it means.
        with connect() as mcp:
            names = {t.name for t in mcp.tools()}
            assert "mcp_server__add" in names
            assert len(names) == 8

    def test_a_tool_runs_over_the_real_wire(self) -> None:
        with connect() as mcp:
            add = next(t for t in mcp.tools() if t.name.endswith("__add"))
            assert add(a=2, b=3) == "5"

    def test_the_declared_schema_is_the_servers_own(self) -> None:
        with connect() as mcp:
            add = next(t for t in mcp.tools() if t.name.endswith("__add"))
            assert add.definition["parameters"]["required"] == ["a", "b"]

    def test_a_tool_that_fails_returns_text_rather_than_raising(self) -> None:
        # The model is the audience for a tool-level failure and can often
        # recover; only the connection failing is our problem.
        with connect() as mcp:
            broken = next(t for t in mcp.tools() if t.name.endswith("__no_such_thing"))
            assert broken() == "Tool error: no_such_thing is unavailable"

    def test_media_survives_as_parts_rather_than_being_flattened(self) -> None:
        with connect() as mcp:
            picture = next(t for t in mcp.tools() if t.name.endswith("__picture"))
            result = picture()
            assert isinstance(result, list)
            assert [p["type"] for p in result] == ["text", "image"]
            assert result[1]["source"]["mimeType"] == "image/png"

    def test_a_paginated_list_is_followed_to_the_end(self) -> None:
        # A client that reads only the first page offers the model a subset,
        # which presents as a model that "forgot" a tool.
        with connect("paged") as mcp:
            assert len(mcp.tools()) == 8

    def test_a_banner_on_stdout_does_not_break_the_connection(self) -> None:
        # Servers do this. One that cannot survive it fails against a server
        # that works everywhere else.
        with connect("noisy") as mcp:
            assert len(mcp.tools()) == 8

    def test_a_slow_server_times_out_saying_so(self) -> None:
        with pytest.raises(McpError) as caught:
            connect("slow", timeout=0.3)
        assert caught.value.code == McpErrorCode.REQUEST_TIMEOUT

    def test_refresh_re_asks_the_server(self) -> None:
        with connect() as mcp:
            assert len(mcp.refresh()) == 8
            assert len(mcp.definitions()) == 8


class TestLazy:
    def test_lazy_marks_every_tool_from_the_server(self) -> None:
        with connect(lazy=True) as mcp:
            assert all(t.lazy for t in mcp.tools())

    def test_the_flag_travels_with_the_tool(self) -> None:
        # It has to: a tool passes through several hands between the server and
        # the loop that reads the flag.
        with connect(lazy=True) as mcp:
            assert all(t.to_wire()["lazy"] for t in mcp.tools())

    def test_lazy_is_a_per_call_override(self) -> None:
        with connect(lazy=True) as mcp:
            assert not any(t.lazy for t in mcp.tools(lazy=False))
            # And the connection's own setting is unchanged by the override.
            assert all(t.lazy for t in mcp.tools())

    def test_a_default_connection_defers_nothing(self) -> None:
        with connect() as mcp:
            assert not any(t.lazy for t in mcp.tools())

    def test_the_wire_form_is_not_what_a_provider_gets(self) -> None:
        # `lazy` is ours; a provider handed it would reject the request.
        with connect(lazy=True) as mcp:
            tool = mcp.tools()[0]
            assert "lazy" not in tool.definition
            assert tool.to_wire()["definition"] == tool.definition


class TestWindowsSpawn:
    """Resolving a command that is not really a program.

    `npx`, `uvx`, `pnpm` -- what every MCP server's README tells you to run --
    are `.cmd` shims on Windows, and `CreateProcess` cannot execute one. The
    fixture server is a real `.exe` plus a script, so nothing here caught it
    until a live corpus run tried `npx` and got `[WinError 2]`.
    """

    PATH_ENV: ClassVar[dict[str, str]] = {"PATH": r"C:\tools", "PATHEXT": ".EXE;.CMD"}

    def plan(self, command: str, args: Any = (), files: Any = ()) -> Any:
        known = set(files)
        return windows_spawn_plan(command, list(args), self.PATH_ENV, lambda p: p in known)

    def test_a_cmd_shim_is_routed_through_cmd_exe(self) -> None:
        plan = self.plan("npx", ["-y", "pkg"], files={r"C:\tools\npx.CMD"})
        assert plan.file.lower().endswith("cmd.exe")
        assert plan.args[:3] == ["/d", "/s", "/c"]
        assert "npx.CMD" in plan.args[3]
        assert plan.verbatim is True

    def test_a_real_executable_is_spawned_directly(self) -> None:
        plan = self.plan("server", ["--port", "1"], files={r"C:\tools\server.EXE"})
        assert plan.file == r"C:\tools\server.EXE"
        assert plan.args == ["--port", "1"]
        assert plan.verbatim is False

    def test_a_path_or_extension_is_taken_at_its_word(self) -> None:
        # Already unambiguous: looking it up on PATH could find a different one.
        assert self.plan(r"C:\bin\thing.exe").file == r"C:\bin\thing.exe"

    def test_an_explicit_bat_still_goes_through_cmd(self) -> None:
        assert self.plan("run.bat").file.lower().endswith("cmd.exe")

    def test_an_unresolvable_name_is_left_to_fail_by_name(self) -> None:
        # The TypeScript hands this to cmd.exe as one more chance at a shim.
        # The lookup already used the same PATH and PATHEXT cmd would, so that
        # chance finds nothing and costs the error: cmd starts, prints "is not
        # recognized" to stderr and exits, which surfaces as "the server exited".
        plan = self.plan("definitely-not-real")
        assert plan.file == "definitely-not-real"
        assert plan.verbatim is False

    def test_an_argument_with_spaces_survives_quoting(self) -> None:
        # The quotes reach the line caret-escaped: `cmd` reads the line before
        # the quoting rules apply, so an unescaped quote is `cmd`'s to interpret.
        plan = self.plan(
            "npx", ["--dir", r"C:\Program Files\x"], files={r"C:\tools\npx.CMD"}
        )
        assert r'^"C:\Program Files\x^"' in plan.args[3]

    def test_an_argument_needing_no_quotes_gets_none(self) -> None:
        assert quote_win_arg("--port") == "--port"
        assert quote_win_arg(r"C:\dir\file") == r"C:\dir\file"

    def test_a_trailing_backslash_does_not_escape_the_closing_quote(self) -> None:
        # The backslash rule, and the case it actually bites: an argument that
        # NEEDS quoting and ends in a backslash. Undoubled, that backslash
        # escapes the closing quote and swallows the rest of the command line.
        assert quote_win_arg("C:\\Program Files\\") == '"C:\\Program Files\\\\"'

    def test_an_embedded_quote_is_escaped_with_its_backslashes(self) -> None:
        assert quote_win_arg('say "hi"') == '"say \\"hi\\""'

    def test_cmd_metacharacters_are_escaped(self) -> None:
        # An unescaped `&` ends the command even inside quotes.
        assert escape_cmd_meta("a&b") == "a^&b"
        assert escape_cmd_meta("%PATH%") == "^%PATH^%"

    def test_pathext_order_decides_which_file_wins(self) -> None:
        both = {r"C:\tools\npx.EXE", r"C:\tools\npx.CMD"}
        assert self.plan("npx", files=both).file == r"C:\tools\npx.EXE"


class TestTheProtocolThroughTheConnection:
    """The corpus drives the protocol off the connection, not off `.client`."""

    def test_the_raw_protocol_is_reachable_without_naming_the_client(self) -> None:
        with connect() as mcp:
            assert len(mcp.list_tools()) == 8
            answered = mcp.call_tool("add", {"a": 1, "b": 2})
            assert answered["content"][0]["text"] == "3"

    def test_invalidate_cache_is_safe_when_nothing_is_cached(self) -> None:
        # Caching is opt-in, so the common connection has no cache at all --
        # and a caller should not have to know that to call this.
        with connect() as mcp:
            mcp.invalidate_cache("tools/list")
            mcp.invalidate_cache()


class TestTheResultCache:
    def test_nothing_is_cached_without_a_hint(self) -> None:
        # Every pre-2026 server sends none, so behaviour must be identical to
        # having no cache at all.
        cache = McpResultCache()
        assert cache.set("tools/list", [1], None) is False
        assert cache.get("tools/list") is None

    def test_a_ttl_is_honoured_and_then_expires(self) -> None:
        cache = McpResultCache()
        assert cache.set("tools/list", [1], {"ttlMs": 1000}, now=0) is True
        assert cache.get("tools/list", now=999) == [1]
        assert cache.get("tools/list", now=1000) is None

    def test_a_zero_ttl_evicts_rather_than_being_ignored(self) -> None:
        # "Stale now" is an instruction, not a missing value. Without the
        # eviction, a server that said 60s and then 0 keeps being answered from
        # the stale entry for the rest of the minute.
        cache = McpResultCache()
        cache.set("tools/list", [1], {"ttlMs": 60_000}, now=0)
        assert cache.set("tools/list", [2], {"ttlMs": 0}, now=1) is False
        assert cache.get("tools/list", now=2) is None

    def test_params_are_part_of_the_key(self) -> None:
        # Two reads of different URIs are different entries; sharing one would
        # serve one document for another.
        first = McpResultCache.key("resources/read", {"uri": "a"})
        second = McpResultCache.key("resources/read", {"uri": "b"})
        assert first != second

    def test_one_method_can_be_cleared_without_the_others(self) -> None:
        cache = McpResultCache()
        cache.set("tools/list", [1], {"ttlMs": 1000}, now=0)
        cache.set("prompts/list", [2], {"ttlMs": 1000}, now=0)
        cache.clear_method("tools/list")
        assert cache.get("tools/list", now=1) is None
        assert cache.get("prompts/list", now=1) == [2]

    def test_a_hinted_list_is_reused_rather_than_refetched(self) -> None:
        # The churn the hint exists to remove: without it, every `list_tools()`
        # goes back to the wire.
        with connect("caching", cache_results=True) as mcp:
            asked: list[str] = []
            original = mcp.client.transport.request

            def spy(method: str, params: Any = None) -> Any:
                asked.append(method)
                return original(method, params)

            mcp.client.transport.request = spy  # type: ignore[method-assign]
            # `connect_mcp` already listed once while connecting, so start from
            # a state this test states rather than one it inherits.
            mcp.invalidate_cache()
            mcp.client.list_tools()
            mcp.client.list_tools()
            assert asked.count("tools/list") == 1

    def test_invalidating_sends_the_next_list_back_to_the_wire(self) -> None:
        with connect("caching", cache_results=True) as mcp:
            asked: list[str] = []
            original = mcp.client.transport.request

            def spy(method: str, params: Any = None) -> Any:
                asked.append(method)
                return original(method, params)

            mcp.client.transport.request = spy  # type: ignore[method-assign]
            mcp.invalidate_cache()
            mcp.client.list_tools()
            mcp.invalidate_cache("tools/list")
            mcp.client.list_tools()
            assert asked.count("tools/list") == 2

    def test_a_server_that_sends_no_hint_is_never_cached(self) -> None:
        # Every pre-2026 server. Behaviour must be identical to no cache at all,
        # even with caching switched on.
        with connect(cache_results=True) as mcp:
            asked: list[str] = []
            original = mcp.client.transport.request

            def spy(method: str, params: Any = None) -> Any:
                asked.append(method)
                return original(method, params)

            mcp.client.transport.request = spy  # type: ignore[method-assign]
            mcp.invalidate_cache()
            mcp.client.list_tools()
            mcp.client.list_tools()
            assert asked.count("tools/list") == 2


class TestNamespacing:
    def test_an_interpreter_defers_to_the_script_it_runs(self) -> None:
        # Naming every python-hosted server `python` would collide on the first
        # second server, which is the one thing the namespace exists to prevent.
        assert _default_namespace("/usr/bin/python3", ["/srv/weather.py"]) == "weather"
        assert _default_namespace("C:\\Python\\python.exe", ["a\\b\\tickets.py"]) == "tickets"

    def test_a_real_command_names_itself(self) -> None:
        assert _default_namespace("/opt/bin/weather-mcp", []) == "weather-mcp"

    def test_a_flag_is_not_mistaken_for_a_script(self) -> None:
        assert _default_namespace("python", ["-u", "-m", "pkg"]) == "python"

    def test_an_explicit_name_wins(self) -> None:
        with connect(name="calc") as mcp:
            assert mcp.namespace == "calc"
            assert all(t.name.startswith("calc__") for t in mcp.tools())

    def test_a_namespace_is_made_safe_for_a_tool_name(self) -> None:
        assert sanitize_namespace("my server/v2") == "my_server_v2"
        assert sanitize_namespace("!!!") == "mcp"


class TestResourcesAndPrompts:
    def test_a_resource_is_read_back(self) -> None:
        with connect() as mcp:
            contents = mcp.client.read_resource("mem://greeting")
            assert contents[0]["text"] == "hello from mem://greeting"

    def test_resources_and_prompts_are_listed(self) -> None:
        with connect() as mcp:
            assert mcp.client.list_resources()[0]["name"] == "greeting"
            assert mcp.client.list_prompts()[0]["name"] == "greet"

    def test_a_prompt_becomes_messages(self) -> None:
        with connect() as mcp:
            messages = mcp_prompt_to_messages(mcp.client.get_prompt("greet"))
            assert messages == [{"role": "user", "content": "Say hello politely."}]

    def test_a_method_the_modern_wire_removed_says_which_version_removed_it(self) -> None:
        # Without the version the caller sees a bare -32601 and cannot tell the
        # method existed until this session happened to negotiate modern.
        with connect("modern") as mcp, pytest.raises(McpError, match="2026-07-28"):
            mcp.client.set_log_level("info")


class TestTheDuplexChannel:
    def test_a_server_notification_reaches_the_handler(self) -> None:
        seen: list[Any] = []
        with connect(on_notification=lambda m, p: seen.append((m, p))) as mcp:
            mcp.client.request("fixture/notify", {"data": "ping"})
        assert seen and seen[-1][0] == "notifications/message"

    def test_a_server_request_is_answered(self) -> None:
        answered: list[str] = []

        def on_request(method: str, params: Any) -> Any:
            answered.append(method)
            return {"roots": []}

        with connect(on_server_request=on_request) as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")  # a round trip, so the answer has landed
        assert answered == ["roots/list"]

    def test_an_unanswerable_server_request_still_gets_a_reply(self) -> None:
        # A dropped request leaves the SERVER blocked forever, which looks from
        # outside like our connection hanging. Asked of the server, because only
        # the server knows whether anything came back.
        with connect() as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")  # a round trip, so our reply has landed
            replies = mcp.client.request("fixture/replies")["replies"]
        assert [r["ok"] for r in replies] == [False]
        assert replies[0]["error"] == "unsupported server request: roots/list"

    def test_a_transport_with_no_handler_at_all_still_replies(self) -> None:
        # `McpClient` always installs a handler, so this branch belongs to a
        # caller driving the transport directly -- and the server is just as
        # blocked either way.
        transport = StdioTransport(sys.executable, [FIXTURE, "normal"])
        transport.start()
        try:
            transport.request("initialize", {"protocolVersion": "2025-11-25"})
            transport.request("fixture/ask")
            transport.request("ping")
            replies = transport.request("fixture/replies")["replies"]
        finally:
            transport.close()
        assert [r["ok"] for r in replies] == [False]
        assert "no request handler" in replies[0]["error"]

    def test_a_handler_that_raises_still_answers_the_server(self) -> None:
        # Same reason, one layer up: a handler that blows up must not turn into
        # silence on the wire.
        def on_request(method: str, params: Any) -> Any:
            raise RuntimeError("the roots are on fire")

        with connect(on_server_request=on_request) as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")
            replies = mcp.client.request("fixture/replies")["replies"]
        assert [r["ok"] for r in replies] == [False]
        assert replies[0]["error"] == "the roots are on fire"

    def test_a_handler_that_answers_is_reported_as_a_result(self) -> None:
        with connect(on_server_request=lambda m, p: {"roots": []}) as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")
            replies = mcp.client.request("fixture/replies")["replies"]
        assert [r["ok"] for r in replies] == [True]

    def test_a_tool_call_is_reported_on_the_hook_bus(self) -> None:
        hooks = HookBus()
        seen: list[Any] = []
        hooks.on("onMcpToolCall", seen.append)
        with connect(hooks=hooks) as mcp:
            next(t for t in mcp.tools() if t.name.endswith("__add"))(a=1, b=1)
        assert seen[-1].tool == "add"
        assert seen[-1].is_error is False


class TestTheProcess:
    def test_closing_ends_a_well_behaved_child(self) -> None:
        mcp = connect()
        transport = mcp.client.transport
        assert isinstance(transport, StdioTransport)
        assert transport.is_running
        mcp.close()
        assert not transport.is_running

    def test_closing_ends_one_that_ignores_eof_too(self) -> None:
        # A server with its own event loop and no stdin handling does not exit
        # when the pipe closes, and `close()` has to escalate. Without the
        # escalation this returns with the child still running.
        mcp = connect("stubborn")
        transport = mcp.client.transport
        assert isinstance(transport, StdioTransport)
        mcp.close()
        assert not transport.is_running

    def test_a_request_after_close_is_refused_rather_than_hanging(self) -> None:
        mcp = connect()
        mcp.close()
        with pytest.raises(McpError, match="not started"):
            mcp.client.request("ping")

    def test_a_request_already_in_flight_wakes_when_the_child_dies(self) -> None:
        # The case the reader thread exists to cover. Otherwise the caller waits
        # out its full timeout and is told "timed out", which sends whoever
        # reads it looking for a slow server rather than a crashed one.
        mcp = connect("slow", timeout=30)
        transport = mcp.client.transport
        assert isinstance(transport, StdioTransport)
        failure: list[BaseException] = []

        def ask() -> None:
            try:
                mcp.client.request("ping")
            except BaseException as exc:  # noqa: BLE001 -- recorded for the assertion
                failure.append(exc)

        caller = threading.Thread(target=ask)
        caller.start()
        time.sleep(0.2)  # long enough for the request to be on the wire
        transport.terminate_now()
        caller.join(timeout=5)
        assert not caller.is_alive(), "the waiter was never woken"
        assert isinstance(failure[0], McpError)
        assert failure[0].code == McpErrorCode.CONNECTION_CLOSED
        mcp.close()

    def test_a_request_after_the_child_died_is_refused(self) -> None:
        mcp = connect(timeout=30)
        transport = mcp.client.transport
        assert isinstance(transport, StdioTransport)
        transport.terminate_now()
        with pytest.raises(McpError) as caught:
            mcp.client.request("ping")
        assert caught.value.code == McpErrorCode.CONNECTION_CLOSED
        mcp.close()

    def test_the_child_cannot_see_the_parents_secrets(self, monkeypatch: Any) -> None:
        # An MCP server is third-party code, and the parent's environment is
        # where the API keys are. Asked of the CHILD: asserting on `safe_env()`
        # here would only check the function, and in a shell with no key set it
        # would pass against a transport that forwarded everything.
        monkeypatch.setenv("LLM_API_KEY", "sk-not-for-the-child")
        monkeypatch.setenv("COMBYCODE_MARKER", "leaked")
        with connect() as mcp:
            names = mcp.client.request("fixture/env")["names"]
        assert "LLM_API_KEY" not in names
        assert "COMBYCODE_MARKER" not in names
        # And it can still find its own runtime, which is the point of a
        # whitelist rather than an empty environment.
        assert any(n.lower() == "path" for n in names)

    def test_an_explicit_env_entry_does_reach_the_child(self) -> None:
        with connect(env={"COMBYCODE_ASKED_FOR": "1"}) as mcp:
            names = mcp.client.request("fixture/env")["names"]
        assert "COMBYCODE_ASKED_FOR" in names

    def test_an_unterminated_flood_is_bounded_rather_than_growing(self) -> None:
        # NDJSON has no length prefix, so a server that never writes a newline
        # grows the buffer until the process dies.
        transport = StdioTransport(
            sys.executable,
            ["-c", "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()"],
            max_buffer_bytes=1024,
        )
        transport.start()
        with pytest.raises(McpError) as caught:
            transport.request("ping")
        assert "newline" in str(caught.value)
        transport.close()

    def test_a_failed_handshake_does_not_leave_a_child_running(
        self, spawned: list[StdioTransport]
    ) -> None:
        # The caller's `with` never ran, so nothing else would close it. Asked
        # of the transport that was actually created rather than by counting
        # processes on the machine, which cannot tell ours from anyone else's.
        with pytest.raises(McpError):
            connect("slow", timeout=0.3)
        assert spawned, "no transport was created, so this proved nothing"
        assert not any(t.is_running for t in spawned)

    def test_a_failed_tool_list_does_not_leave_a_child_running(
        self, spawned: list[StdioTransport]
    ) -> None:
        # A DIFFERENT path from the one above: this server connects perfectly
        # and then refuses the list, so the failure is past the point where
        # `connect()` cleans up after itself.
        with pytest.raises(McpError, match="tool index is offline"):
            connect("listless")
        assert spawned, "no transport was created, so this proved nothing"
        assert not any(t.is_running for t in spawned)


class TestContentMapping:
    def test_text_only_content_reads_as_a_string(self) -> None:
        result = mcp_content_to_result({"content": [{"type": "text", "text": "hi"}]})
        assert result == "hi"

    def test_an_error_result_is_labelled_for_the_model(self) -> None:
        result = mcp_content_to_result(
            {"content": [{"type": "text", "text": "nope"}], "isError": True}
        )
        assert result == "Tool error: nope"

    def test_an_embedded_resource_becomes_its_text(self) -> None:
        result = mcp_content_to_result(
            {"content": [{"type": "resource", "resource": {"uri": "u", "text": "body"}}]}
        )
        assert result == "body"

    def test_a_resource_without_text_becomes_a_reference(self) -> None:
        result = mcp_content_to_result(
            {"content": [{"type": "resource", "resource": {"uri": "mem://x"}}]}
        )
        assert result == "[resource mem://x]"

    def test_an_unknown_block_type_is_dropped_rather_than_crashing(self) -> None:
        result = mcp_content_to_result(
            {"content": [{"type": "hologram"}, {"type": "text", "text": "ok"}]}
        )
        assert result == "ok"

    def test_an_unknown_block_does_not_survive_into_the_parts(self) -> None:
        # Where dropping it actually matters: beside media the parts go to a
        # provider, and a part shaped like nothing it knows is a rejected
        # request. In text-only output a stray empty part is invisible, which
        # is why that case cannot be the one asserted on.
        result = mcp_content_to_result(
            {
                "content": [
                    {"type": "hologram"},
                    {"type": "image", "mimeType": "image/png", "data": "AA=="},
                ]
            }
        )
        assert isinstance(result, list)
        assert [p["type"] for p in result] == ["image"]

    def test_empty_content_is_an_empty_string(self) -> None:
        assert mcp_content_to_result({}) == ""


class TestProtocolVersions:
    def test_an_unknown_version_reads_as_the_older_safer_wire(self) -> None:
        # Versions are an enumerated set, not an ordered scalar: `"zzz" >
        # "2025-11-25"` is true and meaningless.
        assert mcp_era_of("zzz") == "handshake"
        assert mcp_era_of("2026-07-28") == "modern"

    def test_the_newest_shared_modern_version_wins(self) -> None:
        assert newest_mutual_modern_version(["2026-07-28", "2099-01-01"]) == "2026-07-28"
        assert newest_mutual_modern_version(["2025-11-25"]) is None

    def test_a_malformed_supported_list_reads_as_no_information(self) -> None:
        # Not as an empty and therefore disjoint list -- the difference decides
        # whether negotiation falls back or gives up.
        assert supported_versions_from({"supported": "nope"}) is None
        assert supported_versions_from({"supported": []}) is None
        assert supported_versions_from(None) is None
        assert supported_versions_from({"supported": ["2026-07-28"]}) == ("2026-07-28",)


class TestNegotiation:
    """Which wire gets chosen, with a scripted peer rather than a real one.

    The one place here that does not use the fixture server: the interesting
    cases are a server that answers the probe with a specific error, and one
    whose connection drops mid-probe, and a real process cannot be made to do
    either on demand without becoming a mock with extra steps.
    """

    class Scripted:
        """A transport that answers from a script and records what it was asked.

        A method may be scripted with a LIST, which is consumed one call at a
        time -- negotiation can ask the same method twice with different
        expectations, and the second answer is the whole point of that path.
        """

        def __init__(self, answers: dict[str, Any]) -> None:
            self.answers = {
                k: list(v) if isinstance(v, list) else [v] for k, v in answers.items()
            }
            self.asked: list[str] = []

        def start(self) -> None:
            pass

        def set_handlers(self, handlers: Any) -> None:
            pass

        def notify(self, method: str, params: Any = None) -> None:
            self.asked.append(method)

        def close(self) -> None:
            pass

        def request(self, method: str, params: Any = None) -> Any:
            self.asked.append(method)
            scripted = self.answers.get(method) or [None]
            answer = scripted.pop(0) if len(scripted) > 1 else scripted[0]
            if isinstance(answer, BaseException):
                raise answer
            return answer

    def client_for(self, answers: dict[str, Any]) -> tuple[McpClient, Scripted]:
        transport = self.Scripted(answers)
        return McpClient(transport, McpClientOptions()), transport

    def test_a_dead_connection_during_the_probe_is_not_a_downgrade(self) -> None:
        # An outage must not silently choose the older wire. Without the
        # re-raise this connects as handshake and reports nothing wrong, which
        # is the failure mode that only shows up as a mysteriously legacy
        # session against a modern server.
        client, transport = self.client_for(
            {
                "server/discover": McpError("pipe died", code=McpErrorCode.CONNECTION_CLOSED),
                "initialize": {"protocolVersion": "2025-11-25", "serverInfo": {}},
            }
        )
        with pytest.raises(McpError, match="pipe died"):
            client.connect()
        assert "initialize" not in transport.asked

    def test_a_server_that_rejects_the_probe_falls_back(self) -> None:
        client, transport = self.client_for(
            {
                "server/discover": McpError("no", code=McpErrorCode.METHOD_NOT_FOUND),
                "initialize": {"protocolVersion": "2025-11-25", "serverInfo": {}},
            }
        )
        client.connect()
        assert client.era == "handshake"
        assert "initialize" in transport.asked

    def test_a_shared_modern_version_is_adopted_from_the_rejection(self) -> None:
        # The probe is refused with a list naming a version we DO share, so the
        # second probe adopts it rather than dropping to the older wire.
        client, transport = self.client_for(
            {
                "server/discover": [
                    McpError(
                        "not that one",
                        code=McpErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                        data={"supported": ["2026-07-28"]},
                    ),
                    {"capabilities": {}, "_meta": {}},
                ],
                "initialize": {"protocolVersion": "2025-11-25", "serverInfo": {}},
            }
        )
        client.connect()
        assert client.era == "modern"
        assert transport.asked.count("server/discover") == 2
        assert "initialize" not in transport.asked

    def test_a_rejection_naming_no_shared_version_gives_up(self) -> None:
        # Positive evidence that no shared wire exists. Falling back here would
        # just fail a second time, more slowly.
        client, _ = self.client_for(
            {
                "server/discover": McpError(
                    "no",
                    code=McpErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                    data={"supported": ["2099-01-01"]},
                ),
                "initialize": {"protocolVersion": "2025-11-25", "serverInfo": {}},
            }
        )
        with pytest.raises(McpError, match="no"):
            client.connect()

    def test_an_unknown_protocol_mode_is_refused(self) -> None:
        transport = self.Scripted({})
        client = McpClient(transport, McpClientOptions(protocol_mode="sideways"))
        with pytest.raises(McpError, match="unknown MCP protocol mode"):
            client.connect()


class TestToolset:
    def test_several_servers_become_one_flat_namespaced_toolset(self) -> None:
        tools, connections = mcp_toolset(
            [
                {"command": sys.executable, "args": [FIXTURE, "normal"], "name": "first"},
                {"command": sys.executable, "args": [FIXTURE, "normal"], "name": "second"},
            ]
        )
        try:
            assert len(tools) == 16
            assert {t.name.split("__")[0] for t in tools} == {"first", "second"}
        finally:
            for connection in connections:
                connection.close()

    def test_one_server_failing_closes_the_ones_already_open(
        self, spawned: list[StdioTransport]
    ) -> None:
        # The first server connected fine and is nobody's responsibility once
        # the call raises, so the call has to clean it up itself.
        with pytest.raises(McpError):
            mcp_toolset(
                [
                    {"command": sys.executable, "args": [FIXTURE, "normal"]},
                    {"command": sys.executable, "args": [FIXTURE, "slow"], "timeout": 0.3},
                ]
            )
        assert len(spawned) == 2
        assert not any(t.is_running for t in spawned)


def test_a_client_can_be_driven_directly_without_the_helper() -> None:
    """The layer below `connect_mcp`, for a caller who wants the protocol itself."""
    transport = StdioTransport(sys.executable, [FIXTURE, "normal"])
    client = McpClient(transport, McpClientOptions(protocol_mode="legacy"))
    with client:
        client.connect()
        assert len(client.list_tools()) == 8
        assert client.call_tool("add", {"a": 20, "b": 22})["content"][0]["text"] == "42"


class HttpServer:
    """A Streamable HTTP MCP server, scripted at the fetch boundary.

    Not a socket: `handle()` is where every decision this transport makes shows
    up -- which media type came back, which status, which headers -- and a real
    server would let it choose only one of those per run.
    """

    def __init__(
        self,
        *,
        content_type: str = "application/json",
        status: int = 200,
        session: str | None = "sess-1",
        tools: int = 2,
    ) -> None:
        self.content_type = content_type
        self.status = status
        self.session = session
        self.tools = tools
        self.requests: list[dict[str, Any]] = []
        #: How many of the next calls answer 401 before relenting.
        self.refuse_next = 0

    def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
        self.requests.append(request)
        body = request.get("body")
        method = body.get("method") if isinstance(body, Mapping) else None

        if self.refuse_next > 0:
            self.refuse_next -= 1
            return {"status": 401, "headers": {}, "body": ""}

        payload = self._result(method, body)
        if payload is None:
            return {"status": 202, "headers": self._headers(), "body": ""}
        message = {"jsonrpc": "2.0", "id": (body or {}).get("id"), "result": payload}
        return {
            "status": self.status,
            "headers": self._headers(),
            "body": self._encode(message),
        }

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": self.content_type}
        if self.session:
            headers["mcp-session-id"] = self.session
        return headers

    def _encode(self, message: dict[str, Any]) -> str:
        if media_type_essence(self.content_type) == "text/event-stream":
            # Pretty-printed across several `data:` lines, which is what a real
            # multi-line frame looks like: SSE joins them with a newline, so a
            # reader that took only the first would silently truncate it.
            lines = json.dumps(message, indent=2).split("\n")
            body = "".join(f"data: {line}\n" for line in lines)
            return f"event: message\n{body}\n"
        return json.dumps(message)

    def _result(self, method: Any, body: Any) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "http-fixture", "version": "1"},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {"name": f"t{i}", "description": f"tool {i}",
                     "inputSchema": {"type": "object", "properties": {}}}
                    for i in range(self.tools)
                ]
            }
        if method == "tools/call":
            name = (body.get("params") or {}).get("name")
            return {"content": [{"type": "text", "text": f"ran {name}"}]}
        if method == "server/discover":
            return None
        return {}


def http_connect(server: HttpServer, **kwargs: Any) -> McpConnection:
    return connect_mcp(
        url="https://mcp.example.com/mcp",
        engine=SimpleNamespace(fetch=server, fetch_stream=None),
        protocol_mode="legacy",
        **kwargs,
    )


class TestHttpTransport:
    def test_a_url_server_speaks_the_same_protocol(self) -> None:
        with http_connect(HttpServer()) as mcp:
            assert mcp.server_info is not None
            assert mcp.server_info["serverInfo"]["name"] == "http-fixture"
            assert len(mcp.tools()) == 2

    def test_its_tools_are_ordinary_tools(self) -> None:
        # The whole point of the bridge: nothing downstream knows the tool runs
        # on someone else's server.
        with http_connect(HttpServer()) as mcp:
            tool = mcp.tools()[0]
            assert tool.name == "example__t0"
            assert tool() == "ran t0"

    def test_an_sse_body_is_read_as_well_as_a_json_one(self) -> None:
        # A server answers `text/event-stream` when the request produced
        # notifications on the way, and both media types have to be read.
        with http_connect(HttpServer(content_type="text/event-stream")) as mcp:
            assert len(mcp.tools()) == 2

    def test_the_session_id_is_echoed_on_every_later_request(self) -> None:
        # A server issues it on the first answer and expects it back; missing it
        # turns turn two into a new session.
        server = HttpServer()
        with http_connect(server) as mcp:
            mcp.list_tools()
        later = [r for r in server.requests if r["headers"].get("mcp-session-id")]
        assert later, "no request carried the session id"
        assert all(r["headers"]["mcp-session-id"] == "sess-1" for r in later)

    def test_the_first_request_cannot_carry_a_session_it_does_not_have(self) -> None:
        server = HttpServer()
        with http_connect(server):
            pass
        assert "mcp-session-id" not in server.requests[0]["headers"]

    def test_the_declared_protocol_version_reaches_the_header(self) -> None:
        server = HttpServer()
        with http_connect(server) as mcp:
            mcp.list_tools()
        listed = [r for r in server.requests if (r.get("body") or {}).get("method") == "tools/list"]
        assert listed[0]["headers"]["mcp-protocol-version"] == "2025-11-25"

    def test_configured_headers_reach_the_server(self) -> None:
        server = HttpServer()
        with http_connect(server, headers={"x-tenant": "acme"}):
            pass
        assert server.requests[0]["headers"]["x-tenant"] == "acme"

    def test_a_401_is_retried_once_after_re_auth(self) -> None:
        server = HttpServer()
        server.refuse_next = 1
        attempts: list[int] = []

        def re_authed() -> bool:
            attempts.append(1)
            return True

        transport = HttpTransport(
            "https://mcp.example.com/mcp", fetch=server, on_unauthorized=re_authed
        )
        result = transport.request("initialize", {"protocolVersion": "2025-11-25"})
        assert attempts == [1]
        assert result["serverInfo"]["name"] == "http-fixture"

    def test_a_401_that_cannot_be_re_authed_is_reported(self) -> None:
        server = HttpServer()
        server.refuse_next = 99
        transport = HttpTransport("https://mcp.example.com/mcp", fetch=server)
        with pytest.raises(McpError, match="401"):
            transport.request("initialize")

    def test_a_4xx_carrying_a_json_rpc_error_keeps_it(self) -> None:
        # For negotiation the body is the whole point: -32022 is what names the
        # versions a server speaks. Collapsing every 4xx into "connection
        # closed" makes a modern-only server look like a dead socket.
        def refusing(request: Any, options: Any = None) -> dict[str, Any]:
            return {
                "status": 400,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request["body"]["id"],
                        "error": {
                            "code": McpErrorCode.UNSUPPORTED_PROTOCOL_VERSION,
                            "message": "no",
                            "data": {"supported": ["2026-07-28"]},
                        },
                    }
                ),
            }

        transport = HttpTransport("https://mcp.example.com/mcp", fetch=refusing)
        with pytest.raises(McpError) as caught:
            transport.request("server/discover")
        assert caught.value.code == McpErrorCode.UNSUPPORTED_PROTOCOL_VERSION
        assert caught.value.data == {"supported": ["2026-07-28"]}

    def test_a_4xx_with_no_body_is_still_an_error_naming_the_method(self) -> None:
        def refusing(request: Any, options: Any = None) -> dict[str, Any]:
            return {"status": 503, "headers": {}, "body": ""}

        transport = HttpTransport("https://mcp.example.com/mcp", fetch=refusing)
        with pytest.raises(McpError, match="503 for 'tools/list'"):
            transport.request("tools/list")

    def test_an_answer_for_another_id_is_not_mistaken_for_ours(self) -> None:
        # A server may interleave, so the id is the correlation and position is
        # a coincidence.
        def mismatched(request: Any, options: Any = None) -> dict[str, Any]:
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps({"jsonrpc": "2.0", "id": 999, "result": {"not": "ours"}}),
            }

        transport = HttpTransport("https://mcp.example.com/mcp", fetch=mismatched)
        with pytest.raises(McpError, match="no JSON-RPC response"):
            transport.request("tools/list")

    def test_the_modern_wire_adds_the_routing_headers(self) -> None:
        # A modern server rejects a request whose Mcp-Method disagrees with its
        # body, including a missing one.
        seen: list[Any] = []

        def recording(request: Any, options: Any = None) -> dict[str, Any]:
            seen.append(request)
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {"jsonrpc": "2.0", "id": request["body"]["id"], "result": {}}
                ),
            }

        transport = HttpTransport("https://mcp.example.com/mcp", fetch=recording)
        transport.set_era("modern")
        transport.request("tools/call", {"name": "search"})
        assert seen[0]["headers"]["mcp-method"] == "tools/call"
        assert seen[0]["headers"]["mcp-name"] == "search"

    def test_the_handshake_wire_adds_neither(self) -> None:
        # An older server has every right to reject headers it never defined.
        server = HttpServer()
        with http_connect(server) as mcp:
            mcp.call_tool("t0", {})
        called = [r for r in server.requests if (r.get("body") or {}).get("method") == "tools/call"]
        assert "mcp-method" not in called[0]["headers"]
        assert "mcp-name" not in called[0]["headers"]

    def test_a_non_ascii_subject_is_encoded_for_the_header(self) -> None:
        # Header values must be ASCII; a tool name need not be, and a raw one
        # is rejected by the runtime rather than by the server.
        assert encode_mcp_header_value("café") == "caf%C3%A9"
        assert encode_mcp_header_value("search") == "search"

    def test_closing_tells_the_server_the_session_is_over(self) -> None:
        server = HttpServer()
        with http_connect(server):
            pass
        assert any(r["method"] == "DELETE" for r in server.requests)

    def test_closing_a_server_that_refuses_deletion_is_not_an_error(self) -> None:
        # A 405 means the server does not support session termination, which is
        # a valid server and nothing to raise about while shutting down.
        calls: list[Any] = []

        def refusing_delete(request: Any, options: Any = None) -> dict[str, Any]:
            calls.append(request)
            if request["method"] == "DELETE":
                raise RuntimeError("405 Method Not Allowed")
            return HttpServer()(request, options)

        transport = HttpTransport("https://mcp.example.com/mcp", fetch=refusing_delete)
        transport.request("initialize")
        transport.close()
        assert any(r["method"] == "DELETE" for r in calls), "close must still try"


class TestWhatALiveServerFound:
    """Two bugs a real Streamable HTTP server found and no stub had."""

    def test_the_wire_timeout_is_milliseconds(self) -> None:
        # SECONDS on the public surface, MILLISECONDS on the wire. Passing the
        # seconds straight through made `timeout=120` mean 120ms, so every
        # request to a real server timed out before it finished connecting.
        seen: list[Any] = []

        def recording(request: Any, options: Any = None) -> dict[str, Any]:
            seen.append(request)
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {"jsonrpc": "2.0", "id": request["body"]["id"], "result": {}}
                ),
            }

        HttpTransport("https://x/mcp", fetch=recording, timeout=120).request("ping")
        assert seen[0]["timeout"] == 120_000

    def test_a_4xx_the_executor_classified_is_still_read(self) -> None:
        # The executor raises on a 4xx, which is right for a completion and
        # wrong here: an MCP 4xx MAY carry the JSON-RPC error that names which
        # protocol versions the server speaks. DeepWiki answers the 2026 probe
        # exactly this way, and reading it is what lets the handshake fall back
        # instead of the connection failing outright.
        refusal = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "server-error",
                "error": {"code": -32600, "message": "Unsupported protocol version"},
            }
        )

        def classifying(request: Any, options: Any = None) -> dict[str, Any]:
            raise LLMError(refusal, status=400, raw={"status": 400, "body": refusal})

        transport = HttpTransport("https://x/mcp", fetch=classifying)
        with pytest.raises(McpError, match="Unsupported protocol version"):
            transport.request("server/discover")

    def test_an_error_under_another_id_is_still_the_answer(self) -> None:
        # A server rejecting the request outright answers with its OWN id --
        # DeepWiki says `"server-error"` -- and refusing to read that throws
        # away the one thing it told us.
        body = json.dumps(
            {"jsonrpc": "2.0", "id": "server-error", "error": {"code": -32600, "message": "no"}}
        )

        def refusing(request: Any, options: Any = None) -> dict[str, Any]:
            return {"status": 400, "headers": {"content-type": "application/json"}, "body": body}

        transport = HttpTransport("https://x/mcp", fetch=refusing)
        with pytest.raises(McpError) as caught:
            transport.request("server/discover")
        assert caught.value.code == -32600

    def test_a_refusal_falls_back_to_the_handshake_rather_than_failing(self) -> None:
        # The whole point of reading that body: the connection survives.
        class Fussy(HttpServer):
            def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
                body = request.get("body") or {}
                if body.get("method") == "server/discover":
                    return {
                        "status": 400,
                        "headers": {"content-type": "application/json"},
                        "body": json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": "server-error",
                                "error": {"code": -32600, "message": "Unsupported version"},
                            }
                        ),
                    }
                return super().__call__(request, options)

        server = Fussy()
        with connect_mcp(
            url="https://mcp.example.com/mcp",
            engine=SimpleNamespace(fetch=server, fetch_stream=None),
        ) as mcp:
            assert mcp.client.era == "handshake"
            assert len(mcp.tools()) == 2

    def test_a_transport_failure_with_no_status_is_still_a_failure(self) -> None:
        # A dead socket or a DNS failure is not a protocol answer, and reading
        # it as one would turn an outage into a silent downgrade.
        def dead(request: Any, options: Any = None) -> dict[str, Any]:
            raise LLMError("connection reset")

        transport = HttpTransport("https://x/mcp", fetch=dead)
        with pytest.raises(LLMError, match="connection reset"):
            transport.request("ping")

    def test_the_negotiated_version_reaches_the_transport(self) -> None:
        # Bookkeeping over stdio, which has no headers -- and load-bearing over
        # HTTP, where a modern server ROUTES on `Mcp-Protocol-Version`.
        server = HttpServer()
        with http_connect(server) as mcp:
            mcp.list_tools()
        transport = mcp.client.transport
        assert isinstance(transport, HttpTransport)
        listed = [r for r in server.requests if (r.get("body") or {}).get("method") == "tools/list"]
        assert listed[0]["headers"]["mcp-protocol-version"] == "2025-11-25"


class TestVersionHeaders:
    """Which version each request DECLARES -- and it is not always ours.

    Invisible over stdio, which has no headers. Over HTTP a modern server routes
    on `Mcp-Protocol-Version`, so a request that declares the wrong one lands on
    the wrong handler and is rejected for a reason that has nothing to do with
    what it asked.
    """

    def headers_for(self, server: Any, method: str) -> dict[str, str]:
        matching = [
            r for r in server.requests if (r.get("body") or {}).get("method") == method
        ]
        assert matching, f"no {method} request was sent"
        headers: dict[str, str] = matching[0]["headers"]
        return headers

    def test_the_probe_declares_the_modern_version(self) -> None:
        # Told to the transport BEFORE the probe goes out: a modern server routes
        # by this header, so a probe without it lands on the legacy handler and
        # is rejected outright instead of negotiating.
        server = HttpServer()
        with connect_mcp(
            url="https://mcp.example.com/mcp",
            engine=SimpleNamespace(fetch=server, fetch_stream=None),
        ):
            pass
        headers = self.headers_for(server, "server/discover")
        assert headers["mcp-protocol-version"] == "2026-07-28"

    def test_a_failed_probe_does_not_leave_its_version_behind(self) -> None:
        # The `initialize` that follows must declare the HANDSHAKE version --
        # otherwise it is sent to the very handler that just refused us.
        class Refusing(HttpServer):
            def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
                body = request.get("body") or {}
                if body.get("method") == "server/discover":
                    self.requests.append(request)
                    return {
                        "status": 400,
                        "headers": {"content-type": "application/json"},
                        "body": json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": "server-error",
                                "error": {"code": -32600, "message": "Unsupported version"},
                            }
                        ),
                    }
                return super().__call__(request, options)

        server = Refusing()
        with connect_mcp(
            url="https://mcp.example.com/mcp",
            engine=SimpleNamespace(fetch=server, fetch_stream=None),
        ) as mcp:
            assert mcp.client.era == "handshake"
        headers = self.headers_for(server, "initialize")
        assert headers["mcp-protocol-version"] == "2025-11-25"

    def test_the_header_is_the_version_the_server_chose(self) -> None:
        # Not the one we offered. A server may answer `initialize` with an older
        # revision, and every later request has to declare what it actually got.
        class Older(HttpServer):
            def _result(self, method: Any, body: Any) -> Any:
                if method == "initialize":
                    return {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "older", "version": "1"},
                    }
                return super()._result(method, body)

        server = Older()
        with http_connect(server) as mcp:
            mcp.list_tools()
        headers = self.headers_for(server, "tools/list")
        assert headers["mcp-protocol-version"] == "2025-06-18"

    def test_the_probe_is_modern_by_its_declared_version_alone(self) -> None:
        # The era is set only AFTER discovery succeeds, so keying the modern
        # rules on era alone leaves the probe itself half-modern -- which a
        # modern server rejects.
        seen: list[Any] = []

        def recording(request: Any, options: Any = None) -> dict[str, Any]:
            seen.append(request)
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {"jsonrpc": "2.0", "id": request["body"]["id"], "result": {}}
                ),
            }

        transport = HttpTransport("https://x/mcp", fetch=recording)
        transport.set_protocol_version("2026-07-28")
        transport.request("tools/call", {"name": "search"})
        # The era was never set to modern, so only the declared version can have
        # produced this header.
        assert seen[0]["headers"]["mcp-method"] == "tools/call"


class TestTheNameHeader:
    @staticmethod
    def header(method: str, params: dict[str, Any]) -> str | None:
        seen: list[Any] = []

        def recording(request: Any, options: Any = None) -> dict[str, Any]:
            seen.append(request)
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": json.dumps(
                    {"jsonrpc": "2.0", "id": request["body"]["id"], "result": {}}
                ),
            }

        transport = HttpTransport("https://x/mcp", fetch=recording)
        transport.set_era("modern")
        transport.request(method, params)
        header: str | None = seen[0]["headers"].get("mcp-name")
        return header

    def test_each_method_names_its_own_subject(self) -> None:
        # The subject lives under a different parameter per method, so a fixed
        # key would send a tool name where a URI belongs.
        assert self.header("tools/call", {"name": "search"}) == "search"
        assert self.header("prompts/get", {"name": "greet"}) == "greet"
        assert self.header("resources/read", {"uri": "mem://x"}) == "mem://x"

    def test_a_method_with_no_subject_sends_no_header(self) -> None:
        assert self.header("tools/list", {}) is None

    def test_a_subject_under_the_wrong_key_sends_no_header(self) -> None:
        # `resources/read` carries a `uri`, not a `name`; reading the wrong key
        # would put a tool name where a URI belongs.
        assert self.header("resources/read", {"name": "not-a-uri"}) is None


class TestReadingAnHttpBody:
    def test_a_media_type_is_compared_not_searched_for(self) -> None:
        # A substring test misroutes anything that merely CONTAINS the token.
        assert media_type_essence("text/event-stream; charset=utf-8") == "text/event-stream"
        assert media_type_essence("application/json") == "application/json"
        assert media_type_essence("  APPLICATION/JSON ; x=1") == "application/json"

    def test_a_message_split_across_data_lines_is_one_message(self) -> None:
        frames = sse_messages('data: {"a":\ndata: 1}\n\n')
        assert frames == [{"a": 1}]

    def test_a_keep_alive_frame_is_skipped(self) -> None:
        frames = sse_messages(': keep-alive\n\ndata: {"a": 1}\n\n')
        assert frames == [{"a": 1}]

    def test_a_batched_json_body_is_searched_by_id(self) -> None:
        body = json.dumps([{"id": 1, "result": "no"}, {"id": 2, "result": "yes"}])
        assert pick_response("application/json", body, 2) == {"id": 2, "result": "yes"}

    def test_an_empty_body_is_no_answer_rather_than_a_crash(self) -> None:
        assert pick_response("application/json", "   ", 1) is None

    def test_a_body_that_is_not_json_is_no_answer(self) -> None:
        assert pick_response("application/json", "<html>nope</html>", 1) is None

    def test_several_errors_under_no_matching_id_are_not_guessed_between(self) -> None:
        # ONE error under an id we did not send is the server answering us.
        # SEVERAL is a batch we cannot attribute, and picking the first would be
        # reporting one request's failure as another's.
        one = json.dumps({"id": "server-error", "error": {"code": -1, "message": "a"}})
        assert _sole_error("application/json", one) is not None

        several = (
            'data: {"id": "e1", "error": {"code": -1, "message": "a"}}\n\n'
            'data: {"id": "e2", "error": {"code": -2, "message": "b"}}\n\n'
        )
        assert _sole_error("text/event-stream", several) is None


@pytest.fixture
def ws_server() -> Iterator[Callable[..., int]]:
    """A real MCP server on a real socket, torn down with the test."""
    stoppers: list[Callable[[], None]] = []

    def start(mode: str = "normal", events: list[str] | None = None) -> int:
        port, stop = serve_ws(mode, events)
        stoppers.append(stop)
        return port

    yield start
    for stop in stoppers:
        stop()


def ws_connect(port: int, **kwargs: Any) -> McpConnection:
    kwargs.setdefault("timeout", 20)
    return connect_mcp(url=f"ws://127.0.0.1:{port}", **kwargs)


class TestWebSocketTransport:
    """Against a socket that is genuinely a socket.

    The fixture speaks RFC 6455 on the stdlib alone rather than pulling in a
    server library: the point is to exercise OUR client, and a framework in the
    middle would make the test about the framework.
    """

    def test_it_speaks_the_protocol_over_a_socket(self, ws_server: Any) -> None:
        with ws_connect(ws_server()) as mcp:
            assert mcp.server_info is not None
            assert mcp.server_info["serverInfo"]["name"] == "ws-fixture"
            assert len(mcp.tools()) == 2

    def test_a_tool_runs_over_the_socket(self, ws_server: Any) -> None:
        with ws_connect(ws_server()) as mcp:
            add = next(t for t in mcp.tools() if t.name.endswith("__add"))
            assert add(a=2, b=3) == "5"

    def test_the_server_can_speak_first(self, ws_server: Any) -> None:
        # What a socket buys over Streamable HTTP: the server->client direction
        # is the same channel, with no second request held open to carry it.
        seen: list[str] = []
        with ws_connect(ws_server(), on_notification=lambda m, _p: seen.append(m)) as mcp:
            mcp.client.request("fixture/push")
            mcp.client.request("ping")  # a round trip, so the push has landed
        assert seen == ["notifications/message"]

    def test_a_server_request_is_answered_over_the_same_socket(self, ws_server: Any) -> None:
        answered: list[str] = []

        def on_request(method: str, params: Any) -> Any:
            answered.append(method)
            return {"roots": []}

        with ws_connect(ws_server(), on_server_request=on_request) as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")
            replies = mcp.client.request("fixture/replies")["replies"]
        assert answered == ["roots/list"]
        assert [r["ok"] for r in replies] == [True]

    def test_an_unanswerable_server_request_still_gets_a_reply(self, ws_server: Any) -> None:
        # A dropped request leaves the SERVER blocked forever.
        with ws_connect(ws_server()) as mcp:
            mcp.client.request("fixture/ask")
            mcp.client.request("ping")
            replies = mcp.client.request("fixture/replies")["replies"]
        assert [r["ok"] for r in replies] == [False]

    def test_a_modern_server_is_adopted_over_a_socket_too(self, ws_server: Any) -> None:
        with ws_connect(ws_server("modern")) as mcp:
            assert mcp.client.era == "modern"
            assert mcp.server_info is not None
            assert mcp.server_info["serverInfo"]["name"] == "ws-fixture"

    def test_a_port_nobody_is_listening_on_fails_by_name(self) -> None:
        with pytest.raises(McpError, match="could not open"):
            connect_mcp(url="ws://127.0.0.1:1", timeout=5)

    def test_closing_ends_the_socket(self, ws_server: Any) -> None:
        # Asked of the SERVER: a client closing its socket is only observable
        # from the other end, and a transport that merely forgot its session
        # would look identical from here.
        events: list[str] = []
        mcp = ws_connect(ws_server("normal", events))
        transport = mcp.client.transport
        assert isinstance(transport, WsTransport)
        assert transport.is_open
        assert events == ["open"]
        mcp.close()
        assert not transport.is_open
        deadline = time.monotonic() + 5
        while "close" not in events and time.monotonic() < deadline:
            time.sleep(0.02)
        assert events == ["open", "close"], "the server never saw the socket close"

    def test_a_closing_transport_reports_itself_shut(self, ws_server: Any) -> None:
        # `close()` marks itself closed BEFORE it lets the session go, so there
        # is a window where the session is still set. A caller asking during it
        # must be told the transport is going down, not that it is usable.
        mcp = ws_connect(ws_server())
        transport = mcp.client.transport
        assert isinstance(transport, WsTransport)
        transport._closed = True
        assert not transport.is_open
        mcp.close()

    def test_a_request_after_close_is_refused_rather_than_hanging(self, ws_server: Any) -> None:
        mcp = ws_connect(ws_server())
        mcp.close()
        with pytest.raises(McpError, match="not started"):
            mcp.client.request("ping")

    def test_a_request_already_in_flight_wakes_when_the_socket_dies(
        self, ws_server: Any
    ) -> None:
        # The case the reader thread exists to cover. Otherwise the caller waits
        # out its full timeout and is told "timed out", which sends whoever
        # reads it looking for a slow server rather than a vanished one.
        mcp = ws_connect(ws_server(), timeout=30)
        transport = mcp.client.transport
        assert isinstance(transport, WsTransport)
        failure: list[BaseException] = []

        def ask() -> None:
            try:
                # The fixture receives this and deliberately says nothing, so
                # the request is still in flight when the socket goes.
                mcp.client.request("fixture/silence")
            except BaseException as exc:  # noqa: BLE001 -- recorded for the assertion
                failure.append(exc)

        caller = threading.Thread(target=ask)
        caller.start()
        time.sleep(0.2)  # long enough for the frame to be on the wire
        session = transport._session
        assert session is not None
        session.close()
        caller.join(timeout=10)
        assert not caller.is_alive(), "the waiter was never woken"
        assert isinstance(failure[0], McpError)
        assert failure[0].code == McpErrorCode.CONNECTION_CLOSED
        mcp.close()

    def test_a_request_after_the_socket_died_is_refused(self, ws_server: Any) -> None:
        mcp = ws_connect(ws_server())
        transport = mcp.client.transport
        assert isinstance(transport, WsTransport)
        session = transport._session
        assert session is not None
        session.close()
        with pytest.raises(McpError):
            mcp.client.request("ping")
        mcp.close()

    def test_a_frame_that_is_not_json_does_not_kill_the_connection(self) -> None:
        # One bad frame is not a dead socket, for the same reason a banner on
        # stdout is not.
        transport = WsTransport("ws://x", connect=lambda *a, **k: None)
        transport._route_text("not json at all")
        transport._route_text('{"jsonrpc": "2.0", "method": "notifications/x"}')

    def test_a_socket_has_no_headers_to_record(self) -> None:
        # Both are no-ops, and both must EXIST: the client calls them on every
        # transport, and a missing one would be an AttributeError mid-handshake.
        transport = WsTransport("ws://x", connect=lambda *a, **k: None)
        transport.set_protocol_version("2026-07-28")
        transport.set_era("modern")
        transport.listen()


class TestSessionLiveness:
    """`receive_text` says `TimeoutError` for both 'nothing yet' and 'gone'."""

    def test_an_open_session_is_live(self) -> None:
        assert session_is_live(SimpleNamespace(connection=SimpleNamespace(
            state=SimpleNamespace(name="OPEN")
        )))

    def test_a_closing_session_is_not(self) -> None:
        for state in ("LOCAL_CLOSING", "REMOTE_CLOSING", "CLOSED", "REJECTING"):
            assert not session_is_live(SimpleNamespace(connection=SimpleNamespace(
                state=SimpleNamespace(name=state)
            ))), state

    def test_a_session_that_cannot_say_is_assumed_live(self) -> None:
        # A caller's own client need not expose wsproto's shape. Guessing "dead"
        # would tear down a working connection on the first quiet poll, so the
        # only safe default is to believe it is up.
        assert session_is_live(SimpleNamespace())
        assert session_is_live(SimpleNamespace(connection=SimpleNamespace()))
        assert session_is_live(object())


class TestUrlNamespaces:
    def test_a_host_names_the_server(self) -> None:
        assert _default_url_namespace("https://mcp.deepwiki.com/mcp") == "deepwiki"
        assert _default_url_namespace("https://tickets.example.com/") == "tickets"

    def test_an_address_is_not_a_name(self) -> None:
        # `127.0.0.1` would otherwise namespace every local server as `127`,
        # which is meaningless and identical for all of them.
        assert _default_url_namespace("ws://127.0.0.1:9000") == "mcp"
        assert _default_url_namespace("ws://[::1]:9000") == "mcp"
        assert _default_url_namespace("ws://localhost:9000") == "mcp"
