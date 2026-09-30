"""A session the server has forgotten is rebuilt, not reported as dead.

Transposed from
`unified-library-ts/tests/unit/plugins/mcp/session-recovery.test.ts`.

A stateful MCP server answers 404 to a session id it no longer holds -- it
restarted, evicted the session, or let it expire. We turned that into
``ConnectionClosed``, and since the id is held for the life of the transport,
every later request failed exactly the same way. The connection was fine; only
the session was gone, and nothing tried to get a new one.

Three conditions before recovering, each there to avoid making things worse:

* a session id must be HELD. A 404 without one is an ordinary wrong URL, and
  re-initializing against it turns one clear error into two confusing ones.
* the id is dropped BEFORE re-initializing, so the new handshake does not
  present the dead one.
* it does not recurse. Re-initializing goes back through this transport, and a
  404 on that must not try to recover again.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, "src")


from combycode_llm_sdk.mcp.client import DEFAULT_CLIENT_INFO, McpClient, McpClientOptions
from combycode_llm_sdk.mcp.errors import McpError
from combycode_llm_sdk.mcp.http import HttpTransport
from combycode_llm_sdk.version import SDK_VERSION


class _Scripted:
    """A transport whose fetch answers from a script, recording what it was asked."""

    def __init__(
        self,
        steps: list[tuple[int, str | None]],
        recover: Any = None,
    ) -> None:
        self.sent: list[str | None] = []
        self._steps = steps
        self._n = 0
        self.transport = HttpTransport(url="https://a.test/mcp", name="s", fetch=self._fetch)
        if recover is not None:
            self.transport.set_on_session_lost(recover)

    def _fetch(self, req: Mapping[str, Any]) -> dict[str, Any]:
        headers_in = req.get("headers") or {}
        self.sent.append(headers_in.get("mcp-session-id"))
        status, session = self._steps[min(self._n, len(self._steps) - 1)]
        self._n += 1
        headers = {"content-type": "application/json"}
        if session:
            headers["mcp-session-id"] = session
        body = req.get("body") or {}
        return {
            "status": status,
            "headers": headers,
            "body": {"jsonrpc": "2.0", "id": body.get("id", 0), "result": {"ok": True}},
        }

    def establish(self) -> None:
        """Give the transport a session id the way a real server does."""
        self.transport.request("tools/list")


class TestA404WhileHoldingASessionId:
    def test_it_reinitializes_and_replays_the_request(self) -> None:
        recovered = {"n": 0}

        def recover() -> bool:
            recovered["n"] += 1
            return True

        s = _Scripted(
            [
                (200, "sess-1"),  # first call establishes the session
                (404, None),  # the server forgot it
                (200, "sess-2"),  # the replay, on a new session
            ],
            recover,
        )
        s.establish()
        s.transport.request("tools/list")

        assert recovered["n"] == 1
        # Three calls: the original, the one that 404'd, and the replay.
        assert len(s.sent) == 3
        # The replay does NOT carry the dead id -- dropped before recovering.
        assert s.sent[2] is None

    def test_it_does_not_recover_when_no_session_is_held(self) -> None:
        # An ordinary 404: wrong URL. Re-initializing would add a second failure
        # to a perfectly clear first one.
        recovered = {"n": 0}

        def recover() -> bool:
            recovered["n"] += 1
            return True

        s = _Scripted([(404, None)], recover)
        with contextlib.suppress(McpError):
            s.transport.request("tools/list")
        assert recovered["n"] == 0
        assert len(s.sent) == 1

    def test_it_surfaces_the_original_error_when_recovery_fails(self) -> None:
        # The 404 is the better error to report; a failed recovery must not
        # replace it with its own.
        s = _Scripted([(200, "sess-1"), (404, None)], lambda: False)
        s.establish()
        with contextlib.suppress(McpError):
            s.transport.request("tools/list")
        assert len(s.sent) == 2

    def test_it_recovers_at_most_once_per_request(self) -> None:
        recovered = {"n": 0}

        def recover() -> bool:
            recovered["n"] += 1
            return True

        s = _Scripted([(200, "sess-1"), (404, None), (404, None)], recover)
        s.establish()
        with contextlib.suppress(McpError):
            s.transport.request("tools/list")
        assert recovered["n"] == 1

    def test_it_does_nothing_when_no_recovery_was_installed(self) -> None:
        # stdio has no sessions; a transport without the hook behaves as before.
        s = _Scripted([(200, "sess-1"), (404, None)])
        s.establish()
        with contextlib.suppress(McpError):
            s.transport.request("tools/list")
        assert len(s.sent) == 2


class TestRecoveryDoesNotRecurse:
    def test_a_404_during_the_reinitialize_does_not_start_another(self) -> None:
        # Re-initializing goes back through THIS transport. A server that
        # answers that call with a 404 -- and hands out a session header on the
        # error, which is enough to re-arm the "a session is held" condition --
        # would otherwise put the transport in a loop it never leaves.
        recovered = {"n": 0}
        box: dict[str, Any] = {}

        def recover() -> bool:
            recovered["n"] += 1
            # What the client's _recover_session does: re-run the handshake
            # over the same transport.
            with contextlib.suppress(McpError):
                box["t"].request("initialize")
            return False

        s = _Scripted(
            [
                (200, "sess-1"),  # establish
                (404, None),  # the server forgot it
                (404, "sess-2"),  # the re-initialize fails, WITH an id
            ],
            recover,
        )
        box["t"] = s.transport
        s.establish()
        with contextlib.suppress(McpError):
            s.transport.request("tools/list")

        assert recovered["n"] == 1
        assert len(s.sent) == 3


class TestTheVersionWeTellAServer:
    def test_it_is_the_library_version_not_a_placeholder(self) -> None:
        # Was `"0"`, hard-coded, for every release -- so every MCP server this
        # library ever spoke to was told the wrong client version.
        assert DEFAULT_CLIENT_INFO["version"] == SDK_VERSION
        assert DEFAULT_CLIENT_INFO["version"] != "0"

    def test_the_handshake_sends_it(self) -> None:
        sent: list[Mapping[str, Any]] = []

        def fetch(req: Mapping[str, Any]) -> dict[str, Any]:
            body = req.get("body") or {}
            params = body.get("params")
            if isinstance(params, Mapping):
                sent.append(params)
            return {
                "status": 200,
                "headers": {"content-type": "application/json"},
                "body": {
                    "jsonrpc": "2.0",
                    "id": body.get("id", 0),
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "s", "version": "1"},
                    },
                },
            }

        transport = HttpTransport(url="https://a.test/mcp", name="s", fetch=fetch)
        client = McpClient(transport, McpClientOptions(protocol_mode="legacy"))
        with contextlib.suppress(McpError):
            client.connect()

        init = next(p for p in sent if "clientInfo" in p)
        assert init["clientInfo"]["version"] == SDK_VERSION


class TestTheClientOnlyRebuildsWhatASessionIs:
    """A client over a transport that records recovery, with no real server."""

    def build(self, era: str) -> tuple[Any, dict[str, int]]:
        state = {"initializes": 0}
        box: dict[str, Any] = {}

        class _T:
            def start(self) -> None: ...
            def close(self) -> None: ...
            def set_handlers(self, handlers: Any) -> None: ...
            def set_protocol_version(self, version: str) -> None: ...
            def set_era(self, era: str) -> None: ...

            def set_on_session_lost(self, recover: Any) -> None:
                box["recover"] = recover

            def request(self, method: str, params: Any = None) -> Any:
                if method == "initialize":
                    state["initializes"] += 1
                    return {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "s", "version": "1"},
                    }
                if method == "server/discover":
                    return {
                        "protocolVersion": "2026-07-28",
                        "capabilities": {},
                        "serverInfo": {"name": "s", "version": "1"},
                    }
                return {}

            def notify(self, method: str, params: Any = None) -> None: ...

        client = McpClient(
            _T(),
            McpClientOptions(protocol_mode="2026-07-28" if era == "modern" else "legacy"),
        )
        client.connect()
        return box["recover"], state

    def test_it_reruns_the_handshake_and_nothing_more(self) -> None:
        # Not the FULL negotiation: the version question is already settled
        # with this server, and re-probing it would ask again on every dropped
        # session.
        recover, state = self.build("handshake")
        assert state["initializes"] == 1
        assert recover() is True
        assert state["initializes"] == 2

    def test_it_declines_in_the_modern_era(self) -> None:
        # There is no session to rebuild there.
        recover, state = self.build("modern")
        assert recover() is False
        assert state["initializes"] == 0
