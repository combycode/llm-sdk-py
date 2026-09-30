"""A redirect is a credential-leak primitive, so MCP does not follow one blindly.

Transposed from
`unified-library-ts/tests/unit/network/same-origin-redirect.test.ts`.

Everything on an MCP request was configured for ONE endpoint: the bearer token,
the session header, the JSON-RPC body. A 301, 302 or 303 additionally turns the
POST into a body-less GET, so even a same-origin redirect silently drops the
message and the server answers a question nobody asked.

One divergence worth stating, because it runs the OTHER way from the TypeScript:
httpx does not follow redirects by default, so this side never had the leak. What
it also never had was the PERMITTED follow -- a legitimate same-origin 307, a
trailing-slash normalisation, simply failed. So the same rule fixes opposite
symptoms on the two ports and leaves them behaving identically.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")


import pytest

from combycode_llm_sdk.transport import TransportRequest
from combycode_llm_sdk.util.http import follow_same_origin

FROM = "https://a.test/mcp"


def at(status: int, location: str | None, method: str = "POST", from_url: str = FROM) -> str | None:
    return follow_same_origin(from_url, method, status, location)


class TestTheRuleItself:
    def test_it_follows_a_method_preserving_redirect_within_the_origin(self) -> None:
        assert at(307, "https://a.test/mcp/") == "https://a.test/mcp/"
        assert at(308, "/mcp/v2") == "https://a.test/mcp/v2"

    @pytest.mark.parametrize("status", [301, 302, 303])
    def test_it_refuses_those_that_would_become_a_body_less_get(self, status: int) -> None:
        # The message IS the body. A GET to the same URL is not a smaller version
        # of the request; it is a different request that loses the JSON-RPC call.
        assert at(status, "https://a.test/mcp/") is None

    @pytest.mark.parametrize("status", [301, 302, 303])
    def test_but_allows_them_for_a_get(self, status: int) -> None:
        assert at(status, "https://a.test/mcp/", "GET") == "https://a.test/mcp/"

    @pytest.mark.parametrize(
        "location",
        [
            "https://evil.test/mcp",
            "https://a.test.evil.test/mcp",
            "https://a.test:8443/mcp",
        ],
    )
    def test_it_refuses_another_origin_which_is_the_leak(self, location: str) -> None:
        assert at(307, location) is None

    def test_it_allows_the_http_to_https_upgrade_on_default_ports(self) -> None:
        # Strictly an improvement to the same host, and the one exception the
        # reference implementations make.
        assert (
            follow_same_origin("http://a.test/mcp", "POST", 307, "https://a.test/mcp")
            == "https://a.test/mcp"
        )

    def test_it_refuses_the_reverse(self) -> None:
        assert follow_same_origin("https://a.test/mcp", "POST", 307, "http://a.test/mcp") is None

    def test_it_refuses_a_location_that_introduces_credentials(self) -> None:
        # `https://x@host/` is sent as Basic auth, so this is a redirect that
        # changes who we authenticate as.
        assert at(307, "https://attacker@a.test/mcp") is None

    def test_it_keeps_userinfo_the_configured_url_already_had(self) -> None:
        assert (
            follow_same_origin("https://u:p@a.test/mcp", "POST", 307, "/mcp/")
            == "https://u:p@a.test/mcp/"
        )

    def test_it_refuses_a_redirect_with_no_location(self) -> None:
        assert at(307, None) is None
        assert at(307, "") is None

    def test_it_refuses_a_non_redirect_status_outright(self) -> None:
        assert at(200, "https://a.test/mcp/") is None
        assert at(404, "https://a.test/mcp/") is None


class TestTheTransportFollowsOnlyWhatTheRuleAllows:
    """Driven through a stub httpx client, so the loop itself is exercised."""

    def client(self, steps: list[tuple[int, str | None]]) -> Any:
        calls: list[str] = []

        class Res:
            def __init__(self, status: int, location: str | None) -> None:
                self.status_code = status
                self.headers = {"content-type": "application/json"}
                if location:
                    self.headers["location"] = location
                self.text = "{}"

            def json(self) -> Any:
                return {}

        class Client:
            def request(self, method: str, url: str, **kwargs: Any) -> Res:
                calls.append(url)
                status, location = steps[min(len(calls) - 1, len(steps) - 1)]
                return Res(status, location)

        return Client(), calls

    def send(self, steps: list[tuple[int, str | None]], redirect: str) -> list[str]:
        from combycode_llm_sdk.transport import http_transport

        client, calls = self.client(steps)
        transport = http_transport(client)
        transport(
            TransportRequest(
                url=FROM,
                method="POST",
                headers={},
                body={"jsonrpc": "2.0", "method": "ping"},
                provider="mcp",
                model="server",
                response_type="text",
                redirect=redirect,
            )
        )
        return calls

    def test_it_follows_a_same_origin_307(self) -> None:
        calls = self.send([(307, "https://a.test/mcp/"), (200, None)], "same-origin")
        assert calls == [FROM, "https://a.test/mcp/"]

    def test_it_does_not_follow_cross_origin(self) -> None:
        # The point of the whole row: the bearer token never reaches evil.test.
        calls = self.send([(307, "https://evil.test/mcp")], "same-origin")
        assert calls == [FROM]

    def test_it_does_not_follow_a_302_on_a_post(self) -> None:
        calls = self.send([(302, "https://a.test/mcp/")], "same-origin")
        assert calls == [FROM]

    def test_it_stops_after_a_bounded_number_of_hops(self) -> None:
        calls = self.send([(307, FROM)] * 10, "same-origin")
        assert len(calls) <= 4

    def test_a_follow_request_is_left_alone(self) -> None:
        # Nothing about provider traffic changes: httpx's own default applies and
        # this loop does not run.
        calls = self.send([(307, "https://a.test/mcp/")], "follow")
        assert calls == [FROM]


class TestTheMcpPathsAskForIt:
    """The rule is worth nothing if the call sites do not reach it."""

    def test_the_transport_builds_every_request_with_it(self) -> None:
        from combycode_llm_sdk.mcp.http import HttpTransport

        transport = HttpTransport(
            url="https://a.test/mcp", name="s", fetch=lambda _req: {"status": 200, "body": {}}
        )
        req = transport._build("mcp/http.close", {})
        assert req["redirect"] == "same-origin"

    def test_so_does_the_oauth_flow(self) -> None:
        # The token request is the one carrying a client secret and a refresh
        # token; a cross-origin redirect would hand both to whoever set Location.
        from combycode_llm_sdk.mcp.oauth import _oauth_request

        req = _oauth_request(
            "mcp-oauth/token.refresh",
            {
                "tokenEndpoint": "https://as.test/token",
                "refreshToken": "r",
                "clientId": "c",
            },
        )
        assert req["redirect"] == "same-origin"
