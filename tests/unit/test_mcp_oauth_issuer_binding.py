"""Stored MCP OAuth credentials are bound to the authorization server that
issued them, and a refresh names the resource it is for.

Transposed from
`unified-library-ts/tests/unit/plugins/mcp/oauth-issuer-binding.test.ts`.

What this defends: the MCP server tells the client where to authorize. Store a
registration or a token without recording WHICH server it came from, and a server
that later points somewhere else is handed credentials minted for somebody else
-- quietly, and by us. Both official MCP SDKs bind stored credentials for exactly
this reason (mcp-py cites SEP-2352).

Two failure modes, two different answers, and the difference is deliberate:

    a client REGISTRATION bound elsewhere raises. Re-registering silently would
    leave the caller with two registrations and no idea the server moved, and
    presenting the old one is the attack itself.

    TOKENS bound elsewhere are treated as absent. They are disposable, so the
    honest recovery is to authorize again rather than to fail.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl

import pytest

from combycode_llm_sdk.mcp.oauth import (
    McpOAuth,
    McpOAuthClientInfo,
    McpOAuthClientMetadata,
    McpOAuthTokens,
    issuers_match,
)

SERVER = "https://mcp.example.com/mcp"
ORIGIN = "https://mcp.example.com"
OTHER = "https://evil.example.net"


class Server:
    """A scripted authorization server at the fetch boundary."""

    def __init__(self, *, rotate_refresh: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self.rotate_refresh = rotate_refresh

    def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
        self.requests.append(request)
        url = request["url"]
        if "/.well-known" in url:
            return {
                "status": 200,
                "body": {
                    "authorization_endpoint": f"{ORIGIN}/authorize",
                    "token_endpoint": f"{ORIGIN}/token",
                    "registration_endpoint": f"{ORIGIN}/register",
                    "issuer": ORIGIN,
                },
            }
        if url.endswith("/register"):
            return {"status": 200, "body": {"client_id": "dcr-id"}}
        if url.endswith("/token"):
            body: dict[str, Any] = {"access_token": "fresh", "expires_in": 3600}
            if self.rotate_refresh:
                body["refresh_token"] = "rt-new"
            return {"status": 200, "body": body}
        raise AssertionError(f"unexpected url: {url}")

    def form_of(self, suffix: str) -> dict[str, str]:
        matching = [r for r in self.requests if r["url"].endswith(suffix)]
        assert matching, f"no request to {suffix}"
        return dict(parse_qsl(matching[-1]["body"]))


class Provider:
    def __init__(
        self,
        *,
        tokens: McpOAuthTokens | None = None,
        client: McpOAuthClientInfo | None = None,
    ) -> None:
        self.redirect_url = "http://127.0.0.1:8765/callback"
        self.client_metadata = McpOAuthClientMetadata(
            redirect_uris=["http://127.0.0.1:8765/callback"], client_name="test"
        )
        self._client = client
        self._tokens = tokens
        self.saved_clients: list[McpOAuthClientInfo] = []
        self.saved_tokens: list[McpOAuthTokens] = []
        self.redirected_to: list[str] = []

    def client_information(self) -> McpOAuthClientInfo | None:
        return self._client

    def save_client_information(self, info: McpOAuthClientInfo) -> None:
        self._client = info
        self.saved_clients.append(info)

    def tokens(self) -> McpOAuthTokens | None:
        return self._tokens

    def save_tokens(self, tokens: McpOAuthTokens) -> None:
        self._tokens = tokens
        self.saved_tokens.append(tokens)

    def redirect_to_authorization(self, url: str) -> None:
        self.redirected_to.append(url)

    def save_code_verifier(self, verifier: str) -> None: ...

    def code_verifier(self) -> str:
        return "seed-verifier"

    def save_state(self, state: str) -> None: ...

    def state(self) -> str | None:
        return "seed-state"


def stale(**over: Any) -> McpOAuthTokens:
    """An hour old with a one-second lifetime, so it needs refreshing."""
    return McpOAuthTokens(
        access_token="stale", obtained_at=1.0, expires_in=1, **over
    )


class TestIssuersMatch:
    """Lenient ON PURPOSE, and not the RFC 9207 check."""

    def test_it_tolerates_the_trailing_slash_a_parser_adds(self) -> None:
        # A parsed origin is slash-suffixed; an advertised issuer usually is not.
        # A strict compare here would discard a valid registration every other run.
        assert issuers_match("https://as.example.com", "https://as.example.com/")

    def test_it_ignores_spellings_a_parser_normalises(self) -> None:
        assert issuers_match("https://AS.example.com", "https://as.example.com")

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("https://as.example.com", "https://as.example.net"),
            ("https://as.example.com", "https://as.example.com/tenant"),
            ("urn:example:as", "urn:example:other"),
        ],
    )
    def test_it_still_separates_different_servers(self, a: str, b: str) -> None:
        assert not issuers_match(a, b)

    def test_it_compares_non_urls_as_written(self) -> None:
        assert issuers_match("urn:example:as", "urn:example:as")


class TestAClientRegistrationIsBound:
    def test_it_refuses_to_present_one_servers_registration_to_another(self) -> None:
        provider = Provider(client=McpOAuthClientInfo(client_id="elsewhere", issuer=OTHER))
        with pytest.raises(ValueError, match="belongs to https://evil.example.net"):
            McpOAuth(SERVER, provider, Server()).authorize()

    def test_it_uses_an_unstamped_registration_and_stamps_it(self) -> None:
        # Stored before the binding existed. It says nothing about where it came
        # from, so there is nothing to enforce -- but the NEXT run should be bound.
        provider = Provider(client=McpOAuthClientInfo(client_id="legacy"))
        McpOAuth(SERVER, provider, Server()).authorize()
        assert provider.saved_clients == [McpOAuthClientInfo(client_id="legacy", issuer=ORIGIN)]

    def test_it_accepts_its_own_stamp_without_resaving(self) -> None:
        provider = Provider(client=McpOAuthClientInfo(client_id="ours", issuer=ORIGIN))
        McpOAuth(SERVER, provider, Server()).authorize()
        assert provider.saved_clients == []

    def test_a_fresh_registration_is_stamped(self) -> None:
        provider = Provider()
        McpOAuth(SERVER, provider, Server()).authorize()
        assert provider.saved_clients == [McpOAuthClientInfo(client_id="dcr-id", issuer=ORIGIN)]


class TestTokensBoundElsewhereReadAsNone:
    def test_a_token_from_another_server_is_ignored(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=McpOAuthTokens(access_token="at", expires_in=3600, issuer=OTHER),
        )
        # Not "authorized": the stored token is not ours to use, so the flow
        # starts over rather than sending it.
        assert McpOAuth(SERVER, provider, Server()).authorize() == "redirect"

    def test_a_token_stamped_with_this_server_is_used(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=McpOAuthTokens(access_token="at", expires_in=3600, issuer=ORIGIN),
        )
        assert McpOAuth(SERVER, provider, Server()).authorize() == "authorized"

    def test_an_unstamped_token_is_used(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=McpOAuthTokens(access_token="at", expires_in=3600),
        )
        assert McpOAuth(SERVER, provider, Server()).authorize() == "authorized"

    def test_a_foreign_token_is_not_spent_via_refresh_either(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt", issuer=OTHER),
        )
        server = Server()
        McpOAuth(SERVER, provider, server).authorize()
        assert not [r for r in server.requests if r["url"].endswith("/token")]


class TestARefreshNamesItsResource:
    def test_it_sends_resource_which_only_the_exchange_used_to(self) -> None:
        # RFC 8707. Without it an authorization server that scopes tokens per
        # resource hands back one scoped to nothing, and the retry 401s with a
        # token that looks perfectly valid.
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt", issuer=ORIGIN),
        )
        server = Server()
        assert McpOAuth(SERVER, provider, server).authorize() == "authorized"
        form = server.form_of("/token")
        assert form["grant_type"] == "refresh_token"
        # The whole server URL, verbatim -- not its origin, and with no trailing
        # slash added: an exact-match authorization server rejects one that gained
        # one.
        assert form["resource"] == SERVER

    def test_it_stamps_the_refreshed_tokens(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt", issuer=ORIGIN),
        )
        McpOAuth(SERVER, provider, Server()).authorize()
        assert provider.saved_tokens[-1].issuer == ORIGIN

    def test_it_keeps_a_refresh_token_the_server_did_not_replace(self) -> None:
        # RFC 6749 section 6: the server MAY issue a new refresh token, and most
        # do not. Dropping the old one here makes the next refresh impossible and
        # sends a headless client to an interactive flow it cannot complete.
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt", issuer=ORIGIN),
        )
        McpOAuth(SERVER, provider, Server()).authorize()
        assert provider.saved_tokens[-1].refresh_token == "rt"

    def test_it_takes_a_rotated_refresh_token(self) -> None:
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt-old", issuer=ORIGIN),
        )
        McpOAuth(SERVER, provider, Server(rotate_refresh=True)).authorize()
        assert provider.saved_tokens[-1].refresh_token == "rt-new"


class TestTheResourceIndicatorGoesOutAsGiven:
    def test_a_pathless_server_url_gains_no_trailing_slash(self) -> None:
        # The case that bites: a parsed pathless URL is `https://host/`, and an
        # authorization server that exact-matches its resource indicators rejects
        # the slashed form. Nothing here parses it -- the string the caller gave
        # is the string that is sent.
        provider = Provider(
            client=McpOAuthClientInfo(client_id="cid", issuer=ORIGIN),
            tokens=stale(refresh_token="rt", issuer=ORIGIN),
        )
        server = Server()
        McpOAuth(ORIGIN, provider, server).authorize()
        assert server.form_of("/token")["resource"] == ORIGIN
