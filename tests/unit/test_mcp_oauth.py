"""OAuth for MCP, and the guard that decides which URLs it may fetch.

The guard gets the most attention here because it is the security boundary: a
discovered endpoint is chosen by the far side, and a client that fetches one
without checking is a confused deputy inside whatever network it runs in.

Everything is exercised against a scripted authorization server rather than a
live one. That is the right call here and not a compromise: the interesting
cases are a server that lies (an endpoint pointing at the metadata service, a
code minted by somebody else, a stripped `iss`) and no cooperative real server
will produce them on demand.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from combycode_llm_sdk.mcp.oauth import (
    AuthServerMetadata,
    McpOAuth,
    McpOAuthClientInfo,
    McpOAuthClientMetadata,
    McpOAuthTokens,
    McpUnauthorizedError,
    build_authorization_url,
    discover_metadata,
    finish_mcp_auth,
    generate_pkce,
    generate_state,
    register_client,
    validate_authorization_response_iss,
)
from combycode_llm_sdk.mcp.url_guard import (
    McpSsrfError,
    SsrfGuardOptions,
    assert_safe_auth_url,
    is_blocked_host,
    parse_canonical_ipv4,
)

SERVER = "https://mcp.example.com/mcp"
ISSUER = "https://mcp.example.com"


# -- the guard ---------------------------------------------------------------


class TestCanonicalIpv4:
    """Every form the OS resolver accepts, because an attacker will use one.

    Python's own `ipaddress` REJECTS all of these -- measured, not assumed -- so
    a guard that asked it alone would read `0x7f000001` as an ordinary hostname
    and never apply the private-address check at all.
    """

    def test_a_dotted_quad_is_itself(self) -> None:
        assert parse_canonical_ipv4("127.0.0.1") == (127, 0, 0, 1)
        assert parse_canonical_ipv4("8.8.8.8") == (8, 8, 8, 8)

    def test_a_bare_integer_is_an_address(self) -> None:
        assert parse_canonical_ipv4("2130706433") == (127, 0, 0, 1)

    def test_hex_and_octal_are_addresses(self) -> None:
        assert parse_canonical_ipv4("0x7f000001") == (127, 0, 0, 1)
        assert parse_canonical_ipv4("0177.0.0.1") == (127, 0, 0, 1)
        assert parse_canonical_ipv4("0x7f.0x0.0x0.0x1") == (127, 0, 0, 1)

    def test_the_short_forms_fill_from_the_right(self) -> None:
        assert parse_canonical_ipv4("127.1") == (127, 0, 0, 1)
        assert parse_canonical_ipv4("127.0.1") == (127, 0, 0, 1)
        assert parse_canonical_ipv4("10.1") == (10, 0, 0, 1)

    def test_a_hostname_is_not_an_address(self) -> None:
        assert parse_canonical_ipv4("example.com") is None
        assert parse_canonical_ipv4("auth.example.com") is None
        assert parse_canonical_ipv4("") is None

    def test_an_out_of_range_part_is_not_an_address(self) -> None:
        assert parse_canonical_ipv4("256.0.0.1") is None
        assert parse_canonical_ipv4("1.2.3.4.5") is None
        assert parse_canonical_ipv4("999999999999") is None


class TestBlockedHosts:
    def test_loopback_by_every_spelling(self) -> None:
        for host in (
            "localhost",
            "127.0.0.1",
            "127.1",
            "0x7f000001",
            "2130706433",
            "0177.0.0.1",
            "[::1]",
            "::1",
        ):
            assert is_blocked_host(host), host

    def test_the_cloud_metadata_address(self) -> None:
        # The one that matters most: the reason SSRF is worth a guard at all.
        assert is_blocked_host("169.254.169.254")

    def test_private_and_reserved_ranges(self) -> None:
        for host in (
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.254",
            "192.168.1.1",
            "0.0.0.0",
            "100.100.1.1",  # CGNAT -- `ipaddress` does NOT flag this one
            "192.0.2.5",
            "198.51.100.5",
            "203.0.113.5",
            "198.18.0.1",
            "250.1.2.3",
            "fd12::1",
            "fe80::1",
            "fec0::1",  # site-local -- `ipaddress` does NOT flag this one either
            "ff02::1",
            "::ffff:127.0.0.1",
        ):
            assert is_blocked_host(host), host

    def test_a_public_host_is_not_blocked(self) -> None:
        # The guard has to let a real identity provider through, or it is just
        # an outage with a security rationale.
        for host in ("auth.example.com", "93.184.216.34", "2606:2800:220:1::1"):
            assert not is_blocked_host(host), host

    def test_a_hostname_is_never_resolved(self) -> None:
        # No DNS, deliberately: a name that answers differently on the second
        # lookup would make the check theatre. The origin rule is what covers a
        # hostname pointing somewhere unwelcome.
        assert not is_blocked_host("localtest.me")


class TestTheGuard:
    def test_https_to_the_same_host_is_allowed(self) -> None:
        assert_safe_auth_url("https://mcp.example.com/token", SERVER)

    def test_http_is_refused(self) -> None:
        with pytest.raises(McpSsrfError, match="only https"):
            assert_safe_auth_url("http://mcp.example.com/token", SERVER)

    def test_http_can_be_opted_into_for_local_work(self) -> None:
        assert_safe_auth_url(
            "http://mcp.example.com/token",
            SERVER,
            SsrfGuardOptions(allow_insecure_http=True),
        )

    def test_opting_into_http_opts_into_nothing_else(self) -> None:
        # `file:` and `gopher:` are exactly the schemes an SSRF wants.
        for url in ("file:///etc/passwd", "gopher://mcp.example.com/", "ftp://x/"):
            with pytest.raises(McpSsrfError, match="not allowed"):
                assert_safe_auth_url(url, SERVER, SsrfGuardOptions(allow_insecure_http=True))

    def test_the_metadata_service_is_refused(self) -> None:
        with pytest.raises(McpSsrfError, match="loopback, link-local, private or reserved"):
            assert_safe_auth_url("https://169.254.169.254/latest/meta-data/", SERVER)

    def test_an_obfuscated_loopback_is_refused(self) -> None:
        # The whole reason the canonical parser exists.
        for host in ("0x7f000001", "2130706433", "127.1", "0177.0.0.1"):
            with pytest.raises(McpSsrfError):
                assert_safe_auth_url(f"https://{host}/token", SERVER)

    def test_a_different_host_is_refused_without_an_allowlist(self) -> None:
        with pytest.raises(McpSsrfError, match="differs from the MCP server host"):
            assert_safe_auth_url("https://evil.example.net/token", SERVER)

    def test_an_allowlisted_identity_provider_is_permitted(self) -> None:
        # The real deployment shape: the server is api.example.com and the
        # authorization server auth.example.com.
        assert_safe_auth_url(
            "https://auth.example.com/token",
            SERVER,
            SsrfGuardOptions(allowed_hosts=["auth.example.com"]),
        )

    def test_an_allowlist_does_not_admit_everything_else(self) -> None:
        with pytest.raises(McpSsrfError, match="not in allowed_hosts"):
            assert_safe_auth_url(
                "https://evil.example.net/token",
                SERVER,
                SsrfGuardOptions(allowed_hosts=["auth.example.com"]),
            )

    def test_an_allowlist_does_not_lift_the_loopback_rule(self) -> None:
        # Naming a host does not make it safe to be sent inside the network.
        with pytest.raises(McpSsrfError, match="loopback"):
            assert_safe_auth_url(
                "https://127.0.0.1/token",
                SERVER,
                SsrfGuardOptions(allowed_hosts=["127.0.0.1"]),
            )

    def test_loopback_can_be_opted_into_for_local_work(self) -> None:
        assert_safe_auth_url(
            "https://127.0.0.1:8080/token", SERVER, SsrfGuardOptions(allow_loopback=True)
        )

    def test_opting_into_loopback_does_not_open_the_public_internet(self) -> None:
        # It waives the ORIGIN check only for the loopback hosts it was turned
        # on for; a public host still has to be the server's or allowlisted.
        with pytest.raises(McpSsrfError, match="differs from the MCP server host"):
            assert_safe_auth_url(
                "https://evil.example.net/token", SERVER, SsrfGuardOptions(allow_loopback=True)
            )

    def test_the_host_is_compared_case_insensitively(self) -> None:
        assert_safe_auth_url("https://MCP.Example.COM/token", SERVER)

    def test_nonsense_is_refused(self) -> None:
        for url in ("", "not a url", "://missing-scheme", "https://"):
            with pytest.raises(McpSsrfError):
                assert_safe_auth_url(url, SERVER)


# -- PKCE and state ----------------------------------------------------------


class TestPkceAndState:
    def test_a_verifier_and_its_challenge_agree(self) -> None:
        import base64
        import hashlib

        verifier, challenge = generate_pkce()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert challenge == expected

    def test_nothing_repeats(self) -> None:
        # The verifier IS the proof that the client redeeming the code is the
        # one that asked for it.
        assert len({generate_pkce()[0] for _ in range(50)}) == 50
        assert len({generate_state() for _ in range(50)}) == 50

    def test_they_are_url_safe_and_long_enough(self) -> None:
        # RFC 7636 wants 43-128 characters, and a `+` or `/` would be mangled
        # in the query string it travels in.
        for value in (generate_pkce()[0], generate_pkce()[1], generate_state()):
            assert 43 <= len(value) <= 128
            assert "+" not in value and "/" not in value and "=" not in value


class TestIssValidation:
    """RFC 9207: who actually minted this code?"""

    META = AuthServerMetadata(
        authorization_endpoint="https://mcp.example.com/authorize",
        token_endpoint="https://mcp.example.com/token",
        issuer="https://mcp.example.com",
    )

    def test_a_matching_issuer_passes(self) -> None:
        validate_authorization_response_iss("https://mcp.example.com", self.META)

    def test_another_issuer_is_refused(self) -> None:
        # The mix-up attack: a code minted somewhere else, redeemed by us,
        # replaying the user's credentials against a party they never chose.
        with pytest.raises(ValueError, match="iss mismatch"):
            validate_authorization_response_iss("https://evil.example.net", self.META)

    def test_the_comparison_is_exact_not_normalised(self) -> None:
        # A trailing slash is a different string, and that leniency is exactly
        # what an attacker looks for.
        with pytest.raises(ValueError, match="iss mismatch"):
            validate_authorization_response_iss("https://mcp.example.com/", self.META)

    def test_a_missing_iss_is_fine_when_the_server_does_not_send_one(self) -> None:
        validate_authorization_response_iss(None, self.META)

    def test_a_missing_iss_is_refused_when_the_server_says_it_sends_one(self) -> None:
        # Otherwise stripping the parameter would dodge the check entirely.
        import dataclasses

        advertised = dataclasses.replace(
            self.META, authorization_response_iss_parameter_supported=True
        )
        with pytest.raises(ValueError, match="missing the iss parameter|no iss parameter"):
            validate_authorization_response_iss(None, advertised)


# -- the flow ----------------------------------------------------------------


class AuthServer:
    """A scripted authorization server, at the fetch boundary."""

    def __init__(
        self,
        *,
        authorization_endpoint: str = "https://mcp.example.com/authorize",
        token_endpoint: str = "https://mcp.example.com/token",
        registration_endpoint: str | None = "https://mcp.example.com/register",
        issuer: str | None = "https://mcp.example.com",
        iss_supported: bool = False,
        oauth_document: bool = True,
        token_status: int = 200,
    ) -> None:
        self.authorization_endpoint = authorization_endpoint
        self.token_endpoint = token_endpoint
        self.registration_endpoint = registration_endpoint
        self.issuer = issuer
        self.iss_supported = iss_supported
        self.oauth_document = oauth_document
        self.token_status = token_status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
        self.requests.append(request)
        url = request["url"]
        if url.endswith("/.well-known/oauth-authorization-server"):
            if not self.oauth_document:
                return {"status": 404, "body": {}}
            return {"status": 200, "body": self._metadata()}
        if url.endswith("/.well-known/openid-configuration"):
            return {"status": 200, "body": self._metadata()}
        if url == self.registration_endpoint:
            return {"status": 200, "body": {"client_id": "cid-1", "client_secret": "shh"}}
        if url == self.token_endpoint:
            if self.token_status >= 400:
                return {"status": self.token_status, "body": {"error": "nope"}}
            return {
                "status": 200,
                "body": {
                    "access_token": "at-1",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rt-1",
                },
            }
        raise AssertionError(f"unexpected OAuth url: {url}")

    def _metadata(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "issuer": self.issuer,
        }
        if self.registration_endpoint:
            document["registration_endpoint"] = self.registration_endpoint
        if self.iss_supported:
            document["authorization_response_iss_parameter_supported"] = True
        return document

    def form_of(self, url: str) -> dict[str, str]:
        """The form body of the last request to one endpoint."""
        from urllib.parse import parse_qsl

        matching = [r for r in self.requests if r["url"] == url]
        assert matching, f"no request to {url}"
        return dict(parse_qsl(matching[-1]["body"]))


class Provider:
    """The interactive half, remembered in memory."""

    def __init__(self, tokens: McpOAuthTokens | None = None) -> None:
        self.redirect_url = "http://127.0.0.1:8765/callback"
        self.client_metadata = McpOAuthClientMetadata(
            redirect_uris=["http://127.0.0.1:8765/callback"],
            client_name="test",
            scope="mcp:read",
        )
        self._client: McpOAuthClientInfo | None = None
        self._tokens = tokens
        self._verifier = ""
        self._state: str | None = None
        self.redirected_to: list[str] = []

    def client_information(self) -> McpOAuthClientInfo | None:
        return self._client

    def save_client_information(self, info: McpOAuthClientInfo) -> None:
        self._client = info

    def tokens(self) -> McpOAuthTokens | None:
        return self._tokens

    def save_tokens(self, tokens: McpOAuthTokens) -> None:
        self._tokens = tokens

    def redirect_to_authorization(self, authorization_url: str) -> None:
        self.redirected_to.append(authorization_url)

    def save_code_verifier(self, verifier: str) -> None:
        self._verifier = verifier

    def code_verifier(self) -> str:
        return self._verifier

    def save_state(self, state: str) -> None:
        self._state = state

    def state(self) -> str | None:
        return self._state


class TestDiscovery:
    def test_the_oauth_document_is_read(self) -> None:
        server = AuthServer()
        metadata = discover_metadata(server, SERVER)
        assert metadata.token_endpoint == "https://mcp.example.com/token"
        assert metadata.issuer == "https://mcp.example.com"

    def test_the_oidc_document_is_the_fallback(self) -> None:
        # A 404 on the first is expected on every OIDC-only server.
        server = AuthServer(oauth_document=False)
        metadata = discover_metadata(server, SERVER)
        assert metadata.authorization_endpoint.endswith("/authorize")

    def test_discovery_reads_the_servers_own_origin(self) -> None:
        # Never a URL the server supplied: the first fetch has to be somewhere
        # already trusted, or the guard has nothing to anchor to.
        server = AuthServer()
        discover_metadata(server, SERVER)
        assert server.requests[0]["url"].startswith("https://mcp.example.com/.well-known/")

    def test_an_endpoint_pointing_inside_the_network_is_refused(self) -> None:
        server = AuthServer(token_endpoint="https://169.254.169.254/token")
        with pytest.raises(McpSsrfError):
            discover_metadata(server, SERVER)

    def test_every_endpoint_is_checked_not_just_the_first(self) -> None:
        # A server that passes the guard on two of three has still been handed
        # one URL it should never have been allowed to name.
        inside = "https://127.0.0.1/x"
        for server in (
            AuthServer(authorization_endpoint=inside),
            AuthServer(token_endpoint=inside),
            AuthServer(registration_endpoint=inside),
        ):
            with pytest.raises(McpSsrfError):
                discover_metadata(server, SERVER)

    def test_a_server_with_no_metadata_says_so(self) -> None:
        class Empty:
            def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
                return {"status": 404, "body": {}}

        with pytest.raises(ValueError, match="no authorization-server metadata"):
            discover_metadata(Empty(), SERVER)


class TestTheAuthorizationUrl:
    def test_it_carries_pkce_and_the_state(self) -> None:
        from urllib.parse import parse_qs, urlsplit

        url = build_authorization_url(
            "https://mcp.example.com/authorize",
            client_id="cid",
            redirect_uri="http://127.0.0.1/cb",
            code_challenge="chal",
            scope="mcp:read",
            state="st",
            resource=SERVER,
        )
        query = parse_qs(urlsplit(url).query)
        assert query["response_type"] == ["code"]
        assert query["code_challenge"] == ["chal"]
        assert query["state"] == ["st"]
        assert query["resource"] == [SERVER]

    def test_only_s256_is_offered(self) -> None:
        # `plain` is still in RFC 7636 and is worth nothing against anyone who
        # can read the authorization request.
        from urllib.parse import parse_qs, urlsplit

        url = build_authorization_url(
            "https://mcp.example.com/authorize",
            client_id="cid",
            redirect_uri="http://127.0.0.1/cb",
            code_challenge="chal",
        )
        assert parse_qs(urlsplit(url).query)["code_challenge_method"] == ["S256"]


class TestTheFlow:
    def oauth(self, server: AuthServer, provider: Provider, **kwargs: Any) -> McpOAuth:
        return McpOAuth(SERVER, provider, server, **kwargs)

    def test_a_fresh_token_needs_no_work(self) -> None:
        provider = Provider(McpOAuthTokens(access_token="at", expires_in=3600))
        server = AuthServer()
        assert self.oauth(server, provider).authorize() == "authorized"
        assert server.requests == [], "a usable token means no traffic at all"

    def test_no_token_starts_a_redirect(self) -> None:
        provider = Provider()
        assert self.oauth(AuthServer(), provider).authorize() == "redirect"
        assert provider.redirected_to, "the person was never sent anywhere"
        assert provider.state() is not None, "the state must be stored before the redirect"
        assert provider.code_verifier(), "the verifier must be stored before the redirect"

    def test_an_expired_token_is_refreshed_rather_than_re_authorized(self) -> None:
        provider = Provider(
            McpOAuthTokens(access_token="old", expires_in=1, obtained_at=0, refresh_token="rt")
        )
        server = AuthServer()
        assert self.oauth(server, provider).authorize() == "authorized"
        assert not provider.redirected_to, "a refresh must not disturb the person"
        tokens = provider.tokens()
        assert tokens is not None and tokens.access_token == "at-1"

    def test_a_refresh_that_fails_falls_back_to_asking_a_person(self) -> None:
        provider = Provider(
            McpOAuthTokens(access_token="old", expires_in=1, obtained_at=0, refresh_token="rt")
        )
        server = AuthServer(token_status=400)
        assert self.oauth(server, provider).authorize() == "redirect"
        assert provider.redirected_to

    def test_a_refresh_keeps_the_refresh_token_when_none_comes_back(self) -> None:
        # A server that returns no new one means the old one still stands, and
        # dropping it would make the NEXT refresh impossible.
        class NoNewRefresh(AuthServer):
            def __call__(self, request: Any, options: Any = None) -> dict[str, Any]:
                answer = super().__call__(request, options)
                if request["url"] == self.token_endpoint:
                    answer["body"].pop("refresh_token", None)
                return answer

        provider = Provider(
            McpOAuthTokens(access_token="old", expires_in=1, obtained_at=0, refresh_token="keep-me")
        )
        self.oauth(NoNewRefresh(), provider).authorize()
        tokens = provider.tokens()
        assert tokens is not None and tokens.refresh_token == "keep-me"

    def test_a_guard_failure_during_refresh_is_never_swallowed(self) -> None:
        # A refresh that failed the guard is an attempted redirect somewhere it
        # should not go, not a stale token -- and `authorize()` would otherwise
        # report it as "the person needs to log in again".
        provider = Provider(
            McpOAuthTokens(access_token="old", expires_in=1, obtained_at=0, refresh_token="rt")
        )
        server = AuthServer(token_endpoint="https://169.254.169.254/token")
        with pytest.raises(McpSsrfError):
            self.oauth(server, provider).authorize()

    def test_the_bearer_header_carries_the_token(self) -> None:
        provider = Provider(McpOAuthTokens(access_token="at", expires_in=3600))
        assert self.oauth(AuthServer(), provider).auth_header() == {
            "authorization": "Bearer at"
        }

    def test_no_token_means_no_header_rather_than_an_empty_one(self) -> None:
        # An `Authorization: Bearer ` header is worse than none: it looks like
        # a credential to every log and proxy in between.
        assert self.oauth(AuthServer(), Provider()).auth_header() == {}

    def test_a_401_refreshes_and_says_retry(self) -> None:
        provider = Provider(
            McpOAuthTokens(access_token="old", expires_in=1, obtained_at=0, refresh_token="rt")
        )
        assert self.oauth(AuthServer(), provider).reauthorize() is True

    def test_a_401_with_nothing_to_refresh_asks_a_person(self) -> None:
        provider = Provider()
        assert self.oauth(AuthServer(), provider).reauthorize() is False
        assert provider.redirected_to

    def test_a_client_is_registered_when_there_is_none(self) -> None:
        provider = Provider()
        self.oauth(AuthServer(), provider).authorize()
        client = provider.client_information()
        assert client is not None and client.client_id == "cid-1"

    def test_registration_declares_a_native_application(self) -> None:
        # Some servers hold `web` clients to stricter redirect-URI rules, and a
        # server left to guess usually guesses `web`.
        server = AuthServer()
        register_client(
            server, "https://mcp.example.com/register", Provider().client_metadata, SERVER
        )
        body = [r for r in server.requests if r["url"].endswith("/register")][-1]["body"]
        assert body["application_type"] == "native"

    def test_a_server_with_no_registration_endpoint_says_so(self) -> None:
        provider = Provider()
        server = AuthServer(registration_endpoint=None)
        with pytest.raises(ValueError, match="no client is registered"):
            self.oauth(server, provider).authorize()


class TestFinishing:
    def started(self) -> tuple[AuthServer, Provider, McpOAuth]:
        server, provider = AuthServer(), Provider()
        oauth = McpOAuth(SERVER, provider, server)
        oauth.authorize()  # stores the state and verifier, and redirects
        return server, provider, oauth

    def test_the_code_becomes_tokens(self) -> None:
        _, provider, oauth = self.started()
        state = provider.state()
        assert state is not None
        oauth.finish("the-code", state)
        tokens = provider.tokens()
        assert tokens is not None and tokens.access_token == "at-1"

    def test_the_exchange_sends_the_verifier_not_the_challenge(self) -> None:
        # PKCE is worthless the other way round.
        server, provider, oauth = self.started()
        state = provider.state()
        assert state is not None
        oauth.finish("the-code", state)
        form = server.form_of(server.token_endpoint)
        assert form["code_verifier"] == provider.code_verifier()
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "the-code"

    def test_a_wrong_state_is_refused(self) -> None:
        _, _provider, oauth = self.started()
        with pytest.raises(ValueError, match="state mismatch"):
            oauth.finish("the-code", "not-the-state")

    def test_a_missing_state_is_refused(self) -> None:
        server, provider = AuthServer(), Provider()
        with pytest.raises(ValueError, match="no state was stored"):
            McpOAuth(SERVER, provider, server).finish("the-code", "anything")

    def test_a_code_from_another_issuer_is_refused(self) -> None:
        _, provider, oauth = self.started()
        state = provider.state()
        assert state is not None
        with pytest.raises(ValueError, match="iss mismatch"):
            oauth.finish("the-code", state, iss="https://evil.example.net")

    def test_the_issuer_is_checked_before_the_code_is_redeemed(self) -> None:
        # Validating afterwards would mean the credentials had already been
        # replayed, which is the whole attack.
        server, provider, oauth = self.started()
        state = provider.state()
        assert state is not None
        before = len([r for r in server.requests if r["url"] == server.token_endpoint])
        with pytest.raises(ValueError, match="iss mismatch"):
            oauth.finish("the-code", state, iss="https://evil.example.net")
        after = len([r for r in server.requests if r["url"] == server.token_endpoint])
        assert after == before, "the code reached the token endpoint anyway"

    def test_the_helper_finishes_the_same_flow(self) -> None:
        server, provider, _ = self.started()
        state = provider.state()
        assert state is not None
        finish_mcp_auth(SERVER, "the-code", state, provider=provider, fetch=server)
        tokens = provider.tokens()
        assert tokens is not None and tokens.access_token == "at-1"


class TestTokens:
    def test_an_unknown_lifetime_reads_as_valid(self) -> None:
        # The server declined to say, and guessing "expired" would throw away a
        # token that very likely works.
        assert not McpOAuthTokens(access_token="at").expired

    def test_a_token_is_stale_before_its_stated_expiry(self) -> None:
        # One that expires while the request it authorises is in flight is a
        # failure the caller can do nothing about.
        import time

        assert McpOAuthTokens(
            access_token="at", expires_in=30, obtained_at=time.time()
        ).expired
        assert not McpOAuthTokens(
            access_token="at", expires_in=3600, obtained_at=time.time()
        ).expired

    def test_tokens_round_trip_through_storage(self) -> None:
        original = McpOAuthTokens(
            access_token="at", refresh_token="rt", expires_in=60, scope="s", token_type="Bearer"
        )
        assert McpOAuthTokens.of(json.loads(json.dumps(original.as_row()))) == original


class TestConnecting:
    def test_a_server_needing_a_person_says_so_by_name(self) -> None:
        from combycode_llm_sdk import connect_mcp

        provider = Provider()
        with pytest.raises(McpUnauthorizedError, match="requires authorization"):
            connect_mcp(url=SERVER, auth=provider, engine=_engine(AuthServer()))
        assert provider.redirected_to, "the person was never sent anywhere"


def _engine(fetch: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(fetch=fetch, fetch_stream=None)
