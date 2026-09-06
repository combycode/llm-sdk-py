"""OAuth 2.1 + PKCE, for an MCP server that wants to know who is asking.

The split is the design. This library owns everything MECHANICAL -- metadata
discovery, PKCE, dynamic client registration, the code exchange, refresh -- and
delegates everything INTERACTIVE to an `McpAuthProvider` the consumer writes:
where to store tokens, how to send a person to a browser, how the callback code
comes back. A library cannot know any of those, and one that guessed would be
wrong in a different way for every host application.

Three defences here are not optional and each prevents a specific attack:

- **PKCE with S256.** `plain` is still in RFC 7636 and is worth nothing against
  anyone who can read the authorization request, so only S256 is sent.
- **A CSRF `state`, compared in constant time.** Without it an attacker feeds
  the victim's browser their own authorization code and the victim's client
  binds an attacker's account.
- **RFC 9207 `iss` validation, BEFORE the code is redeemed.** This is the
  mix-up defence: a malicious authorization server hands back a code minted by a
  DIFFERENT one, and a client that redeems it replays the user's credentials
  against a party they never meant to authorize. Checking afterwards would mean
  the replay already happened.

Every URL discovery returns is checked by `url_guard` before it is fetched.

Transposed from `unified-library-ts/src/plugins/mcp/oauth.ts`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from ..llm.wire_transforms import make_registry
from ..wire.interpreter import build_from_spec
from .url_guard import McpSsrfError, SsrfGuardOptions, assert_safe_auth_url
from .wire_rules import mcp_spec

#: How long before a token's stated expiry to treat it as already stale. A token
#: that expires while the request it authorises is in flight is a failure the
#: caller cannot do anything about.
EXPIRY_BUFFER_SECONDS = 60.0

#: Bytes of entropy behind the PKCE verifier and the CSRF state.
_ENTROPY_BYTES = 32

_REGISTRY = make_registry({})


class McpUnauthorizedError(Exception):
    """A person has to authorize this, interactively, before it can continue.

    The provider's `redirect_to_authorization` has already been called by the
    time this is raised; finish with `finish_mcp_auth` once the callback lands.
    """


@dataclass
class McpOAuthTokens:
    """What an authorization server gave us."""

    access_token: str
    token_type: str | None = None
    expires_in: float | None = None
    refresh_token: str | None = None
    scope: str | None = None
    #: When WE received them, so expiry can be worked out at all: the server
    #: states a lifetime, never a deadline.
    obtained_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        """Whether these are too close to expiry to use.

        An unknown lifetime reads as valid: the server declined to say, and
        guessing "expired" would throw away a token that very likely works.

        `is None`, not falsiness. `expires_in=0` is a server saying "already
        expired" and `obtained_at=0` is the epoch -- both are ANSWERS, and a
        falsy test reads them as "no answer" and calls the token good. The
        TypeScript tests falsiness and gets away with it only because it always
        stamps both fields itself.
        """
        if self.expires_in is None or self.obtained_at is None:
            return False
        return time.time() > self.obtained_at + self.expires_in - EXPIRY_BUFFER_SECONDS

    def as_row(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "obtained_at": self.obtained_at,
        }

    @staticmethod
    def of(row: Mapping[str, Any]) -> McpOAuthTokens:
        expires = row.get("expires_in")
        return McpOAuthTokens(
            access_token=str(row.get("access_token") or ""),
            token_type=str(row["token_type"]) if row.get("token_type") else None,
            expires_in=float(expires) if isinstance(expires, (int, float)) else None,
            refresh_token=str(row["refresh_token"]) if row.get("refresh_token") else None,
            scope=str(row["scope"]) if row.get("scope") else None,
            obtained_at=float(row.get("obtained_at") or time.time()),
        )


@dataclass(frozen=True)
class McpOAuthClientInfo:
    """The client identity, registered or configured."""

    client_id: str
    client_secret: str | None = None


@dataclass(frozen=True)
class McpOAuthClientMetadata:
    """What dynamic client registration tells the server about us."""

    redirect_uris: Sequence[str]
    client_name: str | None = None
    scope: str | None = None
    grant_types: Sequence[str] = ("authorization_code", "refresh_token")
    response_types: Sequence[str] = ("code",)
    token_endpoint_auth_method: str | None = None
    #: OIDC application type. `native` by default because an MCP client is
    #: normally a local process with a loopback redirect, and a server that has
    #: to guess usually guesses `web` -- which some then hold to stricter
    #: redirect-URI rules.
    application_type: str = "native"


@dataclass(frozen=True)
class AuthServerMetadata:
    """What the authorization server publishes about itself."""

    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    #: RFC 8414. What the RFC 9207 `iss` must equal, exactly.
    issuer: str | None = None
    #: The server says it returns `iss`. When it does, a response WITHOUT one is
    #: refused -- otherwise stripping the parameter would dodge the check.
    authorization_response_iss_parameter_supported: bool = False


class McpAuthProvider(Protocol):
    """The interactive half, which belongs to the host application."""

    @property
    def redirect_url(self) -> str: ...

    @property
    def client_metadata(self) -> McpOAuthClientMetadata: ...

    def client_information(self) -> McpOAuthClientInfo | None: ...

    def save_client_information(self, info: McpOAuthClientInfo) -> None: ...

    def tokens(self) -> McpOAuthTokens | None: ...

    def save_tokens(self, tokens: McpOAuthTokens) -> None: ...

    def redirect_to_authorization(self, authorization_url: str) -> None: ...

    def save_code_verifier(self, verifier: str) -> None: ...

    def code_verifier(self) -> str: ...

    def save_state(self, state: str) -> None: ...

    def state(self) -> str | None: ...


# -- PKCE and CSRF -----------------------------------------------------------


def _base64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def generate_pkce() -> tuple[str, str]:
    """A verifier and its S256 challenge.

    `secrets`, not `random`: the verifier IS the proof that the client
    redeeming the code is the one that asked for it, so a predictable generator
    hands that proof to anyone who can guess the seed.
    """
    verifier = _base64url(secrets.token_bytes(_ENTROPY_BYTES))
    challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def generate_state() -> str:
    """A CSRF state token."""
    return _base64url(secrets.token_bytes(_ENTROPY_BYTES))


def validate_authorization_response_iss(
    iss: str | None, metadata: AuthServerMetadata
) -> None:
    """Check who actually minted the code, before it is redeemed.

    Compared with EXACT string equality per RFC 9207 §2.4, deliberately not
    URL-normalised: normalising would make `https://as.example.com` and
    `https://as.example.com/` compare equal, and that leniency is precisely what
    an attacker looks for.
    """
    if iss is not None:
        if iss != metadata.issuer:
            raise ValueError(
                f"MCP OAuth: authorization response iss mismatch -- got {iss!r}, expected "
                f"{metadata.issuer or '(unknown)'!r}. Refusing to exchange a code that may "
                "have been minted by a different authorization server."
            )
        return
    if metadata.authorization_response_iss_parameter_supported:
        raise ValueError(
            "MCP OAuth: the authorization response has no iss parameter, which this "
            "authorization server advertises that it sends. Refusing to exchange the code."
        )


# -- the wire ----------------------------------------------------------------


def _oauth_request(
    spec_id: str, payload: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One OAuth request, built from its spec.

    A form body arrives from the spec as FIELDS and is encoded here -- the same
    split multipart uses, which keeps the frozen fixture readable as parameters
    rather than as one escaped string.
    """
    built = build_from_spec(
        mcp_spec(spec_id), dict(payload), _REGISTRY, "mcp", None, dict(config or {})
    )
    request: dict[str, Any] = {
        "url": built.url,
        "method": built.method or "POST",
        "headers": dict(built.headers or {}),
        "provider": "mcp",
        "model": "oauth",
        "responseType": "json",
    }
    if getattr(built, "no_body", False):
        return request
    body = built.body
    if getattr(built, "form_body", False) and isinstance(body, Mapping):
        request["body"] = urlencode({k: v for k, v in body.items() if v is not None})
        request["headers"]["content-type"] = "application/x-www-form-urlencoded"
    else:
        request["body"] = body
    return request


def _read(response: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(response, Mapping):
        status, body = response.get("status"), response.get("body")
    else:
        status, body = getattr(response, "status", None), getattr(response, "body", None)
    return (
        int(status) if isinstance(status, int) else 0,
        dict(body) if isinstance(body, Mapping) else {},
    )


def _get_json(fetch: Any, spec_id: str, config: Mapping[str, Any]) -> dict[str, Any] | None:
    """A discovery document, or None when there is not one there.

    A missing document is an ordinary answer, not a failure: the OAuth
    well-known path is probed first and the OIDC one is the fallback, so the
    first 404 is expected on every OIDC-only server.
    """
    try:
        status, body = _read(fetch(_oauth_request(spec_id, {}, config)))
    except Exception:  # noqa: BLE001 -- a 404 may arrive as an exception from the
        # executor; either way it means "not here", and the caller tries the next.
        return None
    return body if status < 400 else None


def _post_form(fetch: Any, spec_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    status, body = _read(fetch(_oauth_request(spec_id, payload)))
    if status >= 400:
        raise ValueError(f"MCP OAuth: the token endpoint answered {status}")
    return body


# -- discovery, registration, tokens -----------------------------------------


def discover_metadata(
    fetch: Any, server_url: str, security: SsrfGuardOptions | None = None
) -> AuthServerMetadata:
    """Find the authorization server for an MCP server, and vet every URL it names.

    Both well-known documents are read from the MCP SERVER'S OWN ORIGIN, never
    from a URL the server supplied. That ordering is what makes the guard
    meaningful: the first fetch has to be somewhere we already trust.
    """
    origin = _origin_of(server_url)
    document = _get_json(fetch, "mcp-oauth/discover.oauth", {"origin": origin}) or _get_json(
        fetch, "mcp-oauth/discover.oidc", {"origin": origin}
    )
    if not document or not document.get("authorization_endpoint") or not document.get(
        "token_endpoint"
    ):
        raise ValueError(f"MCP OAuth: no authorization-server metadata at {origin}")

    authorization = str(document["authorization_endpoint"])
    token = str(document["token_endpoint"])
    registration = (
        str(document["registration_endpoint"]) if document.get("registration_endpoint") else None
    )

    # Every server-chosen URL, before any of them is fetched.
    assert_safe_auth_url(authorization, server_url, security)
    assert_safe_auth_url(token, server_url, security)
    if registration:
        assert_safe_auth_url(registration, server_url, security)

    issuer = document.get("issuer")
    return AuthServerMetadata(
        authorization_endpoint=authorization,
        token_endpoint=token,
        registration_endpoint=registration,
        issuer=str(issuer) if isinstance(issuer, str) else None,
        authorization_response_iss_parameter_supported=(
            document.get("authorization_response_iss_parameter_supported") is True
        ),
    )


def _origin_of(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def register_client(
    fetch: Any,
    registration_endpoint: str,
    metadata: McpOAuthClientMetadata,
    server_url: str,
    security: SsrfGuardOptions | None = None,
) -> McpOAuthClientInfo:
    """Dynamic client registration (RFC 7591)."""
    assert_safe_auth_url(registration_endpoint, server_url, security)
    status, body = _read(
        fetch(
            _oauth_request(
                "mcp-oauth/register",
                {
                    "registrationEndpoint": registration_endpoint,
                    "metadata": {
                        "redirect_uris": list(metadata.redirect_uris),
                        "client_name": metadata.client_name,
                        "scope": metadata.scope,
                        "grant_types": list(metadata.grant_types),
                        "response_types": list(metadata.response_types),
                        "token_endpoint_auth_method": metadata.token_endpoint_auth_method,
                        "application_type": metadata.application_type,
                    },
                },
            )
        )
    )
    if status >= 400:
        raise ValueError(f"MCP OAuth: client registration answered {status}")
    client_id = body.get("client_id")
    if not client_id:
        raise ValueError("MCP OAuth: the registration response carried no client_id")
    secret = body.get("client_secret")
    return McpOAuthClientInfo(
        client_id=str(client_id), client_secret=str(secret) if secret else None
    )


def build_authorization_url(authorization_endpoint: str, **params: Any) -> str:
    """The URL a person opens. Built from the spec like any request."""
    url: str = _oauth_request(
        "mcp-oauth/authorize",
        {"authorizationEndpoint": authorization_endpoint, **params},
    )["url"]
    return url


def exchange_code(fetch: Any, token_endpoint: str, **params: Any) -> McpOAuthTokens:
    """Trade an authorization code for tokens."""
    return McpOAuthTokens.of(
        _post_form(fetch, "mcp-oauth/token.exchange", {"tokenEndpoint": token_endpoint, **params})
    )


def refresh_tokens(fetch: Any, token_endpoint: str, **params: Any) -> McpOAuthTokens:
    """Trade a refresh token for a fresh access token."""
    return McpOAuthTokens.of(
        _post_form(fetch, "mcp-oauth/token.refresh", {"tokenEndpoint": token_endpoint, **params})
    )


# -- the orchestrator --------------------------------------------------------


class McpOAuth:
    """Holds the flow together: discovery, registration, tokens, refresh."""

    def __init__(
        self,
        server_url: str,
        provider: McpAuthProvider,
        fetch: Any,
        security: SsrfGuardOptions | None = None,
    ) -> None:
        self._server_url = server_url
        self._provider = provider
        self._fetch = fetch
        self._security = security
        self._metadata: AuthServerMetadata | None = None

    def authorize(self) -> str:
        """Make sure a usable token exists.

        Returns `"authorized"`, or `"redirect"` when a person has to act -- the
        provider has already been asked to send them.
        """
        tokens = self._provider.tokens()
        if tokens and tokens.access_token and not tokens.expired:
            return "authorized"
        if tokens and tokens.refresh_token and self._try_refresh(tokens.refresh_token):
            return "authorized"
        self._start_redirect()
        return "redirect"

    def auth_header(self) -> dict[str, str]:
        """The bearer header, refreshing first if the token is stale."""
        tokens = self._provider.tokens()
        stale = bool(tokens and tokens.access_token and tokens.expired and tokens.refresh_token)
        if stale and tokens and self._try_refresh(str(tokens.refresh_token)):
            tokens = self._provider.tokens()
        if tokens and tokens.access_token:
            return {"authorization": f"Bearer {tokens.access_token}"}
        return {}

    def reauthorize(self) -> bool:
        """Answer a 401. True when a retry is worth it."""
        tokens = self._provider.tokens()
        if tokens and tokens.refresh_token and self._try_refresh(tokens.refresh_token):
            return True
        self._start_redirect()
        return False

    def finish(self, code: str, returned_state: str, iss: str | None = None) -> None:
        """Complete the interactive flow with the callback's code.

        The state is compared in CONSTANT TIME. A comparison that returns early
        on the first wrong byte leaks the expected value one byte at a time to
        anyone who can measure it, and the whole point of the state is that an
        attacker cannot produce it.
        """
        expected = self._provider.state()
        if not expected:
            raise ValueError(
                "MCP OAuth: no state was stored -- this authorization was not started "
                "by this client"
            )
        if not hmac.compare_digest(expected, returned_state):
            raise ValueError("MCP OAuth: state mismatch -- possible CSRF")

        metadata = self._ensure_metadata()
        # Before the code goes anywhere near the token endpoint: validating
        # afterwards would mean the credentials had already been replayed.
        validate_authorization_response_iss(iss, metadata)

        client = self._ensure_client(metadata)
        tokens = exchange_code(
            self._fetch,
            metadata.token_endpoint,
            code=code,
            code_verifier=self._provider.code_verifier(),
            client_id=client.client_id,
            client_secret=client.client_secret,
            redirect_uri=self._provider.redirect_url,
            resource=self._server_url,
        )
        self._provider.save_tokens(tokens)

    # -- internal ------------------------------------------------------------

    def _start_redirect(self) -> None:
        metadata = self._ensure_metadata()
        client = self._ensure_client(metadata)
        verifier, challenge = generate_pkce()
        state = generate_state()
        # Stored BEFORE the person is sent anywhere: a callback that arrives
        # before the verifier was saved cannot be completed.
        self._provider.save_code_verifier(verifier)
        self._provider.save_state(state)
        self._provider.redirect_to_authorization(
            build_authorization_url(
                metadata.authorization_endpoint,
                client_id=client.client_id,
                redirect_uri=self._provider.redirect_url,
                code_challenge=challenge,
                scope=self._provider.client_metadata.scope,
                state=state,
                resource=self._server_url,
            )
        )

    def _try_refresh(self, refresh_token: str) -> bool:
        try:
            metadata = self._ensure_metadata()
            client = self._ensure_client(metadata)
            tokens = refresh_tokens(
                self._fetch,
                metadata.token_endpoint,
                refresh_token=refresh_token,
                client_id=client.client_id,
                client_secret=client.client_secret,
            )
            # A server that returns no new refresh token means the old one still
            # stands; dropping it here would make the next refresh impossible.
            if not tokens.refresh_token:
                tokens.refresh_token = refresh_token
            self._provider.save_tokens(tokens)
        except McpSsrfError:
            # Never swallowed: a refresh that failed the guard is an attempted
            # redirect somewhere it should not go, not a stale token.
            raise
        except Exception:  # noqa: BLE001 -- any other failure means "cannot refresh",
            # and the caller's answer to that is to ask a person.
            return False
        return True

    def _ensure_metadata(self) -> AuthServerMetadata:
        if self._metadata is None:
            self._metadata = discover_metadata(self._fetch, self._server_url, self._security)
        return self._metadata

    def _ensure_client(self, metadata: AuthServerMetadata) -> McpOAuthClientInfo:
        existing = self._provider.client_information()
        if existing:
            return existing
        if not metadata.registration_endpoint:
            raise ValueError(
                "MCP OAuth: no client is registered and this server offers no "
                "registration endpoint"
            )
        info = register_client(
            self._fetch,
            metadata.registration_endpoint,
            self._provider.client_metadata,
            self._server_url,
            self._security,
        )
        save = getattr(self._provider, "save_client_information", None)
        if save is not None:
            save(info)
        return info


def finish_mcp_auth(
    server_url: str,
    code: str,
    state: str,
    *,
    provider: McpAuthProvider,
    fetch: Any,
    security: SsrfGuardOptions | None = None,
    iss: str | None = None,
) -> None:
    """Finish an interactive grant, after catching `McpUnauthorizedError`.

    **Pass `iss` when the callback URL carried one.** It is validated against
    the authorization server's issuer before the code is redeemed, which is what
    stops a malicious server handing you a code minted elsewhere. Optional so
    existing callers keep working -- but a server that advertises
    `authorization_response_iss_parameter_supported` makes a missing one an
    error, as it should, since otherwise an attacker could strip the parameter
    to skip the check.
    """
    McpOAuth(server_url, provider, fetch, security).finish(code, state, iss)


__all__ = [
    "EXPIRY_BUFFER_SECONDS",
    "AuthServerMetadata",
    "McpAuthProvider",
    "McpOAuth",
    "McpOAuthClientInfo",
    "McpOAuthClientMetadata",
    "McpOAuthTokens",
    "McpUnauthorizedError",
    "build_authorization_url",
    "discover_metadata",
    "exchange_code",
    "finish_mcp_auth",
    "generate_pkce",
    "generate_state",
    "refresh_tokens",
    "register_client",
    "validate_authorization_response_iss",
]
