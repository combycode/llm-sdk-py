"""Small HTTP header helpers shared across the network and server layers.

PARTIAL transposition of `unified-library-ts/src/util/http.ts`. That file also
carries `headers_to_record` (from a WHATWG `Headers`), `is_stream_body`,
`any_signal` (combining AbortSignals) and `parse_response_body` -- all four are
about the fetch implementation itself, and they land with the network engine that
owns one. What is here is what a CONSUMER of a response needs.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import SplitResult, urljoin, urlsplit


def header(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup.

    HTTP header names are case-insensitive (RFC 9110 5.1) but a plain dict is
    not, and the casing a server or fetch implementation actually sends varies --
    so read response headers through here instead of guessing casings at the call
    site.
    """
    lower = name.lower()
    for key in headers:
        if key.lower() == lower:
            return headers[key]
    return None


def parse_int_header(headers: Mapping[str, str], key: str) -> int | None:
    """Parse an integer header value, or None if absent / not a number."""
    value = headers.get(key)
    if not value:
        return None
    try:
        return int(value, 10)
    except ValueError:
        return None


__all__ = ["header", "parse_int_header"]


#: Redirect statuses that PRESERVE the method and body. 307 and 308 are defined
#: to; 301, 302 and 303 are the ones every client turns into a body-less GET.
_METHOD_PRESERVING = frozenset((307, 308))


def _same_origin(from_url: SplitResult, to_url: SplitResult) -> bool:
    """Is `to_url` on `from_url`'s origin -- or its https upgrade on default ports?

    The upgrade is allowed because it is strictly a security improvement to the
    same host, and it is the one exception the reference implementations make.
    """
    if (from_url.scheme, from_url.netloc.lower()) == (to_url.scheme, to_url.netloc.lower()):
        return True
    return (
        (from_url.hostname or "") == (to_url.hostname or "")
        and from_url.scheme == "http"
        and to_url.scheme == "https"
        and from_url.port in (None, 80)
        and to_url.port in (None, 443)
    )


def follow_same_origin(
    request_url: str, method: str, status: int, location: str | None
) -> str | None:
    """Where a redirect wants to go, if we are willing to follow it.

    Willing means all of:

    * the METHOD survives. 307/308 preserve it; 301/302/303 turn a POST into a
      body-less GET, which for a JSON-RPC transport means the message is dropped
      and the server answers a question nobody asked. A GET redirect is fine
      under any of them, since there is no method or body to lose.
    * the target is the SAME ORIGIN (or its https upgrade on default ports).
      Everything on the request -- bearer token, session header, body -- was
      configured for one endpoint, so following cross-origin hands those to
      whoever controls the `Location` header.
    * the target brings no USERINFO of its own. `https://attacker@host/` is sent
      as Basic auth, so a Location that introduces credentials is a redirect that
      changes who we authenticate as.

    None for anything else, including a non-redirect -- the caller then treats the
    redirect response as the non-success it is.
    """
    if not location:
        return None
    verb = method.upper()
    if status not in _METHOD_PRESERVING and verb not in ("GET", "HEAD"):
        return None
    try:
        target = urljoin(request_url, location)
        to_url = urlsplit(target)
        from_url = urlsplit(request_url)
    except ValueError:
        # A Location we cannot even parse is not one to follow.
        return None
    if not to_url.scheme or not to_url.netloc:
        return None
    # Userinfo the CONFIGURED url already carried is fine and survives a relative
    # Location; userinfo the Location INTRODUCES is not.
    if (to_url.username or to_url.password) and (to_url.username, to_url.password) != (
        from_url.username,
        from_url.password,
    ):
        return None
    if not _same_origin(from_url, to_url):
        return None
    return target
