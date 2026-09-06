"""Refusing a URL the SERVER chose to point us at.

Everything OAuth discovery hands back -- the authorization endpoint, the token
endpoint, the registration endpoint -- is chosen by the far side. Fetching one
without checking it is a server-side request forgery: point the client at
`http://169.254.169.254/` and it fetches cloud credentials on the attacker's
behalf, from inside the network, with whatever the process can reach.

So every discovered URL passes `assert_safe_auth_url` before any fetch.

Secure by default, and the escape hatches are deliberately verbose so the risk
is visible at the call site:

- **https only.** `allow_insecure_http` opts into `http:` for local development.
- **No loopback, link-local, private or reserved hosts.** `allow_loopback` opts
  back in, for the same reason.
- **Same host as the MCP server**, unless `allowed_hosts` names the separate
  identity provider a real deployment often uses.

Transposed from `unified-library-ts/src/plugins/mcp/url-guard.ts`.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

#: Names that mean "this machine" whatever DNS would say.
LOOPBACK_HOSTNAMES = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})

#: The largest value a 32-bit address can hold.
MAX_IPV4_INT = 0xFFFFFFFF


@dataclass(frozen=True)
class SsrfGuardOptions:
    """What may be relaxed. Every default is the restrictive one."""

    #: Hosts accepted besides the MCP server's own. This is how a real
    #: deployment names its identity provider: the server is
    #: `api.example.com` and the authorization server `auth.example.com`.
    allowed_hosts: Sequence[str] = field(default_factory=tuple)
    #: Allow `http:`. For local development, never for production.
    allow_insecure_http: bool = False
    #: Allow loopback, private and reserved addresses. Same warning.
    allow_loopback: bool = False


class McpSsrfError(Exception):
    """A server-controlled URL failed the safety check."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"MCP SSRF guard: rejected URL {url!r} -- {reason}")
        self.url = url
        self.reason = reason


def _parse_ipv4_part(part: str) -> int | None:
    """One part of an address, in decimal, octal or hex."""
    if not part:
        return None
    try:
        if part.lower().startswith("0x"):
            value = int(part, 16)
        elif part.startswith("0") and len(part) > 1:
            value = int(part, 8)
        else:
            value = int(part, 10)
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_canonical_ipv4(host: str) -> tuple[int, int, int, int] | None:
    """An IPv4 literal in ANY form `inet_aton` accepts, as four octets.

    This is the security-critical half of the guard, and it is hand-written for
    a measured reason: Python's `ipaddress` REJECTS every obfuscated form --
    `0x7f000001`, `2130706433`, `0177.0.0.1`, `127.1` all raise. A guard that
    asked `ipaddress` alone would read each of those as an ordinary hostname and
    never apply the private-range check at all, while a browser, curl and the
    OS resolver all read them as 127.0.0.1.

    The short forms come from RFC 3986 / POSIX:

        1 part   the whole address as an integer   2130706433 -> 127.0.0.1
        2 parts  first octet + 24-bit remainder    127.1      -> 127.0.0.1
        3 parts  first two octets + 16-bit rest
        4 parts  ordinary dotted quad

    Returns None when the host is not an IPv4 literal at all -- a real hostname,
    which the caller then handles as one.
    """
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    values: list[int] = []
    for part in parts:
        parsed = _parse_ipv4_part(part)
        if parsed is None:
            return None
        values.append(parsed)

    limits = {1: MAX_IPV4_INT, 2: 0xFFFFFF, 3: 0xFFFF, 4: 0xFF}
    # Every part but the last is a single octet; the last absorbs the remainder.
    if any(value > 0xFF for value in values[:-1]):
        return None
    if values[-1] > limits[len(values)]:
        return None

    address = 0
    for index, value in enumerate(values[:-1]):
        address |= value << (8 * (3 - index))
    address |= values[-1]
    address &= MAX_IPV4_INT
    return (
        (address >> 24) & 0xFF,
        (address >> 16) & 0xFF,
        (address >> 8) & 0xFF,
        address & 0xFF,
    )


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is one nobody should be redirected at.

    `ipaddress` classifies almost all of it, and is preferred over a hand-written
    range list because it is maintained as registries change. Two ranges it does
    NOT flag are added explicitly, both measured rather than assumed:

    - `100.64.0.0/10`, carrier-grade NAT (RFC 6598).
    - `fec0::/10`, deprecated IPv6 site-local.

    Python is also STRICTER than the TypeScript in one place: it blocks IPv4
    multicast (`224.0.0.0/4`), which the TypeScript's `a >= 240` test lets
    through. Blocking more is the safe direction for this guard, so the
    difference is kept.
    """
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.IPv4Network("100.64.0.0/10")
    return address in ipaddress.IPv6Network("fec0::/10")


def is_blocked_host(hostname: str) -> bool:
    """Whether this host names something internal."""
    lowered = hostname.lower().strip()
    if lowered in LOOPBACK_HOSTNAMES:
        return True
    if lowered.startswith("[") and lowered.endswith("]"):
        lowered = lowered[1:-1]

    octets = parse_canonical_ipv4(lowered)
    if octets is not None:
        return _is_blocked_address(ipaddress.IPv4Address(bytes(octets)))

    try:
        return _is_blocked_address(ipaddress.ip_address(lowered))
    except ValueError:
        # Not an address at all. A hostname is left to the origin check, which
        # is what stops a name that resolves somewhere unwelcome: this guard
        # never resolves DNS, because a name that answers differently on the
        # second lookup would make the check theatre.
        return False


def assert_safe_auth_url(
    url: str, issuer_url: str, options: SsrfGuardOptions | None = None
) -> None:
    """Refuse a server-chosen URL that is not safe to fetch.

    Three checks, in order: the scheme, the host, and whether the host is one
    this MCP server is entitled to send us to.
    """
    opts = options or SsrfGuardOptions()
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise McpSsrfError(url, "not a valid URL") from exc
    if not parsed.scheme:
        raise McpSsrfError(url, "not a valid URL")

    # The SCHEME is judged before the host, because the schemes worth refusing
    # loudest have no host at all: `file:///etc/passwd` rejected as "not a valid
    # URL" is right by accident and reads as a typo, when what happened is that
    # something tried to make this process read a local file.
    scheme = parsed.scheme.lower()
    if scheme != "https":
        if not opts.allow_insecure_http:
            raise McpSsrfError(
                url, f'scheme "{scheme}" is not allowed; only https is permitted'
            )
        if scheme != "http":
            # `allow_insecure_http` opts into http, and into nothing else --
            # `file:` and `gopher:` are exactly the schemes an SSRF wants.
            raise McpSsrfError(
                url,
                f'scheme "{scheme}" is not allowed; only https, or http with '
                "allow_insecure_http, is permitted",
            )

    if not parsed.hostname:
        raise McpSsrfError(url, "no host")
    hostname = parsed.hostname.lower()
    blocked = is_blocked_host(hostname)
    if blocked and not opts.allow_loopback:
        raise McpSsrfError(
            url, f'host "{hostname}" is a loopback, link-local, private or reserved address'
        )
    # In explicit local-dev mode the authorization server will not share an
    # origin with the MCP server, so the origin check is skipped for exactly
    # the hosts `allow_loopback` was turned on for -- and no others.
    if blocked and opts.allow_loopback:
        return

    issuer_host = (urlsplit(issuer_url).hostname or "").lower()
    allowed = {h.lower() for h in opts.allowed_hosts}
    if allowed:
        if hostname not in allowed and hostname != issuer_host:
            raise McpSsrfError(
                url,
                f'host "{hostname}" is not in allowed_hosts and does not match the '
                f'MCP server host "{issuer_host}"',
            )
        return
    if hostname != issuer_host:
        raise McpSsrfError(
            url,
            f'host "{hostname}" differs from the MCP server host "{issuer_host}"; '
            "pass allowed_hosts to permit a separate authorization server",
        )


__all__ = [
    "LOOPBACK_HOSTNAMES",
    "MAX_IPV4_INT",
    "McpSsrfError",
    "SsrfGuardOptions",
    "assert_safe_auth_url",
    "is_blocked_host",
    "parse_canonical_ipv4",
]
