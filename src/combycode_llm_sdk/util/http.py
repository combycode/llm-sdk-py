"""Small HTTP header helpers shared across the network and server layers.

PARTIAL transposition of `unified-library-ts/src/util/http.ts`. That file also
carries `headers_to_record` (from a WHATWG `Headers`), `is_stream_body`,
`any_signal` (combining AbortSignals) and `parse_response_body` -- all four are
about the fetch implementation itself, and they land with the network engine that
owns one. What is here is what a CONSUMER of a response needs.
"""

from __future__ import annotations

from collections.abc import Mapping


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
