"""Join a base URL and a path without destroying the base's query string.

Transposed from `unified-library-ts/src/llm/join-url.ts`.

`base_url() + path` is correct for every base that is only a host, which is every
base this library ships. It is wrong for the one shape callers configure by hand:
an Azure-style endpoint carrying a query string.

    "https://x.openai.azure.com/openai?api-version=2026-05-01" + "/v1/responses"
    -> "https://x.openai.azure.com/openai?api-version=2026-05-01/v1/responses"

The path has become part of the `api-version` VALUE. The request goes to the base
path with a nonsense version, and the error that comes back is about the version,
not about the URL -- so the one clue points at the wrong thing.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode


def join_url(base: str, path: str) -> str:
    """`base` and `path` joined, with the base's query kept at the end.

    A trailing slash on the base and a leading one on the path collapse to one:
    `//` is a different path to a strict router, and the two halves come from
    different places (our adapter's constant and the caller's config), so neither
    can know what the other ended with.

    A fragment is dropped rather than carried. `#x` has no meaning to an HTTP
    server, it is never sent, and keeping it in the middle of a URL we then append
    to would move it somewhere it means even less.
    """
    without_hash = base.split("#", 1)[0]
    head, _, query = without_hash.partition("?")
    suffix = f"?{query}" if query else ""
    if not path:
        return head + suffix
    if head.endswith("/") and path.startswith("/"):
        return head + path[1:] + suffix
    return head + path + suffix


def ws_url(base: str, path: str, params: dict[str, str | None] | None = None) -> str:
    """A WebSocket URL from an http(s) base, a path and query parameters.

    Three things go wrong when this is spelled out at the call site, and
    Azure-style endpoints hit all three at once
    (`wss://<res>.openai.azure.com/openai/realtime?api-version=...`):

    - the scheme has to change, `https` -> `wss`;
    - the path has to land before the base's query, not inside it (`join_url`);
    - the extra parameters have to MERGE with the base's query rather than start a
      second one -- `?api-version=x?model=y` is one parameter called
      `api-version` whose value ends in `?model=y`.

    A parameter whose value is None is dropped rather than sent empty: a provider
    that validates its query rejects `model=` differently from an absent `model`,
    and the absent one is what "not specified" means.
    """
    joined = join_url(base, path)
    if joined.startswith("http"):
        joined = "ws" + joined[len("http") :]
    head, _, existing = joined.partition("?")
    pairs = parse_qsl(existing, keep_blank_values=True)
    for key, value in (params or {}).items():
        if value is None:
            continue
        pairs = [(k, v) for k, v in pairs if k != key] + [(key, value)]
    query = urlencode(pairs)
    return f"{head}?{query}" if query else head
