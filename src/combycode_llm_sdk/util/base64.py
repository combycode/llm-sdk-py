"""Base64 <-> bytes helpers -- the single source of truth across the SDK.

Transposed from `unified-library-ts/src/util/base64.ts`.

The TypeScript encoder has two branches because the PLATFORM decides which one
exists: Node/Bun's `Buffer` when present, a browser-safe `btoa` otherwise. Python
has one `base64` module everywhere, so the branch has nothing to select between.
That removes a path rather than changing a behaviour, and it is why
`base64-http.test.ts:21` -- which asserts the two branches agree -- is skipped
rather than translated.
"""

from __future__ import annotations

import base64 as _base64

#: Uint8Array in TypeScript. `bytes` is the direct equivalent; every caller here
#: only reads.
Bytes = bytes


def base64_to_bytes(b64: str) -> bytes:
    """`atob`, as raw bytes.

    Three things `atob` does that Python's default does NOT, all of which a
    caller depends on:

      - it THROWS on a character outside the base64 alphabet. `b64decode`
        silently discards them, so `base64_to_bytes("!!!!")` would return `b""`
        instead of raising -- and `normalize_image_source` relies on the raise to
        fall back to the declared mime (source-image.ts:25-32).
      - it tolerates missing padding.
      - it ignores ASCII whitespace.
    """
    stripped = "".join(b64.split())
    return _base64.b64decode(stripped + "=" * (-len(stripped) % 4), validate=True)


def bytes_to_base64(data: Bytes) -> str:
    """Encode bytes as a base64 string."""
    return _base64.b64encode(bytes(data)).decode("ascii")


def base64_to_utf8(b64: str) -> str:
    """Decode a base64 payload to a UTF-8 string.

    `new TextDecoder()` is non-fatal by default: an invalid sequence becomes
    U+FFFD rather than an exception, so `errors="replace"` is the transposition
    and `errors="strict"` would be a new failure mode.
    """
    return base64_to_bytes(b64).decode("utf-8", errors="replace")


__all__ = ["base64_to_bytes", "base64_to_utf8", "bytes_to_base64"]
