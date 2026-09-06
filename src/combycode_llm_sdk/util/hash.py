"""FNV-1a, 32-bit. Deterministic, synchronous and dependency-free -- used where a
stable short id has to be derived from content rather than from a clock.

Not cryptographic. Collisions are acceptable for naming and bucketing; do not use
it for integrity or security.

Transposed from `unified-library-ts/src/util/hash.ts`.

One JS/Python semantic bridge, and it reaches the wire. TypeScript hashes
`input.charCodeAt(i)` -- UTF-16 code UNITS. Python's `for ch in s` yields code
POINTS, so any character above U+FFFF (emoji, most CJK extensions) would hash to
a different number under the obvious Python spelling, and `xaiBatchName` would
name the same batch differently from the TypeScript SDK. Encoding to UTF-16-LE
and reading 16-bit units reproduces `charCodeAt` exactly, surrogate pairs
included.
"""

from __future__ import annotations

_OFFSET_BASIS = 0x811C9DC5
_PRIME = 0x01000193
_MASK = 0xFFFFFFFF


def fnv1a32(value: str) -> int:
    """The 32-bit FNV-1a hash of `value`, unsigned.

    `Math.imul(h, prime) >>> 0` is a multiply truncated to 32 bits and then read
    unsigned; `& 0xFFFFFFFF` is the same operation. The XOR's sign in JavaScript
    is irrelevant because only the low 32 bits survive the multiply.
    """
    h = _OFFSET_BASIS
    units = value.encode("utf-16-le", "surrogatepass")
    for i in range(0, len(units), 2):
        code = units[i] | (units[i + 1] << 8)
        h = ((h ^ code) * _PRIME) & _MASK
    return h


def fnv1a32_hex(value: str) -> str:
    """Same value as 8 lowercase hex characters."""
    return format(fnv1a32(value), "08x")


__all__ = ["fnv1a32", "fnv1a32_hex"]
