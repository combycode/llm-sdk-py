"""Compact duration strings.

Transposed from `unified-library-ts/src/util/duration.ts`.

A reusable pattern across the SDK (server-state retention, cache TTLs, timeouts).
Format is `<number><unit>` where unit is one of s/m/h/d/w: `"30s"`, `"5m"`,
`"72h"`, `"3d"`, `"2w"`.

`parse_duration("72h")` -> 259_200_000 (ms).
"""

from __future__ import annotations

import re

_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}

_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d|w)$", re.IGNORECASE)


def parse_duration(value: str) -> float:
    """Parse a duration string to milliseconds. Raises on malformed input."""
    match = _PATTERN.match(value.strip())
    if not match:
        raise ValueError(
            f'Invalid duration "{value}" -- expected e.g. "30s", "72h", "3d", "2w".'
        )
    return float(match.group(1)) * _UNIT_MS[match.group(2).lower()]


def parse_duration_or_none(value: str | None) -> float | None:
    """Parse to milliseconds, or None for None/empty -- NOT for malformed.

    Malformed input still raises: an unreadable TTL is a bug in a config, and
    swallowing it here would silently disable the expiry it was meant to set.
    """
    if value is None or value == "":
        return None
    return parse_duration(value)


__all__ = ["parse_duration", "parse_duration_or_none"]
