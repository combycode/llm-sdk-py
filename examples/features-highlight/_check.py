"""Assertions for self-verifying examples.

These examples are not illustrations -- they run deterministically (stub
transport, no keys, no network) and FAIL loudly when the behaviour they document
stops being true. That is what lets the gate execute them as tests.
"""

from __future__ import annotations

import json
import sys
from typing import Any

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _failures.append(message)
        print(f"FAIL: {message}", file=sys.stderr)


def report(**fields: Any) -> None:
    """Print the one-line JSON result and exit non-zero if anything failed."""
    print(json.dumps(fields, default=str))
    if _failures:
        raise SystemExit(1)
