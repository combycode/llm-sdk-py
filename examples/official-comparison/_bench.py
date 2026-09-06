"""Shared timing + output shape, so every example's last line is comparable.

The TypeScript corpus inlines `performance.now()` in each file. Here it is one
helper: these examples ARE the API design under review, and repeating six lines
of timing in thirty files would bury the one line that actually matters.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any


def model() -> str:
    return os.environ["LLM_MODEL"]


def api_key() -> str | None:
    return os.environ.get("LLM_API_KEY")


def bench(fn: Callable[[], Any]) -> None:
    t0 = time.perf_counter()
    result = fn()
    ms = round((time.perf_counter() - t0) * 1000)
    print(json.dumps({"result": str(result), "ms": ms}))
