"""Shared test setup.

One fixture, and it exists because of a bug this suite actually had: `Engine`
registers itself as a process-wide default unless told not to, and a test that
left one registered changed what a LATER test's `LLM()` reached for. The suite
passed in file order and failed in any other -- the worst shape of flake, because
it looks like the second test is broken.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from combycode_llm_sdk.helpers.engine import clear_default_engine


@pytest.fixture(autouse=True)
def _no_leaked_default_engine() -> Iterator[None]:
    """No test starts or ends with someone else's default engine."""
    clear_default_engine()
    yield
    clear_default_engine()
