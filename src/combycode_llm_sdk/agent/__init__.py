"""combycode_llm_sdk.agent — stub.

STUB — generated from the reviewed examples, which are the API contract. Every
name here exists so the surface can be checked from day one; every one raises
NotImplementedError, so nothing can pass by accident.

Names are replaced, module by module, as each is transposed from the TypeScript.
See PORTING.md.
"""

from __future__ import annotations

from typing import Any


class GuardrailError(RuntimeError):
    """A guardrail stopped the run.

    Its own class rather than a bare RuntimeError, so an application can catch
    "a rule refused this" separately from "something broke" -- the two call for
    completely different handling, and a caller that cannot tell them apart ends
    up retrying a refusal.
    """

    def __init__(self, message: str, *, result: Any = None, guardrail: str = "") -> None:
        super().__init__(message)
        #: Whatever the guard was judging, kept so the refusal is actionable.
        self.result = result
        self.guardrail = guardrail
