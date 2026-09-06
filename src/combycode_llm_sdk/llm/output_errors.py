"""Typed errors for abnormal agent-run / structured-output outcomes.

Transposed from `unified-library-ts/src/llm/output-errors.ts`.

`AgentRunError` is the shared base so callers can differentiate the failure
reason (`error.reason`, or `isinstance`). Today only `invalid_final_output` is an
exception; `max_steps` / `model_refusal` remain returned results (differentiated
by `finishReason` / `AgentRunReport.reason`). A future run-error-handler config
can add `MaxStepsError` / `ModelRefusalError` under this base without a break.
"""

from __future__ import annotations


class AgentRunError(Exception):
    """`class AgentRunError extends Error` (output-errors.ts:10)."""

    #: Machine-readable failure reason (e.g. `'invalid_final_output'`).
    reason: str

    def __init__(self, reason: str, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        # `new Error(message, { cause })`. Python spells the same thing
        # `raise ... from cause`, but the constructor must work standalone too,
        # so the cause is attached here and `raise X from e` stays available.
        if cause is not None:
            self.__cause__ = cause


class InvalidFinalOutputError(AgentRunError):
    """The model's final output could not be parsed against the requested schema.

    Carries the raw text so a caller can inspect, log, or retry.
    """

    #: The raw model output that failed to parse.
    raw_text: str

    def __init__(self, raw_text: str, cause: BaseException | None = None) -> None:
        detail = str(cause) if isinstance(cause, Exception) else "parse failed"
        super().__init__(
            "invalid_final_output",
            f"Model final output did not match the requested schema: {detail}",
            cause,
        )
        self.raw_text = raw_text


__all__ = ["AgentRunError", "InvalidFinalOutputError"]
