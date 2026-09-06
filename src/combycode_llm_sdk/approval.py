"""Human in the loop: nothing runs until someone says yes, and every default is no.

`ApprovalGate` sits between the tool call the model asked for and the function
that would carry it out. A model that can delete a row, spend money or send an
email needs that seam, and the seam is only worth having if it fails CLOSED:

    a gate with no approver              denies
    a gate with no `requires` predicate  asks about every call
    an approver that raises              denies
    an approver that forgets to `return` denies

A gate whose failure mode is "allow" is not a gate, it is a delay. It passes
every test with a working approver and opens the door on the one run where the
approval channel is down.

The decision is read through `result_for`, never by hand. It returns None for
exactly one case -- a plain approve, no override -- which means "run it". A
caller who instead checks `decision.decision == "approve"` looks right and is
wrong on the override case: an approve carrying an `override_result` is an
answer to feed back to the model INSTEAD of running the tool, and the
hand-rolled check runs it anyway. That is a deleted row nobody authorised.

The TypeScript half of this lives in `agent/approval-types.ts` as a callback the
loop awaits; the gate is the Python shape, because a suspend/resume across a
process restart is what the reviewed example asks for and a bare callback cannot
express it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .results import ToolCall, ToolResult
from .wire.interpreter import js_json

#: What the model is told when the gate had nobody to ask. It says the call was
#: not refused on its merits, so the model stops rephrasing and reports upward.
NO_APPROVER_NOTE = (
    "this tool call needs human approval and no approver is configured, so it was not run"
)

#: The three answers an approver may give.
APPROVE = "approve"
DENY = "deny"
SKIP = "skip"


@dataclass(frozen=True)
class ApprovalRequest:
    """What the approver is shown.

    The arguments are included because an approver shown only a tool NAME cannot
    judge: `delete_row` is routine and `delete_row(table="invoices")` may not be.
    """

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    step: int = 0
    run_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    """What the approver answered.

    `decision` is positional and required: a decision whose verdict defaulted
    would default to something, and every safe default here is `deny` -- which
    is better expressed by the gate refusing than by this pretending to be an
    answer nobody gave.
    """

    decision: str
    #: Feed this back to the model INSTEAD of running the tool. An approve that
    #: carries one is not permission to run.
    override_result: Any = None
    #: Why, in words the model can act on -- it is what stops it asking again.
    note: str | None = None

    @property
    def approved(self) -> bool:
        """True only for a plain approve that runs the tool.

        An approve carrying an override is not one: the human answered the
        question themselves.
        """
        return self.decision == APPROVE and self.override_result is None


@dataclass(frozen=True)
class PendingToolCall:
    """A call suspended awaiting a human, in a form that survives a restart."""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    step: int = 0
    requested_at: float = 0.0
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """TypeScript key names, so either runtime can read the checkpoint.

        A checkpoint written by one and unreadable by the other would make the
        two runtimes unable to hand work to each other, which is most of the
        reason to have a serialisable form at all.
        """
        return {
            "callId": self.call_id,
            "toolName": self.tool_name,
            "arguments": dict(self.arguments),
            "step": self.step,
            "requestedAt": self.requested_at,
            "runId": self.run_id,
        }

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> PendingToolCall:
        return PendingToolCall(
            call_id=str(raw.get("callId") or ""),
            tool_name=str(raw.get("toolName") or ""),
            arguments=dict(raw.get("arguments") or {}),
            step=int(raw.get("step") or 0),
            requested_at=float(raw.get("requestedAt") or 0.0),
            run_id=raw.get("runId"),
        )

    def matches(self, call: ToolCall) -> bool:
        """Whether this record is about that call.

        The id ALONE is not enough. Call ids are per-turn and providers reuse
        them -- turn 2 synthesises `call_0`, `call_1` again -- so a decision
        keyed on the id would approve a different tool that happened to inherit
        the name. The name and arguments are what make it the same call.
        """
        return (
            self.call_id == call.id
            and self.tool_name == call.name
            and js_json(dict(self.arguments)) == js_json(dict(call.arguments or {}))
        )


Approver = Callable[[ApprovalRequest], "ApprovalDecision | None"]


class ApprovalGate:
    """Ask a human, and refuse whenever the answer is not clearly yes."""

    def __init__(
        self,
        approver: Approver | None = None,
        *,
        requires: Callable[[ToolCall], bool] | None = None,
    ) -> None:
        self._approver = approver
        self._requires = requires
        self._pending: dict[str, PendingToolCall] = {}
        self._answers: list[tuple[PendingToolCall, ApprovalDecision]] = []

    def requires_approval(self, call: ToolCall) -> bool:
        """Whether this call has to be asked about.

        With no predicate, everything is gated. A predicate that was meant to be
        passed and was not would otherwise leave the gate wide open while
        looking configured.
        """
        return True if self._requires is None else bool(self._requires(call))

    def check(
        self,
        call: ToolCall,
        *,
        step: int = 0,
        run_id: str | None = None,
        reason: str | None = None,
    ) -> ApprovalDecision:
        """The verdict for one call. Never raises; every failure is a denial."""
        if not self.requires_approval(call):
            return ApprovalDecision(APPROVE)

        answered = self._take_answer(call)
        if answered is not None:
            return answered

        if self._approver is None:
            return ApprovalDecision(DENY, note=NO_APPROVER_NOTE)

        record = PendingToolCall(
            call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments or {}),
            step=step,
            requested_at=time.time() * 1000,
            run_id=run_id,
        )
        # Recorded BEFORE the approver runs and cleared after, so an approver
        # that suspends the run (by raising) can snapshot what is pending. That
        # window is the only moment the record exists.
        self._pending[call.id] = record
        try:
            answer = self._approver(
                ApprovalRequest(
                    call_id=call.id,
                    tool_name=call.name,
                    arguments=dict(call.arguments or {}),
                    step=step,
                    run_id=run_id,
                    reason=reason,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- an approver that failed approved nothing
            return ApprovalDecision(
                DENY, note=f"the approval channel failed ({type(exc).__name__}: {exc})"
            )
        finally:
            self._pending.pop(call.id, None)

        if not isinstance(answer, ApprovalDecision):
            # A missing `return` is the usual way this happens, and it is not an
            # approval.
            return ApprovalDecision(
                DENY, note="the approver returned no decision, which is not an approval"
            )
        return answer

    # -- suspend and resume --------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """The calls awaiting an answer right now, ready to checkpoint."""
        return [record.to_dict() for record in self._pending.values()]

    def restore(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Take back a checkpoint written before the process stopped."""
        for raw in records:
            record = PendingToolCall.from_dict(raw)
            self._pending[record.call_id] = record

    def resume_with(self, call_id: str, decision: ApprovalDecision) -> bool:
        """Feed in the answer a human gave while the process was away.

        Bound to the RESTORED record rather than to the id, so the answer can
        only ever apply to the call it was actually about.
        """
        record = self._pending.pop(call_id, None)
        if record is None:
            return False
        self._answers.append((record, decision))
        return True

    @property
    def pending(self) -> Sequence[PendingToolCall]:
        return tuple(self._pending.values())

    def _take_answer(self, call: ToolCall) -> ApprovalDecision | None:
        """A fed-in answer for this exact call, consumed on use.

        Consumed because a human approved ONE call: leaving it in place would
        turn one yes into a standing permission for every later call that looked
        like it.
        """
        for index, (record, decision) in enumerate(self._answers):
            if record.matches(call):
                del self._answers[index]
                return decision
        return None


def result_for(call: ToolCall, decision: ApprovalDecision) -> ToolResult | None:
    """What to send back INSTEAD of running the tool, or None to run it.

    None for exactly one case -- a plain approve with no override. Read the
    decision this way rather than by hand: `decision.decision == "approve"` is
    true for an override too, and running the tool then is the failure this
    function exists to prevent.
    """
    if decision.approved:
        return None
    if decision.decision == APPROVE:
        # An approve carrying an override: the human answered instead of the tool.
        return ToolResult.for_call(call, decision.override_result)
    note = decision.note or (
        "this tool call was skipped" if decision.decision == SKIP else "this tool call was denied"
    )
    return ToolResult.for_call(call, note, is_error=decision.decision == DENY)


__all__ = [
    "APPROVE",
    "DENY",
    "NO_APPROVER_NOTE",
    "SKIP",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRequest",
    "Approver",
    "PendingToolCall",
    "result_for",
]
