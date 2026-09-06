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

**And the decision is read through `result_for`, never by hand.** It returns
None for exactly one case -- a plain approve, no override -- which means "run
it". A caller who instead checks `decision.decision == "approve"` looks right
and is wrong on the case below: an approve carrying an `override_result` is an
answer to feed back to the model INSTEAD of running the tool, and the hand-rolled
check runs it anyway. That is a deleted row nobody authorised.

Deterministic: no provider, no key, no network. The approvers are plain
functions, and the tool records whether it ran.
"""

from _check import check, report

from combycode_llm_sdk import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    PendingToolCall,
    ToolCall,
    result_for,
)
from combycode_llm_sdk.approval import NO_APPROVER_NOTE

DELETE = ToolCall(id="call_1", name="delete_row", arguments={"table": "invoices", "id": 7})
READ = ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"})
HOLD_NOTE = "invoice 7 is on legal hold"
CANNED = "nothing was deleted; invoice 7 is on legal hold"

executed: list[str] = []
seen: list[ApprovalRequest] = []


def run_tool(call: ToolCall) -> str:
    """The side effect that must not happen without a yes."""
    executed.append(call.name)
    return f"{call.name} executed"


def guarded(gate: ApprovalGate, call: ToolCall) -> str:
    """Wired as the library intends: `result_for` decides, not the caller."""
    decision = gate.check(call, step=1, run_id="run_1")
    blocked = result_for(call, decision)
    return str(blocked.content) if blocked is not None else run_tool(call)


def naive(gate: ApprovalGate, call: ToolCall) -> str:
    """The tempting shortcut: read the verdict, ignore the rest of the answer."""
    decision = gate.check(call, step=1, run_id="run_1")
    return run_tool(call) if decision.decision == "approve" else "blocked"


def attempt(gate: ApprovalGate, call: ToolCall, *, by_hand: bool = False) -> tuple[list[str], str]:
    """One gated call, and what actually ran because of it."""
    executed.clear()
    answer = naive(gate, call) if by_hand else guarded(gate, call)
    return list(executed), answer


def approves(request: ApprovalRequest) -> ApprovalDecision:
    seen.append(request)
    return ApprovalDecision("approve")


def approves_with_a_canned_answer(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision("approve", override_result=CANNED)


def unreachable(request: ApprovalRequest) -> ApprovalDecision:
    raise ConnectionError("the approval channel is down")


def forgot_to_return(request: ApprovalRequest) -> ApprovalDecision:
    """The usual way it happens: the decision is built and never handed back."""
    ApprovalDecision("approve")


def refuses(request: ApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision("deny", note=HOLD_NOTE)


# -- the closed defaults -----------------------------------------------------

without_approver, answer = attempt(ApprovalGate(), DELETE)
check(without_approver == [], "a gate with no approver let the tool run")
check(NO_APPROVER_NOTE in answer, f"the model must be told why it was refused, got {answer!r}")

on_raise, _ = attempt(ApprovalGate(unreachable), DELETE)
check(on_raise == [], "an approver that raised approved nothing; the tool must not run")

on_missing_return, _ = attempt(ApprovalGate(forgot_to_return), DELETE)
check(on_missing_return == [], "a missing `return` is not an approval")

# With no predicate every call is gated, including the harmless-looking one --
# a predicate that was meant to be passed and was not would otherwise leave the
# gate wide open while looking configured.
gate = ApprovalGate(approves)
check(gate.requires_approval(READ) is True, "an unconfigured gate must gate everything")

ran, answer = attempt(gate, DELETE)
check(ran == ["delete_row"], "an approved call must actually run")
check(answer == "delete_row executed", "the tool's own result goes back, not the gate's")
check([r.tool_name for r in seen] == ["delete_row"], "the approver is asked about the real call")
check(seen[0].arguments == DELETE.arguments, "an approver shown no arguments cannot judge")
check(seen[0].run_id == "run_1", "and cannot tell which run it is answering for")

# A predicate can wave a class of calls through without asking anyone.
seen.clear()
narrow = ApprovalGate(approves, requires=lambda call: call.name != "read_file")
ran, _ = attempt(narrow, READ)
check(ran == ["read_file"], "a call the predicate cleared should not be blocked")
check(seen == [], "nobody should be woken up to approve a file read")

# -- a denial the model can learn from ---------------------------------------

ran, answer = attempt(ApprovalGate(refuses), DELETE)
check(ran == [], "a denied call must not run")
check(answer == HOLD_NOTE, "the note is what the model sees, so it stops asking")

# -- the case the hand-rolled check gets wrong -------------------------------

override = ApprovalGate(approves_with_a_canned_answer)

ran_via_result_for, answer = attempt(override, DELETE)
check(ran_via_result_for == [], "an approve carrying an override_result must NOT run the tool")
check(answer == CANNED, "the override is what goes back in place of the tool's result")

ran_by_hand, _ = attempt(override, DELETE, by_hand=True)
check(
    ran_by_hand == ["delete_row"],
    "the hand-rolled check no longer runs an overridden approve -- if reading the field "
    "directly is now safe, this example's warning is stale and should go",
)

# -- suspending for a human, and resuming without widening the answer ---------
#
# A real approver is a person, so the run stops. The approver below snapshots
# the gate while the call is pending -- `check()` clears the record on the way
# out, so during the ask is the only moment it exists -- and raises to suspend.

captured: list[dict] = []


def suspends(request: ApprovalRequest) -> ApprovalDecision:
    captured.extend(suspending.snapshot())
    raise RuntimeError("waiting for a human")


suspending = ApprovalGate(suspends)
ran, _ = attempt(suspending, DELETE)
check(ran == [], "a call that suspended must not have run")
check(len(captured) == 1, "the pending call is what gets checkpointed")

waiting = PendingToolCall.from_dict(captured[0])
check(waiting.tool_name == "delete_row", "the checkpoint knows which tool was asked about")
check(waiting.arguments == {"table": "invoices", "id": 7}, "and with which arguments")
check(captured[0]["callId"] == "call_1", "TypeScript key names, so either runtime reads it")

# The human said yes. A NEW gate -- the process restarted -- picks the answer up.
resumed = ApprovalGate(refuses)  # the approver would deny; the fed answer wins
resumed.restore(captured)
resumed.resume_with(waiting.call_id, ApprovalDecision("approve"))
ran, _ = attempt(resumed, DELETE)
check(ran == ["delete_row"], "the human's yes should carry across the restart")

# The trap: `call_1` is not a unique name. Turn 2 synthesises `call_0`, `call_1`
# again, so a decision keyed on the id alone would approve a DIFFERENT call.
reused = ApprovalGate(refuses)
reused.restore(captured)
reused.resume_with(waiting.call_id, ApprovalDecision("approve"))
imposter = ToolCall(id="call_1", name="wire_funds", arguments={"usd": 90000})
ran, _ = attempt(reused, imposter)
check(ran == [], "an answer about delete_row must not approve wire_funds sharing its id")

# And the answer is still there for the call it was actually about.
ran, _ = attempt(reused, DELETE)
check(ran == ["delete_row"], "the kept decision still belongs to its own call")

report(
    ran_without_approver=bool(without_approver),
    ran_when_the_approver_raised=bool(on_raise),
    ran_when_the_approver_forgot_to_return=bool(on_missing_return),
    override_ran_via_result_for=bool(ran_via_result_for),
    override_ran_via_hand_rolled_check=bool(ran_by_hand),
    suspended_tool=waiting.tool_name,
    resumed_after_restart=True,
    id_reuse_approved_nothing=True,
)
