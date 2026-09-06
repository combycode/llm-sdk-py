"""Default-deny, first match wins, and the deny actually stops the call.

A policy is an ordered list of rules over `(source, target, action)`: who is
asking, what they want to touch, what they want to do to it. The first rule that
matches decides, and a triple no rule matched is DENIED.

Default-deny is the whole design. A policy that allowed what it had not thought
about would hand an agent the filesystem the first time someone added a tool the
rule list predates -- and the rules are written by hand, so that is every new
tool. The failure is silent in the direction that costs something: a rule that
wrongly refuses is reported the first time anyone runs it, a rule that wrongly
allows is found afterwards.

`ask` is a third effect, not a soft deny. It means the run stops and a human
answers, which is a different program flow and not something a boolean carries.

Deterministic: no network. The tools record what they did, so a rule that failed
to stop one is visible here rather than assumed.
"""

from _check import check, report

from combycode_llm_sdk import PermissionPolicy, Rule, tool
from combycode_llm_sdk.permissions import fs_glob
from combycode_llm_sdk.tool_catalog import (
    ALL_TOOLS,
    AgentScope,
    PermissionDenied,
    TargetDeclaration,
    ToolCatalog,
)

AGENT = "report_writer"
NOTES = "workspace/notes.md"
SECRETS = "workspace/.env"

WRITTEN: list[str] = []


@tool
def write_note(path: str, content: str) -> str:
    """Write a note into the workspace."""
    WRITTEN.append(path)
    return f"wrote {path}"


@tool
def write_env(path: str, content: str) -> str:
    """Write a value into the workspace environment file."""
    WRITTEN.append(path)
    return f"wrote {path}"


POLICY = PermissionPolicy(
    [
        Rule(effect="deny", target=fs_glob("**/.env"), reason="credentials are never writable"),
        Rule(effect="ask", action="delete", reason="a human confirms deletions"),
        Rule(effect="allow", action="write", target=fs_glob("workspace/**")),
    ]
)


def decide(path: str, action: str = "write"):
    return POLICY.check(AGENT, {"kind": "fs", "path": path}, action)


allowed = decide(NOTES)
check(allowed.allow is True, "a write inside the workspace is what rule 2 is for")

# Rule 2 would allow this path too. Rule 0 is first, so it never gets asked --
# which is why the deny is written above the allow and not below it.
denied = decide(SECRETS)
check(denied.allow is False, "the deny must win over a later allow")
check(denied.matched_rule == 0, f"the first matching rule decides, matched {denied.matched_rule}")
check(denied.reason == "credentials are never writable", "and it says why")

unruled = decide(NOTES, action="read")
check(unruled.allow is False, "no rule mentions reading, so reading is denied")
check(unruled.matched_rule is None, "denied by the default, not by a rule")
check("default deny" in unruled.reason, f"and it says so, got {unruled.reason!r}")

pending = decide(NOTES, action="delete")
check(pending.allow is False and pending.ask is True, "ask is neither an allow nor a deny")

# Appended, never prepended: a caller handed a policy cannot widen it by putting
# an allow-all in front of the deny that was already there.
widened = POLICY.with_additional([Rule(effect="allow", target=fs_glob("**"), action="write")])
check(
    widened.check(AGENT, {"kind": "fs", "path": SECRETS}, "write").allow is False,
    "an appended allow must not override the deny already in the policy",
)
check(
    widened.check(AGENT, {"kind": "fs", "path": "tmp/scratch"}, "write").allow is True,
    "but it does widen what no earlier rule ruled on",
)

# The policy is not advice: the catalog rules on every target a tool declared
# BEFORE the function runs.
catalog = ToolCatalog(policy=POLICY)
catalog.register(
    write_note,
    declared_targets=[TargetDeclaration(kind="fs", value={"path": NOTES})],
    declared_actions=["write"],
)
catalog.register(
    write_env,
    declared_targets=[TargetDeclaration(kind="fs", value={"path": SECRETS})],
    declared_actions=["write"],
)
catalog.set_agent_scope(AGENT, AgentScope(tool_names=ALL_TOOLS))

catalog.call("write_note", source=AGENT, arguments={"path": NOTES, "content": "hello"})

refusal = None
try:
    catalog.call("write_env", source=AGENT, arguments={"path": SECRETS, "content": "KEY=1"})
except PermissionDenied as exc:
    refusal = exc

check(refusal is not None, "the denied tool call must raise, not return a failure to the model")
check(WRITTEN == [NOTES], f"the denied tool must not have run, but wrote {WRITTEN}")

report(allowed=WRITTEN, refused=str(refusal), rules=len(POLICY), widened_rules=len(widened))
