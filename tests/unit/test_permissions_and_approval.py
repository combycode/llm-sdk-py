"""Default-deny, first match wins, and a gate that fails closed.

Both subsystems are about refusing correctly, and the failures worth testing are
the ones that look like success: an allow that should have been a deny, and a
gate that opens on the one run where the approval channel is down.
"""

from __future__ import annotations

from typing import Any

import pytest

from combycode_llm_sdk import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    PendingToolCall,
    PermissionPolicy,
    Rule,
    ToolCall,
    ToolCatalog,
    result_for,
    tool,
)
from combycode_llm_sdk.approval import NO_APPROVER_NOTE
from combycode_llm_sdk.permissions import (
    any_of_kind,
    fs_glob,
    glob_to_regex,
    shell_glob,
    url_pattern,
)
from combycode_llm_sdk.tool_catalog import (
    ALL_TOOLS,
    DEFAULT_SEARCH_LIMIT,
    AgentScope,
    NoToolAccess,
    PermissionDenied,
    TargetDeclaration,
    ToolNotFound,
    ToolRegistrationError,
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


@tool
def refund_order(order_id: str) -> str:
    """Refund an order that has already shipped."""
    return f"refunded {order_id}"


class TestGlobs:
    def test_a_star_stops_at_a_separator(self) -> None:
        # What makes `workspace/*` mean one level.
        assert glob_to_regex("workspace/*").match("workspace/a")
        assert not glob_to_regex("workspace/*").match("workspace/a/b")

    def test_a_double_star_crosses_them(self) -> None:
        assert glob_to_regex("workspace/**").match("workspace/a/b/c")

    def test_a_double_star_matches_no_segments_at_all(self) -> None:
        # `**/.env` has to match a bare `.env`, or a rule written for "anywhere"
        # misses the root.
        assert glob_to_regex("**/.env").match(".env")
        assert glob_to_regex("**/.env").match("deep/nested/.env")

    def test_a_dot_is_literal(self) -> None:
        assert not glob_to_regex("a.txt").match("axtxt")

    def test_loose_lets_a_star_cross_anything(self) -> None:
        # A command is not a path, so `/` is not a boundary in one.
        assert shell_glob("git *")({"kind": "shell", "command": "git log --oneline"})
        assert url_pattern("https://*.example.com/*")(
            {"kind": "url", "url": "https://a.example.com/x/y"}
        )

    def test_a_matcher_checks_the_kind_too(self) -> None:
        # `fs_glob("**")` must not match a shell command that happens to look
        # like a path.
        assert not fs_glob("**")({"kind": "shell", "command": "rm -rf /"})

    def test_any_of_kind_ignores_the_rest(self) -> None:
        assert any_of_kind("fs", "url")({"kind": "url", "url": "x"})
        assert not any_of_kind("fs")({"kind": "shell", "command": "x"})


class TestThePolicy:
    POLICY = PermissionPolicy(
        [
            Rule(effect="deny", target=fs_glob("**/.env"), reason="credentials are never writable"),
            Rule(effect="ask", action="delete", reason="a human confirms deletions"),
            Rule(effect="allow", action="write", target=fs_glob("workspace/**")),
        ]
    )

    def decide(self, path: str, action: str = "write") -> Any:
        return self.POLICY.check("writer", {"kind": "fs", "path": path}, action)

    def test_a_matching_allow_allows(self) -> None:
        assert self.decide("workspace/notes.md").allow is True

    def test_an_earlier_deny_beats_a_later_allow(self) -> None:
        # Which is why the deny is written above the allow and not below it.
        denied = self.decide("workspace/.env")
        assert denied.allow is False
        assert denied.matched_rule == 0
        assert denied.reason == "credentials are never writable"

    def test_nothing_matching_is_denied_by_default(self) -> None:
        # A policy that allowed what it had not thought about would hand an
        # agent the filesystem the first time someone added a tool.
        unruled = self.decide("workspace/notes.md", action="read")
        assert unruled.allow is False
        assert unruled.matched_rule is None
        assert "default deny" in (unruled.reason or "")

    def test_ask_is_neither_an_allow_nor_a_deny(self) -> None:
        pending = self.decide("workspace/notes.md", action="delete")
        assert pending.allow is False
        assert pending.ask is True

    def test_a_star_action_matches_anything(self) -> None:
        policy = PermissionPolicy([Rule(effect="allow", action="*")])
        assert policy.check("a", {"kind": "fs", "path": "x"}, "whatever").allow is True

    def test_a_source_list_narrows_the_rule(self) -> None:
        policy = PermissionPolicy([Rule(effect="allow", source=["alice", "bob"])])
        assert policy.check("alice", {"kind": "fs"}, "read").allow is True
        assert policy.check("mallory", {"kind": "fs"}, "read").allow is False

    def test_additional_rules_are_appended_not_prepended(self) -> None:
        # A caller handed a policy cannot widen it by putting an allow-all in
        # front of the deny that was already there.
        widened = self.POLICY.with_additional(
            [Rule(effect="allow", target=fs_glob("**"), action="write")]
        )
        assert widened.check("w", {"kind": "fs", "path": "a/.env"}, "write").allow is False
        assert widened.check("w", {"kind": "fs", "path": "tmp/x"}, "write").allow is True
        assert len(widened) == len(self.POLICY) + 1

    def test_the_original_is_not_mutated(self) -> None:
        before = len(self.POLICY)
        self.POLICY.with_additional([Rule(effect="allow")])
        assert len(self.POLICY) == before


class TestTheCatalog:
    def build(self) -> ToolCatalog:
        catalog = ToolCatalog()
        catalog.register(get_weather)
        catalog.register(refund_order)
        return catalog

    def test_a_duplicate_registration_is_refused(self) -> None:
        catalog = self.build()
        with pytest.raises(ToolRegistrationError, match="already registered"):
            catalog.register(get_weather)

    def test_a_sentence_finds_the_tool_it_describes(self) -> None:
        assert [t.name for t in self.build().search("look up the weather")] == ["get_weather"]

    def test_search_is_capped_by_default(self) -> None:
        catalog = ToolCatalog()
        for index in range(12):

            def body(order_id: str, _i: int = index) -> str:
                return ""

            body.__name__ = f"order_report_{index}"
            body.__doc__ = "Produce a report about an order."
            catalog.register(tool(body))
        assert len(catalog.search("order")) == DEFAULT_SEARCH_LIMIT
        assert len(catalog.search("order", limit=None)) > DEFAULT_SEARCH_LIMIT

    def test_a_scoped_search_never_mentions_an_out_of_scope_tool(self) -> None:
        catalog = self.build()
        catalog.set_agent_scope("support", AgentScope(tool_names=("get_weather",)))
        found = [t.name for t in catalog.search("refund an order", agent_id="support")]
        assert "refund_order" not in found

    def test_an_unscoped_search_is_the_whole_catalog(self) -> None:
        # Which is what an operator's own search wants.
        found = [t.name for t in self.build().search("refund an order")]
        assert "refund_order" in found

    def test_an_agent_with_no_scope_discovers_nothing(self) -> None:
        assert self.build().search("weather", agent_id="stranger") == []

    def test_scope_is_enforced_on_call_not_only_on_discovery(self) -> None:
        # The assertion the whole arrangement rests on: a name can be
        # hallucinated without any search at all.
        catalog = self.build()
        catalog.set_agent_scope("support", AgentScope(tool_names=("get_weather",)))
        with pytest.raises(NoToolAccess, match="not in agent scope"):
            catalog.call("refund_order", source="support", arguments={"order_id": "A1"})

    def test_an_agent_with_no_scope_cannot_call(self) -> None:
        with pytest.raises(NoToolAccess, match="no scope registered"):
            self.build().call("get_weather", source="stranger", arguments={"city": "P"})

    def test_a_tool_in_scope_runs_and_returns_its_own_value(self) -> None:
        catalog = self.build()
        catalog.set_agent_scope("support", AgentScope(tool_names=ALL_TOOLS))
        assert catalog.call("get_weather", source="support", arguments={"city": "Paris"}).output == (
            "sunny in Paris"
        )

    def test_an_unknown_tool_is_named(self) -> None:
        catalog = self.build()
        catalog.set_agent_scope("support", AgentScope(tool_names=ALL_TOOLS))
        with pytest.raises(ToolNotFound):
            catalog.call("no_such_tool", source="support")

    def test_an_external_tool_is_off_unless_said_otherwise(self) -> None:
        catalog = ToolCatalog()
        catalog.register(get_weather, category="external")
        catalog.set_agent_scope("a", AgentScope(tool_names=ALL_TOOLS))
        with pytest.raises(NoToolAccess, match="external tools disabled"):
            catalog.call("get_weather", source="a", arguments={"city": "P"})

    def test_the_policy_rules_before_the_function_runs(self) -> None:
        # A policy consulted after the write has already happened is an audit
        # log, not a permission.
        written: list[str] = []

        @tool
        def write_env(path: str) -> str:
            """Write a value into the environment file."""
            written.append(path)
            return "wrote"

        catalog = ToolCatalog(
            policy=PermissionPolicy([Rule(effect="deny", target=fs_glob("**/.env"))])
        )
        catalog.register(
            write_env,
            declared_targets=[TargetDeclaration(kind="fs", value={"path": "workspace/.env"})],
            declared_actions=["write"],
        )
        catalog.set_agent_scope("w", AgentScope(tool_names=ALL_TOOLS))
        with pytest.raises(PermissionDenied):
            catalog.call("write_env", source="w", arguments={"path": "workspace/.env"})
        assert written == []

    def test_no_policy_means_no_opinion(self) -> None:
        # A catalog used only to keep the tool block small should not need a
        # rule list to work at all.
        catalog = self.build()
        catalog.set_agent_scope("a", AgentScope(tool_names=ALL_TOOLS))
        assert catalog.call("get_weather", source="a", arguments={"city": "P"}).output


class TestTheApprovalGate:
    DELETE = ToolCall(id="call_1", name="delete_row", arguments={"table": "invoices", "id": 7})

    def test_a_gate_with_no_approver_denies(self) -> None:
        decision = ApprovalGate().check(self.DELETE)
        assert decision.approved is False
        assert NO_APPROVER_NOTE in (decision.note or "")

    def test_an_approver_that_raises_denies(self) -> None:
        def unreachable(request: ApprovalRequest) -> ApprovalDecision:
            raise ConnectionError("the approval channel is down")

        assert ApprovalGate(unreachable).check(self.DELETE).approved is False

    def test_an_approver_that_forgets_to_return_denies(self) -> None:
        # The usual way it happens: the decision is built and never handed back.
        def forgot(request: ApprovalRequest) -> Any:
            ApprovalDecision("approve")

        assert ApprovalGate(forgot).check(self.DELETE).approved is False

    def test_an_unconfigured_gate_gates_everything(self) -> None:
        # A predicate meant to be passed and not passed would otherwise leave
        # the gate wide open while looking configured.
        assert ApprovalGate(lambda r: ApprovalDecision("approve")).requires_approval(
            ToolCall(id="c", name="read_file", arguments={})
        )

    def test_a_predicate_can_wave_a_class_of_calls_through(self) -> None:
        asked: list[str] = []

        def approver(request: ApprovalRequest) -> ApprovalDecision:
            asked.append(request.tool_name)
            return ApprovalDecision("approve")

        gate = ApprovalGate(approver, requires=lambda call: call.name != "read_file")
        assert gate.check(ToolCall(id="c", name="read_file", arguments={})).approved is True
        assert asked == []

    def test_the_approver_sees_the_arguments_and_the_run(self) -> None:
        # An approver shown only a tool name cannot judge.
        seen: list[ApprovalRequest] = []

        def approver(request: ApprovalRequest) -> ApprovalDecision:
            seen.append(request)
            return ApprovalDecision("approve")

        ApprovalGate(approver).check(self.DELETE, step=1, run_id="run_1")
        assert seen[0].tool_name == "delete_row"
        assert seen[0].arguments == self.DELETE.arguments
        assert seen[0].run_id == "run_1"


class TestReadingTheDecision:
    CALL = ToolCall(id="c1", name="delete_row", arguments={})

    def test_a_plain_approve_means_run_it(self) -> None:
        assert result_for(self.CALL, ApprovalDecision("approve")) is None

    def test_an_approve_with_an_override_does_NOT_mean_run_it(self) -> None:
        # The case the hand-rolled check gets wrong, and a deleted row nobody
        # authorised.
        blocked = result_for(self.CALL, ApprovalDecision("approve", override_result="canned"))
        assert blocked is not None
        assert blocked.content == "canned"

    def test_a_denial_carries_the_note_the_model_learns_from(self) -> None:
        blocked = result_for(self.CALL, ApprovalDecision("deny", note="on legal hold"))
        assert blocked is not None
        assert blocked.content == "on legal hold"
        assert blocked.is_error is True

    def test_a_skip_is_not_an_error(self) -> None:
        blocked = result_for(self.CALL, ApprovalDecision("skip"))
        assert blocked is not None
        assert blocked.is_error is False


class TestSuspendAndResume:
    DELETE = ToolCall(id="call_1", name="delete_row", arguments={"table": "invoices", "id": 7})

    def suspended(self) -> list[dict[str, Any]]:
        captured: list[dict[str, Any]] = []

        def suspends(request: ApprovalRequest) -> ApprovalDecision:
            captured.extend(gate.snapshot())
            raise RuntimeError("waiting for a human")

        gate = ApprovalGate(suspends)
        gate.check(self.DELETE, step=1, run_id="run_1")
        return captured

    def test_the_pending_call_is_what_gets_checkpointed(self) -> None:
        captured = self.suspended()
        assert len(captured) == 1
        assert captured[0]["callId"] == "call_1"
        assert PendingToolCall.from_dict(captured[0]).tool_name == "delete_row"

    def test_the_record_is_cleared_once_the_ask_is_over(self) -> None:
        gate = ApprovalGate(lambda r: ApprovalDecision("deny"))
        gate.check(self.DELETE)
        assert gate.pending == ()

    def test_a_humans_yes_carries_across_a_restart(self) -> None:
        resumed = ApprovalGate(lambda r: ApprovalDecision("deny"))
        resumed.restore(self.suspended())
        assert resumed.resume_with("call_1", ApprovalDecision("approve")) is True
        assert resumed.check(self.DELETE).approved is True

    def test_an_answer_does_not_approve_a_different_call_sharing_its_id(self) -> None:
        # Call ids are per-turn and providers reuse them: turn 2 synthesises
        # `call_0`, `call_1` again.
        gate = ApprovalGate(lambda r: ApprovalDecision("deny"))
        gate.restore(self.suspended())
        gate.resume_with("call_1", ApprovalDecision("approve"))
        imposter = ToolCall(id="call_1", name="wire_funds", arguments={"usd": 90000})
        assert gate.check(imposter).approved is False
        # ...and the answer is still there for the call it was about.
        assert gate.check(self.DELETE).approved is True

    def test_an_answer_is_consumed_by_the_call_it_approved(self) -> None:
        # A human approved ONE call; leaving it in place would turn one yes into
        # a standing permission.
        gate = ApprovalGate(lambda r: ApprovalDecision("deny"))
        gate.restore(self.suspended())
        gate.resume_with("call_1", ApprovalDecision("approve"))
        assert gate.check(self.DELETE).approved is True
        assert gate.check(self.DELETE).approved is False

    def test_resuming_an_unknown_call_reports_that_it_did_nothing(self) -> None:
        assert ApprovalGate().resume_with("never_seen", ApprovalDecision("approve")) is False
