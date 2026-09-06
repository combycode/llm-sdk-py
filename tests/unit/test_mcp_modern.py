"""The three subsystems that exist only at 2026-07-28.

`subscriptions/listen`, tasks, and `input_required` retries. None of them can be
exercised on the handshake wire at all, so everything here runs against the
fixture's `modern` mode -- in a real child process, over a real pipe, because
these are protocol behaviours and a mocked transport would only confirm the
model already in my head.

The properties worth holding on to, and each has a test that fails without it:

- A honoured filter can be NARROWER than the request, so `is_honored()` is the
  answer and the request is not.
- A stream that ends must SAY so. A subscription that stopped delivering is
  otherwise indistinguishable from a server where nothing has changed.
- `requestState` is echoed back byte-exact and never interpreted.
- The `input_required` dispatcher is the same handler that answers a
  handshake-era server's pushed request -- that is the whole point of it.
- A handler that never satisfies the server has to terminate.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from combycode_llm_sdk.helpers.mcp import McpConnection, connect_mcp
from combycode_llm_sdk.mcp import (
    BaseJsonRpcTransport,
    IncomingHandlers,
    McpClient,
    McpClientOptions,
    McpError,
    McpErrorCode,
    McpServerEvent,
    McpSubscription,
    McpSubscriptionFilter,
    event_from_wire,
    is_input_required,
    run_input_required_driver,
    subscription_id_from,
)
from combycode_llm_sdk.mcp.input_required import (
    DEFAULT_INPUT_REQUIRED_MAX_ROUNDS,
    STATE_ONLY_BACKOFF_CAP_SECONDS,
)
from combycode_llm_sdk.mcp.result_cache import McpResultCache
from combycode_llm_sdk.mcp.sampling import (
    McpSamplingViaLLM,
    sampling_handler_with,
    to_internal_messages,
    to_stop_reason,
)
from combycode_llm_sdk.mcp.subscriptions import MCP_SUBSCRIPTION_ID_META_KEY

FIXTURE = str(Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py")


def connect(mode: str = "modern", **kwargs: Any) -> McpConnection:
    return connect_mcp(command=sys.executable, args=[FIXTURE, mode], **kwargs)


@pytest.fixture
def modern() -> Iterator[McpConnection]:
    connection = connect("modern")
    try:
        yield connection
    finally:
        connection.close()


class RecordingClock(threading.Event):
    """A cancel flag that records every back-off instead of serving it.

    The driver waits on this event, so a subclass that returns immediately
    turns a real sleep into a measurement -- which is the only way to assert on
    a back-off curve without the test taking as long as the curve.
    """

    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout or 0.0)
        return True


def wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll a condition rather than sleeping a guessed interval.

    A fixed sleep here would be either flaky or slow, and on a loaded Windows
    box usually both.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# -- the wire helpers --------------------------------------------------------


class TestFramesAndEvents:
    def test_each_change_notification_maps_to_its_event(self) -> None:
        assert event_from_wire("notifications/tools/list_changed", None) == McpServerEvent(
            "tools_list_changed"
        )
        assert event_from_wire("notifications/prompts/list_changed", None) == McpServerEvent(
            "prompts_list_changed"
        )
        assert event_from_wire("notifications/resources/list_changed", None) == McpServerEvent(
            "resources_list_changed"
        )
        assert event_from_wire(
            "notifications/resources/updated", {"uri": "mem://x"}
        ) == McpServerEvent("resource_updated", uri="mem://x")

    def test_a_resource_update_without_a_uri_is_not_an_event(self) -> None:
        # There is nothing to re-fetch, and inventing one would send the caller
        # to a document the server never mentioned.
        assert event_from_wire("notifications/resources/updated", {}) is None
        assert event_from_wire("notifications/resources/updated", {"uri": 7}) is None

    def test_an_unrelated_notification_announces_nothing(self) -> None:
        assert event_from_wire("notifications/message", {"level": "info"}) is None

    def test_a_frame_names_the_subscription_it_belongs_to(self) -> None:
        params = {"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 7}}
        assert subscription_id_from(params) == 7
        assert subscription_id_from({"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: "s-1"}}) == "s-1"

    def test_a_frame_with_no_stamp_belongs_to_no_subscription(self) -> None:
        assert subscription_id_from(None) is None
        assert subscription_id_from({}) is None
        assert subscription_id_from({"_meta": "not-an-object"}) is None
        assert subscription_id_from({"_meta": {}}) is None

    def test_a_boolean_is_not_a_subscription_id(self) -> None:
        # `bool` is an `int` in Python, so `True` would otherwise be attributed
        # to subscription 1 -- a frame delivered to the wrong caller.
        assert subscription_id_from({"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: True}}) is None


class TestTheFilter:
    def test_only_what_was_asked_for_is_sent(self) -> None:
        # A spelled-out `false` asks the server to record a preference it cannot
        # act on, and is one more key a strict server can reject.
        wire = McpSubscriptionFilter(tools_list_changed=True).to_wire()
        assert wire == {"toolsListChanged": True}

    def test_every_kind_survives_the_round_trip(self) -> None:
        wanted = McpSubscriptionFilter(
            tools_list_changed=True,
            prompts_list_changed=True,
            resources_list_changed=True,
            resource_subscriptions=["mem://a", "mem://b"],
        )
        assert McpSubscriptionFilter.of_wire(wanted.to_wire()) == wanted

    def test_an_empty_filter_sends_nothing(self) -> None:
        assert McpSubscriptionFilter().to_wire() == {}

    def test_a_malformed_uri_list_is_dropped_not_coerced(self) -> None:
        parsed = McpSubscriptionFilter.of_wire({"resourceSubscriptions": "mem://a"})
        assert parsed.resource_subscriptions == ()


class TestSubscriptionBookkeeping:
    @staticmethod
    def _subscription(**kwargs: Any) -> tuple[McpSubscription, list[McpServerEvent], list[int]]:
        seen: list[McpServerEvent] = []
        closed: list[int] = []
        subscription = McpSubscription(
            kwargs.get("id", 1),
            kwargs.get("requested", McpSubscriptionFilter(tools_list_changed=True)),
            seen.append,
            lambda: closed.append(1),
        )
        return subscription, seen, closed

    def test_a_frame_for_another_subscription_is_not_taken(self) -> None:
        subscription, seen, _ = self._subscription(id=1)
        taken = subscription.handle_frame(
            "notifications/tools/list_changed",
            {"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 2}},
        )
        assert taken is False
        assert seen == []

    def test_the_ack_records_the_honoured_subset(self) -> None:
        subscription, _, _ = self._subscription(id=1)
        subscription.handle_frame(
            "notifications/subscriptions/acknowledged",
            {
                "notifications": {"toolsListChanged": True},
                "_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 1},
            },
        )
        assert subscription.is_honored("tools_list_changed")
        assert not subscription.is_honored("prompts_list_changed")

    def test_a_missing_filter_is_malformed_not_empty(self) -> None:
        # Reading it as empty would report that the server honoured NOTHING,
        # which a caller cannot tell from a server that really did.
        subscription, _, _ = self._subscription(id=1)
        subscription.handle_frame(
            "notifications/subscriptions/acknowledged",
            {"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 1}},
        )
        assert subscription.honored is None

    def test_an_unacknowledged_kind_is_not_honoured(self) -> None:
        subscription, _, _ = self._subscription()
        assert subscription.is_honored("tools_list_changed") is False

    def test_watched_uris_count_as_honoured_only_when_non_empty(self) -> None:
        subscription, _, _ = self._subscription(id=1)
        subscription.handle_frame(
            "notifications/subscriptions/acknowledged",
            {
                "notifications": {"resourceSubscriptions": []},
                "_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 1},
            },
        )
        assert subscription.is_honored("resource_subscriptions") is False

    def test_a_closed_subscription_takes_no_more_frames(self) -> None:
        subscription, seen, _ = self._subscription(id=1)
        subscription.close()
        taken = subscription.handle_frame(
            "notifications/tools/list_changed",
            {"_meta": {MCP_SUBSCRIPTION_ID_META_KEY: 1}},
        )
        assert taken is False
        assert seen == []

    def test_closing_is_idempotent(self) -> None:
        subscription, _, closed = self._subscription()
        subscription.close()
        subscription.close()
        subscription.mark_ended(RuntimeError("late"))
        assert closed == [1]
        # The FIRST verdict stands: a clean close is not retroactively an error.
        assert subscription.ended is not None
        assert subscription.ended.error is None

    def test_an_error_that_killed_the_stream_is_kept(self) -> None:
        subscription, _, _ = self._subscription()
        boom = RuntimeError("the server hung up")
        subscription.mark_ended(boom)
        assert subscription.active is False
        assert subscription.ended is not None
        assert subscription.ended.error is boom

    def test_a_running_subscription_has_not_ended(self) -> None:
        # The distinction the type exists for: `None` is "still running", and an
        # `McpSubscriptionEnd` with no error is "cleanly over".
        subscription, _, _ = self._subscription()
        assert subscription.ended is None
        assert subscription.active


# -- against the real server -------------------------------------------------


class TestListening:
    def test_a_stream_delivers_the_changes_it_was_opened_for(
        self, modern: McpConnection
    ) -> None:
        seen: list[McpServerEvent] = []
        subscription = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), seen.append
        )
        assert wait_for(lambda: subscription.honored is not None)
        modern.client.request("fixture/push_event", {})
        assert wait_for(lambda: len(seen) == 1)
        assert seen[0].type == "tools_list_changed"
        subscription.close()

    def test_the_honoured_subset_can_be_narrower_than_the_request(
        self, modern: McpConnection
    ) -> None:
        # The property that makes `honored` worth reading at all: the fixture
        # deliberately drops prompts from whatever is asked for.
        subscription = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True, prompts_list_changed=True),
            lambda _event: None,
        )
        assert wait_for(lambda: subscription.honored is not None)
        assert subscription.requested.prompts_list_changed is True
        assert subscription.is_honored("tools_list_changed") is True
        assert subscription.is_honored("prompts_list_changed") is False
        subscription.close()

    def test_a_resource_update_names_the_document(self, modern: McpConnection) -> None:
        seen: list[McpServerEvent] = []
        subscription = modern.client.listen(
            McpSubscriptionFilter(resource_subscriptions=["mem://greeting"]), seen.append
        )
        assert wait_for(lambda: subscription.honored is not None)
        modern.client.request(
            "fixture/push_event",
            {"event": "notifications/resources/updated", "uri": "mem://greeting"},
        )
        assert wait_for(lambda: len(seen) == 1)
        assert seen[0] == McpServerEvent("resource_updated", uri="mem://greeting")
        subscription.close()

    def test_a_stream_that_ends_says_so(self, modern: McpConnection) -> None:
        subscription = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), lambda _event: None
        )
        assert wait_for(lambda: subscription.honored is not None)
        modern.client.request("fixture/end_listen", {})
        assert wait_for(lambda: not subscription.active)
        assert subscription.ended is not None
        assert subscription.ended.error is None

    def test_a_stream_the_server_rejects_reports_the_error(
        self, modern: McpConnection
    ) -> None:
        subscription = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), lambda _event: None
        )
        assert wait_for(lambda: subscription.honored is not None)
        modern.client.request("fixture/end_listen", {"error": "subscription revoked"})
        assert wait_for(lambda: not subscription.active)
        assert subscription.ended is not None
        assert isinstance(subscription.ended.error, McpError)
        assert "revoked" in str(subscription.ended.error)

    def test_closing_the_connection_ends_every_open_stream(self) -> None:
        # Otherwise a caller keeps a handle to a stream that will never deliver
        # again and is never told -- which looks exactly like a quiet server.
        connection = connect("modern")
        subscription = connection.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), lambda _event: None
        )
        assert wait_for(lambda: subscription.honored is not None)
        connection.close()
        assert subscription.active is False
        assert subscription.ended is not None
        assert isinstance(subscription.ended.error, McpError)

    def test_two_streams_each_get_only_their_own_frames(
        self, modern: McpConnection
    ) -> None:
        first: list[McpServerEvent] = []
        second: list[McpServerEvent] = []
        one = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), first.append
        )
        two = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), second.append
        )
        assert wait_for(lambda: one.honored is not None and two.honored is not None)
        modern.client.request("fixture/push_event", {"subscriptionId": two.id})
        assert wait_for(lambda: len(second) == 1)
        assert first == []
        one.close()
        two.close()

    def test_listening_needs_the_modern_wire(self) -> None:
        # And the error says what to use instead, rather than a bare -32601 from
        # a server that never had the method.
        with connect("normal") as mcp, pytest.raises(McpError, match="2026-07-28"):
            mcp.client.listen(McpSubscriptionFilter(tools_list_changed=True), lambda _e: None)

    def test_an_ended_stream_is_forgotten_by_the_client(
        self, modern: McpConnection
    ) -> None:
        # Or every dead subscription is offered every later frame forever.
        subscription = modern.client.listen(
            McpSubscriptionFilter(tools_list_changed=True), lambda _event: None
        )
        assert wait_for(lambda: subscription.honored is not None)
        modern.client.request("fixture/end_listen", {})
        assert wait_for(lambda: not subscription.active)
        assert modern.client._subscriptions == {}


class TestListenInvalidatesTheCache:
    def test_a_change_event_drops_the_list_it_invalidates(self) -> None:
        # Dropped BEFORE `on_event` runs, so a handler that immediately re-lists
        # is not answered out of the entry the server has just called stale.
        with connect("modern", cache_results=True) as mcp:
            client = mcp.client
            cache = client._cache
            assert cache is not None
            cache.set("tools/list", [{"name": "x"}], {"ttlMs": 60000})
            during: list[Any] = []
            subscription = client.listen(
                McpSubscriptionFilter(tools_list_changed=True),
                lambda _event: during.append(cache.get("tools/list")),
            )
            assert wait_for(lambda: subscription.honored is not None)
            client.request("fixture/push_event", {})
            assert wait_for(lambda: len(during) == 1)
            assert during[0] is None
            subscription.close()


class TestTasks:
    def test_a_task_call_returns_before_the_work_is_done(
        self, modern: McpConnection
    ) -> None:
        task = modern.client.call_tool_task("add", {"a": 2, "b": 3})
        assert task["status"] == "working"
        assert task["taskId"]

    def test_polling_runs_to_a_terminal_status(self, modern: McpConnection) -> None:
        task = modern.client.call_tool_task("add", {"a": 2, "b": 3})
        finished = modern.client.await_task(task["taskId"], poll_interval=0.01)
        assert finished["status"] == "completed"

    def test_the_result_is_fetched_separately(self, modern: McpConnection) -> None:
        # The point of a task: the answer outlives the request that started it.
        task = modern.client.call_tool_task("add", {"a": 2, "b": 3})
        modern.client.await_task(task["taskId"], poll_interval=0.01)
        result = modern.client.get_task_result(task["taskId"])
        assert result["content"][0]["text"] == "5"

    def test_a_cancelled_task_is_terminal(self, modern: McpConnection) -> None:
        task = modern.client.call_tool_task("add", {"a": 1, "b": 1})
        modern.client.cancel_task(task["taskId"])
        assert modern.client.await_task(task["taskId"], poll_interval=0.01)["status"] == (
            "cancelled"
        )

    def test_tasks_are_listed(self, modern: McpConnection) -> None:
        modern.client.call_tool_task("add", {"a": 1, "b": 1})
        listed = modern.client.list_tasks()
        assert len(listed) >= 1
        assert all("taskId" in task for task in listed)

    def test_waiting_gives_up_rather_than_hanging(self, modern: McpConnection) -> None:
        # A task that never finishes must surface as a timeout naming the last
        # status, not as a process that stopped responding.
        task = modern.client.call_tool_task("never_satisfied", {})
        with pytest.raises(McpError, match="did not reach a terminal status"):
            modern.client.await_task(task["taskId"], poll_interval=0.01, timeout=0.1)

    def test_the_wait_never_sleeps_past_its_own_deadline(
        self, modern: McpConnection
    ) -> None:
        # A server suggesting a five-minute interval must not make a
        # one-second timeout mean five minutes.
        task = modern.client.call_tool_task("never_satisfied", {})
        started = time.monotonic()
        with pytest.raises(McpError):
            modern.client.await_task(task["taskId"], poll_interval=30.0, timeout=0.2)
        assert time.monotonic() - started < 5.0

    def test_a_server_that_forgets_the_task_says_so(self, modern: McpConnection) -> None:
        with pytest.raises(McpError):
            modern.client.get_task("no-such-task")

    def test_a_task_call_answered_without_a_task_is_an_error(
        self, modern: McpConnection
    ) -> None:
        # There would be nothing to poll, and shrugging leaves the caller
        # holding a result that never arrives.
        with pytest.raises(McpError, match="nothing to poll"):
            modern.client.call_tool_task("task_less", {})


# -- input_required ----------------------------------------------------------


class TestTheDriver:
    def test_a_result_without_a_result_type_is_complete(self) -> None:
        # Every pre-2026 server omits the field, so any other reading would turn
        # every legacy result into a question.
        assert is_input_required({"content": []}) is False
        assert is_input_required({"resultType": "complete"}) is False
        assert is_input_required({"resultType": "input_required"}) is True
        assert is_input_required(None) is False

    def test_it_returns_a_terminal_result_untouched(self) -> None:
        final = {"content": [{"type": "text", "text": "done"}]}
        assert (
            run_input_required_driver(
                final, dispatch=lambda *_a: None, retry=lambda *_a: None
            )
            is final
        )

    def test_every_question_is_answered_before_the_retry(self) -> None:
        asked: list[str] = []
        sent: list[Any] = []

        def dispatch(key: str, request: Mapping[str, Any]) -> Any:
            asked.append(key)
            return {"answer": request.get("method")}

        def retry(responses: Any, state: str | None) -> Any:
            sent.append((responses, state))
            return {"content": []}

        run_input_required_driver(
            {
                "resultType": "input_required",
                "requestState": "s1",
                "inputRequests": {
                    "a": {"method": "sampling/createMessage"},
                    "b": {"method": "roots/list"},
                },
            },
            dispatch=dispatch,
            retry=retry,
        )
        assert sorted(asked) == ["a", "b"]
        responses, state = sent[0]
        assert responses == {
            "a": {"answer": "sampling/createMessage"},
            "b": {"answer": "roots/list"},
        }
        assert state == "s1"

    def test_a_state_only_leg_retries_with_no_responses(self) -> None:
        # `None`, not `{}`: "I have nothing to add" and "here are zero answers"
        # are different claims, and only one of them is true.
        sent: list[Any] = []
        legs = iter(
            [{"resultType": "input_required", "requestState": "s2"}, {"content": []}]
        )

        def retry(responses: Any, state: str | None) -> Any:
            sent.append((responses, state))
            return next(legs)

        run_input_required_driver(
            {"resultType": "input_required", "requestState": "s1"},
            dispatch=lambda *_a: None,
            retry=retry,
        )
        assert [responses for responses, _ in sent] == [None, None]
        assert [state for _, state in sent] == ["s1", "s2"]

    def test_it_gives_up_rather_than_looping_forever(self) -> None:
        rounds = {"n": 0}

        def retry(_responses: Any, _state: str | None) -> Any:
            rounds["n"] += 1
            return {"resultType": "input_required", "inputRequests": {"q": {"method": "x"}}}

        with pytest.raises(McpError, match="more than 3 rounds"):
            run_input_required_driver(
                {"resultType": "input_required", "inputRequests": {"q": {"method": "x"}}},
                dispatch=lambda *_a: None,
                retry=retry,
                max_rounds=3,
            )
        assert rounds["n"] == 3

    def test_the_default_round_cap_matches_every_other_sdk(self) -> None:
        assert DEFAULT_INPUT_REQUIRED_MAX_ROUNDS == 10

    def test_the_state_only_backoff_is_bounded(self) -> None:
        # A slow tool must not become a spin loop against the server, and the
        # backoff must not become a hang either.
        clock = RecordingClock()
        legs: Iterator[dict[str, Any]] = iter(
            [{"resultType": "input_required"}] * 6 + [{"content": []}]
        )
        run_input_required_driver(
            {"resultType": "input_required"},
            dispatch=lambda *_a: None,
            retry=lambda *_a: next(legs),
            cancel=clock,
        )
        assert clock.waits[0] < clock.waits[1]
        assert max(clock.waits) <= STATE_ONLY_BACKOFF_CAP_SECONDS

    def test_a_real_question_resets_the_backoff(self) -> None:
        # A server that is asking is not a server that is stalling, so the
        # climb starts over rather than continuing from where it left off.
        clock = RecordingClock()
        legs: Iterator[dict[str, Any]] = iter(
            [
                {"resultType": "input_required"},
                {"resultType": "input_required", "inputRequests": {"q": {"method": "x"}}},
                {"resultType": "input_required"},
                {"content": []},
            ]
        )
        run_input_required_driver(
            {"resultType": "input_required"},
            dispatch=lambda *_a: None,
            retry=lambda *_a: next(legs),
            cancel=clock,
        )
        assert clock.waits[-1] == clock.waits[0]


class TestInputRequiredAgainstTheServer:
    def test_a_tool_that_asks_first_is_driven_to_an_answer(self) -> None:
        # The caller only ever sees a finished call.
        answers: list[Mapping[str, Any]] = []

        def sample(params: Mapping[str, Any]) -> dict[str, Any]:
            answers.append(params)
            return {
                "role": "assistant",
                "content": {"type": "text", "text": "42"},
                "model": "stub",
                "stopReason": "endTurn",
            }

        with connect(
            "modern",
            on_server_request=lambda method, params: sample(params)
            if method == "sampling/createMessage"
            else None,
        ) as mcp:
            result = mcp.client.call_tool("needs_input", {})
        assert result["content"][0]["text"] == "the model said: 42"
        # The question reached the SAME handler a pushed request would.
        assert answers[0]["maxTokens"] == 16

    def test_the_request_state_comes_back_byte_exact(self) -> None:
        with connect(
            "modern",
            on_server_request=lambda _m, _p: {
                "role": "assistant",
                "content": {"type": "text", "text": "ok"},
            },
        ) as mcp:
            mcp.client.call_tool("needs_input", {})
            states = mcp.client.request("fixture/states", {})
        assert states["states"] == ["state-token-1"]

    def test_a_server_still_working_is_polled_to_completion(self) -> None:
        with connect("modern") as mcp:
            result = mcp.client.call_tool("slow_work", {})
        assert result["content"][0]["text"] == "done after 2 rounds"

    def test_a_handler_that_never_satisfies_the_server_terminates(self) -> None:
        with connect(
            "modern",
            on_server_request=lambda _m, _p: {"content": {"type": "text", "text": "no"}},
            input_required_max_rounds=2,
        ) as mcp, pytest.raises(McpError, match="more than 2 rounds"):
            mcp.client.call_tool("never_satisfied", {})

    def test_the_handshake_wire_pays_nothing_for_it(self) -> None:
        # No `resultType` anywhere, so the driver returns without a single extra
        # request -- which is what keeps every pre-2026 server unaffected.
        with connect("normal") as mcp:
            assert mcp.client.call_tool("add", {"a": 1, "b": 2})["content"][0]["text"] == "3"


# -- the rest of the modern surface ------------------------------------------


class TestServerRequests:
    def test_a_ping_is_answered_without_a_handler(self) -> None:
        # A liveness probe has one correct answer, and dropping the connection
        # because nobody configured a handler would be a strange way to fail.
        with connect("normal") as mcp:
            assert mcp.client._handle_server_request("ping", None) == {}

    def test_an_unhandled_request_is_refused_by_name(self) -> None:
        with connect("normal") as mcp, pytest.raises(McpError, match="roots/list"):
            mcp.client._handle_server_request("roots/list", None)

    def test_roots_are_declared_and_answered(self) -> None:
        with connect("normal", roots=[{"uri": "file:///work", "name": "work"}]) as mcp:
            answer = mcp.client._handle_server_request("roots/list", None)
            assert answer == {"roots": [{"uri": "file:///work", "name": "work"}]}
            assert mcp.client._options.capabilities["roots"] == {"listChanged": False}

    def test_roots_may_be_computed_per_request(self) -> None:
        calls = {"n": 0}

        def roots() -> list[dict[str, str]]:
            calls["n"] += 1
            return [{"uri": f"file:///run-{calls['n']}"}]

        with connect("normal", roots=roots) as mcp:
            mcp.client._handle_server_request("roots/list", None)
            second = mcp.client._handle_server_request("roots/list", None)
        assert second == {"roots": [{"uri": "file:///run-2"}]}

    def test_elicitation_is_declared_and_answered(self) -> None:
        with connect("normal", elicit=lambda _p: {"action": "accept"}) as mcp:
            assert mcp.client._options.capabilities["elicitation"] == {}
            answer = mcp.client._handle_server_request("elicitation/create", {})
            assert answer == {"action": "accept"}

    def test_a_capability_is_not_declared_when_it_cannot_be_answered(self) -> None:
        # A capability we advertise is one the server may use, so advertising an
        # unanswerable one turns our configuration gap into its failed request.
        with connect("normal") as mcp:
            assert mcp.client._options.capabilities == {}

    def test_a_caller_s_own_handler_still_runs_alongside(self) -> None:
        # Configuring `roots` must not silently disconnect a handler somebody
        # already passed. The TypeScript cannot combine the two at all.
        with connect(
            "normal",
            roots=[{"uri": "file:///work"}],
            on_server_request=lambda method, _p: {"handled": method},
        ) as mcp:
            assert mcp.client._handle_server_request("roots/list", None) == {
                "roots": [{"uri": "file:///work"}]
            }
            assert mcp.client._handle_server_request("anything/else", None) == {
                "handled": "anything/else"
            }


class TestResourceSubscriptionsOnTheOldWire:
    def test_the_handshake_wire_still_subscribes_per_resource(self) -> None:
        with connect("normal") as mcp:
            mcp.client.subscribe_resource("mem://greeting")
            mcp.client.unsubscribe_resource("mem://greeting")

    def test_the_modern_wire_refuses_and_names_the_replacement(self) -> None:
        with connect("modern") as mcp, pytest.raises(McpError, match="listen"):
            mcp.client.subscribe_resource("mem://greeting")


class TestCompletionArgument:
    def test_it_returns_the_server_s_suggestions(self) -> None:
        with connect("normal") as mcp:
            answer = mcp.client.complete_argument(
                {"type": "ref/prompt", "name": "greet"}, {"name": "who", "value": "al"}
            )
        assert answer["values"] == ["al-one", "al-two"]

    def test_a_server_with_nothing_to_suggest_still_returns_a_list(self) -> None:
        # "No suggestions" and "did not answer the question" both leave nothing
        # to offer, and a caller iterating the values should not have to tell
        # them apart -- or guard against None.
        with connect("normal") as mcp:
            answer = mcp.client.complete_argument(
                {"type": "ref/prompt", "name": "silent"}, {"name": "who", "value": "al"}
            )
        assert answer == {"values": []}


class TestCacheInvalidation:
    def test_a_resource_update_drops_only_the_named_document(self) -> None:
        # Clearing every read would throw away good entries for documents the
        # server said nothing about, turning one change into a stampede.
        with connect("normal", cache_results=True) as mcp:
            cache = mcp.client._cache
            assert cache is not None
            one = McpResultCache.key("resources/read", {"uri": "mem://a"})
            two = McpResultCache.key("resources/read", {"uri": "mem://b"})
            cache.set(one, [{"text": "a"}], {"ttlMs": 60000})
            cache.set(two, [{"text": "b"}], {"ttlMs": 60000})
            mcp.client._invalidate_on_change(
                "notifications/resources/updated", {"uri": "mem://a"}
            )
            assert cache.get(one) is None
            assert cache.get(two) is not None

    def test_a_resource_list_change_drops_the_templates_too(self) -> None:
        # Templates are a view of the same resource set; one that survives its
        # own list going stale is the more misleading half.
        with connect("normal", cache_results=True) as mcp:
            cache = mcp.client._cache
            assert cache is not None
            cache.set("resources/list", [{"uri": "mem://a"}], {"ttlMs": 60000})
            cache.set("resources/templates/list", [{"uriTemplate": "x"}], {"ttlMs": 60000})
            mcp.client._invalidate_on_change("notifications/resources/list_changed")
            assert cache.get("resources/list") is None
            assert cache.get("resources/templates/list") is None

    def test_a_read_is_cached_per_uri(self) -> None:
        # Per URI, not per method: two reads of different documents are
        # different entries, and sharing one key would serve one document for
        # another -- the worst kind of cache hit.
        with connect("caching", cache_results=True) as mcp:
            first = mcp.client.read_resource("mem://one")
            second = mcp.client.read_resource("mem://two")
            assert first[0]["text"] == "hello from mem://one"
            assert second[0]["text"] == "hello from mem://two"
            assert mcp.client.read_resource("mem://one")[0]["text"] == "hello from mem://one"
            # The second read of `mem://one` never reached the server, which is
            # the whole point -- and the only way to see it, since the answer is
            # identical whether it came from the wire or the cache.
            assert mcp.client.request("fixture/reads", {})["reads"] == {
                "mem://one": 1,
                "mem://two": 1,
            }

    def test_no_hint_means_no_caching(self) -> None:
        # Every pre-2026 server sends no `ttlMs`, so behaviour has to be
        # identical to having no cache at all.
        with connect("normal", cache_results=True) as mcp:
            cache = mcp.client._cache
            assert cache is not None
            mcp.client.read_resource("mem://greeting")
            assert cache.get(McpResultCache.key("resources/read", {"uri": "mem://greeting"})) is (
                None
            )


# -- sampling ----------------------------------------------------------------


class TestSamplingMapping:
    def test_text_collapses_to_a_bare_string(self) -> None:
        assert to_internal_messages(
            [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        ) == [{"role": "user", "content": "hi"}]

    def test_media_becomes_a_part_list(self) -> None:
        converted = to_internal_messages(
            [
                {
                    "role": "user",
                    "content": {"type": "image", "mimeType": "image/png", "data": "AAA="},
                }
            ]
        )
        assert converted[0]["content"] == [
            {
                "type": "image",
                "source": {"type": "base64", "mimeType": "image/png", "data": "AAA="},
            }
        ]

    def test_audio_is_carried_the_same_way(self) -> None:
        converted = to_internal_messages(
            [
                {
                    "role": "user",
                    "content": {"type": "audio", "mimeType": "audio/wav", "data": "BBB="},
                }
            ]
        )
        assert converted[0]["content"][0]["type"] == "audio"

    def test_an_unknown_block_is_dropped_not_passed_through(self) -> None:
        # Our adapters would reject it further down, where the error names our
        # own message shape rather than the MCP block that caused it.
        converted = to_internal_messages([{"role": "user", "content": {"type": "hologram"}}])
        assert converted[0]["content"] == ""

    def test_only_the_two_renamed_stop_reasons_are_translated(self) -> None:
        assert to_stop_reason("length") == "maxTokens"
        assert to_stop_reason("stop") == "endTurn"
        # An unfamiliar reason is still information; flattening it to `endTurn`
        # would tell the server the model stopped cleanly when it may not have.
        assert to_stop_reason("content_filter") == "content_filter"

    def test_a_handler_is_passed_through_unchanged(self) -> None:
        def mine(_params: Mapping[str, Any]) -> dict[str, Any]:
            return {"role": "assistant"}

        assert sampling_handler_with(lambda **_k: None, mine) is mine

    def test_a_model_id_is_wired_to_a_completion(self) -> None:
        seen: dict[str, Any] = {}

        class FakeCompletion:
            text = "the answer"
            model = "stub-model"
            finish_reason = "length"

        def fake_complete(**options: Any) -> FakeCompletion:
            seen.update(options)
            return FakeCompletion()

        handler = sampling_handler_with(fake_complete, McpSamplingViaLLM(model="a/b"))
        answer = handler(
            {
                "messages": [{"role": "user", "content": {"type": "text", "text": "q"}}],
                "systemPrompt": "be brief",
                "maxTokens": 32,
                "temperature": 0.2,
            }
        )
        assert seen["model"] == "a/b"
        assert seen["system"] == "be brief"
        assert seen["max_tokens"] == 32
        assert answer["content"] == {"type": "text", "text": "the answer"}
        assert answer["stopReason"] == "maxTokens"

    def test_options_the_server_did_not_send_are_left_alone(self) -> None:
        # Passing `None` through would override a model's own default with "no
        # value", which is not the same as not saying anything.
        seen: dict[str, Any] = {}

        class FakeCompletion:
            text = ""
            model = "m"
            finish_reason = "stop"

        def fake_complete(**options: Any) -> FakeCompletion:
            seen.update(options)
            return FakeCompletion()

        sampling_handler_with(fake_complete, McpSamplingViaLLM(model="a/b"))({"messages": []})
        assert "system" not in seen
        assert "max_tokens" not in seen
        assert "temperature" not in seen


class TestTaskOnlyTools:
    def test_a_tool_that_must_be_a_task_is_run_as_one(self, modern: McpConnection) -> None:
        # `taskSupport: "required"` means a plain `tools/call` is refused. A
        # tool that says how it must be invoked and is then invoked the other
        # way is a server error the client already knows how to avoid.
        tool = next(t for t in modern.tools() if t.name.endswith("long_job"))
        assert tool.func() == "long_job({})"

    def test_a_task_that_did_not_complete_reaches_the_model(
        self, modern: McpConnection
    ) -> None:
        # A failed task is a disappointing answer, not a broken connection, so
        # it becomes text the model can work around rather than an exception it
        # never sees -- the same rule `isError` follows.
        tool = next(t for t in modern.tools() if t.name.endswith("doomed_job"))
        assert tool.func() == "Tool error: task failed: the worker died"

    def test_an_ordinary_tool_is_still_a_plain_call(self, modern: McpConnection) -> None:
        tool = next(t for t in modern.tools() if t.name.endswith("__add"))
        assert tool.func(a=2, b=3) == "5"


class TestTheRegistrationOrder:
    """A fast server answers before the send returns.

    This is not hypothetical: the stdio fixture acknowledges a listen request
    from inside the same `handle()` call, and the reader thread routed that
    frame against a registry the subscription had not been added to yet -- so
    the acknowledgement was dropped and `honored` stayed None for good. It cost
    a flaky test to find, and the fix is an ordering guarantee rather than a
    narrower window.
    """

    def test_the_id_is_handed_out_before_anything_is_sent(self) -> None:
        order: list[str] = []

        class Recording(BaseJsonRpcTransport):
            def _send_message(self, message: Mapping[str, Any]) -> None:
                order.append("sent")

        Recording().send_long_lived_request(
            "subscriptions/listen", {}, None, lambda _id: order.append("registered")
        )
        assert order == ["registered", "sent"]

    def test_an_acknowledgement_during_the_send_is_not_lost(self) -> None:
        class InstantAck(BaseJsonRpcTransport):
            # The four methods the transport protocol needs, none of which this
            # test exercises: the subject here is the ORDER of registration.
            def start(self) -> None: ...

            def close(self) -> None: ...

            def request(self, method: str, params: Any = None) -> Any: ...

            def notify(self, method: str, params: Any = None) -> None: ...

            def _send_message(self, message: Mapping[str, Any]) -> None:
                # Answers from inside the send, exactly as a local server does.
                self._route_incoming(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/subscriptions/acknowledged",
                        "params": {
                            "notifications": {"toolsListChanged": True},
                            "_meta": {MCP_SUBSCRIPTION_ID_META_KEY: message["id"]},
                        },
                    }
                )

        transport = InstantAck()
        client = McpClient(transport)
        client._negotiated_version = "2026-07-28"
        transport.set_handlers(IncomingHandlers(on_notification=client._handle_notification))
        subscription = client.listen(
            McpSubscriptionFilter(tools_list_changed=True), lambda _event: None
        )
        assert subscription.is_honored("tools_list_changed")

    def test_a_stream_settles_exactly_once(self) -> None:
        # Both the server's own response and the transport's teardown end a
        # stream, and they can both happen. The second must be a no-op rather
        # than a second `on_end` -- or a KeyError on a dict that no longer has
        # the entry.
        ended: list[BaseException | None] = []

        class Quiet(BaseJsonRpcTransport):
            def _send_message(self, message: Mapping[str, Any]) -> None: ...

        transport = Quiet()
        request_id = transport.send_long_lived_request(
            "subscriptions/listen", {}, ended.append
        )
        assert transport._resolve_long_lived(request_id, None) is True
        assert transport._resolve_long_lived(request_id, RuntimeError("late")) is False
        assert ended == [None]

    def test_a_stream_that_fails_to_send_is_not_left_registered(self) -> None:
        # Otherwise the id sits in the transport forever and the caller holds a
        # subscription for a request that never went out.
        class Broken(BaseJsonRpcTransport):
            def _send_message(self, message: Mapping[str, Any]) -> None:
                raise RuntimeError("the pipe is gone")

        transport = Broken()
        with pytest.raises(RuntimeError):
            transport.send_long_lived_request("subscriptions/listen", {})
        assert transport._long_lived == {}


class TestKeepAlive:
    """An idle connection that nothing sits on gets reaped.

    Not by the server usually -- by whatever is between: a proxy, a container
    runtime, an idle-socket timeout nobody configured on purpose. A ping on an
    interval is how the connection stays worth having.
    """

    def test_it_pings_on_the_interval(self) -> None:
        with connect("normal", keep_alive=0.05) as mcp:
            assert wait_for(lambda: mcp.client.request("fixture/pings", {})["pings"] >= 2)

    def test_it_is_off_by_default(self) -> None:
        with connect("normal") as mcp:
            assert mcp.client._keep_alive is None
            time.sleep(0.15)
            assert mcp.client.request("fixture/pings", {})["pings"] == 0

    def test_a_zero_interval_is_off_rather_than_a_spin(self) -> None:
        with connect("normal", keep_alive=0) as mcp:
            assert mcp.client._keep_alive is None

    def test_a_modern_session_starts_none(self) -> None:
        # `ping` does not exist at 2026-07-28, so a timer there would send a
        # method the server may reject -- on an interval, forever.
        with connect("modern", keep_alive=0.05) as mcp:
            assert mcp.client._keep_alive is None

    def test_closing_stops_it(self) -> None:
        connection = connect("normal", keep_alive=0.05)
        thread = connection.client._keep_alive
        assert thread is not None
        connection.close()
        assert thread.is_alive() is False
        assert connection.client._keep_alive is None

    def test_closing_does_not_wait_out_the_interval(self) -> None:
        # An Event, not a sleep: a keep-alive measured in minutes would
        # otherwise hold `close()` for minutes.
        connection = connect("normal", keep_alive=120.0)
        assert connection.client._keep_alive is not None
        started = time.monotonic()
        connection.close()
        assert time.monotonic() - started < 5.0

    def test_a_failed_ping_does_not_end_the_keep_alive(self) -> None:
        # There is nobody to report it to, and the next real request will meet
        # the same failure where someone is actually waiting for an answer.
        attempts: list[int] = []

        class Refusing(BaseJsonRpcTransport):
            def start(self) -> None: ...

            def close(self) -> None: ...

            def notify(self, method: str, params: Any = None) -> None: ...

            def request(self, method: str, params: Any = None) -> Any:
                attempts.append(1)
                if len(attempts) >= 3:
                    client._keep_alive_stop.set()
                raise McpError("the pipe is gone", code=McpErrorCode.CONNECTION_CLOSED)

            def _send_message(self, message: Mapping[str, Any]) -> None: ...

        client = McpClient(Refusing(), McpClientOptions(keep_alive=0.01))
        client._keep_alive_loop(0.01)
        assert len(attempts) >= 3

    def test_a_negative_interval_is_off_not_a_spin(self) -> None:
        # `Event.wait(-1)` returns immediately, so a negative interval that got
        # through would be a ping loop running as fast as the pipe allows.
        with connect("normal", keep_alive=-1.0) as mcp:
            assert mcp.client._keep_alive is None
            time.sleep(0.1)
            assert mcp.client.request("fixture/pings", {})["pings"] == 0

    def test_the_interval_is_actually_waited_out(self) -> None:
        # Not just "some pings happened": a loop that never waits also pings,
        # and does it as fast as the connection allows.
        with connect("normal", keep_alive=30.0) as mcp:
            time.sleep(0.2)
            assert mcp.client.request("fixture/pings", {})["pings"] == 0

    def test_closing_waits_for_a_ping_already_in_flight(self) -> None:
        # `close()` has to MEAN the thread has stopped, not that it is about to
        # -- otherwise a ping outlives the client and writes to a transport
        # that is being torn down.
        class SlowPing(BaseJsonRpcTransport):
            def start(self) -> None: ...

            def close(self) -> None: ...

            def notify(self, method: str, params: Any = None) -> None: ...

            def request(self, method: str, params: Any = None) -> Any:
                time.sleep(0.3)
                return {}

            def _send_message(self, message: Mapping[str, Any]) -> None: ...

        client = McpClient(SlowPing(), McpClientOptions(keep_alive=0.01))
        client._start_keep_alive()
        thread = client._keep_alive
        assert thread is not None
        time.sleep(0.05)  # long enough to be inside a ping
        client.close()
        assert thread.is_alive() is False
