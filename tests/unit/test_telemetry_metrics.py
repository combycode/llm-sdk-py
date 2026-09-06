"""Counters, sampling, and subscriptions that only want some spans.

Three failures worth pinning, all of which look fine from the outside: a gauge
that drifts negative and stays there, a trace half-kept because sampling was
random rather than hashed, and a filtered subscriber handed a parent it will
never receive -- which a backend draws as a second root, silently turning one
trace into several.
"""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.telemetry import Span, TelemetryAdapter, TelemetryMetrics


class Event:
    """What the bus hands a subscriber."""

    def __init__(self, name: str, **ctx: Any) -> None:
        self.type = name
        self.ctx = ctx


def feed(adapter: TelemetryAdapter, *events: Event) -> None:
    for event in events:
        adapter._record(event)


class TestCounters:
    def test_requests_and_in_flight_move_together(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onRequestStart"), Event("onRequestStart"))
        assert (a.metrics.requests, a.metrics.in_flight) == (2, 2)
        feed(a, Event("onRequestComplete"))
        assert (a.metrics.requests, a.metrics.in_flight) == (2, 1)

    def test_in_flight_never_goes_negative(self) -> None:
        # A complete with no matching start -- a replayed retry, an adapter that
        # emits one side only -- would otherwise drive the gauge below zero and
        # leave it there for the life of the process.
        a = TelemetryAdapter()
        feed(a, Event("onRequestComplete"), Event("onRequestComplete"))
        assert a.metrics.in_flight == 0

    def test_errors_come_from_every_kind_of_failure(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onModelError"), Event("onRunError"), Event("onToolCallError"))
        assert a.metrics.errors == 3

    def test_retries_and_rate_limits_are_separate(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onRetry"), Event("onRetry"), Event("onRateLimitHit"))
        assert (a.metrics.retries, a.metrics.rate_limit_hits) == (2, 1)

    def test_tokens_add_up_across_completions(self) -> None:
        a = TelemetryAdapter()
        feed(
            a,
            Event("onCompletion", usage={"inputTokens": 100, "outputTokens": 20}),
            Event("onCompletion", usage={"inputTokens": 5, "outputTokens": 2}),
        )
        assert a.metrics.completions == 2
        assert (a.metrics.input_tokens, a.metrics.output_tokens) == (105, 22)

    def test_a_completion_with_no_usage_still_counts(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onCompletion"))
        assert a.metrics.completions == 1
        assert a.metrics.input_tokens == 0

    def test_queue_depth_is_a_gauge_not_a_counter(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onEnqueue", queueLength=5), Event("onDequeue", queueLength=3))
        assert a.metrics.queue_depth == 3

    def test_cost_accumulates(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onCostEntry", entry={"cost": {"total": 0.002}}))
        feed(a, Event("onCostEntry", entry={"cost": {"total": 0.003}}))
        assert round(a.metrics.cost_usd, 6) == 0.005

    def test_a_cost_entry_with_no_total_is_not_a_crash(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onCostEntry", entry={}), Event("onCostEntry"))
        assert a.metrics.cost_usd == 0.0

    def test_media_counts_a_batch_as_its_size(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onMediaGenerated", count=4), Event("onMediaGenerated"))
        assert a.metrics.media_generated == 5


class TestLatency:
    def test_it_summarises_rather_than_keeping_samples(self) -> None:
        # A list of samples grows without bound in the only kind of process
        # anyone points a metrics backend at.
        a = TelemetryAdapter()
        for ms in (10.0, 30.0, 20.0):
            feed(a, Event("onRequestComplete", durationMs=ms))
        latency = a.metrics.latency
        assert (latency.count, latency.min, latency.max) == (3, 10.0, 30.0)
        assert latency.avg == 20.0

    def test_the_first_sample_is_the_minimum(self) -> None:
        # Starting `min` at zero would make every minimum zero for ever.
        a = TelemetryAdapter()
        feed(a, Event("onRequestComplete", durationMs=42.0))
        assert a.metrics.latency.min == 42.0

    def test_a_completion_with_no_timing_is_not_recorded_as_zero(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onRequestComplete"))
        assert a.metrics.latency.count == 0


class TestSampling:
    def test_everything_is_kept_by_default(self) -> None:
        a = TelemetryAdapter(traces=True)
        feed(a, Event("onCompletion", runId="r1"))
        assert len(a.spans) == 1

    def test_zero_keeps_nothing(self) -> None:
        a = TelemetryAdapter(traces=True, sample=0.0)
        feed(a, Event("onCompletion", runId="r1"))
        assert a.spans == []

    def test_the_decision_is_the_same_in_every_process(self) -> None:
        # Hashed, not random: a trace shared by two services must be kept by
        # both or dropped by both. Random sampling gives you half a trace,
        # which is worse than none.
        first = TelemetryAdapter(traces=True, sample=0.5)
        second = TelemetryAdapter(traces=True, sample=0.5)
        kept_by = []
        for adapter in (first, second):
            for run in [f"run-{i}" for i in range(30)]:
                feed(adapter, Event("onCompletion", runId=run))
            kept_by.append({s.trace_id for s in adapter.spans})
        assert kept_by[0] == kept_by[1]

    def test_a_rate_between_keeps_some_and_drops_some(self) -> None:
        a = TelemetryAdapter(traces=True, sample=0.5)
        for run in [f"run-{i}" for i in range(60)]:
            feed(a, Event("onCompletion", runId=run))
        assert 0 < len(a.spans) < 60

    def test_metrics_count_a_dropped_trace_too(self) -> None:
        # Sampling decides what is STORED, not what happened. A cost total that
        # only sees a tenth of runs is not a cost total.
        a = TelemetryAdapter(traces=True, sample=0.0)
        feed(a, Event("onCompletion", usage={"inputTokens": 10, "outputTokens": 1}))
        assert a.spans == []
        assert a.metrics.completions == 1
        assert a.metrics.input_tokens == 10


def _span(**over: Any) -> Span:
    fields: dict[str, Any] = {
        "name": "n",
        "kind": "llm",
        "trace_id": "t",
        "span_id": "s",
        "started_at": 0.0,
        "ended_at": 1.0,
    }
    fields.update(over)
    return Span(**fields)


class TestFilteredSubscriptions:
    def test_a_subscriber_gets_every_span_by_default(self) -> None:
        a = TelemetryAdapter()
        seen: list[Span] = []
        a.on_trace(seen.append)
        a._dispatch(_span(kind="llm"))
        a._dispatch(_span(kind="http"))
        assert [s.kind for s in seen] == ["llm", "http"]

    def test_a_filter_narrows_it(self) -> None:
        a = TelemetryAdapter()
        seen: list[Span] = []
        a.on_trace(seen.append, kinds=["llm"])
        a._dispatch(_span(kind="llm"))
        a._dispatch(_span(kind="http"))
        assert [s.kind for s in seen] == ["llm"]

    def test_unsubscribing_stops_it(self) -> None:
        a = TelemetryAdapter()
        seen: list[Span] = []
        stop = a.on_trace(seen.append)
        stop()
        a._dispatch(_span())
        assert seen == []

    def test_a_filtered_child_is_reparented_to_a_span_it_receives(self) -> None:
        # THE failure this exists for. Filtering out `http` would otherwise
        # leave its children pointing at a span that never arrives, and a
        # backend draws a dangling parent as a separate root -- one trace
        # silently becomes several.
        a = TelemetryAdapter()
        a._lineage["agent-1"] = ("agent", None)
        a._lineage["http-1"] = ("http", "agent-1")
        seen: list[Span] = []
        a.on_trace(seen.append, kinds=["agent", "llm"])
        a._dispatch(_span(kind="llm", span_id="llm-1", parent_id="http-1"))
        assert [s.parent_id for s in seen] == ["agent-1"]

    def test_an_unknown_ancestor_roots_the_span(self) -> None:
        # A root is honest; a parent nobody will receive is a broken trace.
        a = TelemetryAdapter()
        seen: list[Span] = []
        a.on_trace(seen.append, kinds=["llm"])
        a._dispatch(_span(kind="llm", parent_id="evicted"))
        assert seen[0].parent_id is None

    def test_reparenting_does_not_edit_the_stored_span(self) -> None:
        # Two subscribers with different filters would otherwise fight over one
        # object, and whichever ran last would win.
        a = TelemetryAdapter()
        a._lineage["agent-1"] = ("agent", None)
        a._lineage["http-1"] = ("http", "agent-1")
        original = _span(kind="llm", parent_id="http-1")
        a.on_trace(lambda s: None, kinds=["agent", "llm"])
        a._dispatch(original)
        assert original.parent_id == "http-1"


class TestTheNumbersReachTheExports:
    def test_a_snapshot_carries_them(self) -> None:
        a = TelemetryAdapter()
        feed(a, Event("onRequestStart"))
        assert a.snapshot()["metrics"]["requests"] == 1

    def test_the_bundle_carries_them(self) -> None:
        import json

        a = TelemetryAdapter()
        feed(a, Event("onRetry"))
        assert json.loads(a.serialize())["metrics"]["retries"] == 1

    def test_serialising_a_run_that_recorded_events_works(self) -> None:
        # This is what caught two bugs: serialize() read `.time` and `.payload`
        # off an event that carries `.timestamp` and `.attributes`, so it raised
        # for any run that had recorded anything. The earlier test only ever put
        # SPANS in, so it never touched the events branch.
        import json

        a = TelemetryAdapter()
        feed(a, Event("onRequestStart", model="gpt-4.1"))
        bundle = json.loads(a.serialize())
        assert bundle["events"][0]["model"] == "gpt-4.1"
        assert bundle["exportedAt"] > 0

    def test_the_row_is_wire_spelled(self) -> None:
        # camelCase on the way out, like everything else that leaves here.
        row = TelemetryMetrics().as_row()
        assert "rateLimitHits" in row
        assert "inFlight" in row
