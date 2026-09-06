"""Telemetry at the edge: OTLP, W3C traceparent, and the debug bundle.

The in-memory model is deliberately ours -- readable ids, a domain kind, a trace
id a human can grep. None of that is legal OTLP, so the conversion happens on
the way out, and everything that can go wrong there is silent: an id of the
wrong length that a collector drops, a token count exported as a string that no
backend will sum, a hashed parent that no longer matches its child.
"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, "src")

from combycode_llm_sdk.telemetry import (
    OTLP_SPAN_KIND_CLIENT,
    OTLP_SPAN_KIND_INTERNAL,
    Span,
    TelemetryAdapter,
    TelemetryResource,
    non_zero_id,
    otlp_span_name,
    parse_traceparent,
    to_otlp_id,
    to_otlp_value,
    trimmed,
)

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def span(**over: object) -> Span:
    fields: dict[str, object] = {
        "name": "llm.request",
        "kind": "llm",
        "trace_id": "sess:req",
        "span_id": "llm:1",
        "started_at": 1_000.0,
        "ended_at": 1_250.0,
        "status": "ok",
        "attributes": {},
    }
    fields.update(over)
    return Span(**fields)  # type: ignore[arg-type]


class TestIds:
    def test_a_trace_id_is_sixteen_bytes_of_hex(self) -> None:
        value = to_otlp_id("sess:req", 16)
        assert len(value) == 32
        assert int(value, 16) >= 0

    def test_a_span_id_is_eight(self) -> None:
        assert len(to_otlp_id("sess:req|llm:1", 8)) == 16

    def test_the_same_input_gives_the_same_id_every_time(self) -> None:
        # The point of hashing rather than generating: a trace stitched together
        # from two exports, or from two services sharing a session, still joins
        # up in the backend.
        assert to_otlp_id("a:b", 16) == to_otlp_id("a:b", 16)

    def test_different_inputs_give_different_ids(self) -> None:
        assert to_otlp_id("a:b", 16) != to_otlp_id("a:c", 16)

    def test_each_word_of_an_id_uses_its_own_seed(self) -> None:
        # Mutation-driven. One seed repeated gives four identical words: a
        # 32-character id carrying 32 bits of entropy, so unrelated traces
        # collide. Length and determinism tests all pass in that state.
        trace = to_otlp_id("sess:req", 16)
        words = {trace[i : i + 8] for i in range(0, 32, 8)}
        assert len(words) == 4, f"id repeats a word: {trace}"

    def test_an_all_zero_id_is_never_handed_out(self) -> None:
        # Invalid OTLP -- a collector drops it without saying so. Tested on the
        # guard itself because no input anyone can supply reaches it, which is
        # what makes it the kind of guard that rots.
        assert non_zero_id("0" * 16) == "0" * 15 + "1"
        assert non_zero_id("00ff00ff") == "00ff00ff"


class TestValues:
    def test_an_integer_goes_out_as_a_string_under_int_value(self) -> None:
        # That is how OTLP/JSON encodes 64-bit integers. Send it as a plain
        # string and no backend will sum it -- which is what a token count is for.
        assert to_otlp_value(1200) == {"intValue": "1200"}

    def test_a_float_stays_a_double(self) -> None:
        assert to_otlp_value(3.5) == {"doubleValue": 3.5}

    def test_a_whole_float_is_still_an_integer_to_a_backend(self) -> None:
        assert to_otlp_value(4.0) == {"intValue": "4"}

    def test_a_bool_is_not_an_integer(self) -> None:
        # bool is an int subclass in Python, so order of checks matters here.
        assert to_otlp_value(True) == {"boolValue": True}

    def test_nan_and_infinity_do_not_poison_the_batch(self) -> None:
        # They are not valid JSON numbers; a collector rejects the whole export.
        assert to_otlp_value(float("nan"))["stringValue"] == "nan"
        assert "inf" in to_otlp_value(float("inf"))["stringValue"]

    def test_a_structure_is_json_not_a_python_repr(self) -> None:
        assert to_otlp_value({"a": 1}) == {"stringValue": '{"a":1}'}

    def test_nothing_becomes_an_empty_string_not_the_word_none(self) -> None:
        assert to_otlp_value(None) == {"stringValue": ""}


class TestTraceparent:
    def test_a_valid_header_is_parsed(self) -> None:
        assert parse_traceparent(TRACEPARENT) == {
            "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
            "spanId": "00f067aa0ba902b7",
        }

    def test_case_and_padding_are_tolerated(self) -> None:
        assert parse_traceparent(f"  {TRACEPARENT.upper()}  ") is not None

    def test_an_all_zero_trace_is_refused(self) -> None:
        # The spec forbids it, and a bad header must not silently reroute a run
        # into a garbage trace the backend then drops.
        assert parse_traceparent("00-" + "0" * 32 + "-00f067aa0ba902b7-01") is None

    def test_an_all_zero_span_is_refused(self) -> None:
        assert parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01") is None

    def test_junk_and_absence_are_both_none(self) -> None:
        assert parse_traceparent("nonsense") is None
        assert parse_traceparent(None) is None
        assert parse_traceparent("") is None

    def test_a_truncated_header_is_refused(self) -> None:
        assert parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa-01") is None


class TestSpanNames:
    def test_a_known_operation_is_named_by_convention(self) -> None:
        # A backend that speaks the conventions recognises this and can chart
        # it; `llm.request` is a name only we understand.
        named = span(attributes={"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"})
        assert otlp_span_name(named) == "chat m"

    def test_an_operation_with_no_subject_keeps_the_bare_operation(self) -> None:
        assert otlp_span_name(span(attributes={"gen_ai.operation.name": "chat"})) == "chat"

    def test_a_span_outside_the_conventions_keeps_its_own_name(self) -> None:
        assert otlp_span_name(span(name="http.request", attributes={})) == "http.request"


class TestTheExport:
    def _exported(
        self, *spans: Span, resource: TelemetryResource | None = None
    ) -> dict[str, Any]:
        adapter = TelemetryAdapter(resource=resource)
        adapter.spans.extend(spans)
        return adapter.to_otlp_traces()

    def test_the_shape_is_resource_spans(self) -> None:
        out = self._exported(span())
        assert list(out) == ["resourceSpans"]
        assert out["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"] == "combycode.telemetry"

    def test_the_service_is_stamped_on_every_export(self) -> None:
        out = self._exported(
            span(), resource=TelemetryResource(service_name="billing", service_version="3.2")
        )
        keys = {a["key"]: a["value"]["stringValue"] for a in out["resourceSpans"][0]["resource"]["attributes"]}
        assert keys["service.name"] == "billing"
        assert keys["service.version"] == "3.2"

    def test_an_unset_service_still_names_itself(self) -> None:
        out = self._exported(span())
        keys = [a["key"] for a in out["resourceSpans"][0]["resource"]["attributes"]]
        assert keys == ["service.name"]

    def test_times_are_nanoseconds(self) -> None:
        exported = self._exported(span())["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert exported["startTimeUnixNano"] == 1_000_000_000
        assert exported["endTimeUnixNano"] == 1_250_000_000

    def test_an_unfinished_span_ends_when_it_started(self) -> None:
        exported = self._exported(span(ended_at=None))["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert exported["endTimeUnixNano"] == exported["startTimeUnixNano"]

    def test_a_call_out_is_a_client_span_and_our_own_work_is_internal(self) -> None:
        out = self._exported(span(kind="llm"), span(kind="agent"), span(kind="nonsense"))
        kinds = [s["kind"] for s in out["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        assert kinds == [OTLP_SPAN_KIND_CLIENT, OTLP_SPAN_KIND_INTERNAL, OTLP_SPAN_KIND_INTERNAL]

    def test_status_becomes_a_code(self) -> None:
        out = self._exported(span(status="ok"), span(status="error"), span(status="unset"))
        codes = [s["status"]["code"] for s in out["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        assert codes == [1, 2, 0]

    def test_an_app_supplied_trace_id_is_adopted_verbatim(self) -> None:
        # It is already a real 32-hex id. Hashing it would invent a different
        # trace and defeat the entire point of accepting a parent.
        real = "4bf92f3577b34da6a3ce929d0e0e4736"
        exported = self._exported(span(trace_id=real))["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert exported["traceId"] == real

    def test_one_of_our_readable_trace_ids_is_hashed(self) -> None:
        exported = self._exported(span(trace_id="sess:req"))["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert exported["traceId"] != "sess:req"
        assert len(exported["traceId"]) == 32

    def test_span_ids_are_scoped_by_trace(self) -> None:
        # Two conversations can each hold a span with the same readable id, and
        # colliding them would merge unrelated traces in the backend.
        out = self._exported(
            span(trace_id="a", span_id="llm:1"), span(trace_id="b", span_id="llm:1")
        )
        ids = [s["spanId"] for s in out["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        assert ids[0] != ids[1]

    def test_a_parent_link_survives_the_hashing(self) -> None:
        # The child's parentSpanId must equal the parent's own exported spanId,
        # or the trace arrives as a flat pile of unrelated spans.
        parent = span(span_id="agent:1", kind="agent")
        child = span(span_id="llm:1", parent_id="agent:1")
        out = self._exported(parent, child)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert out[1]["parentSpanId"] == out[0]["spanId"]

    def test_an_app_supplied_parent_passes_through(self) -> None:
        exported = self._exported(span(parent_id="00f067aa0ba902b7"))["resourceSpans"][0][
            "scopeSpans"
        ][0]["spans"][0]
        assert exported["parentSpanId"] == "00f067aa0ba902b7"

    def test_a_root_span_carries_no_parent_key(self) -> None:
        exported = self._exported(span(parent_id=None))["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert "parentSpanId" not in exported

    def test_attributes_keep_their_types(self) -> None:
        exported = self._exported(span(attributes={"tokens": 1200, "ok": True}))["resourceSpans"][
            0
        ]["scopeSpans"][0]["spans"][0]
        by_key = {a["key"]: a["value"] for a in exported["attributes"]}
        assert by_key["tokens"] == {"intValue": "1200"}
        assert by_key["ok"] == {"boolValue": True}

    def test_the_whole_thing_is_json(self) -> None:
        assert json.loads(json.dumps(self._exported(span())))["resourceSpans"]


class TestTheDebugBundle:
    def test_a_snapshot_hands_out_copies(self) -> None:
        # A reader that mutates what it was handed would corrupt the record of a
        # run still in progress.
        adapter = TelemetryAdapter()
        adapter.spans.append(span())
        snap = adapter.snapshot()
        snap["spans"].clear()
        assert len(adapter.spans) == 1

    def test_it_serialises_to_json(self) -> None:
        adapter = TelemetryAdapter(resource=TelemetryResource(service_name="svc"))
        adapter.spans.append(span())
        bundle = json.loads(adapter.serialize())
        assert bundle["resource"]["serviceName"] == "svc"
        assert bundle["spans"][0]["name"] == "llm.request"

    def test_a_base64_blob_is_cut_down(self) -> None:
        adapter = TelemetryAdapter()
        adapter.spans.append(span(attributes={"image": "A" * 5000}))
        text = adapter.serialize()
        assert "chars trimmed" in text
        assert len(text) < 3000

    def test_an_exception_keeps_its_message(self) -> None:
        # `vars()` on one returns nothing, so a naive dump drops the single most
        # useful field -- the provider's actual reason.
        out = trimmed(ValueError("the model refused"))
        assert out == {"name": "ValueError", "message": "the model refused"}

    def test_a_short_string_is_left_alone(self) -> None:
        assert trimmed("fine") == "fine"

    def test_trimming_reaches_inside_structures(self) -> None:
        out = trimmed({"a": ["x" * 5000]})
        assert "chars trimmed" in out["a"][0]


class TestStatusIsDerived:
    def test_a_failed_span_says_so_without_the_producer_remembering(self) -> None:
        # Every producer would otherwise have to set it, and the one that forgets
        # reports a failed run as a healthy one.
        adapter = TelemetryAdapter(traces=True)
        adapter._span_for("onRequestComplete", {"error": "boom"})
        assert adapter.spans[-1].status == "error"

    def test_a_clean_span_is_ok(self) -> None:
        adapter = TelemetryAdapter(traces=True)
        adapter._span_for("onRequestComplete", {"status": 200})
        assert adapter.spans[-1].status == "ok"
