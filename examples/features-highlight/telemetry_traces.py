"""Traces are the live feed; `events` is the buffer you read afterwards.

`TelemetryAdapter.events` answers "what happened in that run?" after it is over.
Traces answer "what is happening now?" -- a feed of spans (agent, llm, tool,
http) shaped for an OTLP exporter, with parent/child links so a slow run can be
read as a tree rather than a list.

Deterministic: spans are produced by a stubbed run, no network.
"""

from _check import check, report

from combycode_llm_sdk import Engine, TelemetryAdapter

telemetry = TelemetryAdapter(traces=True)
engine = Engine(register_as_default=False, plugins=[telemetry])

spans: list[object] = []


@telemetry.on_span_end
def collect(span) -> None:
    spans.append(span)


engine.emit_completion(provider="openai", model="gpt-4.1", input_tokens=10, output_tokens=5)

check(len(spans) > 0, "expected at least one span")

llm_spans = [s for s in spans if s.kind == "llm"]
check(len(llm_spans) == 1, f"expected one llm span, got {len(llm_spans)}")

span = llm_spans[0]
check(span.duration_ms >= 0, "a finished span must carry a duration")
check(span.attributes["provider"] == "openai", "span attributes carry the provider")

# Parenting is what makes a trace a tree. A root span has no parent; anything
# under an agent run points at it.
check(span.trace_id is not None, "every span belongs to a trace")

report(spans=len(spans), llm_spans=len(llm_spans), kinds=sorted({s.kind for s in spans}))
