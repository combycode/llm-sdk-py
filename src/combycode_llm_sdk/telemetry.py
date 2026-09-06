"""What happened, in two shapes -- and what must never leave the process.

`events` is the buffer you read AFTERWARDS: what happened in that run. Traces
are the live feed -- spans (agent, llm, tool, http) shaped for an OTLP exporter,
with parent/child links so a slow run reads as a tree rather than a list.

Redaction is the half worth reading twice, and it has two tiers.

URLs and headers are ALWAYS redacted. API keys live in URLs (Google) and headers
(everyone else), and there is no legitimate reason to export one. No switch
reaches that tier, because a switch is a thing somebody eventually flips.

Free text is governed by `redact_free_text`, default ON. A provider's
`error.message` routinely echoes request content straight back: a moderation
refusal quotes the prompt, a validation error names the offending field and its
value. That text IS the payload, so whether it reaches a telemetry backend has
to be a decision by someone who knows what that backend stores -- not an
accident of what an error happened to contain.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .wire.interpreter import js_json

#: What replaces anything redacted. One token, so a reader can grep for it and
#: see that redaction happened rather than that a field was empty.
REDACTED = "[redacted]"

#: Header names whose value is a credential whatever else it looks like.
SECRET_HEADERS = frozenset(
    {"authorization", "x-api-key", "api-key", "x-goog-api-key", "cookie", "set-cookie"}
)

#: Query parameters that carry a key.
SECRET_QUERY = frozenset({"key", "api_key", "apikey", "access_token", "token"})

_QUERY = re.compile(r"([?&])([^=&]+)=([^&]*)")

#: Which hook becomes which span kind. A hook with no entry is recorded as an
#: event and produces no span: a span per warning would make a trace unreadable.
SPAN_KINDS: dict[str, str] = {
    "onCompletion": "llm",
    "onRunStart": "agent",
    "onRunComplete": "agent",
    "onToolCallStart": "tool",
    "onToolCallComplete": "tool",
    "onRequestStart": "http",
    "onRequestComplete": "http",
}


@dataclass
class Span:
    """One unit of work, with the links that make a trace a tree."""

    name: str
    kind: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    #: `unset` until something says otherwise. OTLP wants a code, and "did this
    #: fail" is the first thing anyone filters a trace by.
    status: str = "unset"

    @property
    def duration_ms(self) -> float:
        """How long it took. 0 while still open, never negative."""
        if self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    def __repr__(self) -> str:
        return f"<Span {self.kind}:{self.name} {self.duration_ms:.1f}ms>"


@dataclass
class TelemetryEvent:
    """One recorded hook, redacted."""

    name: str
    #: The snake_case name, so it can be pasted back into a subscription.
    type: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __getattr__(self, item: str) -> Any:
        # Fields of the payload read as attributes -- `event.code` rather than
        # `event.attributes["code"]`, which is what a reader reaches for.
        attributes = object.__getattribute__(self, "attributes")
        if item in attributes:
            return attributes[item]
        raise AttributeError(
            f"this event has no {item!r}. It carries: {', '.join(sorted(attributes))}."
        )

    def __str__(self) -> str:
        return f"{self.type} {self.attributes}"


def redact_url(url: str) -> str:
    """A URL with any credential-bearing query value replaced."""

    def scrub(match: re.Match[str]) -> str:
        lead, name, value = match.groups()
        hidden = REDACTED if name.lower() in SECRET_QUERY else value
        return f"{lead}{name}={hidden}"

    return _QUERY.sub(scrub, url)


def redact_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: (REDACTED if name.lower() in SECRET_HEADERS else value)
        for name, value in headers.items()
    }


# -- OTLP, at the edge --------------------------------------------------------
#
# The in-memory model is OURS: readable span ids, a domain `kind` a sidebar can
# group by, and a trace id a human can grep for. None of that is legal OTLP, and
# rewriting it would break every reader of `snapshot()` to satisfy a wire format
# they never look at. So the conversion happens HERE, on the way out.
#
# What the spec wants, against what the in-memory model holds:
#   trace id     16 bytes of hex        ours is readable
#   span id       8 bytes of hex        ours is readable
#   kind         an int enum            ours is a word
#   attributes   a typed AnyValue       a plain string for everything means a
#                                       token count arrives as text and no
#                                       backend can sum it

OTLP_SPAN_KIND_INTERNAL = 1
OTLP_SPAN_KIND_CLIENT = 3

#: Which of ours is a call out, and which is work we did ourselves.
OTLP_KIND_BY_SPAN: dict[str, int] = {
    "llm": OTLP_SPAN_KIND_CLIENT,
    "http": OTLP_SPAN_KIND_CLIENT,
    "mcp": OTLP_SPAN_KIND_CLIENT,
    "media": OTLP_SPAN_KIND_CLIENT,
    "agent": OTLP_SPAN_KIND_INTERNAL,
    "tool": OTLP_SPAN_KIND_INTERNAL,
    "other": OTLP_SPAN_KIND_INTERNAL,
}

OTLP_STATUS_UNSET = 0
OTLP_STATUS_OK = 1
OTLP_STATUS_ERROR = 2

#: What each operation names itself after, per the GenAI conventions. A backend
#: that speaks them recognises `execute_tool search` as a tool call and can
#: chart it; `tool.call` is a name only we understand.
SPAN_NAME_SUBJECT: dict[str, str] = {
    "chat": "gen_ai.request.model",
    "invoke_agent": "gen_ai.agent.name",
    "execute_tool": "gen_ai.tool.name",
}

#: Beyond this a string in an exported bundle is a base64 blob, not information.
MAX_SERIALIZED_STRING = 512
SERIALIZED_HEAD = 256

_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

#: FNV-1a's offset basis, and the golden-ratio step between per-word seeds.
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_SEED_STEP = 0x9E3779B9
_MASK32 = 0xFFFFFFFF


def _fnv1a32(value: str, seed: int) -> int:
    """FNV-1a, one 32-bit word per seed.

    Seeded here rather than reusing `util.hash.fnv1a32` because an OTLP id needs
    several independent words from one input, and deterministic because the same
    logical trace must map to the same id in every process -- otherwise a trace
    stitched from two exports does not join up.
    """
    h = seed & _MASK32
    for char in value:
        h = ((h ^ ord(char)) * _FNV_PRIME) & _MASK32
    return h


def to_otlp_id(value: str, byte_len: int) -> str:
    """A conformant hex id derived from one of ours. 16 bytes for a trace, 8 for a span."""
    # A DIFFERENT seed per word. One seed repeated would make every word
    # identical -- a 32-character id carrying 32 bits of entropy, so two
    # unrelated traces collide roughly as often as a birthday.
    out = "".join(
        format(_fnv1a32(value, (_FNV_OFFSET + i * _SEED_STEP) & _MASK32), "08x")
        for i in range(byte_len // 4)
    )
    return non_zero_id(out)


def non_zero_id(hex_id: str) -> str:
    """An id a collector will accept.

    All-zero is invalid OTLP and gets dropped without a word, so the last digit
    is forced. Its own function because no input anyone can supply reaches it --
    which makes it exactly the kind of guard that rots untested.
    """
    return f"{hex_id[:-1]}1" if hex_id and set(hex_id) == {"0"} else hex_id


def to_otlp_value(value: Any) -> dict[str, Any]:
    """One attribute in OTLP's AnyValue shape, keeping numbers numeric.

    An integer goes out as `intValue` carrying a STRING, which is how OTLP/JSON
    encodes 64-bit integers. Send it as a plain string instead and no backend
    will sum it -- which is exactly what you want from a token count.
    """
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        # NaN and the infinities are not valid JSON numbers, so they go out as
        # text rather than as something a collector rejects the whole batch for.
        if not math.isfinite(value):
            return {"stringValue": str(value)}
        return {"intValue": str(int(value))} if value.is_integer() else {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if value is None:
        return {"stringValue": ""}
    if isinstance(value, (Mapping, list, tuple)):
        return {"stringValue": js_json(value)}
    return {"stringValue": str(value)}


def parse_traceparent(value: str | None) -> dict[str, str] | None:
    """A W3C `traceparent`, or None for anything malformed.

    The all-zero ids the spec forbids are rejected too: a bad header must not
    silently reroute a trace into a garbage one that a backend then drops.
    """
    if not value:
        return None
    match = _TRACEPARENT.match(value.strip().lower())
    if not match:
        return None
    trace_id, span_id = match.group(1), match.group(2)
    if set(trace_id) == {"0"} or set(span_id) == {"0"}:
        return None
    return {"traceId": trace_id, "spanId": span_id}


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdef" for c in value.lower())


def otlp_span_name(span: Span) -> str:
    """The name the conventions ask for, applied only on EXPORT.

    The internal names stay put: a span's identity should not depend on which
    attributes happen to be set, and anything grouping by them still works.
    """
    operation = span.attributes.get("gen_ai.operation.name")
    if not isinstance(operation, str):
        return span.name
    subject_key = SPAN_NAME_SUBJECT.get(operation)
    subject = span.attributes.get(subject_key) if subject_key else None
    return f"{operation} {subject}" if isinstance(subject, str) and subject else operation


def trimmed(value: Any) -> Any:
    """A value fit for an exported bundle: huge strings cut, exceptions unwrapped.

    An exception is unwrapped by hand because its message is the useful part and
    a plain `vars()` returns nothing. Only the safe fields go out -- an
    exception's arbitrary attributes can carry anything the caller attached.
    """
    if isinstance(value, BaseException):
        out: dict[str, Any] = {"name": type(value).__name__, "message": str(value)}
        code = getattr(value, "code", None)
        if code is not None:
            out["code"] = code
        cause = value.__cause__
        if cause is not None and not isinstance(cause, BaseException):
            out["cause"] = cause
        return out
    if isinstance(value, str) and len(value) > MAX_SERIALIZED_STRING:
        return f"{value[:SERIALIZED_HEAD]}... ({len(value)} chars trimmed)"
    if isinstance(value, Mapping):
        return {k: trimmed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [trimmed(v) for v in value]
    return value


@dataclass
class LatencySummary:
    """Latency, as a running summary rather than a list of samples.

    A list would grow without bound in a process that runs for weeks, which is
    the only kind of process anyone points a metrics backend at.
    """

    count: int = 0
    min: float = 0.0
    max: float = 0.0
    avg: float = 0.0
    #: Kept so the mean stays exact; recomputing it from `avg` accumulates drift.
    _total: float = 0.0

    def record(self, ms: float) -> None:
        self.count += 1
        self._total += ms
        self.min = ms if self.count == 1 else min(self.min, ms)
        self.max = max(self.max, ms)
        self.avg = self._total / self.count


@dataclass
class TelemetryMetrics:
    """What has happened so far, as numbers a dashboard can chart.

    Updated ON the events rather than polled: a gauge sampled on a timer misses
    every spike shorter than its interval, and an in-flight count is mostly
    spikes.
    """

    requests: int = 0
    errors: int = 0
    retries: int = 0
    rate_limit_hits: int = 0
    completions: int = 0
    media_generated: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Gauges, not counters: what is true right now.
    in_flight: int = 0
    queue_depth: int = 0
    latency: LatencySummary = field(default_factory=LatencySummary)

    def as_row(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "retries": self.retries,
            "rateLimitHits": self.rate_limit_hits,
            "completions": self.completions,
            "mediaGenerated": self.media_generated,
            "costUsd": self.cost_usd,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "inFlight": self.in_flight,
            "queueDepth": self.queue_depth,
            "latency": {
                "count": self.latency.count,
                "min": self.latency.min,
                "max": self.latency.max,
                "avg": self.latency.avg,
            },
        }


#: Fixed so a sampling decision is reproducible -- see `TelemetryAdapter._sampled`.
SAMPLE_SEED = 0x9E3779B9

#: Counters that are a plain increment, keyed by the hook that means it happened.
_COUNTER_BY_HOOK = {
    "onRetry": "retries",
    "onRateLimitHit": "rate_limit_hits",
    "onModelError": "errors",
    "onRunError": "errors",
    "onInternalError": "errors",
    "onToolCallError": "errors",
}


@dataclass
class TelemetryResource:
    """Who is reporting. Stamped on every export."""

    service_name: str = "unknown_service"
    service_namespace: str | None = None
    service_instance_id: str | None = None
    service_version: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


class TelemetryAdapter:
    """Records what the engine emits, redacting on the way in.

    Redacting on the way IN rather than on the way out: an adapter that stored
    the raw payload and cleaned it at export time is one exporter away from
    leaking, and the buffer itself is readable by anything in the process.
    """

    #: Payload fields that are free text -- a provider's own prose, which may
    #: quote the request back.
    FREE_TEXT_FIELDS = frozenset({"message", "error", "reason", "text", "detail", "note"})

    def __init__(
        self,
        *,
        traces: bool = False,
        redact_free_text: bool = True,
        max_events: int = 1000,
        resource: TelemetryResource | None = None,
        sample: float = 1.0,
    ) -> None:
        self.resource = resource or TelemetryResource()
        #: 1.0 keeps everything; 0.0 keeps nothing. Anything between is decided
        #: per TRACE, not per span, so a sampled trace arrives whole.
        self.sample = sample
        self.metrics = TelemetryMetrics()
        self._sinks: list[tuple[frozenset[str] | None, Callable[[Span], Any]]] = []
        #: span id -> (kind, parent id), so a filtered subscriber can be given a
        #: parent it will actually receive.
        self._lineage: dict[str, tuple[str, str | None]] = {}
        self.traces = traces
        self.redact_free_text = redact_free_text
        self.max_events = max_events
        self.events: list[TelemetryEvent] = []
        self.spans: list[Span] = []
        self._open: dict[str, Span] = {}
        self._span_handlers: list[Callable[[Span], Any]] = []
        self._unsubscribe: Callable[[], None] | None = None

    # -- wiring --------------------------------------------------------------

    def attach(self, engine: Any) -> Callable[[], None]:
        """Subscribe to everything the engine emits."""
        self._unsubscribe = engine.hooks.on_any(self._record)
        return self._unsubscribe

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_span_end(self, handler: Callable[[Span], Any]) -> Callable[[Span], Any]:
        """Called as each span closes. The live feed."""
        self._span_handlers.append(handler)
        return handler

    # -- recording -----------------------------------------------------------

    def _record(self, event: Any) -> None:
        name = getattr(event, "type", "")
        ctx = getattr(event, "ctx", {})
        attributes = self.redact(dict(ctx) if isinstance(ctx, Mapping) else {"value": ctx})

        self.events.append(
            TelemetryEvent(
                name=name, type=name, attributes=attributes, timestamp=time.time() * 1000
            )
        )
        if len(self.events) > self.max_events:
            # A bounded buffer, because an unbounded one in a long-lived process
            # is a memory leak that only shows up in production.
            del self.events[: len(self.events) - self.max_events]

        # Counted before sampling: a sampled-out trace still happened, and a
        # cost total that only sees 10% of runs is not a cost total.
        self._count(name, attributes)

        if self.traces:
            self._span_for(name, attributes)

    def redact(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One payload, cleaned to whichever tier applies."""
        out: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = key.lower()
            if lowered == "url" and isinstance(value, str):
                out[key] = redact_url(value)
            elif lowered == "headers" and isinstance(value, Mapping):
                out[key] = redact_headers(value)
            elif isinstance(value, Mapping):
                out[key] = self.redact(value)
            elif self.redact_free_text and lowered in self.FREE_TEXT_FIELDS and value:
                out[key] = REDACTED
            else:
                out[key] = value
        return out

    def _span_for(self, name: str, attributes: Mapping[str, Any]) -> None:
        """Open, close, or complete a span for this event.

        A `*_start` opens one and the matching `*_complete` closes it. Anything
        else that maps to a kind is a single instant of work, opened and closed
        at once -- `onCompletion` is one call that already happened, and holding
        it open waiting for an end that never comes would leak a span per call.
        """
        camel = _camel_of(name)
        kind = SPAN_KINDS.get(camel)
        if kind is None:
            return

        now = time.time() * 1000
        if camel.endswith("Start"):
            self._open[kind] = Span(
                name=camel,
                kind=kind,
                trace_id=str(attributes.get("runId") or uuid.uuid4().hex[:16]),
                span_id=uuid.uuid4().hex[:16],
                parent_id=self._current_parent(kind),
                started_at=now,
                attributes=dict(attributes),
            )
            return

        span = self._open.pop(kind, None)
        if span is None:
            span = Span(
                name=camel,
                kind=kind,
                trace_id=str(attributes.get("runId") or uuid.uuid4().hex[:16]),
                span_id=uuid.uuid4().hex[:16],
                parent_id=self._current_parent(kind),
                started_at=now,
                attributes=dict(attributes),
            )
        span.attributes.update(attributes)
        span.ended_at = now
        # Derived rather than asked for: every producer would otherwise have to
        # remember to set it, and the one that forgets reports a failed run as
        # a healthy one.
        span.status = "error" if _looks_failed(attributes) else "ok"
        self._lineage[span.span_id] = (span.kind, span.parent_id)
        if not self._sampled(span.trace_id):
            return
        self.spans.append(span)
        for handler in list(self._span_handlers):
            handler(span)
        self._dispatch(span)

    def _current_parent(self, kind: str) -> str | None:
        """An open agent span is the parent of everything under it.

        Parenting is what makes a trace a tree; a flat list of spans is a log
        with extra fields.
        """
        agent = self._open.get("agent")
        return agent.span_id if agent is not None and kind != "agent" else None

    # -- export ---------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Copies, not the live lists: a reader that mutates what it was handed
        would corrupt the record of a run still in progress."""
        return {
            "spans": list(self.spans),
            "events": list(self.events),
            "metrics": self.metrics.as_row(),
        }

    def serialize(self) -> str:
        """A debug bundle, with base64 blobs cut down to something readable."""
        return json.dumps(
            {
                "resource": {
                    "serviceName": self.resource.service_name,
                    "serviceNamespace": self.resource.service_namespace,
                    "serviceInstanceId": self.resource.service_instance_id,
                    "serviceVersion": self.resource.service_version,
                    "attributes": dict(self.resource.attributes),
                },
                "exportedAt": self.events[-1].timestamp if self.events else 0,
                "metrics": self.metrics.as_row(),
                "spans": [self._span_row(s) for s in self.spans],
                "events": [trimmed(dict(e.attributes)) for e in self.events],
            },
            indent=2,
            default=str,
        )

    def _span_row(self, span: Span) -> dict[str, Any]:
        return {
            "name": span.name,
            "kind": span.kind,
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "parentId": span.parent_id,
            "startedAt": span.started_at,
            "endedAt": span.ended_at,
            "status": span.status,
            "attributes": trimmed(dict(span.attributes)),
        }

    def resource_attributes(self) -> list[dict[str, Any]]:
        out = [{"key": "service.name", "value": {"stringValue": self.resource.service_name}}]
        for key, value in (
            ("service.namespace", self.resource.service_namespace),
            ("service.instance.id", self.resource.service_instance_id),
            ("service.version", self.resource.service_version),
        ):
            if value:
                out.append({"key": key, "value": {"stringValue": value}})
        for key, value in self.resource.attributes.items():
            out.append({"key": key, "value": {"stringValue": str(value)}})
        return out

    def to_otlp_traces(self) -> dict[str, Any]:
        """Completed spans as OTLP resourceSpans, for someone else's exporter.

        No SDK here: the shape is small, stable, and importing an OpenTelemetry
        distribution to build one dictionary would cost every consumer of this
        library a dependency they did not ask for.
        """
        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": self.resource_attributes()},
                    "scopeSpans": [
                        {
                            "scope": {"name": "combycode.telemetry"},
                            "spans": [self._otlp_span(s) for s in self.spans],
                        }
                    ],
                }
            ]
        }

    def _otlp_span(self, span: Span) -> dict[str, Any]:
        # An app-supplied trace id is ALREADY a real 32-hex id; hashing it would
        # invent a different trace and defeat the point of accepting a parent.
        trace_id = span.trace_id if _is_hex(span.trace_id, 32) else to_otlp_id(span.trace_id, 16)
        out: dict[str, Any] = {
            "traceId": trace_id,
            # Scoped by trace: two conversations can each hold a span with the
            # same readable id, and colliding them would merge unrelated traces.
            "spanId": to_otlp_id(f"{span.trace_id}|{span.span_id}", 8),
            "name": otlp_span_name(span),
            "startTimeUnixNano": round(span.started_at * 1e6),
            "endTimeUnixNano": round((span.ended_at or span.started_at) * 1e6),
            "kind": OTLP_KIND_BY_SPAN.get(span.kind, OTLP_SPAN_KIND_INTERNAL),
            "status": {"code": _OTLP_STATUS.get(span.status, OTLP_STATUS_UNSET)},
            "attributes": [
                {"key": k, "value": to_otlp_value(v)} for k, v in span.attributes.items()
            ],
        }
        if span.parent_id:
            # The app's own parent arrives as hex and passes through; one of ours
            # is hashed exactly as it was when emitted, so the link still matches.
            out["parentSpanId"] = (
                span.parent_id
                if _is_hex(span.parent_id, 16)
                else to_otlp_id(f"{span.trace_id}|{span.parent_id}", 8)
            )
        return out

    # -- subscriptions --------------------------------------------------------

    def on_trace(
        self,
        handler: Callable[[Span], Any],
        *,
        kinds: Sequence[str] | None = None,
    ) -> Callable[[], None]:
        """Receive completed spans, optionally only some kinds.

        Returns the unsubscribe. A filtered subscriber gets spans whose parent
        has been REPARENTED to the nearest ancestor it also receives -- see
        `_surviving_parent`.
        """
        sink = (frozenset(kinds) if kinds else None, handler)
        self._sinks.append(sink)

        def unsubscribe() -> None:
            if sink in self._sinks:
                self._sinks.remove(sink)

        return unsubscribe

    def _sampled(self, trace_id: str) -> bool:
        """Whether this trace is kept.

        Hashed, not random: the same trace samples the same way in every
        process, so a trace shared by two services is kept by both or dropped by
        both. Random sampling gives you half a trace, which is worse than none.
        """
        if self.sample >= 1:
            return True
        if self.sample <= 0:
            return False
        return _fnv1a32(trace_id, SAMPLE_SEED) / 0x1_0000_0000 < self.sample

    def _surviving_parent(self, parent_id: str | None, kinds: frozenset[str]) -> str | None:
        """The nearest ancestor this subscriber actually receives.

        Without it, filtering out `http` leaves its children pointing at a span
        that never arrives, and a backend draws a dangling parent as a separate
        root -- so one trace silently becomes several. An unknown ancestor roots
        the span instead: a root is honest, a dangling parent is a broken trace.
        """
        current = parent_id
        while current:
            node = self._lineage.get(current)
            if node is None:
                return None
            kind, grandparent = node
            if kind in kinds:
                return current
            current = grandparent
        return None

    def _dispatch(self, span: Span) -> None:
        for kinds, handler in list(self._sinks):
            if kinds is not None and span.kind not in kinds:
                continue
            if kinds is None or span.parent_id is None:
                handler(span)
                continue
            surviving = self._surviving_parent(span.parent_id, kinds)
            if surviving == span.parent_id:
                handler(span)
            else:
                handler(replace(span, parent_id=surviving))

    # -- metrics --------------------------------------------------------------

    def _count(self, name: str, attributes: Mapping[str, Any]) -> None:
        """Fold one event into the running numbers."""
        counter = _COUNTER_BY_HOOK.get(name)
        if counter:
            setattr(self.metrics, counter, getattr(self.metrics, counter) + 1)
            return

        if name == "onRequestStart":
            self.metrics.requests += 1
            self.metrics.in_flight += 1
        elif name == "onRequestComplete":
            # Floored at zero: a complete with no matching start -- a retry
            # replayed, or an adapter that emits one side only -- would
            # otherwise drive the gauge negative and stay there.
            self.metrics.in_flight = max(0, self.metrics.in_flight - 1)
            latency = attributes.get("durationMs") or attributes.get("latencyMs")
            if isinstance(latency, (int, float)):
                self.metrics.latency.record(float(latency))
        elif name == "onCompletion":
            self.metrics.completions += 1
            usage = attributes.get("usage")
            if isinstance(usage, Mapping):
                self.metrics.input_tokens += int(usage.get("inputTokens") or 0)
                self.metrics.output_tokens += int(usage.get("outputTokens") or 0)
            elif usage is not None:
                self.metrics.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.metrics.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        elif name in ("onEnqueue", "onDequeue"):
            depth = attributes.get("queueLength")
            if isinstance(depth, int):
                self.metrics.queue_depth = depth
        elif name == "onCostEntry":
            self.metrics.cost_usd += _total_cost(attributes)
        elif name == "onMediaGenerated":
            count = attributes.get("count")
            self.metrics.media_generated += count if isinstance(count, int) else 1

    def clear(self) -> None:
        self.events.clear()
        self.spans.clear()
        self._open.clear()
        self._lineage.clear()

    def __repr__(self) -> str:
        return f"<TelemetryAdapter {len(self.events)} event(s), {len(self.spans)} span(s)>"


def _camel_of(name: str) -> str:
    """`on_completion` -> `onCompletion`, which is what the catalog is keyed by."""
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)


__all__ = [
    "REDACTED",
    "SECRET_HEADERS",
    "SECRET_QUERY",
    "SPAN_KINDS",
    "Span",
    "TelemetryAdapter",
    "TelemetryEvent",
    "redact_headers",
    "redact_url",
]


_OTLP_STATUS = {"ok": OTLP_STATUS_OK, "error": OTLP_STATUS_ERROR, "unset": OTLP_STATUS_UNSET}

#: Attribute keys that mean the work failed, whichever producer set them.
_FAILURE_KEYS = ("error", "errorMessage", "failed")


def _looks_failed(attributes: Mapping[str, Any]) -> bool:
    return any(attributes.get(key) for key in _FAILURE_KEYS)


def _total_cost(attributes: Mapping[str, Any]) -> float:
    """The total off a cost entry, whichever shape it arrived in."""
    entry = attributes.get("entry", attributes)
    cost = entry.get("cost") if isinstance(entry, Mapping) else getattr(entry, "cost", None)
    total = cost.get("total") if isinstance(cost, Mapping) else getattr(cost, "total", None)
    return float(total) if isinstance(total, (int, float)) else 0.0
