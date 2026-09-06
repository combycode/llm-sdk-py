"""Everything the library already says, in one place a caller can read.

The hook bus is the observability surface, but subscribing to eleven hooks to
find out what happened is work every application would otherwise repeat. `Logger`
does it once: `attach(engine.hooks)` and warnings, completions, retries, rate
limits, media, budget breaches and cost entries all arrive as `LogEvent`s at
whatever sinks are configured.

**A sink that raises must not silence the log.** A failing sink is reported
straight to stderr rather than through this logger, because logging a logging
failure through the thing that just failed is how a stack trace becomes an
infinite loop.

Levels filter before sinks see anything, so a `min_level="warn"` logger costs one
integer comparison per event rather than a formatted string nobody reads.

Transposed from `unified-library-ts/src/plugins/logger/`.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

LogLevel = Literal["trace", "debug", "info", "warn", "error"]

#: Higher is more severe. Ranked rather than ordered by name so a comparison is
#: a number, and so a level can be added between two existing ones.
LOG_LEVEL_RANK: Mapping[LogLevel, int] = {
    "trace": 0,
    "debug": 10,
    "info": 20,
    "warn": 30,
    "error": 40,
}

DEFAULT_MIN_LEVEL: LogLevel = "info"

#: Which levels go to stderr. The rest go to stdout, so a shell pipeline can
#: separate what went wrong from what the program produced.
STDERR_LEVELS: frozenset[str] = frozenset({"error", "warn"})


@dataclass(frozen=True)
class LogEvent:
    """One thing that happened."""

    level: LogLevel
    #: Where it came from: a provider name, `network`, `agent`, `cost`, or a
    #: plugin's own name.
    source: str
    #: What kind of thing it was -- `warning`, `completion`, `retry`. Stable
    #: enough to filter on, unlike the message.
    kind: str
    message: str = ""
    #: Epoch seconds. Seconds, not milliseconds: this is the public surface, and
    #: `time.time()` is what a Python caller compares it against.
    timestamp: float = field(default_factory=time.time)
    #: The request's accumulating ids, when the hook carried them.
    ctx: Mapping[str, Any] | None = None
    #: Whatever else the event knows. Sinks decide how to render it.
    data: Mapping[str, Any] | None = None


class LogSink(Protocol):
    """Somewhere log events go."""

    def log(self, event: LogEvent) -> None: ...


def default_format(event: LogEvent) -> str:
    """`[ts] [LEVEL] [source] kind: message {ctx} {data}`."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(event.timestamp))
    head = f"[{stamp}Z] [{event.level.upper()}] [{event.source}] {event.kind}"
    tail: list[str] = []
    if event.message:
        tail.append(event.message)
    if event.ctx:
        tail.append(json.dumps({"ctx": dict(event.ctx)}, default=str))
    if event.data:
        tail.append(json.dumps(dict(event.data), default=str))
    return f"{head}: {' '.join(tail)}" if tail else head


class ConsoleSink:
    """Lines on stdout and stderr."""

    def __init__(
        self,
        format: Callable[[LogEvent], str] | None = None,
        stdout: Any = None,
        stderr: Any = None,
    ) -> None:
        self._format = format or default_format
        # Resolved per WRITE, not captured here, so a test that redirects
        # `sys.stdout` after construction still sees the output -- which is what
        # pytest's capture does.
        self._stdout = stdout
        self._stderr = stderr

    def log(self, event: LogEvent) -> None:
        stream = (
            (self._stderr or sys.stderr)
            if event.level in STDERR_LEVELS
            else (self._stdout or sys.stdout)
        )
        stream.write(self._format(event) + "\n")


class Logger:
    """Hook events in, log events out."""

    def __init__(
        self,
        sinks: Sequence[LogSink],
        min_level: LogLevel = DEFAULT_MIN_LEVEL,
    ) -> None:
        if not sinks:
            raise ValueError(
                "Logger: give it at least one sink. A logger with none is a "
                "silent failure that looks like a quiet system."
            )
        self.sinks = list(sinks)
        self.min_level = min_level
        self._min_rank = LOG_LEVEL_RANK[min_level]
        self._detachers: list[Callable[[], None]] = []

    # -- logging -------------------------------------------------------------

    def log(self, event: LogEvent) -> None:
        """Send one event to every sink. Never raises."""
        if LOG_LEVEL_RANK[event.level] < self._min_rank:
            return
        for sink in self.sinks:
            try:
                sink.log(event)
            except Exception as exc:  # noqa: BLE001 -- a broken sink must not take
                # the others down with it, nor the caller that logged.
                self._sink_failed(exc, event)

    @staticmethod
    def _sink_failed(exc: BaseException, event: LogEvent) -> None:
        """Report a sink failure WITHOUT going through this logger.

        Routing it back through `log()` would hand it to the same broken sink,
        which is how one bad write becomes a recursion.
        """
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        sys.stderr.write(
            f"[{stamp}Z] [ERROR] [logger] sink-error: {exc} (original_kind={event.kind})\n"
        )

    # -- subscribing ---------------------------------------------------------

    def attach(self, hooks: Any) -> Logger:
        """Subscribe to every hook this logger knows how to render."""
        for name, handler in self._handlers().items():
            self._detachers.append(hooks.on(name, handler))
        return self

    def detach(self) -> None:
        for unsubscribe in self._detachers:
            unsubscribe()
        self._detachers = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()

    def _handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "onWarning": self._on_warning,
            "onCompletion": self._on_completion,
            "onModelError": self._on_model_error,
            "onRetry": self._on_retry,
            "onRateLimitHit": self._on_rate_limit,
            "onMediaGenerated": self._on_media,
            "onMediaError": self._on_media_error,
            "onInternalError": self._on_internal_error,
            "onBudgetWarning": self._on_budget_warning,
            "onBudgetExceeded": self._on_budget_exceeded,
            "onCostEntry": self._on_cost_entry,
        }

    # -- one renderer per hook -----------------------------------------------

    def _on_warning(self, ctx: Any) -> None:
        self.log(
            LogEvent(
                level="warn",
                source=str(_get(ctx, "source") or "unknown"),
                kind="warning",
                message=str(_get(ctx, "message") or ""),
                data={"code": _get(ctx, "code"), **dict(_get(ctx, "details") or {})},
            )
        )

    def _on_completion(self, ctx: Any) -> None:
        response = _get(ctx, "response") or {}
        usage = _get(response, "usage") or {}
        provider = str(_get(ctx, "provider") or "")
        model = str(_get(ctx, "model") or "")
        # Counts, so `_count` and not `_num`: a float renders as `12.0->3.0`,
        # which reads as a measurement rather than a number of tokens.
        into = _count(_get(usage, "inputTokens", "input_tokens"))
        out_of = _count(_get(usage, "outputTokens", "output_tokens"))
        self.log(
            LogEvent(
                level="info",
                source=provider,
                kind="completion",
                message=f"{provider}/{model} {into}->{out_of} tok",
                ctx=_get(ctx, "ctx"),
                data={
                    "finishReason": _get(response, "finishReason", "finish_reason"),
                    "latencyMs": _get(response, "latencyMs", "latency_ms"),
                },
            )
        )

    def _on_model_error(self, ctx: Any) -> None:
        error = _get(ctx, "error")
        will_retry = bool(_get(ctx, "willRetry", "will_retry"))
        self.log(
            LogEvent(
                # A retryable failure is not an error yet: the call may still
                # succeed, and logging it at error level trains people to ignore
                # the level.
                level="warn" if will_retry else "error",
                source=str(_get(ctx, "provider") or ""),
                kind="model_error",
                message=f"{_message_of(error)} ({_get(ctx, 'queueName', 'queue_name')})",
                ctx=_get(ctx, "trace"),
                data={
                    "attempt": _get(ctx, "attempt"),
                    "willRetry": will_retry,
                    "errorKind": _get(error, "kind"),
                },
            )
        )

    def _on_retry(self, ctx: Any) -> None:
        attempt = _get(ctx, "attempt")
        reason = _get(ctx, "reason")
        backoff = _get(ctx, "backoffMs", "backoff_ms")
        self.log(
            LogEvent(
                level="warn",
                source=str(_get(ctx, "provider") or ""),
                kind="retry",
                message=f"retry #{attempt} ({reason}) after {backoff}ms",
                ctx=_get(ctx, "trace"),
                data={"attempt": attempt, "reason": reason, "backoffMs": backoff},
            )
        )

    def _on_rate_limit(self, ctx: Any) -> None:
        self.log(
            LogEvent(
                level="warn",
                source=str(_get(ctx, "provider") or ""),
                kind="rate_limit",
                message=f"rate limited (HTTP {_get(ctx, 'status')})",
                ctx=_get(ctx, "trace"),
                data={"retryAfterMs": _get(ctx, "retryAfterMs", "retry_after_ms")},
            )
        )

    def _on_media(self, ctx: Any) -> None:
        kind = _get(ctx, "mediaType", "media_type") or "media"
        count = _get(ctx, "count") or 1
        self.log(
            LogEvent(
                level="info",
                source=str(_get(ctx, "provider") or ""),
                kind="media",
                message=f"{kind} x{count}",
                ctx=_get(ctx, "trace"),
            )
        )

    def _on_media_error(self, ctx: Any) -> None:
        self.log(
            LogEvent(
                level="error",
                source=str(_get(ctx, "provider") or ""),
                kind="media_error",
                message=_message_of(_get(ctx, "error")),
            )
        )

    def _on_internal_error(self, ctx: Any) -> None:
        self.log(
            LogEvent(
                level="error",
                source=str(_get(ctx, "source") or "internal"),
                kind="internal_error",
                message=_message_of(_get(ctx, "error")),
                data={
                    "queueName": _get(ctx, "queueName", "queue_name"),
                    "provider": _get(ctx, "provider"),
                },
            )
        )

    def _on_budget_warning(self, ctx: Any) -> None:
        percentage = _num(_get(ctx, "percentage"))
        self.log(
            LogEvent(
                level="warn",
                source="cost",
                kind="budget_warning",
                message=f"budget {_get(ctx, 'budgetId', 'budget_id')} at {percentage:.0f}%",
                data={"current": _get(ctx, "current"), "limit": _get(ctx, "limit")},
            )
        )

    def _on_budget_exceeded(self, ctx: Any) -> None:
        current = _num(_get(ctx, "current"))
        self.log(
            LogEvent(
                level="error",
                source="cost",
                kind="budget_exceeded",
                message=(
                    f"budget {_get(ctx, 'budgetId', 'budget_id')} exceeded "
                    f"(${current:.4f} / ${_get(ctx, 'limit')})"
                ),
            )
        )

    def _on_cost_entry(self, entry: Any) -> None:
        """One priced call.

        The payload is the `CostEntry` ITSELF, where the TypeScript passes
        `{entry, runningTotal}` -- so there is no running total to log here. A
        reader who wants one asks `engine.cost.running_total`, which is the
        ledger rather than a snapshot of it.
        """
        cost = _get(entry, "cost")
        if cost is None:
            # Unpriced is not free: `cost` is None precisely so the two cannot be
            # confused, and logging $0.000000 would confuse them.
            self.log(
                LogEvent(
                    level="debug",
                    source=str(_get(entry, "provider") or ""),
                    kind="cost",
                    message=f"unpriced ({_get(entry, 'model')})",
                )
            )
            return
        self.log(
            LogEvent(
                level="debug",
                source=str(_get(entry, "provider") or ""),
                kind="cost",
                message=f"${_num(_get(cost, 'total')):.6f} ({_get(cost, 'source')})",
                data={"model": _get(entry, "model")},
            )
        )

    def __repr__(self) -> str:
        return f"<Logger {len(self.sinks)} sink(s) at {self.min_level}>"


def _get(obj: Any, *names: str) -> Any:
    """One field, whether the payload is a mapping or an object.

    Hook contexts arrive as mappings and the cost entry as a dataclass, and both
    spellings of a name reach here because the bus translates camel to snake at
    delivery -- so a renderer that reads only one form works on one of the two.
    """
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _message_of(error: Any) -> str:
    if error is None:
        return ""
    message = _get(error, "message")
    return str(message) if message else str(error)


__all__ = [
    "DEFAULT_MIN_LEVEL",
    "LOG_LEVEL_RANK",
    "STDERR_LEVELS",
    "ConsoleSink",
    "LogEvent",
    "LogLevel",
    "LogSink",
    "Logger",
    "default_format",
]
