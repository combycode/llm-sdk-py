"""`AgentLoop` -- the durable loop behind `Agent`.

Transposed from `unified-library-ts/src/agent/loop.ts`.

The difference from `complete(tools=...)`, which is also a loop: that one is a
function and forgets everything when it returns. This is an OBJECT. It keeps the
conversation, it has an identity that hooks and telemetry are bound to, it
reports what each step cost, and it can be stopped, observed and resumed.

Ported here: the step machine in both shapes -- `complete()` buffers, `stream()`
delivers the same run as it happens -- tool execution with per-call reports,
hooks, guardrails, reflect-and-retry, lazy tool discovery, and the run report.

Still seams rather than silent omissions: the permission policy, the human
approval gate, and durable checkpoints. The layered context registry used to be
listed here and no longer is -- `history.registry` carries it now, and this loop
reads its `system`-tagged layers when it composes a step.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..approval import (
    NO_APPROVER_NOTE,
    ApprovalGate,
    PendingToolCall,
    result_for,
)
from ..bus.hook_bus import HookBus
from ..llm.output_errors import AgentRunError
from ..results import Completion, ToolCall, Usage
from ..util.json_schema import validate_json_schema
from ..wire.interpreter import js_json
from .fallback import (
    FallbackNotice,
    FallbackRun,
    client_chain,
    complete_with_fallback,
    stream_with_fallback,
)
from .history import ConversationHistory
from .lazy_tools import (
    LazySearchState,
    LazyToolsConfig,
    create_lazy_tools,
    unwrap_lazy_call,
)
from .loop_stream import (
    StepState,
    accumulate_stream_event,
    build_step_completion,
    finalize_unended_tool_calls,
    tool_call_end_events,
    tool_call_start_events,
)
from .reflect_retry import ReflectAndRetry, ReflectAndRetryPolicy, reflection_guidance
from .reports import (
    AgentRunReport,
    StepReport,
    ToolCallReport,
    ToolExecutionContext,
    add_usage,
)
from .tool_key import describe_tool, tool_key
from .tool_run import ToolRunContext, tool_run

#: How many tool-followup rounds one run may take.
#:
#: Values <= 0 mean "use the default", not "unlimited". There is deliberately no
#: way to disable the cap: an agent that never stops is a bill that never stops.
DEFAULT_MAX_STEPS = 16

#: Seconds a single tool body may take before the loop gives up on it.
DEFAULT_TOOL_TIMEOUT = 30.0


def _now_ms() -> float:
    return time.perf_counter() * 1000


#: Where a suspended run is stored, namespaced so a checkpoint store shared
#: with other subsystems cannot collide with an agent id.
CHECKPOINT_KEY_PREFIX = "agent-loop:"

#: Said when a policy refuses without giving a reason.
DENIAL_DEFAULT_REASON = "denied by policy"


class ToolNameCollision(ValueError):
    """Two tools claimed the same name and the policy said to refuse."""


class AgentLoop:
    """The loop. See the module docstring for what is and is not here."""

    def __init__(
        self,
        client: Any,
        *,
        hooks: HookBus | None = None,
        system: str | Callable[[], str] = "",
        context: str = "",
        tools: Sequence[Any] = (),
        history: ConversationHistory | Mapping[str, Any] | None = None,
        label: str | None = None,
        source: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking: Any = None,
        cache: Any = None,
        max_steps: int | None = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        model_timeout: float | None = None,
        parallel_tool_calls: bool = True,
        reflect_and_retry: ReflectAndRetry | None = None,
        lazy_tools: LazyToolsConfig | None = None,
        before: Sequence[Callable[[Any], Any]] = (),
        after: Sequence[Callable[[Any], Any]] = (),
        tool_input_guardrails: Sequence[Any] = (),
        tool_output_guardrails: Sequence[Any] = (),
        tool_output_blocked_message: Any = None,
        tool_name_collision: str = "warn",
        policy: Any = None,
        approve: Any = None,
        checkpoint: Any = None,
        metadata: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        fallback_clients: Sequence[Any] | None = None,
        fallback_on: Sequence[str] | None = None,
        validate_tool_arguments: bool = False,
    ) -> None:
        if client is None:
            raise ValueError("AgentLoop: a client is required")

        self.client = client
        #: The primary plus its backups, in order. One entry when none were
        #: given, so every step takes the same path whether or not fallback is
        #: configured.
        #:
        #: `self.model` still reports the PRIMARY, as it must: it is read before
        #: any request is made. Each step's report and span name whoever served.
        self._client_chain = client_chain(client, fallback_clients)
        self._fallback_on = tuple(fallback_on) if fallback_on is not None else None
        #: Opt-in: check a tool call's arguments against the tool's own schema
        #: before running it. Off by default because the bundled validator reads
        #: the common JSON Schema keywords and not all of Draft 2020-12 -- on by
        #: default it would refuse calls that are valid under a schema it cannot
        #: fully read.
        self._validate_tool_arguments = validate_tool_arguments
        self.hooks = hooks if hooks is not None else HookBus()
        self.label = label
        self.source = source

        self._system_thunk = system if callable(system) else None
        self._system = "" if callable(system) else str(system or "")
        self._context = context or ""

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._thinking = thinking
        self._cache = cache
        self._options = dict(options or {})

        self._max_steps = max_steps if max_steps and max_steps > 0 else DEFAULT_MAX_STEPS
        self._tool_timeout = tool_timeout
        #: Default timeout for each MODEL call in a run, in seconds.
        #:
        #: `tool_timeout` already bounded the tool half of a step; the model half
        #: was bounded only by whatever the client was configured with, so on a
        #: long run one slow step could hold the whole run open past any deadline
        #: the caller thought they had set.
        #:
        #: Applied per STEP, not per run. A run-wide budget is a different thing
        #: and the caller already has it. A per-call `timeout` still wins, and the
        #: timeout surfaces as the existing timeout error rather than a new class.
        self._model_timeout = model_timeout
        self._parallel = parallel_tool_calls

        self._before = list(before)
        self._after = list(after)
        #: Per-CALL guardrails. The message-level ones above halt a run; these
        #: act on one tool call and leave the run going.
        self._tool_input_guardrails = list(tool_input_guardrails)
        self._tool_output_guardrails = list(tool_output_guardrails)
        self._tool_output_blocked_message = tool_output_blocked_message

        self._collision_policy = tool_name_collision
        #: Consulted before every tool call. `None` means every call runs.
        self._policy = policy
        #: Asked when the policy says `ask`. Wrapped in the gate rather than
        #: called directly: the gate is where "the approver raised" and "the
        #: approver returned nothing" already become denials instead of
        #: exceptions that would take down the run.
        self._approve = approve
        self._gate = ApprovalGate(approve) if approve is not None else None
        #: Where a suspended run is written so it can outlive the process.
        self._checkpoint = checkpoint
        #: Caller-owned scratch space, carried through dump/restore.
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._pending_tool_calls: list[PendingToolCall] = []
        #: Input staged by `add_input`, waiting for the next model call.
        self._pending_input: list[dict[str, Any]] = []
        self._reflect = ReflectAndRetryPolicy(reflect_and_retry) if reflect_and_retry else None

        self._lazy_config = lazy_tools or LazyToolsConfig()
        self._lazy_state = LazySearchState()
        self._lazy_installed = False

        self._tools: dict[str, Any] = {}
        for item in tools:
            self.register_tool(item)

        if isinstance(history, ConversationHistory):
            self._history = history
        elif history:
            self._history = ConversationHistory.restore(history)
        else:
            self._history = ConversationHistory()

        self.id = self._history.id
        self._history.system = self._system
        self._history.context = self._context

        self._running = False
        self._stop_requested = False
        #: A caller's stop, inherited by a nested run. See `_stopping`.
        self._stop_when: Callable[[], bool] | None = None
        self._reports: list[AgentRunReport] = []

        self.hooks.emit_sync(
            "onAgentCreate",
            {
                "agentId": self.id,
                "clientId": getattr(client, "id", None),
                "provider": getattr(client, "provider", None),
                "model": getattr(client, "model", None),
            },
        )

    # -- identity and state --------------------------------------------------

    @property
    def model(self) -> str:
        return str(getattr(self.client, "model", ""))

    @property
    def system(self) -> str:
        return self._system

    @property
    def context(self) -> str:
        return self._context

    @property
    def history(self) -> ConversationHistory:
        return self._history

    @property
    def running(self) -> bool:
        return self._running

    @property
    def metadata(self) -> dict[str, Any]:
        """Caller-owned scratch space. Mutable on purpose: it survives dump()."""
        return self._metadata

    @property
    def pending_approvals(self) -> Sequence[PendingToolCall]:
        """Calls suspended awaiting a decision, for a resume after a restart."""
        return tuple(self._pending_tool_calls)

    def add_input(self, input_: Any) -> None:
        """Stage input for admission immediately before the next model call.

        The case it exists for: a run suspends at an approval gate, and while
        the human is deciding the user adds something -- "use staging, not
        prod". There was nowhere to put it. Appending to history directly lands
        it BEFORE the repair of the unanswered tool call that every run passes
        through, so the model read the correction and then a tool result, in
        that order; and it was lost if the process restarted between the gate
        and the resume. Staged input rides in the snapshot and is admitted
        last, where a correction belongs.

        Calls preserve insertion order. Admission clears the staging.
        """
        if self._running:
            raise RuntimeError(
                "a run is in flight, and its input was admitted when it started -- "
                "staging now would reach the NEXT run, not this one. Stage input "
                "before the run, or after it ends."
            )
        self._pending_input.extend(_as_messages(input_))

    @property
    def pending_input(self) -> Sequence[Mapping[str, Any]]:
        """Input staged for the next model call, in the order it was added."""
        return tuple(self._pending_input)

    def clear_pending_input(self) -> None:
        """Discard staged input -- the user thought better of it."""
        self._pending_input = []

    @property
    def reports(self) -> Sequence[AgentRunReport]:
        return tuple(self._reports)

    @property
    def last_report(self) -> AgentRunReport | None:
        return self._reports[-1] if self._reports else None

    def destroy(self) -> None:
        self.hooks.emit_sync(
            "onAgentDestroy", {"agentId": self.id, "clientId": getattr(self.client, "id", None)}
        )
        destroy = getattr(self.client, "destroy", None)
        if callable(destroy):
            destroy()

    def stop(self) -> None:
        """Ask the run to end after the current step.

        Not an abort: a tool already running is left to finish, because killing
        it mid-write is how a half-applied side effect happens.
        """
        self._stop_requested = True

    def _stopping(self) -> bool:
        """This run's own stop, OR the one it inherited from the run above it.

        A nested run has to honour both: a sub-agent whose caller gave up is
        answering a question nobody will read, and the parent cannot reach into
        the child's flag. `stop_when` is how the caller's answer travels --
        checked rather than fired, because cancellation on this side is
        cooperative by design (see `stop`).
        """
        if self._stop_requested:
            return True
        return bool(self._stop_when and self._stop_when())

    # -- tools ---------------------------------------------------------------

    def register_tool(self, item: Any) -> None:
        """Add one tool, refusing or warning about a name already taken."""
        from ..helpers.tool import Tool
        from ..helpers.tool import tool as as_tool

        resolved: Tool = item if isinstance(item, Tool) else as_tool(item)
        # Not `resolved.name`: a builtin tool -- `{"type": "web_search"}` -- has
        # no name to read, and the provider matches it on its type instead.
        key = tool_key(resolved)
        existing = self._tools.get(key)
        # `existing is not resolved`: registering the SAME tool twice is
        # idempotent, not a collision. Warning there would fire on any code that
        # re-adds a tool defensively, and nothing has been shadowed.
        if existing is not None and existing is not resolved:
            message = (
                f"two tools are named {key!r}; the second replaced the first, and the "
                f"model was told about a tool that will not run"
            )
            if self._collision_policy == "error":
                raise ToolNameCollision(message)
            self.hooks.emit_sync(
                "onWarning",
                {
                    "source": "agent",
                    "code": "tool_name_collision",
                    "message": message,
                    "details": {
                        "agentId": getattr(self, "id", None),
                        "key": key,
                        # The KIND, not just the key: "a function tool shadowed a
                        # builtin" and "two functions share a name" read the same
                        # otherwise, and they need different fixes.
                        "shadowed": describe_tool(existing),
                        "winner": describe_tool(resolved),
                    },
                },
            )
        self._tools[key] = resolved
        if resolved.lazy:
            self._install_lazy_tools()

    add_tool = register_tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def has_tool(self, name: str) -> bool:
        """Whether the tool is REGISTERED -- declared or not.

        A lazy tool answers True: it is callable, merely undeclared, and a
        `has_tool` that said otherwise would be answering a different question
        from the one every caller is asking.
        """
        return name in self._tools

    def declared_tools(self) -> list[dict[str, Any]]:
        """What actually goes on the wire.

        Lazy tools are absent by definition -- that filter IS the feature.
        `tool_search` and `call_tool` appear here as soon as any tool is lazy,
        because a lazy tool with no way to find it is just a missing one.
        """
        return [dict(t.definition) for t in self._tools.values() if not t.lazy]

    def _lazy_registered(self) -> list[Any]:
        return [t for t in self._tools.values() if t.lazy]

    def _eager_names(self) -> list[str]:
        return [name for name, t in self._tools.items() if not t.lazy]

    def _install_lazy_tools(self) -> None:
        """Declare `tool_search` + `call_tool` once, on the first lazy tool.

        Never removed afterwards: the declared array must stay identical for the
        whole conversation or the cached prefix is invalidated, which is the
        cost the feature exists to avoid.
        """
        if self._lazy_installed:
            return
        self._lazy_installed = True
        for built_in in create_lazy_tools(
            lazy_tools=self._lazy_registered,
            eager_names=self._eager_names,
            state=self._lazy_state,
            config=self._lazy_config,
            on_search=lambda info: self.hooks.emit_sync(
                "onToolSearch", {"agentId": self.id, **info}
            ),
        ):
            self._tools.setdefault(built_in.name, built_in)

    # -- the run -------------------------------------------------------------

    def complete(self, input_: Any = None, **options: Any) -> Completion:
        """One run: steps until the model stops asking for tools.

        Returns the FINAL completion, with the run's total usage rather than the
        last step's -- a caller billing on `result.usage` after a three-step run
        would otherwise be told about a third of it.
        """
        stop_when = options.pop("stop_when", None)
        run_id, trace = self._begin_run(input_, options.get("ctx"), stop_when)

        steps: list[StepReport] = []
        total_usage = Usage()
        total_llm_ms = 0.0
        total_tool_ms = 0.0
        step_count = 0
        tool_call_count = 0
        last: Completion | None = None
        citations: dict[str, Any] = {}
        reason = "done"
        error_message: str | None = None
        raised: BaseException | None = None
        guardrail_reason: str | None = None
        started_at = time.time() * 1000
        start = _now_ms()

        try:
            while True:
                if self._stopping():
                    reason = "stopped"
                    break

                step_type = "initial" if step_count == 0 else "tool_followup"
                self.hooks.emit_sync(
                    "onStepStart",
                    {
                        "runId": run_id,
                        "agentId": self.id,
                        "step": step_count,
                        "type": step_type,
                        "messageCount": len(self._history),
                        "estimatedInputTokens": self._history.estimated_tokens(),
                        "trace": trace,
                    },
                )

                self._run_guardrails(self._before, self._input_guard_ctx(step_count, trace))

                step_start = _now_ms()
                last, served_by = complete_with_fallback(
                    self._step_fallback(),
                    self._history.messages(),
                    dict(self._step_options(options, trace)),
                )
                step_latency = _now_ms() - step_start
                total_llm_ms += step_latency
                total_usage = add_usage(total_usage, last.usage)
                for cite in last.citations:
                    citations[cite.url] = cite

                # A recoverable MODEL failure: feed back guidance and try again
                # rather than end the run on a mistake it could fix. The usage
                # above is already counted -- a wasted turn still costs money and
                # must show in the ledger. Deliberately BEFORE the history
                # append, so the model does not learn from its own broken output.
                if self._reflect and self._reflect.handles(last.finish_reason):
                    verdict = self._reflect.record_failure()
                    self.hooks.emit_sync(
                        "onWarning",
                        {
                            "source": "agent",
                            "code": "model_failure_retry",
                            "message": (
                                f'Step {step_count} ended with "{last.finish_reason}"; '
                                + (
                                    f"retrying with reflection guidance (attempt "
                                    f"{verdict.attempt} of {self._reflect.max_attempts})"
                                    if verdict.retry
                                    else "retry budget exhausted"
                                )
                                + "."
                            ),
                            "details": {
                                "finishReason": last.finish_reason,
                                "attempt": verdict.attempt,
                                "maxAttempts": self._reflect.max_attempts,
                                "agentId": self.id,
                            },
                        },
                    )
                    if verdict.retry:
                        self._history.append(
                            {
                                "role": "user",
                                "content": reflection_guidance(
                                    last.finish_reason,
                                    verdict.attempt,
                                    self._reflect.max_attempts,
                                ),
                            }
                        )
                        continue
                    if self._reflect.raise_if_exceeded:
                        raise AgentRunError(
                            "model_failure_retry_exhausted",
                            f'The model returned "{last.finish_reason}" {verdict.attempt} '
                            f"times in a row (reflect_and_retry.max_attempts = "
                            f"{self._reflect.max_attempts}). Set raise_if_exceeded=False "
                            f"to receive the last response instead.",
                        )
                elif self._reflect:
                    self._reflect.record_success()

                calls = self._tool_calls_of(last)
                # Stamped with WHO SERVED, not with the primary. Provenance is
                # model-bound: a stateful continuation is only valid against the
                # model that issued the state, so recording the primary on a turn
                # a backup produced would have the next step offer the backup's
                # server state to the primary.
                self._history.append(
                    self._assistant_message(last, calls),
                    model=getattr(served_by, "model", self.model),
                    usage=last.usage,
                    latency_ms=step_latency,
                )

                self.hooks.emit_sync(
                    "onStepComplete",
                    {
                        "runId": run_id,
                        "agentId": self.id,
                        "step": step_count,
                        "response": last,
                        "hasToolCalls": bool(calls),
                        "toolCalls": list(calls),
                        "willContinue": bool(calls) and not self._stopping(),
                        "trace": trace,
                    },
                )

                self._run_guardrails(self._after, self._output_guard_ctx(step_count, last, trace))

                step_reports: list[ToolCallReport] = []
                step_tool_ms = 0.0
                if calls:
                    results = self._execute_tool_calls(
                        run_id, step_count, calls, step_reports, trace
                    )
                    step_tool_ms = sum(r.latency_ms for r in step_reports)
                    total_tool_ms += step_tool_ms
                    tool_call_count += len(calls)
                    self._history.append({"role": "tool", "content": results})

                steps.append(
                    StepReport(
                        index=step_count,
                        type=step_type,
                        llm_latency_ms=step_latency,
                        usage=last.usage,
                        finish_reason=last.finish_reason,
                        tool_calls=tuple(step_reports),
                        tool_total_ms=step_tool_ms,
                    )
                )
                step_count += 1

                if not calls:
                    break
                if step_count >= self._max_steps:
                    reason = "max_steps"
                    break

        except _GuardrailStop as stop:
            reason = "guardrail"
            guardrail_reason = str(stop.reason)
        except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised below
            reason = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            raised = exc
        finally:
            self._running = False

        answer = self._final_completion(
            last, reason, guardrail_reason, total_usage, list(citations.values())
        )
        self._settle_run(
            run_id=run_id,
            started_at=started_at,
            start=start,
            user_message=input_,
            answer=answer,
            reason=reason,
            error=error_message,
            steps=steps,
            step_count=step_count,
            tool_call_count=tool_call_count,
            total_usage=total_usage,
            total_llm_ms=total_llm_ms,
            total_tool_ms=total_tool_ms,
            trace=trace,
        )

        if raised is not None:
            # Re-raised so a failed run never silently returns empty text. The
            # report is already recorded, so a caller that catches this can
            # still read what happened from `last_report`.
            raise raised
        return answer

    #: `run` is `complete` under the name the delegation example uses.
    run = complete

    # -- run scaffolding -----------------------------------------------------

    def stream(self, input_: Any = None, **options: Any) -> Iterator[dict[str, Any]]:
        """One run, delivered as it happens.

        The same loop as `complete()` with different plumbing: same steps, same
        guardrails, same tools, same reports. What differs is that the model's
        text arrives as deltas and the tool calls have to be reassembled from
        them, which `loop_stream` does.

        The last event is always `done`, carrying the same `Completion`
        `complete()` would have returned -- run totals and all. A caller that
        only wants the answer can ignore everything before it; a caller that
        streamed the text still needs it, because the deltas carry no usage and
        no cost.
        """
        stop_when = options.pop("stop_when", None)
        run_id, trace = self._begin_run(input_, options.get("ctx"), stop_when)

        steps: list[StepReport] = []
        total_usage = Usage()
        total_llm_ms = 0.0
        total_tool_ms = 0.0
        step_count = 0
        tool_call_count = 0
        last: Completion | None = None
        citations: dict[str, Any] = {}
        reason = "done"
        error_message: str | None = None
        raised: BaseException | None = None
        guardrail_reason: str | None = None
        started_at = time.time() * 1000
        start = _now_ms()

        try:
            while True:
                if self._stopping():
                    reason = "stopped"
                    break

                step_type = "initial" if step_count == 0 else "tool_followup"
                yield {"type": "step_start", "step": step_count}

                self.hooks.emit_sync(
                    "onStepStart",
                    {
                        "runId": run_id,
                        "agentId": self.id,
                        "step": step_count,
                        "type": step_type,
                        "messageCount": len(self._history),
                        "estimatedInputTokens": self._history.estimated_tokens(),
                        "trace": trace,
                    },
                )

                self._run_guardrails(self._before, self._input_guard_ctx(step_count, trace))

                step_start = _now_ms()
                state = StepState(step=step_count)
                # Who served is read back AFTER the stream, not assumed: a step
                # the backup answered must be stamped with the backup's model, or
                # the report and the span both name a model that produced none of
                # this.
                stream_served_by: list[Any] = [self.client]
                for event in stream_with_fallback(
                    self._step_fallback(),
                    self._history.messages(),
                    dict(self._step_options(options, trace)),
                    stream_served_by,
                ):
                    forward = accumulate_stream_event(event, state)
                    if forward is not None:
                        yield forward

                # Anthropic and OpenAI both stream calls that end only when the
                # message does, so without this the step's last tool call is
                # dropped and the model is answered as if it had not asked.
                finalize_unended_tool_calls(state)

                step_latency = _now_ms() - step_start
                last = build_step_completion(
                    state, getattr(stream_served_by[0], "model", self.model), step_latency
                )
                total_llm_ms += step_latency
                total_usage = add_usage(total_usage, last.usage)
                for cite in last.citations:
                    citations[cite.url] = cite

                calls = self._tool_calls_of(last)
                self._history.append(
                    self._assistant_message(last, calls),
                    model=getattr(stream_served_by[0], "model", self.model),
                    usage=last.usage,
                    latency_ms=step_latency,
                )

                self.hooks.emit_sync(
                    "onStepComplete",
                    {
                        "runId": run_id,
                        "agentId": self.id,
                        "step": step_count,
                        "response": last,
                        "hasToolCalls": bool(calls),
                        "toolCalls": list(calls),
                        "willContinue": bool(calls) and not self._stopping(),
                        "trace": trace,
                    },
                )

                self._run_guardrails(self._after, self._output_guard_ctx(step_count, last, trace))

                yield {
                    "type": "step_end",
                    "step": step_count,
                    "usage": last.usage,
                    "latencyMs": step_latency,
                }

                step_reports: list[ToolCallReport] = []
                step_tool_ms = 0.0
                if calls:
                    yield from tool_call_start_events(step_count, state.tool_calls)
                    results = self._execute_tool_calls(
                        run_id, step_count, calls, step_reports, trace
                    )
                    step_tool_ms = sum(r.latency_ms for r in step_reports)
                    total_tool_ms += step_tool_ms
                    tool_call_count += len(calls)
                    yield from tool_call_end_events(step_count, step_reports)
                    self._history.append({"role": "tool", "content": results})

                steps.append(
                    StepReport(
                        index=step_count,
                        type=step_type,
                        llm_latency_ms=step_latency,
                        usage=last.usage,
                        finish_reason=last.finish_reason,
                        tool_calls=tuple(step_reports),
                        tool_total_ms=step_tool_ms,
                    )
                )
                step_count += 1

                if not calls:
                    break
                if step_count >= self._max_steps:
                    reason = "max_steps"
                    break

        except _GuardrailStop as stop:
            reason = "guardrail"
            guardrail_reason = str(stop.reason)
        except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised below
            reason = "error"
            error_message = f"{type(exc).__name__}: {exc}"
            raised = exc
        finally:
            self._running = False

        answer = self._final_completion(
            last, reason, guardrail_reason, total_usage, list(citations.values())
        )
        self._settle_run(
            run_id=run_id,
            started_at=started_at,
            start=start,
            user_message=input_,
            answer=answer,
            reason=reason,
            error=error_message,
            steps=steps,
            step_count=step_count,
            tool_call_count=tool_call_count,
            total_usage=total_usage,
            total_llm_ms=total_llm_ms,
            total_tool_ms=total_tool_ms,
            trace=trace,
        )
        if raised is not None:
            raise raised

        yield {"type": "done", "response": answer}

    def _repair_unanswered_tool_calls(self) -> None:
        """Answer any tool call the history never answered.

        The synthetic result says the call was INTERRUPTED rather than claiming
        it succeeded: the model can see the work did not happen and decide
        whether to ask again. Dropping the call would also satisfy the providers
        and would erase the fact it was ever made.
        """
        messages = self._history.messages()
        answered = {
            part.get("id")
            for message in messages
            for part in (message.get("content") or [])
            if isinstance(part, Mapping) and part.get("type") == "tool_result"
        }

        orphans: list[dict[str, Any]] = []
        seen: list[str] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for part in message.get("content") or []:
                if not isinstance(part, Mapping) or part.get("type") != "tool_call":
                    continue
                call_id = part.get("id")
                # A duplicate id is answered by the first result; counting it
                # twice would append a second and create the opposite imbalance.
                if call_id in answered or call_id in seen:
                    continue
                seen.append(str(call_id))
                orphans.append(
                    {
                        "type": "tool_result",
                        "id": call_id,
                        "content": (
                            f"Tool {part.get('name')!r} was never run: the previous turn "
                            "ended before it could execute."
                        ),
                        "isError": True,
                    }
                )
        if not orphans:
            return

        self._history.append({"role": "tool", "content": orphans})
        self.hooks.emit_sync(
            "onWarning",
            {
                "source": "agent",
                "code": "unanswered_tool_calls_repaired",
                "message": (
                    f"{len(orphans)} tool call(s) from an interrupted turn had no result "
                    "and were answered synthetically, because a provider rejects a history "
                    "that leaves one open."
                ),
                "details": {"callIds": list(seen)},
            },
        )

    def _begin_run(
        self,
        input_: Any,
        caller_ctx: Mapping[str, Any] | None = None,
        stop_when: Callable[[], bool] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._running:
            raise RuntimeError(
                "this agent is already running. One AgentLoop drives one conversation; "
                "for concurrent work build a second agent, or a second run will "
                "interleave two turns into one history."
            )
        self._running = True
        self._stop_requested = False
        self._stop_when = stop_when
        # Per RUN, not per agent.
        self._lazy_state.searches = 0
        if self._reflect:
            self._reflect.reset()

        if self._system_thunk is not None:
            # Re-evaluated every run so a live-reload prompt picks up a change.
            fresh = self._system_thunk()
            if fresh != self._system:
                self._system = fresh
                self._history.system = fresh

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        # A caller's trace wins, and it has to win HERE rather than only at the
        # step: `_step_options` already let a caller's ctx through to the LLM
        # calls, so those joined the caller's trace while the run that made them
        # -- onRunStart, every tool call -- sat in a second, unrelated one. That
        # is the same split the TypeScript side measured against a live collector.
        #
        # sessionId and requestId are resolved TOGETHER for the same reason: a
        # caller supplying one of them and not the other is how one run ends up
        # reported as two.
        caller = dict(caller_ctx or {})
        trace = {
            "sessionId": caller.get("sessionId") or self.id,
            "requestId": caller.get("requestId") or run_id,
            "conversationId": self.id,
        }
        if caller.get("traceparent"):
            # The span this run hangs under, when the caller runs inside one.
            trace["traceparent"] = caller["traceparent"]

        # Before a single new message goes in. A turn can end between "the model
        # asked for a tool" and "the tool answered" in several ways -- an early
        # break, an output guardrail tripping, continue_on_error=False, a pending
        # approval, a caller who simply stopped -- and each leaves the
        # assistant's call as the last thing in history.
        #
        # Anthropic and OpenAI both REJECT a request whose history holds a tool
        # call with no result, so the next run died on send, one turn away from
        # the cause and naming neither. Repairing it in each of the places it can
        # happen means missing the next one; this is the gate every run passes.
        self._repair_unanswered_tool_calls()

        if input_ is not None:
            for message in _as_messages(input_):
                self._history.append(message)

        # Staged input goes in LAST, after the run's own. A correction made while
        # the run was suspended ("use staging, not prod") has to sit after the
        # message it corrects to be read as one. Cleared on admission: it is the
        # transcript now, and a second run must not say it again.
        while self._pending_input:
            self._history.append(self._pending_input.pop(0))

        self.hooks.emit_sync(
            "onRunStart",
            {
                "runId": run_id,
                "agentId": self.id,
                "model": self.model,
                "input": input_,
                "trace": trace,
            },
        )
        return run_id, trace

    def _step_options(self, options: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
        """What every step sends, with the caller's own options on top."""
        step: dict[str, Any] = dict(self._options)
        step.update(options)
        step.setdefault("system", self._history.composed_system())
        for key, value in (
            ("max_tokens", self._max_tokens),
            ("temperature", self._temperature),
            ("top_p", self._top_p),
            ("thinking", self._thinking),
            ("cache", self._cache),
            # The caller's per-call value wins: `model_timeout` is the run's
            # DEFAULT for a step, not a cap on what one call may ask for.
            ("timeout", self._model_timeout),
        ):
            if value is not None and step.get(key) is None:
                step[key] = value

        declared = self.declared_tools()
        if declared:
            step["tools"] = declared

        # The RUN's trace, handed down to every LLM call it makes. Without it
        # each `client.complete()` mints a fresh requestId and one conversation
        # arrives at the collector as several unrelated traces -- correlation
        # being the whole point of a trace id, this is the one thing it must not
        # get wrong. `conversationId` is also how an Observer tells one agent's
        # completions from another's on a shared engine.
        step["ctx"] = {**dict(trace), **dict(step.get("ctx") or {})}
        return {k: v for k, v in step.items() if v is not None}

    def _settle_run(self, **fields: Any) -> None:
        report = AgentRunReport(
            id=fields["run_id"],
            model=self.model,
            started_at=fields["started_at"],
            completed_at=time.time() * 1000,
            total_ms=_now_ms() - fields["start"],
            reason=fields["reason"],
            user_message=fields["user_message"],
            final_text=fields["answer"].text,
            steps=tuple(fields["steps"]),
            step_count=fields["step_count"],
            tool_call_count=fields["tool_call_count"],
            total_usage=fields["total_usage"],
            total_llm_time_ms=fields["total_llm_ms"],
            total_tool_time_ms=fields["total_tool_ms"],
            error=fields["error"],
        )
        self._reports.append(report)
        self.hooks.emit_sync(
            "onRunError" if report.reason == "error" else "onRunComplete",
            {
                "runId": report.id,
                "agentId": self.id,
                "report": report,
                "response": fields["answer"],
                "error": report.error,
                "trace": fields["trace"],
            },
        )

    def _final_completion(
        self,
        last: Completion | None,
        reason: str,
        guardrail_reason: str | None,
        total_usage: Usage,
        citations: Sequence[Any],
    ) -> Completion:
        """The run's answer: the last step's content, the run's totals.

        A guardrail trip replaces the text with its reason. Returning the
        model's half-finished answer instead would hand the caller something
        that reads like a result and was never allowed to be one.
        """
        from dataclasses import replace

        if last is None:
            return Completion(
                text=guardrail_reason or "",
                model=self.model,
                finish_reason=reason,
                usage=total_usage,
                parts=(),
            )
        text = guardrail_reason if reason == "guardrail" else last.text
        return replace(
            last,
            text=text or "",
            usage=total_usage,
            finish_reason=reason if reason != "done" else last.finish_reason,
            citations=tuple(citations),
        )

    # -- guardrails ----------------------------------------------------------

    def _input_guard_ctx(self, step: int, trace: Mapping[str, Any]) -> Any:
        return GuardrailContext(
            kind="input",
            step=step,
            agent_id=self.id,
            trace=dict(trace),
            messages=self._history.messages(),
            system=self._history.composed_system(),
        )

    def _output_guard_ctx(
        self, step: int, response: Completion, trace: Mapping[str, Any]
    ) -> Any:
        return GuardrailContext(
            kind="output",
            step=step,
            agent_id=self.id,
            trace=dict(trace),
            response=response,
            usage=response.usage,
            cost=response.cost,
        )

    def _run_guardrails(self, guards: Sequence[Callable[[Any], Any]], ctx: Any) -> None:
        """Run each guard in order.

        A guard signals a problem by RAISING -- its own exception, not one of
        ours. That is the Python divergence the examples ask for: a budget
        ceiling is a policy this library has no opinion about, and making the
        caller build a decision object to express "stop" adds a framework to a
        one-line rule. A guard may also return a falsy `pass` decision, which is
        the shape the TypeScript uses.
        """
        for guard in guards:
            decision = guard(ctx)
            if decision is None or decision is True:
                continue
            passed = getattr(decision, "pass_", None)
            if passed is None and isinstance(decision, Mapping):
                passed = decision.get("pass")
            if passed is False:
                reason = (
                    decision.get("reason")
                    if isinstance(decision, Mapping)
                    else getattr(decision, "reason", "")
                )
                name = getattr(guard, "__name__", "guardrail")
                self.hooks.emit_sync(
                    "onGuardrailTriggered",
                    {
                        "agentId": self.id,
                        "step": getattr(ctx, "step", 0),
                        "guardrailName": name,
                        "kind": getattr(ctx, "kind", ""),
                        "reason": reason,
                    },
                )
                raise _GuardrailStop(reason or f"{name} stopped the run")

    # -- tool execution ------------------------------------------------------

    def _tool_calls_of(self, response: Completion) -> list[Any]:
        if response.finish_reason not in ("tool_use", "tool_calls"):
            # Some providers report `stop` alongside tool calls; the calls are
            # what decide, not the label.
            return [p for p in response.tool_calls if p.type == "tool_call" and p.name]
        return [p for p in response.tool_calls if p.type == "tool_call" and p.name]

    def _step_fallback(self) -> FallbackRun:
        """The chain a step runs against, with the warning wired to this bus.

        Built per step rather than once, so a consumer reading `onWarning` can
        tell WHICH step fell over -- one warning per run would have been enough
        to notice the fallback and not enough to find it.
        """

        def warn(notice: FallbackNotice) -> None:
            self.hooks.emit_sync(
                "onWarning",
                {
                    "source": "agent",
                    "code": "model_fallback",
                    "message": (
                        f'Model "{notice.from_model}" failed '
                        f"({notice.kind or 'error'}: {notice.message}); "
                        f'falling back to "{notice.to_model}".'
                    ),
                    "details": {
                        "from": notice.from_model,
                        "to": notice.to_model,
                        "kind": notice.kind,
                    },
                },
            )

        return FallbackRun(chain=self._client_chain, kinds=self._fallback_on, on_fallback=warn)

    @staticmethod
    def _assistant_message(response: Completion, calls: Sequence[Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for part in response.parts:
            if part.type == "text" and part.text:
                content.append({"type": "text", "text": part.text})
        for call in calls:
            item: dict[str, Any] = {
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments or {}),
            }
            meta = call.raw.get("_meta") if isinstance(call.raw, Mapping) else None
            if meta:
                item["_meta"] = meta
            content.append(item)
        return {"role": "assistant", "content": content}

    def _execute_tool_calls(
        self,
        run_id: str,
        step: int,
        calls: Sequence[Any],
        reports: list[ToolCallReport],
        trace: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Answer every call in the round, concurrently unless told not to."""
        if len(calls) == 1 or not self._parallel:
            results = [self._execute_one(run_id, step, c, reports, trace) for c in calls]
            return results
        with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
            # `reports` is appended to from several threads; list.append is
            # atomic under the GIL, and the order is restored by sorting on the
            # call order below rather than relied on here.
            futures = [
                pool.submit(self._execute_one, run_id, step, c, reports, trace) for c in calls
            ]
            return [f.result() for f in futures]

    def _execute_one(
        self,
        run_id: str,
        step: int,
        call: Any,
        reports: list[ToolCallReport],
        trace: Mapping[str, Any],
    ) -> dict[str, Any]:
        arguments = dict(call.arguments or {})
        name = str(call.name)

        # Refused BEFORE any hook or dispatch. `onToolCallStart` intercepts a
        # call that is about to happen, and this one never will -- handing it
        # arguments that are empty only because they failed to parse would
        # invite a hook to act on a request the model never made.
        #
        # It still gets a RESULT: providers require every call in the history to
        # be answered on the next turn, and the message tells the model what to
        # fix rather than leaving it to guess why nothing happened.
        if getattr(call, "malformed", False):
            message = (
                f"Tool call arguments for {name!r} were not valid JSON, so the tool was "
                "not run. Re-issue the call with complete, well-formed arguments."
            )
            reports.append(
                ToolCallReport(
                    call_id=str(call.id or ""),
                    tool_name=name,
                    arguments=arguments,
                    latency_ms=0.0,
                    error="malformed_tool_call",
                )
            )
            return {"type": "tool_result", "id": call.id, "content": message, "isError": True}

        # The tool that actually ran, not the router that reached it.
        inner = unwrap_lazy_call(name, arguments)
        reported_name = inner or name

        self.hooks.emit_sync(
            "onToolCallStart",
            {
                "runId": run_id,
                "agentId": self.id,
                "step": step,
                "callId": call.id,
                "toolName": reported_name,
                "arguments": arguments,
                "trace": trace,
            },
        )

        started = _now_ms()
        found = self._tools.get(name)
        ctx = ToolExecutionContext(
            step=step, call_id=str(call.id or ""), trace={**dict(trace), "callId": call.id}
        )

        if found is None:
            latency = _now_ms() - started
            message = (
                f"No tool named {name!r}. Available: "
                f"{', '.join(sorted(self._tools)) or '(none)'}."
            )
            reports.append(
                ToolCallReport(
                    call_id=str(call.id or ""),
                    tool_name=reported_name,
                    arguments=arguments,
                    latency_ms=latency,
                    error=message,
                    discovered_via="search" if inner else None,
                )
            )
            self.hooks.emit_sync(
                "onToolCallError",
                {
                    "runId": run_id,
                    "agentId": self.id,
                    "step": step,
                    "callId": call.id,
                    "toolName": reported_name,
                    "error": message,
                    "trace": trace,
                },
            )
            return {"type": "tool_result", "id": call.id, "content": message, "isError": True}

        # BEFORE the permission gate: a call refused on its arguments must not
        # reach a person to be approved.
        denied = self._check_tool_input(call, reported_name, arguments, step, trace, reports)
        if denied is not None:
            return denied

        refusal = self._gate_call(run_id, step, call, reported_name, arguments, reports, trace)
        if refusal is not None:
            return refusal

        # What the tool can learn about the call it is serving -- the run's trace
        # and whether a stop has been asked for. A ContextVar rather than an
        # argument because a tool here is a plain function whose signature IS the
        # model's parameter list; see `agent/tool_run.py`.
        run_ctx = ToolRunContext(
            tool_name=reported_name,
            call_id=str(call.id or ""),
            step=step,
            trace=dict(trace),
            is_stopping=self._stopping,
        )
        try:
            with tool_run(run_ctx):
                value = _with_timeout(lambda: found.func(**arguments), self._tool_timeout, found)
            content = _as_content(value)
            error: str | None = None
        except Exception as exc:  # noqa: BLE001 -- a tool body may raise anything, and by
            # contract every failure goes back to the model as its result.
            content = f"{type(exc).__name__}: {exc}"
            error = content

        # Inspected BEFORE anything keeps it. The tool has already run, so the
        # only thing left to control is what its output touches: the model's
        # view, the conversation, and any checkpoint written from them.
        if error is None:
            withheld = self._check_tool_output(
                call, reported_name, arguments, content, step, trace, found
            )
            if withheld is not None:
                content = withheld

        latency = _now_ms() - started
        reports.append(
            ToolCallReport(
                call_id=str(call.id or ""),
                tool_name=reported_name,
                arguments=arguments,
                result_size_bytes=len(content) if isinstance(content, str) else 0,
                latency_ms=latency,
                error=error,
                metrics=dict(ctx.metrics),
                discovered_via="search" if inner else None,
            )
        )
        self.hooks.emit_sync(
            "onToolCallError" if error else "onToolCallComplete",
            {
                "runId": run_id,
                "agentId": self.id,
                "step": step,
                "callId": call.id,
                "toolName": reported_name,
                "latencyMs": latency,
                "error": error,
                "trace": trace,
            },
        )
        part: dict[str, Any] = {"type": "tool_result", "id": call.id, "content": content}
        if error:
            part["isError"] = True
        return part

    def _check_tool_input(
        self,
        call: Any,
        reported_name: str,
        arguments: Mapping[str, Any],
        step: int,
        trace: Mapping[str, Any],
        reports: list[ToolCallReport],
    ) -> dict[str, Any] | None:
        """Validate a call's arguments before it runs. Returns a denial result,
        or None to proceed.

        A trip denies THIS call -- the model gets the reason as an error result
        -- and leaves the run going. The message-level guardrails halt; this one
        does not, because one bad call is not a reason to end a conversation.
        """
        if not self._tool_input_guardrails:
            return None
        ctx = ToolGuardrailContext(
            tool_name=reported_name,
            arguments=dict(arguments),
            call_id=str(call.id or ""),
            step=step,
            trace=dict(trace),
        )
        for guard in self._tool_input_guardrails:
            try:
                decision = guard.check(ctx)
            except Exception as exc:  # noqa: BLE001 -- a checker that crashed approved nothing
                decision = {"pass": False, "reason": f"guardrail failed: {type(exc).__name__}: {exc}"}
            if _passed(decision):
                continue
            reason = _decision_reason(decision) or "refused by a tool-input guardrail"
            reports.append(
                ToolCallReport(
                    call_id=str(call.id or ""),
                    tool_name=reported_name,
                    arguments=dict(arguments),
                    latency_ms=0.0,
                    error=reason,
                )
            )
            return {"type": "tool_result", "id": call.id, "content": reason, "isError": True}
        return None

    def _check_tool_output(
        self,
        call: Any,
        reported_name: str,
        arguments: Mapping[str, Any],
        content: Any,
        step: int,
        trace: Mapping[str, Any],
        found: Any,
    ) -> str | None:
        """Inspect what the tool returned. Returns the replacement text when one
        withheld it, or None when it passes.

        FAIL CLOSED, twice over. A guardrail that throws is treated as having
        tripped -- a checker that crashed has approved nothing, and the one time
        that matters is the one where it crashed ON the output it would have
        caught. And a message formatter that throws or returns nothing falls back
        to the default sentence rather than to the output it was deciding about.
        """
        # The loop's own first, then the ones the tool brought with it. Either
        # can withhold; the first that does decides, so a source-level rule
        # cannot be talked out of by a later one.
        guards = [*self._tool_output_guardrails, *(getattr(found, "output_guardrails", None) or [])]
        if not guards:
            return None
        # A guardrail reads TEXT. A tool that answered with content parts used to
        # skip the check entirely on this side -- and a tool returning media is
        # precisely the case a rule wants to see -- so the parts are serialised
        # for the checker, exactly as the TypeScript does. A trip still replaces
        # the WHOLE result, media included.
        ctx = ToolGuardrailContext(
            tool_name=reported_name,
            arguments=dict(arguments),
            call_id=str(call.id or ""),
            step=step,
            trace=dict(trace),
            result=content if isinstance(content, str) else js_json(content),
        )
        for guard in guards:
            name = getattr(guard, "name", "guardrail")
            try:
                decision = guard.check(ctx)
            except Exception as exc:  # noqa: BLE001 -- see the docstring
                decision = {"pass": False, "reason": f"guardrail failed: {type(exc).__name__}: {exc}"}
            if _passed(decision):
                continue
            replacement = _decision_replacement(decision) or self._blocked_message(name, reported_name, call)
            self.hooks.emit_sync(
                "onWarning",
                {
                    "source": "agent",
                    "code": "tool_output_withheld",
                    "message": (
                        f"{name} withheld the output of {reported_name}: "
                        f"{_decision_reason(decision) or 'no reason given'}"
                    ),
                    "details": {
                        "callId": call.id,
                        "toolName": reported_name,
                        "guardrail": name,
                        "step": step,
                    },
                },
            )
            return replacement
        return None

    def _blocked_message(self, guardrail_name: str, tool_name: str, call: Any) -> str:
        """The placeholder a withheld output is replaced by."""
        configured = self._tool_output_blocked_message
        if isinstance(configured, str):
            return configured or TOOL_OUTPUT_WITHHELD
        if callable(configured):
            # Fails closed: a formatter that raises gives the default sentence,
            # never the output it was deciding about.
            with contextlib.suppress(Exception):
                resolved = configured(
                    {
                        "defaultMessage": TOOL_OUTPUT_WITHHELD,
                        "guardrailName": guardrail_name,
                        "toolName": tool_name,
                        "callId": str(call.id or ""),
                    }
                )
                if isinstance(resolved, str) and resolved:
                    return resolved
        return TOOL_OUTPUT_WITHHELD

    def _gate_call(
        self,
        run_id: str,
        step: int,
        call: Any,
        reported_name: str,
        arguments: Mapping[str, Any],
        reports: list[ToolCallReport],
        trace: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """A refusal to return instead of running the tool, or `None` to run it.

        Refusals are tool RESULTS, not exceptions: a denied call is an answer
        the model can work around, and raising would end a run over one tool the
        operator happened to forbid.
        """
        # The model's arguments against the schema it was given. A failure is a
        # tool RESULT carrying the errors, not an exception: the model asked for
        # something its own schema forbids, which it can fix on the next step, and
        # ending the run would discard every step before it. The bound is
        # `max_steps`, the loop's existing one -- a second retry budget would be a
        # second number to tune and the same answer.
        if self._validate_tool_arguments:
            schema = self._argument_schema(reported_name)
            if schema is not None:
                errors = validate_json_schema(schema, dict(arguments or {}))
                if errors:
                    self.hooks.emit_sync(
                        "onWarning",
                        {
                            "source": "agent",
                            "code": "tool_arguments_invalid",
                            "message": (
                                f'Tool "{reported_name}" was called with arguments its '
                                f"schema rejects: {'; '.join(errors)}. The errors were "
                                f"returned to the model."
                            ),
                            "details": {
                                "toolName": reported_name,
                                "callId": call.id,
                                "errors": list(errors),
                            },
                        },
                    )
                    return self._refuse(
                        call,
                        reported_name,
                        arguments,
                        reports,
                        # Addressed to the MODEL, so it says what to DO rather than
                        # only what went wrong. A bare validator message reads as an
                        # internal error, and models answer those by apologising
                        # instead of re-calling the tool.
                        f'Invalid arguments for "{reported_name}": {"; ".join(errors)}. '
                        f"Call the tool again with arguments matching its schema.",
                    )

        if self._policy is None:
            return None

        # The arguments of THIS call travel with the target, so a rule can
        # depend on them -- "a transfer over 1000 needs a human" rather than
        # "every transfer does". A policy that saw only the tool name had to
        # choose between asking about every call and asking about none, and a
        # gate that fires on every call is one people learn to click through.
        decision = self._policy.check(
            "agent",
            {"kind": "tool", "toolName": reported_name, "arguments": dict(arguments or {})},
            "execute",
        )
        if getattr(decision, "ask", False):
            return self._await_approval(run_id, step, call, reported_name, arguments, reports, trace,
                                        getattr(decision, "reason", None))
        if getattr(decision, "allow", True):
            return None
        return self._refuse(call, reported_name, arguments, reports,
                            getattr(decision, "reason", None) or DENIAL_DEFAULT_REASON)

    def _await_approval(
        self,
        run_id: str,
        step: int,
        call: Any,
        reported_name: str,
        arguments: Mapping[str, Any],
        reports: list[ToolCallReport],
        trace: Mapping[str, Any],
        reason: str | None,
    ) -> dict[str, Any] | None:
        """Suspend at the gate. Returns `None` when the answer was yes."""
        if self._gate is None:
            # A policy that asks with nobody to ask is a denial, not a pass:
            # the alternative is a rule that reads as a guard and is not one.
            return self._refuse(call, reported_name, arguments, reports, NO_APPROVER_NOTE)

        pending = PendingToolCall(
            call_id=str(call.id or ""),
            tool_name=reported_name,
            arguments=dict(arguments),
            step=step,
            requested_at=_now_ms(),
            run_id=run_id,
        )
        self._pending_tool_calls.append(pending)
        self.hooks.emit_sync(
            "onApprovalRequested",
            {"runId": run_id, "agentId": self.id, "step": step, "callId": call.id,
             "toolName": reported_name, "arguments": dict(arguments), "reason": reason,
             "trace": trace},
        )
        # Written BEFORE the approver is asked: the whole point of a checkpoint
        # is to survive a process that does not come back from that wait.
        if self._checkpoint is not None:
            self._checkpoint.set(CHECKPOINT_KEY_PREFIX + self.id, self.dump())

        as_call = _as_tool_call(call)
        decision = self._gate.check(as_call, step=step, run_id=run_id, reason=reason)
        self._pending_tool_calls = [
            p for p in self._pending_tool_calls if p.call_id != str(call.id or "")
        ]
        self.hooks.emit_sync(
            "onApprovalResolved",
            {"runId": run_id, "agentId": self.id, "step": step, "callId": call.id,
             "toolName": reported_name, "decision": decision.decision,
             "note": decision.note, "trace": trace},
        )

        # `result_for` rather than reading `decision.decision`: an APPROVE can
        # carry an override, and running the tool then is the exact failure it
        # exists to prevent. `None` is the one case that means "run it".
        answer = result_for(as_call, decision)
        if answer is None:
            return None
        part = answer.to_part()
        reports.append(
            ToolCallReport(
                call_id=str(call.id or ""),
                tool_name=reported_name,
                arguments=dict(arguments),
                latency_ms=0.0,
                error=str(part["content"]) if answer.is_error else None,
            )
        )
        return part

    def _argument_schema(self, reported_name: str) -> Any:
        """The registered tool's own parameter schema, or None.

        None for a BUILTIN tool, which carries no parameters at all -- reading a
        schema off `{"type": "web_search"}` would refuse a tool that is perfectly
        well formed.
        """
        tool = self._tools.get(reported_name)
        definition = getattr(tool, "definition", None) if tool is not None else None
        if not isinstance(definition, Mapping):
            return None
        kind = definition.get("type")
        if kind is not None and kind != "function":
            return None
        return definition.get("parameters")

    def _refuse(
        self,
        call: Any,
        reported_name: str,
        arguments: Mapping[str, Any],
        reports: list[ToolCallReport],
        reason: str,
    ) -> dict[str, Any]:
        """A refused call, recorded and handed back to the model as an error."""
        reports.append(
            ToolCallReport(
                call_id=str(call.id or ""),
                tool_name=reported_name,
                arguments=dict(arguments),
                latency_ms=0.0,
                error=reason,
            )
        )
        return {"type": "tool_result", "id": call.id, "content": reason, "isError": True}

    # -- serialisation -------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        """Enough to rebuild the conversation, not the wiring.

        Tools are recorded by NAME only: a snapshot that claimed to carry them
        would be a snapshot that cannot be restored, because the bodies are
        Python functions and the transport is a live socket.
        """
        return {
            "version": 1,
            "id": self.id,
            "system": self._system,
            "context": self._context,
            "history": self._history.dump(),
            "toolNames": self.tool_names(),
            "metadata": dict(self._metadata),
            "createdAt": self._history.created_at,
            "savedAt": time.time() * 1000,
            # Only when there are any: a checkpoint of a run that was not
            # suspended should not carry an empty list that reads as "resumed
            # from nothing" when it comes back.
            **(
                {"pendingToolCalls": [_pending_row(p) for p in self._pending_tool_calls]}
                if self._pending_tool_calls
                else {}
            ),
            # Same reasoning: absent rather than empty, so a snapshot that staged
            # nothing does not look like one whose staging was consumed.
            **(
                {"pendingInput": [dict(m) for m in self._pending_input]}
                if self._pending_input
                else {}
            ),
        }

    @classmethod
    def restore(
        cls,
        snapshot: Mapping[str, Any],
        *,
        client: Any,
        tools: Sequence[Any] = (),
        hooks: HookBus | None = None,
        **options: Any,
    ) -> AgentLoop:
        """A dumped conversation, wired to a LIVE client and live tools.

        The wiring is supplied, never restored: a snapshot carries tool NAMES,
        because the bodies are Python functions and the transport is a socket.

        Which makes the two sets worth comparing out loud. A tool the saved run
        used and this one does not provide will be called again -- the model
        learned it exists from the transcript it is about to be shown -- and it
        will fail as an unknown tool several turns later, far from the restore.
        A tool that is new is the milder half of the same surprise. Both are
        reported rather than reconciled: only the caller knows whether the
        difference was deliberate.
        """
        agent = cls(
            client,
            hooks=hooks,
            system=str(snapshot.get("system") or ""),
            context=str(snapshot.get("context") or ""),
            tools=tools,
            history=snapshot.get("history"),
            metadata=snapshot.get("metadata") or {},
            **options,
        )
        # Restored so `pending_approvals` answers the same after a restart as it
        # did before: that list IS the resume queue, and an empty one would say
        # the run had nothing outstanding.
        agent._pending_tool_calls = [
            PendingToolCall(
                call_id=str(row.get("callId") or ""),
                tool_name=str(row.get("toolName") or ""),
                arguments=dict(row.get("arguments") or {}),
                step=int(row.get("step") or 0),
                requested_at=float(row.get("requestedAt") or 0.0),
                run_id=row.get("runId"),
            )
            for row in snapshot.get("pendingToolCalls") or ()
        ]
        # Restored for the same reason, and it matters more: the staged message
        # exists NOWHERE else. A dropped approval can be asked for again; a
        # correction the user typed once is simply gone.
        agent._pending_input = [dict(m) for m in snapshot.get("pendingInput") or ()]
        saved = {str(name) for name in snapshot.get("toolNames") or ()}
        current = set(agent.tool_names())
        for name in sorted(saved - current):
            agent.hooks.emit_sync(
                "onWarning",
                {
                    "source": "agent",
                    "code": "tool_removed",
                    "message": (
                        f"Tool {name!r} was used in the saved conversation but is not "
                        f"provided now"
                    ),
                    "details": {"agentId": agent.id, "toolName": name},
                },
            )
        for name in sorted(current - saved):
            agent.hooks.emit_sync(
                "onWarning",
                {
                    "source": "agent",
                    "code": "tool_added",
                    "message": f"Tool {name!r} is new (not in the saved conversation)",
                    "details": {"agentId": agent.id, "toolName": name},
                },
            )
        return agent


def _as_tool_call(call: Any) -> ToolCall:
    """The loop's `Part` as the `ToolCall` the approval layer is written against.

    A conversion rather than a looser signature on that layer: `ToolCall` is what
    `result_for` and the gate take.

    The provider's opaque signature is deliberately NOT carried across. It rides
    on the assistant message's tool_call part, which is what gets echoed back;
    `ToolResult.to_part()` does not emit it, so setting it here would be a line
    that reads as preserving a token the next request depends on while doing
    nothing at all.
    """
    return ToolCall(
        id=str(getattr(call, "id", "") or ""),
        name=str(getattr(call, "name", "") or ""),
        arguments=dict(getattr(call, "arguments", None) or {}),
    )


def _pending_row(pending: PendingToolCall) -> dict[str, Any]:
    """One suspended call, in the camelCase a snapshot travels in."""
    return {
        "callId": pending.call_id,
        "toolName": pending.tool_name,
        "arguments": dict(pending.arguments),
        "step": pending.step,
        "requestedAt": pending.requested_at,
        "runId": pending.run_id,
    }


class _GuardrailStop(Exception):
    """Internal: a guardrail returned a failing decision."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


#: The data-free placeholder, used when nothing else resolves.
TOOL_OUTPUT_WITHHELD = "Output withheld by an output guardrail."


@dataclass
class ToolGuardrailContext:
    """What a per-call guardrail is shown.

    One shape for both sides: `result` is set on the output side and None on the
    input side, so a rule that reads both is written once. Attributes rather
    than a dict for the same reason `GuardrailContext` uses them -- a one-line
    rule should read like one.
    """

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    step: int = 0
    trace: Mapping[str, Any] = field(default_factory=dict)
    #: What the tool returned, on the output side only.
    result: str | None = None


def _passed(decision: Any) -> bool:
    """Whether a guardrail decision says yes.

    Accepts an object with `.pass_`/`.passed` or a mapping with `"pass"`,
    because `pass` is a keyword here and a Python rule should not have to
    contort to say the same thing the TypeScript one says.
    """
    if isinstance(decision, Mapping):
        return bool(decision.get("pass"))
    for attr in ("pass_", "passed"):
        if hasattr(decision, attr):
            return bool(getattr(decision, attr))
    return bool(decision)


def _decision_reason(decision: Any) -> str | None:
    if isinstance(decision, Mapping):
        reason = decision.get("reason")
    else:
        reason = getattr(decision, "reason", None)
    return str(reason) if isinstance(reason, str) and reason else None


def _decision_replacement(decision: Any) -> str | None:
    if isinstance(decision, Mapping):
        value = decision.get("replaceWith") or decision.get("replace_with")
    else:
        value = getattr(decision, "replace_with", None) or getattr(decision, "replaceWith", None)
    return str(value) if isinstance(value, str) and value else None


@dataclass
class GuardrailContext:
    """What a guardrail is shown.

    Attributes rather than a dict, because the examples read `ctx.cost.total` --
    and `ctx["cost"]["total"]` in a one-line budget rule is the kind of noise
    that makes people not write the rule at all.
    """

    #: `"input"` before the LLM call, `"output"` after it.
    kind: str
    step: int
    agent_id: str
    trace: Mapping[str, Any] = field(default_factory=dict)
    #: Input side.
    messages: Sequence[Mapping[str, Any]] = ()
    system: str | None = None
    #: Output side.
    response: Completion | None = None
    usage: Usage | None = None
    #: None when the model is unpriced -- never 0.0, which would read as free.
    cost: Any = None


def _as_messages(input_: Any) -> list[dict[str, Any]]:
    """Whatever the caller passed, as messages to append."""
    if isinstance(input_, str):
        return [{"role": "user", "content": [{"type": "text", "text": input_}]}]
    if isinstance(input_, list) and input_ and isinstance(input_[0], Mapping):
        if "role" in input_[0]:
            return [dict(m) for m in input_]
        return [{"role": "user", "content": list(input_)}]
    return [{"role": "user", "content": input_}]


def _as_content(value: Any) -> Any:
    """A tool's return value, in the shape a `tool_result` part carries."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _with_timeout(call: Callable[[], Any], timeout: float, tool: Any) -> Any:
    """Run a tool body, giving up after `timeout` seconds.

    The worker thread is NOT killed on timeout -- Python cannot, and pretending
    otherwise would leak a half-finished side effect while reporting a clean
    stop. The loop stops WAITING; the body runs to completion in the background.
    """
    import asyncio
    import contextvars
    import inspect

    if tool.is_async:
        return asyncio.run(asyncio.wait_for(_await_call(call), timeout))
    with ThreadPoolExecutor(max_workers=1) as pool:
        # A worker thread starts with an EMPTY context, so the tool would not see
        # `current_tool_run()` -- and a delegating tool would silently fall back
        # to inheriting nothing, which is the bug this exists to fix. Copying the
        # caller's context is what carries it across the thread boundary.
        future = pool.submit(contextvars.copy_context().run, call)
        value = future.result(timeout=timeout)
    if inspect.isawaitable(value):
        return asyncio.run(_await_value(value))
    return value


async def _await_call(call: Callable[[], Any]) -> Any:
    return await call()


async def _await_value(value: Any) -> Any:
    return await value


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_TOOL_TIMEOUT",
    "AgentLoop",
    "GuardrailContext",
    "ToolNameCollision",
]
