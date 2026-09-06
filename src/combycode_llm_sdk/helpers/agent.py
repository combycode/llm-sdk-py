"""`Agent` -- an agent you hold, rather than a call you make.

Transposed from `unified-library-ts/src/helpers/agent.ts` (`createAgent`), over
the ported `AgentLoop`.

    agent = Agent(model="anthropic/claude-haiku-4.5", api_key=key, tools=[get_weather])
    agent.complete("What is the weather in Paris?").text
    agent.complete("And tomorrow?").text        # it remembers the first turn

The difference from `complete(tools=...)` is the object. This one keeps the
conversation, has an identity hooks and telemetry bind to, records what each
step cost, and can be watched, stopped and handed to another agent as a tool.

Hooks are subscribed with a decorator per event -- `@agent.on_warning` -- rather
than a stringly-typed `on("onWarning", ...)`. Fifty-one explicit methods is more
lines here and one fewer thing to misspell at every call site; the same choice
`Engine` already made.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

from ..agent.loop import DEFAULT_TOOL_TIMEOUT, AgentLoop
from ..agent.reflect_retry import ReflectAndRetry
from ..bus.hook_bus import HookBus
from ..results import Completion

Handler = Callable[[Any], Any]


class Agent:
    """The public façade over `AgentLoop`.

    Synchronous, like `LLM`. There is no `AsyncAgent` in the reviewed contract,
    so there is none here: a second core with no caller is two implementations
    to keep in step and one of them never exercised.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        client: Any = None,
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        engine: Any = None,
        transport: Any = None,
        api: str | None = None,
        hooks: HookBus | None = None,
        system: str | Callable[[], str] = "",
        context: str = "",
        tools: Sequence[Any] = (),
        history: Any = None,
        label: str | None = None,
        source: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning: Any = None,
        cache: Any = None,
        max_steps: int | None = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        parallel_tool_calls: bool = True,
        reflect_and_retry: ReflectAndRetry | None = None,
        lazy_tools: Any = None,
        before: Sequence[Handler] = (),
        after: Sequence[Handler] = (),
        tool_name_collision: str = "warn",
        **options: Any,
    ) -> None:
        if client is None and not model:
            raise TypeError("Agent: pass either client= or model=")

        if client is None:
            from .llm import LLM

            client = LLM(
                **{
                    k: v
                    for k, v in {
                        "model": model,
                        "provider": provider,
                        "api_key": api_key,
                        "base_url": base_url,
                        "engine": engine,
                        "transport": transport,
                        "api": api,
                        "hooks": hooks,
                    }.items()
                    if v is not None
                }
            )

        self._loop = AgentLoop(
            client,
            # The loop and the client share one bus, or a hook subscribed on the
            # agent would miss every `onCompletion` the client emits -- which is
            # most of what there is to watch.
            hooks=hooks if hooks is not None else client.hooks,
            system=system,
            context=context,
            tools=tools,
            history=history,
            label=label,
            source=source,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=reasoning,
            cache=cache,
            max_steps=max_steps,
            tool_timeout=tool_timeout,
            parallel_tool_calls=parallel_tool_calls,
            reflect_and_retry=reflect_and_retry,
            lazy_tools=lazy_tools,
            before=before,
            after=after,
            tool_name_collision=tool_name_collision,
            options=options,
        )

    # -- identity ------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._loop.id

    @property
    def model(self) -> str:
        return self._loop.model

    @property
    def client(self) -> Any:
        return self._loop.client

    @property
    def engine(self) -> Any:
        return getattr(self._loop.client, "engine", None)

    @property
    def hooks(self) -> HookBus:
        return self._loop.hooks

    @property
    def history(self) -> Any:
        return self._loop.history

    @property
    def system(self) -> str:
        return self._loop.system

    @property
    def running(self) -> bool:
        return self._loop.running

    @property
    def reports(self) -> Sequence[Any]:
        return self._loop.reports

    @property
    def last_report(self) -> Any:
        return self._loop.last_report

    def __repr__(self) -> str:
        return f"<Agent {self._loop.label or self.id} model={self.model!r}>"

    # -- running -------------------------------------------------------------

    def complete(self, input_: Any = None, **options: Any) -> Completion:
        """One run: steps until the model stops asking for tools."""
        return self._loop.complete(input_, **options)

    def run(self, input_: Any = None, **options: Any) -> Completion:
        """`complete` under the name the delegation examples use."""
        return self._loop.complete(input_, **options)

    def stop(self) -> None:
        self._loop.stop()

    def destroy(self) -> None:
        self._loop.destroy()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

    # -- tools ---------------------------------------------------------------

    def add_tool(self, tool: Any) -> None:
        self._loop.register_tool(tool)

    def remove_tool(self, name: str) -> None:
        self._loop.remove_tool(name)

    def tool_names(self) -> list[str]:
        """Every REGISTERED tool, lazy ones included."""
        return self._loop.tool_names()

    def has_tool(self, name: str) -> bool:
        """Whether the tool is registered -- declared or not."""
        return self._loop.has_tool(name)

    def declared_tools(self) -> list[dict[str, Any]]:
        """What actually goes on the wire. Lazy tools are absent by design."""
        return self._loop.declared_tools()

    def dump(self) -> dict[str, Any]:
        return self._loop.dump()

    # -- hooks ---------------------------------------------------------------

    def _decorate(self, name: str, handler: Handler) -> Handler:
        unsubscribe = self._loop.hooks.on(name, handler)
        # The handler comes back, so the decorated name still refers to the
        # function; `.unsubscribe()` rides along for the caller who needs it.
        handler.unsubscribe = unsubscribe  # type: ignore[attr-defined]
        return handler

    def on_agent_create(self, handler: Handler) -> Handler:
        """Subscribe to `onAgentCreate`."""
        return self._decorate("onAgentCreate", handler)

    def on_agent_destroy(self, handler: Handler) -> Handler:
        """Subscribe to `onAgentDestroy`."""
        return self._decorate("onAgentDestroy", handler)

    def on_run_start(self, handler: Handler) -> Handler:
        """Subscribe to `onRunStart`."""
        return self._decorate("onRunStart", handler)

    def on_run_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onRunComplete`."""
        return self._decorate("onRunComplete", handler)

    def on_run_error(self, handler: Handler) -> Handler:
        """Subscribe to `onRunError`."""
        return self._decorate("onRunError", handler)

    def on_step_start(self, handler: Handler) -> Handler:
        """Subscribe to `onStepStart`."""
        return self._decorate("onStepStart", handler)

    def on_step_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onStepComplete`."""
        return self._decorate("onStepComplete", handler)

    def on_tool_call_start(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallStart`."""
        return self._decorate("onToolCallStart", handler)

    def on_tool_call_complete(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallComplete`."""
        return self._decorate("onToolCallComplete", handler)

    def on_tool_call_error(self, handler: Handler) -> Handler:
        """Subscribe to `onToolCallError`."""
        return self._decorate("onToolCallError", handler)

    def on_tool_search(self, handler: Handler) -> Handler:
        """Subscribe to `onToolSearch`."""
        return self._decorate("onToolSearch", handler)

    def on_completion(self, handler: Handler) -> Handler:
        """Subscribe to `onCompletion` -- one per LLM call, not per run."""
        return self._decorate("onCompletion", handler)

    def on_warning(self, handler: Handler) -> Handler:
        """Subscribe to `onWarning`."""
        return self._decorate("onWarning", handler)

    def on_guardrail_triggered(self, handler: Handler) -> Handler:
        """Subscribe to `onGuardrailTriggered`."""
        return self._decorate("onGuardrailTriggered", handler)


__all__ = ["Agent"]
