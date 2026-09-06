"""Resolve a tool, check what it was given, pick a model, run it.

The order is the contract, and each step is where it is for a reason:

- the INPUT is checked before a model is chosen, so a call that was never going
  to work costs nothing;
- the KEYS are checked before the first attempt, so a tool whose entire model
  chain is unreachable fails once with a sentence naming the providers, rather
  than three times with "no API key";
- the OUTPUT is checked after, and a mismatch WARNS rather than raises -- the
  answer has already been paid for, and throwing it away turns a shape problem
  into a shape problem plus a lost response.

Clients are pooled per provider, so a hundred calls to five tools open five
clients and share one rate limiter.

Transposed from `unified-library-ts/src/plugins/internal-tools/runner/runner.ts`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..bus.hook_bus import HookBus
from .registry import ToolRegistry
from .types import (
    CompatFile,
    InternalTool,
    InternalToolContext,
    InternalToolError,
    JsonSchema,
)

#: What each declared schema type accepts. Structural only -- a full validator
#: is a dependency, and the shape is what actually breaks a caller reading
#: `output["label"]` off a list.
_TYPES: Mapping[str, type | tuple[type, ...]] = {
    "object": Mapping,
    "array": (list, tuple),
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


@dataclass
class InternalToolRunnerConfig:
    """Everything the runner needs that is not the tool itself."""

    registry: ToolRegistry
    #: Provider -> key. A tool's chosen model needs the matching one.
    api_keys: Mapping[str, str] = field(default_factory=dict)
    hooks: HookBus | None = None
    catalog: Any = None
    #: An `Engine`, when the tools should share a fleet's queue and hooks.
    engine: Any = None
    #: A transport, when they should not -- a stub in a test, a proxy for one
    #: tenant. Passing both is a real combination: the engine keeps its policy
    #: and the transport does the sending.
    transport: Any = None
    #: Used by a tool that declared no model preference.
    default_model: str | None = None
    #: Benchmark-derived recommendations, keyed by tool id.
    compat: CompatFile | None = None
    #: Applied to every pooled client.
    client_options: Mapping[str, Any] = field(default_factory=dict)
    #: For definitions that size their own output. Built on demand when absent.
    counter: Any = None


class InternalToolRunner:
    """Runs registered tools, pooling a client per provider."""

    def __init__(self, config: InternalToolRunnerConfig) -> None:
        self._config = config
        self._hooks = config.hooks or HookBus()
        self._clients: dict[str, Any] = {}
        self._counter = config.counter

    # -- public --------------------------------------------------------------

    @property
    def registry(self) -> ToolRegistry:
        """The registry, so adjacent tooling can resolve a tool without a second one."""
        return self._config.registry

    @property
    def hooks(self) -> HookBus:
        """The bus this runner and its clients emit on."""
        return self._hooks

    @property
    def pool_size(self) -> int:
        """How many clients are open."""
        return len(self._clients)

    def run(self, tool_id: str, value: Any = None) -> Any:
        """Run the tool with this id."""
        tool = self._config.registry.get(tool_id)
        if tool is None:
            raise InternalToolError(f"tool not found in registry: {tool_id}", tool_id=tool_id)
        return self.run_direct(tool, value)

    def run_direct(self, tool: InternalTool, value: Any = None) -> Any:
        """Run a tool instance, bypassing the registry."""
        self._validate_input(tool, value)
        models = self._resolve_models(tool)
        if not models:
            return self._run_without_model(tool, value)
        self._assert_keys(tool, models)
        return self._run_with_model(tool, value, models)

    def destroy(self) -> None:
        """Close every pooled client."""
        for client in self._clients.values():
            client.destroy()
        self._clients.clear()

    # -- validation ----------------------------------------------------------

    def _validate_input(self, tool: InternalTool, value: Any) -> None:
        """Refuse a call that cannot work, before it costs anything.

        Required fields only, and only for an object schema: this is the check
        that pays for itself, and a fuller validator would be a dependency to
        catch what the model's own answer already reports.
        """
        schema = tool.input_schema
        if not schema or schema.get("type") != "object":
            return
        if not isinstance(value, Mapping):
            got = type(value).__name__
            raise InternalToolError(
                f"tool {tool.id} expects an object as input, got {got}", tool_id=tool.id
            )
        required = schema.get("required") or ()
        missing = [key for key in required if key not in value]
        if missing:
            raise InternalToolError(
                f"tool {tool.id} is missing required input: {', '.join(missing)}",
                tool_id=tool.id,
            )

    def _validate_output(self, tool: InternalTool, output: Any) -> None:
        """Report a shape the tool promised and did not deliver.

        A warning, not an exception. The call is already paid for and the value
        may well be usable; what must not happen is that it reaches a caller
        who was told to expect an object, silently, as a list.
        """
        expected = self._declared_type(tool.output_schema)
        if expected is None:
            return
        accepted = _TYPES.get(expected)
        if accepted is None:
            return
        # `bool` is an `int` in Python, so a plain number check accepts `True`
        # and a tool that promised a confidence would be handed a flag.
        if expected == "boolean":
            matches = isinstance(output, bool)
        else:
            matches = isinstance(output, accepted) and not isinstance(output, bool)
        if matches:
            return
        self._hooks.emit_sync(
            "onWarning",
            {
                "source": "agent",
                "code": "output_schema_mismatch",
                "message": (
                    f"tool {tool.id} output does not match its schema: "
                    f"expected {expected}, got {type(output).__name__}"
                ),
                "details": {"toolId": tool.id, "expected": expected},
            },
        )

    @staticmethod
    def _declared_type(schema: JsonSchema | None) -> str | None:
        if not schema:
            return None
        declared = schema.get("type")
        return declared if isinstance(declared, str) else None

    # -- models --------------------------------------------------------------

    def _resolve_models(self, tool: InternalTool) -> list[str]:
        """The chain to try, best first.

        Benchmark recommendations outrank the tool's own preference: the tool
        says what it was written against, and a measurement of what actually
        works is worth more than that.
        """
        chain: list[str] = []
        compat = (self._config.compat or {}).get(tool.id)
        if compat is not None:
            chain.extend(compat.recommended)
        preference = tool.model_preference
        if preference is not None:
            if preference.preferred_model:
                chain.append(preference.preferred_model)
            chain.extend(preference.fallback_models)
        if not chain and self._config.default_model:
            chain.append(self._config.default_model)
        return list(dict.fromkeys(chain))

    def _assert_keys(self, tool: InternalTool, models: Sequence[str]) -> None:
        """One clear refusal, rather than one per model in the chain."""
        wanted = {self._split(model)[0] for model in models}
        available = {p for p, key in (self._config.api_keys or {}).items() if key}
        if wanted & available:
            return
        raise InternalToolError(
            f"tool {tool.id} needs an API key for one of "
            f"[{', '.join(sorted(wanted))}]; the runner has keys for "
            f"[{', '.join(sorted(available)) or 'nothing'}]",
            tool_id=tool.id,
        )

    @staticmethod
    def _split(model_id: str) -> tuple[str, str]:
        provider, _, model = model_id.partition("/")
        if not model:
            raise InternalToolError(
                f"invalid model id {model_id!r}: name the provider, as 'provider/model'"
            )
        return provider, model

    def _client_for(self, model_id: str) -> Any:
        """A pooled client for this model.

        Pooled by "provider/model", NOT by provider as the TypeScript does. A
        client is pinned to its model at construction, so a pool keyed on the
        provider hands the second model of a chain the client built for the
        first -- and a same-provider fallback then silently re-runs the model
        that just failed, reports the fallback's name, and fails again.

        Pooling by provider was there to keep one rate limiter per provider.
        That job belongs to the engine, which every client here shares when one
        is configured; without an engine there is no limiter to share either
        way.
        """
        from ..helpers.llm import LLM

        provider, _ = self._split(model_id)
        api_key = (self._config.api_keys or {}).get(provider)
        if not api_key:
            raise InternalToolError(f"no API key for provider {provider!r}")

        client = self._clients.get(model_id)
        if client is None:
            client = LLM(
                model=model_id,
                api_key=api_key,
                hooks=self._hooks,
                engine=self._config.engine,
                transport=self._config.transport,
                catalog=self._config.catalog,
                **dict(self._config.client_options),
            )
            self._clients[model_id] = client
        return client

    def _token_counter(self) -> Any:
        if self._counter is None:
            from ..tokens import HybridCounter

            self._counter = HybridCounter(self._config.catalog)
        return self._counter

    # -- execution -----------------------------------------------------------

    def _run_without_model(self, tool: InternalTool, value: Any) -> Any:
        """A tool whose body is code. It still gets the hooks and the counter."""
        started = time.perf_counter()
        self._emit_sync(
            "onInternalToolCallStart",
            {"toolId": tool.id, "input": value, "chosenModel": "", "attempt": 1},
        )
        try:
            output = tool.execute(
                value,
                InternalToolContext(
                    hooks=self._hooks, tool_id=tool.id, counter=self._token_counter()
                ),
            )
        except Exception as exc:
            self._emit_sync(
                "onInternalToolCallError",
                {
                    "toolId": tool.id,
                    "input": value,
                    "chosenModel": "",
                    "error": exc,
                    "attempt": 1,
                    "willRetry": False,
                },
            )
            raise
        self._validate_output(tool, output)
        self._emit_sync(
            "onInternalToolCallComplete",
            {
                "toolId": tool.id,
                "input": value,
                "output": output,
                "chosenModel": "",
                "latencyMs": (time.perf_counter() - started) * 1000,
                "attempts": 1,
            },
        )
        return output

    def _run_with_model(self, tool: InternalTool, value: Any, models: Sequence[str]) -> Any:
        """Try each model in turn; the first that answers wins."""
        started = time.perf_counter()
        failures: list[tuple[str, Exception]] = []

        for index, model_id in enumerate(models):
            attempt = index + 1
            provider = model_id.partition("/")[0]
            if not (self._config.api_keys or {}).get(provider):
                failures.append((model_id, InternalToolError(f"no API key for {provider}")))
                continue
            try:
                client = self._client_for(model_id)
            except Exception as exc:  # noqa: BLE001 -- one unbuildable client must
                # not end the chain; the next model may well be reachable.
                failures.append((model_id, exc))
                continue

            self._emit_sync(
                "onInternalToolCallStart",
                {"toolId": tool.id, "input": value, "chosenModel": model_id, "attempt": attempt},
            )

            captured: list[Any] = []
            try:
                output = tool.execute(
                    value,
                    InternalToolContext(
                        hooks=self._hooks,
                        client=client,
                        model_id=model_id,
                        tool_id=tool.id,
                        counter=self._token_counter(),
                        record_completion=captured.append,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- a tool body may raise anything,
                # and the whole point of a model chain is that one failure is not final.
                failures.append((model_id, exc))
                will_retry = index < len(models) - 1
                self._emit_sync(
                    "onInternalToolCallError",
                    {
                        "toolId": tool.id,
                        "input": value,
                        "chosenModel": model_id,
                        "error": exc,
                        "attempt": attempt,
                        "willRetry": will_retry,
                    },
                )
                if will_retry:
                    self._emit_sync(
                        "onWarning",
                        {
                            "source": "agent",
                            "code": "internal_tool_fallback",
                            "message": (
                                f"tool {tool.id} failed on {model_id}, trying the next model"
                            ),
                            "details": {
                                "toolId": tool.id,
                                "failedModel": model_id,
                                "errorMessage": str(exc),
                            },
                        },
                    )
                continue

            self._validate_output(tool, output)
            self._emit_sync(
                "onInternalToolCallComplete",
                {
                    "toolId": tool.id,
                    "input": value,
                    "output": output,
                    "chosenModel": model_id,
                    "latencyMs": (time.perf_counter() - started) * 1000,
                    "attempts": attempt,
                    "usage": captured[-1].usage if captured else None,
                },
            )
            return output

        summary = "; ".join(f"{model}: {error}" for model, error in failures)
        raise InternalToolError(
            f"tool {tool.id} failed on all {len(failures)} model(s): {summary}", tool_id=tool.id
        )

    def _emit_sync(self, name: str, payload: Mapping[str, Any]) -> None:
        self._hooks.emit_sync(name, dict(payload))


__all__ = ["InternalToolRunner", "InternalToolRunnerConfig"]
