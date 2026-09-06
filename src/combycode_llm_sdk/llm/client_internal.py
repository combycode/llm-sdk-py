"""LLMClient internals.

Transposed from `unified-library-ts/src/llm/client-internal.ts`.

Request normalization, system extraction, context building, provider/adapter
resolution, and structured-output parsing. Split out of `client.py` to keep the
class file focused on the public surface.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ..types.request_context import RequestContext
from .output_errors import InvalidFinalOutputError
from .types.messages import Message
from .types.provider import ApiType, ProviderAdapter, ProviderName

if TYPE_CHECKING:  # `import type { LLMClient }` -- the cycle exists in TS too.
    from .client_base import BaseLLMClient

PRIORITY_INTERACTIVE = 1
PRIORITY_BACKGROUND = 2

#: `provider -> ApiType` when neither the caller nor the catalog names one.
#:
#: `generate` is Google's stable, production-recommended API. The Interactions
#: API (`api='interactions'`) is Beta with frequent breaking schema changes (e.g.
#: May 2026 `turn_list` -> `step_list`) -- opt in explicitly when you need its
#: server-side state / agentic steps.
#: Every API surface, which is NOT the same set as the defaults below: no
#: provider DEFAULTS to `interactions` (Google's is Beta and opt-in), so
#: validating a caller's `api=` against the defaults rejected the one value that
#: can only ever be asked for explicitly.
API_TYPES: frozenset[str] = frozenset(
    {"completions", "responses", "messages", "interactions", "generate"}
)

_API_DEFAULTS: dict[str, ApiType] = {
    "anthropic": "messages",
    "openai": "responses",
    "google": "generate",
    "xai": "responses",
    "openrouter": "completions",
}


def _now_ms() -> int:
    """`Date.now()` -- integer milliseconds since the epoch."""
    return int(time.time() * 1000)


def build_assistant_message(
    response: Mapping[str, Any],
    origin: Mapping[str, Any],
) -> Message:
    """Build a history assistant message from a response, stamped with provenance.

    `origin` is `{provider, model, api}`. On a stateful API (responses /
    interactions) the message origin carries the server-state id, so a later turn
    can continue server-side instead of resending the transcript. Shared by
    `LLMClient.assistant_message` (manual multi-turn) and the agent loop (so its
    tool turns also chain server-side).
    """
    stateful = origin.get("api") in ("responses", "interactions")
    message_origin: dict[str, Any] = {
        "provider": origin.get("provider"),
        "model": origin.get("model"),
    }
    if stateful and response.get("id"):
        message_origin["serverStateId"] = response["id"]
    return {
        "role": "assistant",
        "content": response.get("content"),
        "id": response.get("id") or str(uuid.uuid4()),
        "createdAt": _now_ms(),
        "origin": message_origin,
    }


def normalize_input(input_: str | list[Any]) -> list[Message]:
    """`string | ContentPart[] | Message[]` -> `Message[]`.

    The discriminator is `'role' in input[0]` -- a Message has a role, a
    ContentPart has a `type`. An EMPTY list falls through to the content-part
    branch and yields one user message with empty content, exactly as in the
    TypeScript: `input.length > 0` guards the `in` test, not the branch.

    A message list is returned BY IDENTITY (`return input as Message[]`), not
    copied. The client stamps provenance onto these messages and the agent loop
    appends to the same list; copying here would detach both.
    """
    if isinstance(input_, str):
        return [{"role": "user", "content": input_}]
    if input_ and "role" in input_[0]:
        return input_
    return [{"role": "user", "content": input_}]


def _content_to_text(content: Sequence[Mapping[str, Any]]) -> str:
    """The system-message join, which is NEWLINE-separated.

    Deliberately not `content_text` from `types.messages`: that one joins with
    the empty string because it reassembles streamed fragments of one message.
    This one concatenates separate system PARTS, which run together into one
    unreadable line without the newline.
    """
    return "\n".join(p.get("text") or "" for p in content if p.get("type") == "text")


def extract_system(messages: Sequence[Message]) -> dict[str, Any]:
    """Lift any `role='system'` messages out of the input array.

    Anthropic and some other providers expect `system` as a top-level parameter,
    not as a message role. Extracting here in the client gives callers a single,
    provider-neutral way to set per-call system text: either pass
    `options['system']`, or include `role='system'` messages in the input (they
    get concatenated). **Adapters never see `role='system'`.**

    Returns `{system, messages}`; `system` is absent when there were none, never
    an empty string.
    """
    system_texts: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content")
            text = content if isinstance(content, str) else _content_to_text(content or [])
            if text:
                system_texts.append(text)
        else:
            rest.append(m)
    out: dict[str, Any] = {"messages": rest}
    if system_texts:
        out["system"] = "\n\n".join(system_texts)
    return out


_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"```\s*$", re.IGNORECASE)


def parse_structured(text: str) -> Any:
    """Strip leading/trailing markdown fences and parse.

    Exported so AgentLoop and the helper layer share the same parsing rules.
    """
    stripped = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip())).strip()
    try:
        return json.loads(stripped)
    except (ValueError, TypeError) as cause:
        # Typed, differentiated failure -- callers can catch
        # `InvalidFinalOutputError` and inspect `.raw_text`, instead of catching
        # a bare JSONDecodeError.
        raise InvalidFinalOutputError(text, cause) from cause


@dataclass(frozen=True)
class ClientRouting:
    """The routing names `build_context` needs from a client.

    These were read with `as unknown as {queueName: string}` casts straight into
    LLMClient's privates -- which compiles, and silently returns `undefined` the
    day a field is renamed. LLMClient now exposes them deliberately as
    `client.routing`, so a rename is an error instead.
    """

    queue_name: str
    config_name: str
    cache_name: str


def build_context(client: BaseLLMClient, options: Mapping[str, Any]) -> RequestContext:
    """Assemble the RequestContext for one call.

    Precedence differs per field and is not uniform: `configName`, `cacheName`
    and `cacheKey` let the OPTION win over a caller-supplied ctx, while
    `sessionId`, `clientId` and `queueName` let the CTX win over the client's
    own. That is deliberate -- the first three are semantic and the agent may
    override them; `queueName` is infrastructure and the agent may not.
    """
    provided: Mapping[str, Any] = options.get("ctx") or {}
    ctx: RequestContext = dict(provided)
    ctx["sessionId"] = provided.get("sessionId") or client.session_id
    ctx["clientId"] = provided.get("clientId") or client.id
    ctx["queueName"] = provided.get("queueName") or client.routing.queue_name
    ctx["configName"] = (
        options.get("configName") or provided.get("configName") or client.routing.config_name
    )
    ctx["cacheName"] = (
        options.get("cacheName") or provided.get("cacheName") or client.routing.cache_name
    )
    ctx["cacheKey"] = options.get("cacheKey") or provided.get("cacheKey")
    if not ctx.get("callId"):
        ctx["callId"] = f"call_{str(uuid.uuid4())[:8]}"
    # Mint-if-absent: server/agent set requestId upstream; a direct LLM call
    # mints it here so every request carries one (the request half of the trace
    # id).
    if not ctx.get("requestId"):
        ctx["requestId"] = f"req_{str(uuid.uuid4())[:12]}"
    return ctx


def resolve_api(
    provider: ProviderName,
    api: str | None = None,
    preferred: str | None = None,
) -> ApiType:
    """Which API surface this call goes to.

    `preferred` is the catalog's per-model preference, when the model is a known
    one. A model's own preference beats the provider default: the catalog has
    carried `preferredApi` per model from the start and nothing consulted it
    here, so every OpenAI model routed to Responses -- including the six that
    cannot use it. `gpt-audio` came back "not supported with the Responses API";
    the three `*-search` models are Chat Completions-only for the same reason.
    """
    named = api if api and api != "auto" else (preferred or None)
    if named:
        if named not in API_TYPES:
            raise ValueError(f'unknown api "{named}" -- expected one of {sorted(API_TYPES)}')
        return cast("ApiType", named)
    return _API_DEFAULTS.get(provider, "completions")


def resolve_adapter(config: Mapping[str, Any], api: ApiType) -> ProviderAdapter:
    """A pre-built adapter, or a factory called with (provider, apiKey, api, baseURL)."""
    adapter = config.get("adapter")
    if not adapter:
        raise ValueError("LLMClient: adapter or AdapterFactory must be supplied")
    if callable(adapter) and not _is_adapter_instance(adapter):
        factory: Callable[..., ProviderAdapter] = adapter
        return factory(config["provider"], config["apiKey"], api, config.get("baseURL"))
    built: ProviderAdapter = adapter
    return built


def _is_adapter_instance(value: Any) -> bool:
    """`typeof a === 'function'` does not translate directly.

    A JavaScript class instance is never `typeof 'function'`, but a Python
    adapter INSTANCE can be callable if it defines `__call__`, and every adapter
    CLASS is callable. The discriminator is therefore what the object has, not
    whether it can be called: an adapter answers `build_request`, a factory does
    not.
    """
    return hasattr(value, "build_request")


__all__ = [
    "API_TYPES",
    "PRIORITY_BACKGROUND",
    "PRIORITY_INTERACTIVE",
    "ClientRouting",
    "build_assistant_message",
    "build_context",
    "extract_system",
    "normalize_input",
    "parse_structured",
    "resolve_adapter",
    "resolve_api",
]
