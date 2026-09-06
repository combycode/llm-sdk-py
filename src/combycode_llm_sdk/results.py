"""What a completed call gives back: frozen dataclasses, snake_case throughout.

Python-native, and the API contract's own words for why::

    A dict would lose autocomplete and type checking, which is most of the value
    of shipping `py.typed`.

This is the boundary between the wire vocabulary and the Python one. Inside,
`LLMClient` speaks the camelCase dicts the specs and the recorded corpus are
written against; from here out everything is `finish_reason` and `input_tokens`.
`Completion.of` is the ONLY crossing, so a renamed field is one edit rather than
a hunt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .cost import Cost, compute_cost
from .llm.client_internal import build_assistant_message
from .llm.types.messages import Message, final_answer_text


@dataclass(frozen=True)
class Usage:
    """Tokens billed for one call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    #: Audio models only. Absent (None), not 0, when the call carried no audio --
    #: a zero would price silence as if audio had been sent.
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    #: The tier the provider actually billed, raw as it reported it, and the
    #: catalog key it normalises to. Both absent when the provider said nothing.
    service_tier: str | None = None
    pricing_tier: str | None = None

    @staticmethod
    def of(raw: Mapping[str, Any] | None) -> Usage:
        raw = raw or {}
        return Usage(
            input_tokens=_int(raw.get("inputTokens")),
            output_tokens=_int(raw.get("outputTokens")),
            total_tokens=_int(raw.get("totalTokens")),
            cached_tokens=_int(raw.get("cachedTokens")),
            cache_write_tokens=_int(raw.get("cacheWriteTokens")),
            reasoning_tokens=_int(raw.get("reasoningTokens")),
            audio_input_tokens=_opt_int(raw.get("audioInputTokens")),
            audio_output_tokens=_opt_int(raw.get("audioOutputTokens")),
            service_tier=raw.get("serviceTier"),
            pricing_tier=raw.get("pricingTier"),
        )


@dataclass(frozen=True)
class Part:
    """One piece of the assistant's reply.

    A single shape with optional fields rather than a class per kind: the parts
    are read by branching on `type` (`p.type == "tool_call"`), and the kinds are
    an OPEN set -- a provider adding one must not break a consumer, which a
    closed union of classes would guarantee it did.
    """

    type: str
    text: str | None = None
    #: `commentary` or `final_answer` on models that distinguish them (the codex
    #: family); None everywhere else, which reads as "this is the answer".
    phase: str | None = None
    #: Tool calls.
    id: str | None = None
    name: str | None = None
    arguments: Mapping[str, Any] | None = None
    #: Anything the unified part carried that has no field here -- media ids,
    #: provider metadata. Kept rather than dropped: this is a view, not a filter.
    raw: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def of(raw: Mapping[str, Any]) -> Part:
        return Part(
            type=str(raw.get("type") or ""),
            text=raw.get("text"),
            phase=raw.get("phase"),
            id=raw.get("id"),
            name=raw.get("name"),
            arguments=raw.get("arguments"),
            raw=raw,
        )


@dataclass(frozen=True)
class ToolCall:
    """One tool call, as the model asked for it.

    Two fields travel with a tool round-trip and are easy to miss, because most
    turns leave them absent -- so code written against the simple case works
    until the day a provider sends them, and then fails in a way that reads as a
    model error rather than a dropped field.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    #: An opaque provider token (Gemini's thought signature) that must be echoed
    #: back VERBATIM or the next turn is rejected. Opaque means opaque: it is
    #: never parsed, normalised or shortened here.
    signature: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """The answer to one `ToolCall`."""

    call_id: str
    content: Any
    is_error: bool = False
    #: Echoed back from the call this answers.
    signature: str | None = None

    @staticmethod
    def for_call(call: ToolCall, content: Any, *, is_error: bool = False) -> ToolResult:
        """The result of a specific call, carrying its id and signature.

        Built from the call rather than assembled by hand: correlating by
        POSITION is wrong the moment a turn contains several calls that complete
        out of order, and reconstructing history by hand is exactly where the
        signature gets dropped.
        """
        return ToolResult(
            call_id=call.id, content=content, is_error=is_error, signature=call.signature
        )

    def to_part(self) -> dict[str, Any]:
        """The `tool_result` content part this becomes on the wire."""
        part: dict[str, Any] = {"type": "tool_result", "id": self.call_id, "content": self.content}
        if self.is_error:
            part["isError"] = True
        return part


@dataclass(frozen=True)
class WarningNote:
    """Something the request had to give up, reported rather than swallowed.

    `WarningNote`, not `Warning`: shadowing the builtin in a module callers do
    `from ... import *` on would silently replace the exception base class.
    """

    #: Which layer noticed. `"llm"` for a build note.
    source: str
    #: A stable identifier to branch on -- `"request_adjusted"` today.
    code: str
    #: The human-readable sentence, which is what the corpus prints.
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def of(raw: Mapping[str, Any]) -> WarningNote:
        return WarningNote(
            source=str(raw.get("source") or ""),
            code=str(raw.get("code") or ""),
            message=str(raw.get("message") or ""),
            details=raw.get("details") or {},
        )


@dataclass(frozen=True)
class Citation:
    """A source the answer cited."""

    url: str
    title: str | None = None
    #: The passage the source supports. Only Anthropic reports this today;
    #: absent elsewhere rather than faked from the answer text.
    text: str | None = None

    @staticmethod
    def of(raw: Mapping[str, Any]) -> Citation:
        return Citation(url=str(raw.get("url") or ""), title=raw.get("title"), text=raw.get("text"))


@dataclass(frozen=True)
class FileOutput:
    """A file a hosted tool produced (a code-execution chart, a data file)."""

    id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    #: Inline base64, when the provider returned the bytes with the response.
    data: str | None = None
    url: str | None = None
    source: str | None = None
    #: Provider-specific retrieval hints, needed to fetch by `id`.
    ref: Mapping[str, Any] | None = None
    #: How to fetch the bytes, bound by the client that produced this. Excluded
    #: from equality and repr: it is wiring, and two descriptors of the same file
    #: are the same file whichever client can fetch them.
    fetch: Callable[[FileOutput], bytes] | None = field(
        default=None, compare=False, repr=False
    )

    @property
    def filename(self) -> str | None:
        """`name`, under the word a caller reaches for.

        The wire calls it `name` and so does the field; a file has a filename,
        and every reviewed example asks for it that way.
        """
        return self.name

    def read(self) -> bytes:
        """The bytes -- fetched now, not when the response arrived.

        A chart or a CSV can be large and a provider may hand back an id rather
        than inline data, so eagerly downloading every file would make an
        innocuous call slow and occasionally enormous. This is the explicit step.
        """
        if self.fetch is None:
            raise RuntimeError(
                "this file descriptor is not attached to a client, so it cannot "
                "fetch. Pass it to `llm.retrieve_file(...)` instead."
            )
        return self.fetch(self)

    def save(self, path: Any) -> Any:
        """Write the bytes to `path`, and return the path written.

        Saving is the common case, so it is one call rather than an
        open/write dance. A directory is accepted: the file lands inside it
        under its own name, which is what a caller passing one meant.
        """
        from pathlib import Path

        target = Path(path)
        if target.is_dir():
            target = target / (self.filename or self.id or "download")
        target.write_bytes(self.read())
        return target

    @staticmethod
    def of(raw: Mapping[str, Any]) -> FileOutput:
        return FileOutput(
            id=raw.get("id"),
            name=raw.get("name"),
            mime_type=raw.get("mimeType"),
            data=raw.get("data"),
            url=raw.get("url"),
            source=raw.get("source"),
            ref=raw.get("ref"),
        )


@dataclass(frozen=True)
class BuiltinToolCall:
    """A provider-run tool the model invoked. Nothing for the caller to execute."""

    tool: str
    id: str | None = None
    code: str | None = None
    output: str | None = None
    query: str | None = None
    url: str | None = None

    @staticmethod
    def of(raw: Mapping[str, Any]) -> BuiltinToolCall:
        return BuiltinToolCall(
            tool=str(raw.get("tool") or ""),
            id=raw.get("id"),
            code=raw.get("code"),
            output=raw.get("output"),
            query=raw.get("query"),
            url=raw.get("url"),
        )


@dataclass(frozen=True)
class Completion:
    """One completed call."""

    #: The ANSWER. On models that narrate before answering (the codex family)
    #: the commentary is excluded -- concatenating everything would make an
    #: agent's output include its own thinking-out-loud. The narration is still
    #: reachable through `parts`.
    text: str
    model: str
    finish_reason: str
    usage: Usage
    parts: Sequence[Part]
    #: The structured object, when `structured=` asked for one and it parsed.
    parsed: Any = None
    #: Reasoning text. `None` when the model did not think or does not expose
    #: it -- distinguishable from `""`, which means it thought and returned
    #: nothing, and that difference is how you tell whether a model supports the
    #: feature at all.
    thinking: str | None = None
    tool_calls: Sequence[Part] = ()
    citations: Sequence[Citation] = ()
    files: Sequence[FileOutput] = ()
    builtin_tool_calls: Sequence[BuiltinToolCall] = ()
    #: What the build left out on purpose. ALWAYS a sequence, empty when
    #: nothing was adjusted, so a caller never branches on None to read it.
    warnings: Sequence[WarningNote] = ()
    #: USD, or `None` when the model is not priced. Never `0.0` for an unpriced
    #: model -- see `cost.py`.
    cost: Cost | None = None
    #: The server-side conversation id to continue from, on a stateful API.
    #: Pass it as `state=` on the next call. `None` on a stateless one, where
    #: sending an id would be a 400.
    state: str | None = None
    latency_ms: float = 0.0
    #: The provider's own response body, untouched.
    raw: Any = None
    #: What the call was moderated as, when moderation was requested.
    moderation: Mapping[str, Any] | None = None
    #: The wire response this view was built from, kept so `assistant_message`
    #: can stamp provenance without the caller holding two objects.
    _wire: Mapping[str, Any] = field(default_factory=dict, repr=False)
    _origin: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def of(
        wire: Mapping[str, Any],
        *,
        provider: str,
        model: str,
        api: str,
        cost: Cost | None = None,
        parsed: Any = None,
    ) -> Completion:
        content = list(wire.get("content") or [])
        stateful = api in ("responses", "interactions")
        return Completion(
            # `final_answer_text`, NOT the wire's own `text`: the adapters
            # concatenate every text part, and on a codex-family model that
            # includes the commentary. The contract is that `result.text` is the
            # answer alone.
            text=final_answer_text(content),
            model=str(wire.get("model") or model),
            finish_reason=str(wire.get("finishReason") or ""),
            usage=Usage.of(wire.get("usage")),
            parts=tuple(Part.of(p) for p in content),
            parsed=parsed,
            thinking=wire.get("thinking"),
            tool_calls=tuple(Part.of(p) for p in (wire.get("toolCalls") or [])),
            citations=tuple(Citation.of(c) for c in (wire.get("citations") or [])),
            files=tuple(FileOutput.of(f) for f in (wire.get("files") or [])),
            warnings=tuple(WarningNote.of(w) for w in (wire.get("warnings") or [])),
            builtin_tool_calls=tuple(
                BuiltinToolCall.of(b) for b in (wire.get("builtinToolCalls") or [])
            ),
            cost=cost,
            state=wire.get("id") if stateful and wire.get("id") else None,
            latency_ms=float(wire.get("latencyMs") or 0),
            raw=wire.get("raw"),
            moderation=wire.get("moderation"),
            _wire=wire,
            _origin={"provider": provider, "model": model, "api": api},
        )

    def assistant_message(self) -> Message:
        """This turn as a history message, ready to append.

        Stamped with provenance, so the next call can continue server-side
        instead of resending the transcript. A method rather than a property
        because it MINTS something -- an id and a timestamp -- and a property
        that returns a different object each read is a trap.
        """
        return build_assistant_message(self._wire, self._origin)


def build_cost(
    catalog: Any, provider: str, model: str, wire: Mapping[str, Any]
) -> Cost | None:
    """The cost of a completed call, at the tier the provider actually billed."""
    usage: Mapping[str, Any] = wire.get("usage") or {}
    return compute_cost(
        catalog,
        provider,
        model,
        usage,
        provider_evidence=wire.get("raw") if isinstance(wire.get("raw"), Mapping) else {},
        tier=usage.get("pricingTier"),
    )


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _opt_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


__all__ = [
    "BuiltinToolCall",
    "Citation",
    "Completion",
    "Cost",
    "FileOutput",
    "Part",
    "Usage",
    "build_cost",
]
