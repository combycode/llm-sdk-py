"""Runtime shape check -- tell me when a provider's response stops looking like
the one we learned to read.

Transposed from `unified-library-ts/src/llm/response-shape.ts`.

Parsing is the half of the library with the least warning before a failure. A
request that goes wrong comes back as a 400. A RESPONSE that goes wrong comes
back as a 200 with a field we do not read: the parse succeeds, the number is
absent, and the first sign is a cost dashboard that quietly reports zero or a
tool call that never arrives. `google/generate` shipped for months discarding a
`responseId` that had been there all along, under a comment saying it did not
exist.

So this compares the body against a description of the shape we understand:

- **unknown path** -- a field we have never seen. Either the provider added it
  (possibly the one carrying something we now want), or we are talking to
  something that is not the API we think.
- **missing path** -- a field present in EVERY recording is absent from this one.
  That is the shape of a rename, and a rename is what silently turns a token
  count into nothing.
- **unknown value** -- a discriminator (`type`, `role`, `finish_reason`, ...)
  carries a value we do not branch on. A new content-block type is the most
  expensive kind of drift there is, because the content is simply dropped and
  nothing errors.

OFF by default, and it never changes what is parsed: it only emits `onWarning`.
A check that could alter a response would be a new way to break one.

The descriptions in `response-shapes.json` are DERIVED, not hand-written --
`scripts/derive-response-shapes.ts` builds them from the recorded corpus, and a
test asserts every recorded body produces no unknown paths. That is what keeps
the description honest: it cannot drift from real responses without a test going
red, and when a provider does add a field, re-recording surfaces it as a decision
rather than as a silent difference. The file is vendored byte-identical here and
is etalon.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..bus.hook_bus import HookBus

#: Keys whose VALUE selects a branch. A new value here is not a new field -- it
#: is a case nothing handles.
_DISCRIMINATORS = frozenset(
    {"type", "object", "role", "finish_reason", "finishReason", "stop_reason", "status", "event"}
)

#: Guards against a pathological body -- a map keyed by id would otherwise
#: produce an unbounded set of paths and turn a diagnostic into a memory leak.
_MAX_DEPTH = 10
_MAX_PATHS = 2000

#: `interface ShapeDecl` (response-shape.ts:56):
#: `{known, expected, values?}`. `known` is every path seen across the recordings
#: for this target; `expected` is the paths present in EVERY recording, so
#: absence is worth a warning; `values` maps a discriminator path to the values
#: actually seen.
ShapeDecl = dict[str, Any]

#: `interface ShapeSet` (response-shape.ts:64): `{response?, stream?}`.
#:
#: `stream` is one declaration PER SSE event type. Pooling every event type into
#: one declaration was the first attempt, and it cost the missing-field check
#: entirely: `message_start` and `content_block_delta` share almost no fields, so
#: the intersection across a whole stream came to a single path and nothing could
#: ever be reported absent. Per type, "this event always carries usage" becomes a
#: statement worth making.
ShapeSet = dict[str, Any]

#: `type ShapeBook` (response-shape.ts:88) -- `provider/api` -> ShapeSet.
ShapeBook = dict[str, Any]

#: `interface ShapeFindings` (response-shape.ts:90) --
#: `{unknown, missing, unknownValues}`.
ShapeFindings = dict[str, Any]

_SHAPES_PATH = Path(__file__).resolve().parent / "response-shapes.json"


def load_response_shapes() -> ShapeBook:
    """The vendored shape book.

    Read from disk rather than imported statically, as the catalog is -- which is
    why `pyproject.toml` lists it as a wheel artifact.
    """
    with _SHAPES_PATH.open(encoding="utf-8") as handle:
        book: ShapeBook = json.load(handle)
        return book


def stream_event_key(event: Mapping[str, Any], data: Any) -> str:
    """How an SSE event names its own kind.

    The SSE `event:` line wins -- Anthropic routes on it, and its payload `type`
    merely repeats it -- then the payload's `type`, then a single bucket for
    providers that discriminate on neither (chat-completions chunks are all one
    shape).
    """
    if event.get("event"):
        return str(event["event"])
    kind = data.get("type") if isinstance(data, Mapping) else None
    return kind if isinstance(kind, str) else "*"


def paths_of(
    value: Any, prefix: str = "", out: set[str] | None = None, depth: int = 0
) -> set[str]:
    """Collect the paths of a value.

    Array indices collapse to `[]` -- otherwise a three-element list would
    describe a different shape than a four-element one.
    """
    out = set() if out is None else out
    if depth > _MAX_DEPTH or len(out) > _MAX_PATHS:
        return out
    # `list` before `Mapping`, and `str` is neither: JavaScript's
    # `Array.isArray` / `typeof === 'object'` split has no third case for
    # strings, which are primitives there.
    if isinstance(value, (list, tuple)):
        path = f"{prefix}[]"
        out.add(path)
        for item in value:
            paths_of(item, path, out, depth + 1)
        return out
    if isinstance(value, Mapping):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.add(path)
            paths_of(val, path, out, depth + 1)
        return out
    return out


def discriminators_of(
    value: Any,
    prefix: str = "",
    out: list[tuple[str, str]] | None = None,
    depth: int = 0,
) -> list[tuple[str, str]]:
    """Collect `(path, value)` for the discriminator keys only."""
    out = [] if out is None else out
    if depth > _MAX_DEPTH or len(out) > _MAX_PATHS:
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            discriminators_of(item, f"{prefix}[]", out, depth + 1)
        return out
    if isinstance(value, Mapping):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in _DISCRIMINATORS and isinstance(val, str):
                out.append((path, val))
            discriminators_of(val, path, out, depth + 1)
        return out
    return out


def check_shape(body: Any, decl: ShapeDecl) -> ShapeFindings:
    """Compare one body against a declaration. Pure -- the caller decides what to do."""
    seen = paths_of(body)
    known = set(decl.get("known") or ())
    unknown = [path for path in seen if path not in known]

    missing = [path for path in decl.get("expected") or () if path not in seen]

    unknown_values: list[tuple[str, str]] = []
    values: Mapping[str, Sequence[str]] | None = decl.get("values")
    if values:
        for path, value in discriminators_of(body):
            allowed = values.get(path)
            # A path with no recorded values is not a claim about values -- only
            # a path we HAVE seen values for can have an unexpected one.
            if allowed and value not in allowed:
                unknown_values.append((path, value))

    return {
        "unknown": sorted(unknown),
        "missing": sorted(missing),
        "unknownValues": unknown_values,
    }


class ResponseShapeChecker:
    """Per-client checker.

    Holds what it has already reported, because the alternative is the same
    warning on every request for the rest of the process -- which is how a
    diagnostic gets muted by the person reading it.
    """

    def __init__(self, hooks: HookBus, provider: str, api: str, book: ShapeBook) -> None:
        self._hooks = hooks
        self._provider = provider
        self._api = api
        self._book = book
        self._reported: set[str] = set()

    @property
    def _set(self) -> ShapeSet | None:
        """The declaration for this client, or None when nothing was recorded for
        it -- in which case the check stays silent rather than calling every field
        unknown."""
        return self._book.get(f"{self._provider}/{self._api}")

    def check_response(self, body: Any) -> None:
        shape_set = self._set
        decl = shape_set.get("response") if shape_set else None
        if decl:
            self._report("response", check_shape(body, decl))

    def check_stream_event(self, event: Mapping[str, Any]) -> None:
        shape_set = self._set
        stream = shape_set.get("stream") if shape_set else None
        if not stream:
            return
        try:
            data = json.loads(event["data"])
        except (ValueError, TypeError, KeyError):
            # A non-JSON payload is the stream's own business -- `[DONE]`
            # sentinels and keep-alives are not shape drift.
            return
        key = stream_event_key(event, data)
        decl = stream.get(key)
        if not decl:
            # A whole event type nobody has seen. This is the one that matters
            # most in a stream: an unhandled event is skipped silently, so the
            # reply is simply missing a piece and nothing anywhere errors.
            self._warn("stream", "unknown_event", f"stream:@{key}", f'unhandled event type "{key}"')
            return
        self._report(f"stream {key}", check_shape(data, decl))

    def _report(self, kind: str, findings: ShapeFindings) -> None:
        for path in findings["unknown"]:
            self._warn(kind, "unknown_field", f"{kind}:{path}", f"new field {path}")
        for path in findings["missing"]:
            self._warn(
                kind,
                "missing_field",
                f"{kind}:!{path}",
                f"field {path} was always present and is now absent",
            )
        for path, value in findings["unknownValues"]:
            self._warn(
                kind,
                "unknown_value",
                f"{kind}:{path}={value}",
                f'{path} carries an unhandled value "{value}"',
            )

    def _warn(self, kind: str, code: str, dedupe_key: str, message: str) -> None:
        if dedupe_key in self._reported:
            return
        self._reported.add(dedupe_key)
        self._hooks.emit_sync(
            "onWarning",
            {
                "source": "llm",
                "code": f"response_shape_{code}",
                "message": f"{self._provider}/{self._api} {kind}: {message}",
                "details": {"provider": self._provider, "api": self._api, "kind": kind},
            },
        )


__all__ = [
    "ResponseShapeChecker",
    "ShapeBook",
    "ShapeDecl",
    "ShapeFindings",
    "ShapeSet",
    "check_shape",
    "discriminators_of",
    "load_response_shapes",
    "paths_of",
    "stream_event_key",
]
