"""Context as named layers the agent assembles, not one string several authors edit.

Transposed from `unified-library-ts/src/agent/context-registry/`.

A conversation's system prompt has more than one author: the caller pins a
persona, the run supplies its scenario, the agent learns facts as it goes, the
context guard leaves a summary behind after compacting. Concatenating those by
hand is how the memory notes sit above the persona on Tuesday and below it on
Wednesday -- which moves the cache prefix and quietly stops every prompt-cache
hit.

So each contributor writes its own named layer and never touches anyone else's,
and precedence is PRIORITY, then age, then name -- never insertion order. Stable
layers get low numbers and render first; churny ones land after the prefix the
cache depends on.

A registry can have a PARENT, which is how orchestrator-wide context reaches one
conversation without being copied into it. A child layer replaces the parent's
by default, or merges with it when it says so -- because an override that
silently appends is a layer you cannot get rid of.

And layers are not MESSAGES. A fact stated once in message 3 dies the moment
message 3 is summarised away; compaction rewrites the message list and has
nothing here to drop.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..wire.interpreter import js_json

#: Where a layer renders when nobody says. Deliberately mid-range: a writer that
#: forgets to state one lands between the stable prefix and the churny tail
#: rather than in front of the persona.
DEFAULT_PRIORITY = 100

DEFAULT_SEPARATOR = "\n\n"


def _now_ms() -> float:
    return time.time() * 1000


@dataclass(frozen=True)
class ContextLayer:
    """A named, versioned slice of context. Every mutation bumps `version`."""

    name: str
    #: Free-form. A string is the common case; content parts support the
    #: multi-modal and structured hints. Rendering flattens either to text.
    content: Any
    priority: int = DEFAULT_PRIORITY
    tags: Sequence[str] = ()
    #: Who wrote this last, for audit and per-owner filtering.
    owner: str | None = None
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    #: When true AND a parent has a same-named layer, a composed render puts the
    #: parent's content BEFORE this one's instead of replacing it. For layers
    #: that accumulate -- facts, memory.
    merge_parent: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedPart:
    """One layer, as it appears in a render."""

    name: str
    content: str
    priority: int
    tags: Sequence[str] = ()
    owner: str | None = None
    #: Which registry this layer came from -- this one, or a parent. The thing
    #: you need when a cascade is not composing the way you expected.
    registry: str = ""


@dataclass(frozen=True)
class RenderResult:
    """A composed view. `flat` is the string to send."""

    parts: Sequence[RenderedPart]
    flat: str
    total_chars: int
    rendered: float


@dataclass(frozen=True)
class ContextRegistryEvent:
    """A change, including ones bubbled up from a parent."""

    type: str
    name: str
    registry: str
    previous: ContextLayer | None = None
    current: ContextLayer | None = None
    size_before: int = 0
    size_after: int = 0
    timestamp: float = 0.0


def layer_to_text(layer: ContextLayer) -> str:
    """A layer's content as text, whatever shape it was written in."""
    if isinstance(layer.content, str):
        return layer.content
    return _parts_to_text(layer.content)


def _parts_to_text(content: Any) -> str:
    if not isinstance(content, Sequence) or isinstance(content, str):
        return str(content or "")
    out: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            out.append(str(part))
            continue
        kind = part.get("type")
        if kind == "text":
            out.append(str(part.get("text") or ""))
        elif kind == "tool_call":
            out.append(f"[tool_call {part.get('name')}]({js_json(part.get('arguments') or {})})")
        elif kind == "tool_result":
            inner = part.get("content")
            out.append(f"[tool_result] {inner if isinstance(inner, str) else js_json(inner)}")
        # Images, audio and video contribute no text and are skipped.
    return "\n".join(out)


def _concat(first: Any, second: Any) -> Any:
    """Parent content, then child's. Two lists stay a list; anything else joins as text."""
    if isinstance(first, str) and isinstance(second, str):
        return f"{first}\n\n{second}"
    if isinstance(first, list) and isinstance(second, list):
        return [*first, *second]
    return f"{_as_text(first)}\n\n{_as_text(second)}"


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else _parts_to_text(value)


class ContextRegistry:
    """Named layers, composed by priority, optionally inheriting from a parent."""

    def __init__(
        self,
        id: str | None = None,
        *,
        parent: ContextRegistry | None = None,
        counter: Any = None,
        default_owner: str | None = None,
        separator: str = DEFAULT_SEPARATOR,
    ) -> None:
        self.id = id or f"ctx_{uuid.uuid4().hex[:8]}"
        self._parent = parent
        self._counter = counter
        self._default_owner = default_owner
        self.separator = separator
        self._layers: dict[str, ContextLayer] = {}
        self._handlers: list[Callable[[ContextRegistryEvent], Any]] = []

    @property
    def parent(self) -> ContextRegistry | None:
        return self._parent

    # -- writing -------------------------------------------------------------

    def set(
        self,
        name: str,
        content: Any,
        *,
        priority: int | None = None,
        tags: Sequence[str] | None = None,
        owner: str | None = None,
        merge_parent: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextLayer:
        """Write a layer, inheriting anything not restated.

        Inheriting matters more than it looks: a writer that updates CONTENT
        without restating the priority would otherwise fall back to the default
        and jump in front of layers it is meant to follow -- moving the cache
        prefix on a routine edit.
        """
        before = self._local_chars()
        previous = self._layers.get(name)
        now = _now_ms()

        layer = ContextLayer(
            name=name,
            content=content,
            priority=_first(priority, previous.priority if previous else None, DEFAULT_PRIORITY),
            tags=tuple(_first(tags, previous.tags if previous else None, ())),
            owner=_first(owner, previous.owner if previous else None, self._default_owner),
            version=(previous.version if previous else 0) + 1,
            created_at=previous.created_at if previous else now,
            updated_at=now,
            merge_parent=bool(
                _first(merge_parent, previous.merge_parent if previous else None, False)
            ),
            metadata=dict(_first(metadata, previous.metadata if previous else None, {})),
        )
        self._layers[name] = layer
        self._fire(
            ContextRegistryEvent(
                type="update" if previous else "set",
                name=name,
                registry=self.id,
                previous=previous,
                current=layer,
                size_before=before,
                size_after=self._local_chars(),
                timestamp=now,
            )
        )
        return layer

    def patch(self, name: str, fn: Callable[[ContextLayer | None], Any]) -> ContextLayer:
        """Rewrite a layer from its current value."""
        previous = self._layers.get(name)
        result = fn(previous)
        if isinstance(result, ContextLayer):
            return self.set(
                name,
                result.content,
                priority=result.priority,
                tags=result.tags,
                owner=result.owner,
                merge_parent=result.merge_parent,
                metadata=result.metadata,
            )
        return self.set(name, result)

    def remove(self, name: str) -> bool:
        previous = self._layers.pop(name, None)
        if previous is None:
            return False
        self._fire(
            ContextRegistryEvent(
                type="remove",
                name=name,
                registry=self.id,
                previous=previous,
                size_before=self._local_chars(),
                size_after=self._local_chars(),
                timestamp=_now_ms(),
            )
        )
        return True

    def clear(self) -> None:
        for name in list(self._layers):
            self.remove(name)

    # -- reading -------------------------------------------------------------

    def get(self, name: str) -> ContextLayer | None:
        """This registry's own layer. Does NOT walk the parent chain.

        Composition is a RENDER, not a write: a merged layer exists only in the
        rendered output, so `get` answering the merged value would report
        content this registry does not hold and cannot edit.
        """
        return self._layers.get(name)

    def has(self, name: str) -> bool:
        return name in self._layers

    def names(self) -> list[str]:
        return list(self._layers)

    def list(
        self,
        *,
        tag: str | None = None,
        tags: Sequence[str] | None = None,
        owner: str | None = None,
    ) -> list[ContextLayer]:
        found = list(self._layers.values())
        if tag:
            found = [x for x in found if tag in x.tags]
        if tags:
            found = [x for x in found if any(t in x.tags for t in tags)]
        if owner:
            found = [x for x in found if x.owner == owner]
        return found

    # -- rendering -----------------------------------------------------------

    def render(
        self,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        tag: str | None = None,
        tags: Sequence[str] | None = None,
        owner: str | None = None,
        separator: str | None = None,
        include_parent: bool = True,
    ) -> RenderResult:
        """Compose the layers, parent chain first, into one ordered view."""
        options = {
            "include": include,
            "exclude": exclude,
            "tag": tag,
            "tags": tags,
            "owner": owner,
            "include_parent": include_parent,
        }
        collected = self._collect(options)
        entries = sorted(
            collected.values(),
            # Priority, then age, then name. Never insertion order: two layers
            # written in a different sequence must render the same way, or the
            # cache prefix depends on the order the code happened to run in.
            key=lambda e: (e[0].priority, e[0].updated_at, e[0].name),
        )
        parts = [
            RenderedPart(
                name=layer.name,
                content=layer_to_text(layer),
                priority=layer.priority,
                tags=tuple(layer.tags),
                owner=layer.owner,
                registry=source,
            )
            for layer, source in entries
        ]
        joiner = separator if separator is not None else self.separator
        flat = joiner.join(p.content for p in parts if p.content)
        return RenderResult(
            parts=tuple(parts), flat=flat, total_chars=len(flat), rendered=_now_ms()
        )

    def flat(self, **options: Any) -> str:
        return self.render(**options).flat

    def size_chars(self, **options: Any) -> int:
        return self.render(**options).total_chars

    def size_tokens(
        self, provider: str = "", model: str = "", **options: Any
    ) -> int:
        text = self.flat(**options)
        if self._counter is None:
            return -(-len(text) // 4)
        return int(self._counter.count(text, provider, model, exact=False).tokens)

    def _collect(
        self, options: Mapping[str, Any]
    ) -> dict[str, tuple[ContextLayer, str]]:
        result: dict[str, tuple[ContextLayer, str]] = {}
        if options.get("include_parent", True) and self._parent is not None:
            result.update(self._parent._collect(options))
        for name, layer in self._layers.items():
            if not _passes(layer, options):
                continue
            existing = result.get(name)
            if existing is not None and layer.merge_parent:
                result[name] = (
                    replace(layer, content=_concat(existing[0].content, layer.content)),
                    self.id,
                )
            else:
                result[name] = (layer, self.id)
        return result

    def _local_chars(self) -> int:
        return sum(len(layer_to_text(layer)) for layer in self._layers.values())

    # -- events --------------------------------------------------------------

    def subscribe(self, handler: Callable[[ContextRegistryEvent], Any]) -> Callable[[], None]:
        """Watch changes, including ones bubbled up from a parent."""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    def _fire(self, event: ContextRegistryEvent) -> None:
        for handler in list(self._handlers):
            handler(event)

    # -- serialisation -------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        """A snapshot. The PARENT is not serialised -- it is a runtime relationship."""
        return {
            "v": 1,
            "id": self.id,
            "separator": self.separator,
            "layers": [
                {
                    "name": x.name,
                    "content": x.content,
                    "priority": x.priority,
                    "tags": list(x.tags),
                    "owner": x.owner,
                    "version": x.version,
                    "createdAt": x.created_at,
                    "updatedAt": x.updated_at,
                    "mergeParent": x.merge_parent,
                    "metadata": dict(x.metadata),
                }
                for x in self._layers.values()
            ],
        }

    @staticmethod
    def restore(snapshot: Mapping[str, Any]) -> ContextRegistry:
        registry = ContextRegistry(
            id=str(snapshot.get("id") or "") or None,
            separator=str(snapshot.get("separator") or DEFAULT_SEPARATOR),
        )
        for raw in snapshot.get("layers") or []:
            registry._layers[str(raw["name"])] = ContextLayer(
                name=str(raw["name"]),
                content=raw.get("content"),
                priority=int(raw.get("priority", DEFAULT_PRIORITY)),
                tags=tuple(raw.get("tags") or ()),
                owner=raw.get("owner"),
                version=int(raw.get("version", 1)),
                created_at=float(raw.get("createdAt") or 0.0),
                updated_at=float(raw.get("updatedAt") or 0.0),
                merge_parent=bool(raw.get("mergeParent")),
                metadata=dict(raw.get("metadata") or {}),
            )
        return registry

    def __repr__(self) -> str:
        return f"<ContextRegistry {self.id} {len(self._layers)} layer(s)>"


def _first(*values: Any) -> Any:
    """The first value that was actually given."""
    for value in values:
        if value is not None:
            return value
    return None


def _passes(layer: ContextLayer, options: Mapping[str, Any]) -> bool:
    include = options.get("include")
    if include and layer.name not in include:
        return False
    exclude = options.get("exclude")
    if exclude and layer.name in exclude:
        return False
    tag = options.get("tag")
    if tag and tag not in layer.tags:
        return False
    tags = options.get("tags")
    if tags and not any(t in layer.tags for t in tags):
        return False
    owner = options.get("owner")
    return not (owner and layer.owner != owner)


__all__ = [
    "DEFAULT_PRIORITY",
    "DEFAULT_SEPARATOR",
    "ContextLayer",
    "ContextRegistry",
    "ContextRegistryEvent",
    "RenderResult",
    "RenderedPart",
    "layer_to_text",
]
