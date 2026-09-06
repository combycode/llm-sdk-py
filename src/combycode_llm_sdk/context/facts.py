"""The things a conversation must not forget when its transcript is cut.

Compaction throws away words. Most of them do not matter -- but the account id
stated in message one, the deadline, the file path, the amount: those are the
reason the conversation exists, and a summary that paraphrases them has lost
them. A fact is pulled out VERBATIM and carried forward on its own, so what
survives compaction is not a retelling.

Two places facts live, and the difference is not cosmetic:

- as a **layer** in the `ContextRegistry`, which is where a running agent keeps
  them and where they are re-rendered every turn;
- as a **marker-bounded block inside a system prompt**, for callers with no
  registry -- a plain string they can hold, store and hand back. The markers
  are what make the block replaceable: without them a second write appends a
  second copy and the model reads both.

Values are stored exactly as they appeared. A fact rewritten into nicer prose
is a fact nobody can check against the source, which defeats the purpose of
extracting it.

Transposed from `unified-library-ts/src/plugins/context-guard/facts.ts` and the
marker-bounded half of `context-guard/tools.ts`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: What kind of thing a fact is. A closed list, so a renderer can rely on it and
#: a reader can scan by kind -- `other` is the honest bucket rather than an
#: invitation to invent categories per extractor.
FACT_CATEGORIES: tuple[str, ...] = (
    "name",
    "date",
    "time",
    "path",
    "url",
    "email",
    "phone",
    "address",
    "amount",
    "number",
    "identifier",
    "other",
)

#: The registry layer facts live in for a running agent.
LAYER_CHAT_FACTS = "chat.facts"

#: The block's boundaries inside a system prompt. HTML comments because every
#: model treats them as text and no model treats them as instructions.
FACTS_OPEN = "<!-- orxa:facts -->"
FACTS_CLOSE = "<!-- /orxa:facts -->"
FACTS_HEADER = "## Key facts (preserved across compaction)"

#: `- key [category]: value`. Anchored at both ends so a line of prose that
#: happens to contain a bracket is not read as a fact.
_FACT_LINE = re.compile(r"^-\s+(\S.*?)\s+\[([^\]]+)\]:\s+(.+)$")


@dataclass(frozen=True)
class ExtractedFact:
    """One thing worth keeping, and where it came from."""

    #: A short label, lowercase, `snake_or.dotted`. It is the identity: two
    #: facts with the same key are the same fact, and the later one wins.
    key: str
    #: Verbatim from the source. Not normalised, not reworded.
    value: str
    category: str = "other"
    #: The surrounding phrase, when the value alone is ambiguous. `"March 3"`
    #: means little; `"the migration lands March 3"` means something.
    span: str | None = None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "key": self.key,
            "value": self.value,
            "category": self.category,
        }
        if self.span:
            row["span"] = self.span
        return row

    @staticmethod
    def of(raw: Mapping[str, Any]) -> ExtractedFact:
        category = str(raw.get("category") or "other")
        return ExtractedFact(
            key=str(raw.get("key") or ""),
            value=str(raw.get("value") or ""),
            # An unknown category is kept as `other` rather than refused: a
            # producer this build has never seen must not cost the fact.
            category=category if category in FACT_CATEGORIES else "other",
            span=str(raw["span"]) if raw.get("span") else None,
        )

    def __str__(self) -> str:
        return f"- {self.key} [{self.category}]: {self.value}"


def _sorted(facts: Iterable[ExtractedFact]) -> list[ExtractedFact]:
    """By key, always.

    Stable order is what makes two renderings comparable -- a diff of a system
    prompt should show a fact that CHANGED, not the order the extractor
    happened to emit them in.
    """
    return sorted(facts, key=lambda fact: fact.key)


def render_facts_layer(facts: Sequence[ExtractedFact]) -> str:
    """The bare block, for a registry layer that has its own boundaries."""
    return "\n".join([FACTS_HEADER, *(str(fact) for fact in _sorted(facts))])


def render_facts_block(facts: Sequence[ExtractedFact], bare_block: bool = False) -> str:
    """The block, wrapped in its markers unless the caller says otherwise."""
    body = render_facts_layer(facts)
    return body if bare_block else f"{FACTS_OPEN}\n{body}\n{FACTS_CLOSE}"


def parse_facts_lines(text: str) -> list[ExtractedFact]:
    """Every fact line in a piece of text, and nothing else in it."""
    facts: list[ExtractedFact] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line.startswith("- "):
            continue
        match = _FACT_LINE.match(line)
        if match is None:
            continue
        facts.append(
            ExtractedFact.of(
                {"key": match[1], "category": match[2], "value": match[3]}
            )
        )
    return facts


def parse_facts_block(system: str) -> list[ExtractedFact]:
    """The facts inside the marker region of a system prompt.

    Only inside it. A fact-shaped line a user wrote elsewhere in the prompt is
    the user's text, not the agent's memory, and reading it as one would let a
    prompt inject entries into a store the agent trusts.
    """
    start = system.find(FACTS_OPEN)
    if start < 0:
        return []
    inner_start = start + len(FACTS_OPEN)
    end = system.find(FACTS_CLOSE, inner_start)
    if end < 0:
        return []
    return parse_facts_lines(system[inner_start:end])


def write_facts_block(system: str, facts: Sequence[ExtractedFact]) -> str:
    """A system prompt with exactly one facts block in it.

    Replaces the region when there is one and appends when there is not, which
    is what keeps this idempotent: writing twice leaves one block, not two.
    An OPEN marker with no CLOSE is treated as a truncated block and everything
    from it onward is replaced -- the alternative is appending a second block
    after a broken one and letting the model read both.
    """
    block = render_facts_block(facts)
    start = system.find(FACTS_OPEN)
    if start < 0:
        return f"{system}\n\n{block}" if system else block
    end = system.find(FACTS_CLOSE, start + len(FACTS_OPEN))
    if end < 0:
        return system[:start] + block
    return system[:start] + block + system[end + len(FACTS_CLOSE) :]


def read_facts_layer(registry: Any) -> list[ExtractedFact] | None:
    """The facts a registry is carrying, or None when it carries no layer.

    None and `[]` are different answers: no layer means facts were never
    extracted, an empty layer means they were and there were none. A caller
    deciding whether to run an extractor needs to tell those apart.

    The layer's `metadata["facts"]` is preferred over its rendered text --
    that is the structured original, and re-parsing the rendering would lose
    any `span` the renderer does not print.
    """
    layer = registry.get(LAYER_CHAT_FACTS) if hasattr(registry, "get") else None
    if layer is None:
        return None
    metadata = getattr(layer, "metadata", None)
    if metadata is None and isinstance(layer, Mapping):
        metadata = layer.get("metadata")
    if isinstance(metadata, Mapping):
        rows = metadata.get("facts")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            return [
                fact if isinstance(fact, ExtractedFact) else ExtractedFact.of(fact)
                for fact in rows
                if isinstance(fact, (ExtractedFact, Mapping))
            ]
    content = getattr(layer, "content", None)
    if content is None and isinstance(layer, Mapping):
        content = layer.get("content")
    return parse_facts_lines(content) if isinstance(content, str) else []


def merge_facts(
    prior: Sequence[ExtractedFact], fresh: Sequence[ExtractedFact]
) -> list[ExtractedFact]:
    """Prior facts, updated by fresh ones. Later wins on the same key.

    The key is the identity, so a value that was corrected in conversation
    REPLACES the old one rather than sitting beside it -- two entries for
    `deadline` is worse than either alone, because nothing says which is now
    true.
    """
    merged: dict[str, ExtractedFact] = {fact.key: fact for fact in prior}
    for fact in fresh:
        merged[fact.key] = fact
    return _sorted(merged.values())


def render_prior_facts_for_extraction(facts: Sequence[ExtractedFact]) -> str:
    """What to show an extractor so it carries forward instead of starting over.

    A different rendering from the stored one, and deliberately: this is an
    instruction to a model, not a record. Empty when there is nothing prior, so
    a caller can concatenate it without a guard.
    """
    if not facts:
        return ""
    header = (
        "## Previously extracted facts "
        "(carry forward; merge with the new content below)"
    )
    lines = [header]
    lines.extend(f"- {fact.key} ({fact.category}): {fact.value}" for fact in facts)
    return "\n".join(lines)


__all__ = [
    "FACTS_CLOSE",
    "FACTS_HEADER",
    "FACTS_OPEN",
    "FACT_CATEGORIES",
    "LAYER_CHAT_FACTS",
    "ExtractedFact",
    "merge_facts",
    "parse_facts_block",
    "parse_facts_lines",
    "read_facts_layer",
    "render_facts_block",
    "render_facts_layer",
    "render_prior_facts_for_extraction",
    "write_facts_block",
]
