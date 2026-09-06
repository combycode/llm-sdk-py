"""Splitting a document into overlapping windows.

Two decisions carry the quality of every local retrieval result:

**Chunks overlap.** A fact that straddles a boundary is otherwise in neither
chunk and retrievable by neither -- the answer is in the corpus and the search
cannot find it. The overlap costs storage and buys recall.

**Boundaries snap to whitespace.** Cutting mid-word produces a fragment that
embeds as noise, and the two halves each embed as something the document never
said.

Token counts are approximate by design. An exact counter means a tokeniser
dependency, and chunking is a heuristic anyway -- so the default is a
characters-per-token ratio, and a caller who has a real counter can inject it.

Transposed from `unified-library-ts/src/plugins/retrieval/chunker.ts`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: Big enough to hold a paragraph or two, small enough that a hit points at
#: something a person can read.
DEFAULT_CHUNK_MAX_TOKENS = 512
#: Roughly a sentence of run-on between neighbours.
DEFAULT_CHUNK_OVERLAP_TOKENS = 64

#: Characters per token when nothing better is injected. English prose sits
#: near four; the number is a heuristic and named so it is not mistaken for one.
CHARS_PER_TOKEN_HEURISTIC = 4

#: How much of a window a snap-back must leave behind for it to be worth taking.
#:
#: Not a matter of taste: a window whose last space sits in its first half has no
#: space at all in its second half -- 1024 characters at the defaults -- so there
#: is no word there to split. Refusing the snap costs one boundary inside a
#: base64 blob or a minified payload, where a token means nothing; taking it
#: costs a run of runt chunks, because the step is derived from the snapped
#: length and collapses to its floor of 1 once that length drops below the
#: overlap. Measured on a README with one embedded image: 58 chunks, 42 of them
#: runts, the smallest 8 characters.
MIN_SNAP_FRACTION = 0.5


@dataclass(frozen=True)
class TextChunk:
    """One window, and where it came from."""

    text: str
    #: Character offset into the source. Used to build a citation, so a hit can
    #: point at a place rather than just a document.
    offset: int
    index: int


EstimateTokens = Callable[[str], int]


def _default_estimate(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN_HEURISTIC)


def _snap_to_word(raw: str, has_more: bool) -> str:
    """Trim back to the last space, unless this is the final window.

    A boundary that would leave less than `MIN_SNAP_FRACTION` of the window is
    refused: the window is taken whole and the cut lands mid-token, which is the
    cheaper of the two damages where the text has no words in it anyway.
    """
    if not has_more:
        return raw
    last_space = raw.rfind(" ")
    if last_space > 0 and last_space >= len(raw) * MIN_SNAP_FRACTION:
        return raw[:last_space]
    return raw


def _step_from(text: str, offset: int, step: int, limit: int) -> int:
    """How far to advance, landing on a word boundary where there is one.

    A space counts as a boundary only while it is within `limit` -- how far the
    chunk just emitted actually reaches. The next space after a base64 blob or a
    minified payload can be thousands of characters away, and snapping to it
    steps over every one of them: they land in no chunk and are retrievable by
    no query.

    With no usable boundary the advance is the STEP, not the rest of the text.
    Jumping to the end there looks like termination and is actually data loss:
    a document with no whitespace -- CJK prose, minified JSON, one long token --
    would produce a single chunk and silently drop everything after it.
    """
    target = offset + step
    if target >= len(text):
        return len(text) - offset
    next_space = text.find(" ", target)
    if next_space >= 0 and next_space - offset + 1 <= limit:
        return next_space - offset + 1
    return step


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    estimate: EstimateTokens | None = None,
) -> list[TextChunk]:
    """Split `text` into overlapping windows of roughly `max_tokens` each."""
    limit = DEFAULT_CHUNK_MAX_TOKENS if max_tokens is None else max_tokens
    overlap = DEFAULT_CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
    measure = estimate or _default_estimate

    if not text:
        return []
    # Whole thing fits: one chunk, no boundary to get wrong.
    if measure(text) <= limit:
        return [TextChunk(text=text, offset=0, index=0)]

    max_chars = limit * CHARS_PER_TOKEN_HEURISTIC
    overlap_chars = overlap * CHARS_PER_TOKEN_HEURISTIC

    chunks: list[TextChunk] = []
    offset = 0
    index = 0
    while offset < len(text):
        end = min(offset + max_chars, len(text))
        window = _snap_to_word(text[offset:end], end < len(text))
        chunks.append(TextChunk(text=window, offset=offset, index=index))
        # This window reached the end, so it already holds every character that
        # is left. Walking on would re-emit the same tail as a run of
        # ever-shorter chunks -- duplicate index entries, each one paid for at
        # the embedding endpoint and each one competing for a result slot.
        if end >= len(text):
            break
        # At least one character of progress, always: a window that snapped to
        # nothing would otherwise loop here forever on a string with no spaces.
        # The chunk just emitted reaches `offset + len(window)`, and the walk
        # may consume the single space separating it from the next window -- so
        # that, plus one, is the furthest the cursor may legally land.
        step = max(len(window) - overlap_chars, 1)
        offset += _step_from(text, offset, step, len(window) + 1)
        index += 1
    return chunks


__all__ = [
    "CHARS_PER_TOKEN_HEURISTIC",
    "DEFAULT_CHUNK_MAX_TOKENS",
    "DEFAULT_CHUNK_OVERLAP_TOKENS",
    "MIN_SNAP_FRACTION",
    "EstimateTokens",
    "TextChunk",
    "chunk_text",
]
