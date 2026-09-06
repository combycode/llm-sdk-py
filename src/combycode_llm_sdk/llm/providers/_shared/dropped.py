"""Saying so when a content part cannot be carried.

Every adapter has kinds it cannot express. Anthropic has no audio block, the
Responses API has no form for one, and the Interactions API takes audio only as
base64. Each of those paths already did the right thing with the CONTENT --
substituting a placeholder, or leaving the part out so the request stays valid
-- and then said nothing at all about it.

Which is the failure this module exists to end. A caller who attaches a wav to
Anthropic gets back "I'm unable to listen to audio files", and every reading of
that sentence blames the model. The audio never left this library: it was
replaced with `[unsupported: audio]` five frames earlier, silently. The evidence
that would have said so was thrown away at the only point that had it.

`ProviderHttpRequest.notes` is where that evidence belongs -- the runtime turns
each note into an `onWarning` AND into `Completion.warnings`, so a caller reads
it whether or not they subscribed to anything. This module just writes them in
one voice.

De-duplicated per request: three dropped audio parts are one fact about the
request, and three identical warnings would only bury it.
"""

from __future__ import annotations

#: Where a note goes. `None` means nobody is collecting -- a build outside a
#: request, or a caller that predates this -- and the note is discarded rather
#: than forcing every call site to construct a list it will not read.
NoteSink = list[str] | None


def note_dropped(notes: NoteSink, provider: str, kind: str | None) -> None:
    """A part this provider has no form for, left out of the request."""
    _add(notes, provider, kind, "so the request was sent without it")


def note_replaced(notes: NoteSink, provider: str, kind: str | None) -> None:
    """A part swapped for a text placeholder, which is not the same as dropped.

    Worth its own wording: the model still sees something where the content was,
    so an answer that mentions it is the placeholder talking, not the content.
    """
    _add(notes, provider, kind, "so it was replaced with a text placeholder")


def _add(notes: NoteSink, provider: str, kind: str | None, consequence: str) -> None:
    if notes is None or not kind:
        return
    note = f"{provider}: {kind} content is not supported by this API, {consequence}"
    # Membership on a short list, not a set: notes are ORDERED, and the order is
    # the order the build met them.
    if note not in notes:
        notes.append(note)


__all__ = ["NoteSink", "note_dropped", "note_replaced"]
