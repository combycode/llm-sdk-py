"""The sources an answer cited, read from four different wire shapes.

Transposed from `unified-library-ts/src/llm/providers/_shared/citations.ts`.

`builtinToolCalls` already records what the model INVOKED -- that it searched,
and for what. This is the other half: what it ended up CITING. They are not the
same list. A model can run three searches and cite one page, or open a page and
cite nothing, and a caller rendering footnotes needs the second list.

This function always returns a list. The RESPONSE FIELD it feeds is optional and
omitted when empty, like the `files` / `builtinToolCalls` fields beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

Citation = dict[str, Any]


def _rows(v: Any) -> list[Mapping[str, Any]]:
    """A list of mappings, or nothing. Providers omit these keys freely."""
    return [x for x in v if isinstance(x, Mapping)] if isinstance(v, list) else []


def _cite(url: Any, title: Any = None, text: Any = None) -> Citation | None:
    if not isinstance(url, str) or not url:
        return None
    out: Citation = {"url": url}
    if title:
        out["title"] = title
    if text:
        out["text"] = text
    return out


def _from_anthropic(raw: Mapping[str, Any]) -> list[Citation]:
    """Anthropic: text blocks carry a `citations[]`, with the passage they support."""
    out: list[Citation] = []
    for block in _rows(raw.get("content")):
        for cite in _rows(block.get("citations")):
            # Anthropic is the only provider that reports the cited passage.
            c = _cite(cite.get("url"), cite.get("title"), cite.get("cited_text"))
            if c:
                out.append(c)
    return out


def _from_google(raw: Mapping[str, Any]) -> list[Citation]:
    """Google: grounding metadata hangs off the candidate, not the parts."""
    out: list[Citation] = []
    for candidate in _rows(raw.get("candidates")):
        grounding = candidate.get("groundingMetadata")
        grounding = grounding if isinstance(grounding, Mapping) else {}
        for chunk in _rows(grounding.get("groundingChunks")):
            web = chunk.get("web")
            web = web if isinstance(web, Mapping) else {}
            c = _cite(web.get("uri"), web.get("title"))
            if c:
                out.append(c)
    return out


def _from_responses(raw: Mapping[str, Any]) -> list[Citation]:
    """OpenAI Responses: annotations sit on the output text they annotate."""
    out: list[Citation] = []
    for item in _rows(raw.get("output")):
        for part in _rows(item.get("content")):
            for note in _rows(part.get("annotations")):
                # `file_citation` annotations exist too; they are not web
                # sources and carry no URL, so they are skipped rather than
                # emitted URL-less.
                if note.get("type") == "url_citation":
                    c = _cite(note.get("url"), note.get("title"))
                    if c:
                        out.append(c)
    return out


def _from_completions(raw: Mapping[str, Any]) -> list[Citation]:
    """Chat completions: OpenAI annotates the message; xAI lists bare URLs at the
    top level. BOTH are read -- handling only one reports zero citations for the
    other provider, which is indistinguishable from a model that never searched.
    """
    out: list[Citation] = []
    choices = _rows(raw.get("choices"))
    message = choices[0].get("message") if choices else None
    message = message if isinstance(message, Mapping) else {}
    for note in _rows(message.get("annotations")):
        # OpenAI nests the fields under `url_citation`; some gateways flatten.
        detail = note.get("url_citation")
        detail = detail if isinstance(detail, Mapping) else note
        if note.get("type") == "url_citation":
            c = _cite(detail.get("url"), detail.get("title"))
            if c:
                out.append(c)
    for url in raw.get("citations") or []:
        if isinstance(url, str):
            out.append({"url": url})
    return out


#: Which reader a surface needs. Keyed by the API rather than the provider: xAI
#: and OpenRouter serve OpenAI's shapes, and share its readers.
#:
#: `interactions` is DELIBERATELY absent. Google's Interactions API returns a
#: `steps[]` machine, not `candidates[]`, so `_from_google` would read a key that
#: is never there -- and no recorded Interactions response carries grounding, so
#: there is nothing to write a reader against. Guessing the shape would produce a
#: reader that is confidently wrong and passes a test built from the same guess.
_READERS: dict[str, Callable[[Mapping[str, Any]], list[Citation]]] = {
    "messages": _from_anthropic,
    "generate": _from_google,
    "responses": _from_responses,
    "completions": _from_completions,
}


def extract_citations(api: str, raw: Any) -> list[Citation]:
    """Every source the answer cited, or `[]`.

    An unknown surface yields `[]` rather than raising: a response we cannot read
    citations from is still a valid answer.
    """
    if not isinstance(raw, Mapping):
        return []
    reader = _READERS.get(api)
    return reader(raw) if reader else []


__all__ = ["extract_citations"]
