"""A tool may answer with content parts rather than a string.

`AgentTool.execute` has always been allowed to return parts, and the loop always
carried them into the tool result -- then every adapter serialised the list into
the provider's text slot. An image came through as a wall of base64: paid for as
text, unreadable as an image, and often large enough to blow the context on its
own.

Each API has a place for this, and they disagree about where:

===========================  ==================================================
API                          media travels as
===========================  ==================================================
Anthropic Messages           blocks inside ``tool_result.content``
OpenAI Responses             items inside ``function_call_output.output``
Google ``generateContent``   ``functionResponse.parts[].inlineData``
OpenAI Completions           nowhere -- a tool message is text, so it follows in
                             its own user message
Google Interactions          the same, for the same reason
===========================  ==================================================

What they agree on is the split: some of a tool's answer is text for the result
slot, the rest is media. That split happens once, here, so an adapter only has to
say how its API spells the two halves.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

MEDIA_KINDS = ("image", "document", "audio", "video")


def _is_media(part: Mapping[str, Any]) -> bool:
    return part.get("type") in MEDIA_KINDS


def split_tool_result(content: Any) -> tuple[str, list[Mapping[str, Any]]]:
    """Split a tool result into its text half and its media half.

    A part that is neither is serialised into the text rather than dropped -- a
    tool returning something unexpected should reach the model looking odd, not
    vanish on the way.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, Sequence):
        return json.dumps(content, separators=(",", ":")), []

    text: list[str] = []
    media: list[Mapping[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            text.append(json.dumps(part, separators=(",", ":")))
        elif part.get("type") == "text":
            text.append(str(part.get("text") or ""))
        elif _is_media(part):
            media.append(part)
        else:
            text.append(json.dumps(part, separators=(",", ":")))
    return "\n".join(text), media


def has_tool_result_media(content: Any) -> bool:
    """Whether this tool result has anything that must travel as media.

    The question every adapter asks first, because the answer decides whether it
    can keep taking the cheap path -- a plain string result must build exactly
    the body it built before this existed.
    """
    if isinstance(content, str) or not isinstance(content, Sequence):
        return False
    return any(isinstance(p, Mapping) and _is_media(p) for p in content)
