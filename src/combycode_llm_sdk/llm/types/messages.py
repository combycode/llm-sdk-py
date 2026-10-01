"""Universal message and content types -- shared by LLM, Agent, Server.

Transposed from `unified-library-ts/src/llm/types/messages.ts`.

The TypeScript file is almost entirely `interface` declarations, which have no
runtime form. Following the convention `schema_utils.py` set (`JsonSchema =
dict[str, Any]`), the shapes are documented here and typed as plain dicts rather
than restated as TypedDicts: these dicts are handed straight to the wire
interpreter, which reads them by string key, and a second declaration of the
same shape is a second thing to keep in sync.

**Keys stay camelCase.** These parts are wire-facing: the request specs address
them by path (`req.messages[].content[].mimeType`), the 7700-case corpus is
frozen against those names, and a `mime_type` here would silently build a
request the provider rejects. snake_case belongs to the ergonomic layer that
sits above this one.

The shapes, field for field with the TypeScript:

- `MessageOrigin`   `{provider, model?, serverStateId?, signatures?}`
- `TextPart`        `{type:'text', text, cache?, phase?}`
- `ImagePart`       `{type:'image', source, detail?}`
- `DocumentPart`    `{type:'document', source, citations?}`
- `AudioPart`       `{type:'audio', source}`
- `VideoPart`       `{type:'video', source}`
- `ToolCaller`      `{type, callerId?}`
- `ToolCallPart`    `{type:'tool_call', id, name, arguments, caller?, _meta?}`
- `ToolResultPart`  `{type:'tool_result', id, content, isError?, namespace?, caller?}`
- `ProgramCallPart` `{type:'program_call', id, code, fingerprint, _meta?}`
- `ProgramResultPart` `{type:'program_result', id, result, status?, _meta?}`
- `ConfigurationUpdatePart` `{type:'configuration_update', reasoning:{effort}, id?}`
  The configuration a STORED conversation runs under, from this point on.
  OpenAI Responses only (`gpt-5.6-sol`, `gpt-5.6-luna`; every other model
  answers 400). It exists because the top-level effort option applies to ITS
  request and nothing else -- measured 2026-10-01 on `gpt-5.6-luna`, three
  runs each, setting the effort in turn 1 and naming nothing in turn 2: via
  this item 0/0/0 reasoning tokens, via the option 244/189/172, with no
  configuration at all 155/129/198. `reasoning` and `reasoning.effort` are
  both REQUIRED and `effort: null` is refused, all three contrary to the
  official SDK's types. `effort` admits `none` and `minimal` on top of the
  unified ladder, because `none` is the value that proves the item works.
- `ImageOutputPart` `{type:'image_output', mediaId, mimeType, revisedPrompt?, width?, height?, _data?}`
- `AudioOutputPart` `{type:'audio_output', mediaId, mimeType, durationMs?, sampleRate?, _data?}`
- `VideoOutputPart` `{type:'video_output', mediaId, mimeType, durationMs?, width?, height?, _data?}`
- `DataSource`      one of `{type:'base64', mimeType, data}`, `{type:'url', url}`,
  `{type:'file', fileId}`, `{type:'path', mimeType, path}`,
  `{type:'buffer', mimeType, data}`, `{type:'provider_ref', mimeType, refId}`
- `Message`         `{role, content, cache?, id?, createdAt?, origin?}`
"""

from __future__ import annotations

from typing import Any, Literal

#: `type Role` (messages.ts:5).
Role = Literal["system", "user", "assistant", "tool"]

#: `type AssistantPhase = 'commentary' | 'final_answer' | (string & {})`
#: (messages.ts:44). An OPEN union (CONSTITUTION.md R1): a provider adding a
#: third phase must not break consumers, so this is `str`, not a `Literal`, and
#: every consumer needs a default branch.
AssistantPhase = str

#: `type ToolCallerType = 'direct' | 'program' | (string & {})` (messages.ts:85).
#: Open for the same reason.
ToolCallerType = str

#: Any of the twelve part shapes listed in the module docstring.
ContentPart = dict[str, Any]

#: `type DataSource` (messages.ts:196).
DataSource = dict[str, Any]

#: `type Content = string | ContentPart[]` (messages.ts:206).
Content = str | list[ContentPart]

#: `interface Message` (messages.ts:208).
Message = dict[str, Any]

#: `interface MessageOrigin` (messages.ts:14).
MessageOrigin = dict[str, Any]

#: `interface ToolCaller` (messages.ts:94).
ToolCaller = dict[str, Any]


def content_parts(content: str | list[ContentPart]) -> list[ContentPart]:
    """Normalize any content to a ContentPart list.

    The part list is returned BY IDENTITY, not copied -- callers rely on it
    (`contentParts(parts) === parts`), and copying here would silently detach
    mutations made through the returned list.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content


def _joined(parts: list[ContentPart]) -> str:
    """`.map(p => p.text).join('')`.

    `Array.prototype.join` renders `null` and `undefined` as the empty string
    rather than as `"None"`, which is what a bare `str(p.get("text"))` would
    produce for a part built without its text.
    """
    return "".join("" if p.get("text") is None else str(p.get("text")) for p in parts)


def content_text(content: str | list[ContentPart]) -> str:
    """Extract plain text from content."""
    if isinstance(content, str):
        return content
    return _joined([p for p in content if p.get("type") == "text"])


def final_answer_text(content: str | list[ContentPart]) -> str:
    """The assistant's ANSWER, with commentary removed.

    Codex-family models narrate before answering and mark the narration
    `phase: 'commentary'`. Concatenating everything makes an agent's final
    output include its own thinking-out-loud.

    Excludes only what is explicitly `'commentary'` rather than keeping only
    `'final_answer'`: the phase vocabulary is open (R1), and a phase we do not
    recognise yet must never cause us to drop the answer. Text with no phase at
    all -- every other model -- is returned unchanged, so this is identical to
    `content_text` outside the codex family.
    """
    if isinstance(content, str):
        return content
    return _joined(
        [p for p in content if p.get("type") == "text" and p.get("phase") != "commentary"]
    )


__all__ = [
    "AssistantPhase",
    "Content",
    "ContentPart",
    "DataSource",
    "Message",
    "MessageOrigin",
    "Role",
    "ToolCaller",
    "ToolCallerType",
    "content_parts",
    "content_text",
    "final_answer_text",
]
