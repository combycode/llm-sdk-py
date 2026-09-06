"""Universal streaming event types.

Transposed from `unified-library-ts/src/llm/types/stream.ts`.

A discriminated union on `type`, as plain camelCase dicts -- the stream specs
build these by name and the 30-stream corpus is frozen against them, so the
event shapes below are not free to drift:

- `{type:'text', text, itemId?, phase?}` -- `itemId` identifies WHICH output item
  a delta belongs to, when the provider reports one (OpenAI Responses forwards
  `item_id`; chat-completions has no per-item concept, so it is absent there). A
  single turn can interleave deltas from several output items, so a consumer that
  reassembles them per item -- rather than just concatenating into one string --
  needs this. `phase` mirrors the buffered `TextPart.phase`: whether this delta is
  commentary or the answer proper, reported only by models that distinguish them
  (codex family).
- `{type:'thinking', text, itemId?}`
- `{type:'tool_call_start', id, name, _meta?}`
- `{type:'tool_call_delta', id, arguments}`  -- `arguments` is a JSON *string*
  fragment, not an object.
- `{type:'tool_call_end', id}`
- `{type:'usage', usage}`
- `{type:'done', finishReason}`
- `{type:'error', error}`
- `{type:'media_start', mediaType, mimeType}`
- `{type:'media_chunk', data, progress?}`
- `{type:'media_end', mediaId?}`
- `{type:'file', file}` -- a hosted-tool output file (a code-execution chart or
  CSV) became available. Carries the `FileOutput` DESCRIPTOR, not the bytes;
  fetch those via `retrieve_file` / `stream_file`. Also collected onto the
  streamed final response's `files`.
- `{type:'citation', citation}` -- emitted as the citation arrives, which is NOT
  when the search ran: a provider searches early and cites while it writes, so
  these interleave with `text` deltas. Distinct from `builtin_tool_end`, which
  reports the search itself. Also collected onto the final response's
  `citations`, deduped by url.
- `{type:'builtin_tool_start', tool, id?}` -- a hosted (provider-run) builtin
  began executing server-side. Informational progress: unlike `tool_call_*` (a
  function call the CLIENT must run), the provider runs these itself, so there is
  nothing to execute or return.
- `{type:'builtin_tool_end', tool, id?, code?, output?, query?, url?}`
- `{type:'moderation', phase:'input'|'output', result, source:'native'|'emulated'}`
"""

from __future__ import annotations

from typing import Any, Literal

#: `type MediaStreamType` (stream.ts:7).
MediaStreamType = Literal["image", "audio", "video"]

#: `type StreamEvent` (stream.ts:9) -- the union documented above.
StreamEvent = dict[str, Any]

__all__ = ["MediaStreamType", "StreamEvent"]
