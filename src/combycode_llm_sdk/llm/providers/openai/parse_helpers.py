"""The parts of OpenAI's parse that are not shape.

Transposed from the module-level helpers in
`unified-library-ts/src/llm/providers/openai/completions.ts` and
`.../openai/responses.ts`.

Both APIs live here because their registries and both stream registries need
them, and keeping them out of an adapter module avoids the import cycle the
TypeScript side had to live with.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .._shared.builtin_tools import unified_builtin_tool
from ..anthropic.parse_helpers import empty_usage


def _rows(v: Any) -> list[Mapping[str, Any]]:
    return [x for x in v if isinstance(x, Mapping)] if isinstance(v, list) else []


def openai_usage(u: Mapping[str, Any] | None) -> dict[str, Any]:
    """Token usage, from either Chat Completions or Responses naming."""
    if not u:
        return empty_usage()
    inp = u.get("prompt_tokens")
    if inp is None:
        inp = u.get("input_tokens")
    inp = inp or 0
    out = u.get("completion_tokens")
    if out is None:
        out = u.get("output_tokens")
    out = out or 0
    details = u.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        details = u.get("input_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    out_details = u.get("completion_tokens_details")
    if not isinstance(out_details, Mapping):
        out_details = u.get("output_tokens_details")
    out_details = out_details if isinstance(out_details, Mapping) else {}
    return {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": inp + out,
        "cachedTokens": details.get("cached_tokens") or 0,
        "cacheWriteTokens": details.get("cache_write_tokens") or 0,
        "reasoningTokens": out_details.get("reasoning_tokens") or 0,
    }


def openai_responses_usage(u: Mapping[str, Any] | None) -> dict[str, Any]:
    """Responses-API token usage, which names its fields differently."""
    if not u:
        return empty_usage()
    inp = u.get("input_tokens") or 0
    out = u.get("output_tokens") or 0
    in_details = u.get("input_tokens_details")
    in_details = in_details if isinstance(in_details, Mapping) else {}
    out_details = u.get("output_tokens_details")
    out_details = out_details if isinstance(out_details, Mapping) else {}
    total = u.get("total_tokens")
    return {
        "inputTokens": inp,
        "outputTokens": out,
        "totalTokens": total if total is not None else inp + out,
        "cachedTokens": in_details.get("cached_tokens") or 0,
        "cacheWriteTokens": in_details.get("cache_write_tokens") or 0,
        "reasoningTokens": out_details.get("reasoning_tokens") or 0,
    }


def from_wire_caller(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    kind = raw.get("type")
    if not isinstance(kind, str):
        return None
    out: dict[str, Any] = {"type": kind}
    if isinstance(raw.get("caller_id"), str):
        out["callerId"] = raw["caller_id"]
    return out


def _code_output_from_responses_item(item: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    for out in _rows(item.get("outputs")):
        logs = out.get("logs")
        if out.get("type") != "logs" or not isinstance(logs, str):
            continue
        try:
            j = json.loads(logs)
        except (ValueError, TypeError):
            parts.append(logs)  # not JSON -> plain logs
            continue
        if isinstance(j, Mapping) and isinstance(j.get("stdout"), str):
            parts.append(j["stdout"])
            continue
        parts.append(logs)
    return "".join(parts) if parts else None


def _search_action_payload(item: Mapping[str, Any]) -> dict[str, str]:
    action = item.get("action")
    if not isinstance(action, Mapping):
        return {}
    out: dict[str, str] = {}
    # OpenAI deprecated the singular `action.query` in favour of `action.queries[]`.
    # Prefer the array; fall back to the legacy scalar for older/streamed items.
    queries = action.get("queries")
    if isinstance(queries, list) and queries and isinstance(queries[0], str):
        out["query"] = queries[0]
    elif isinstance(action.get("query"), str):
        out["query"] = action["query"]
    if isinstance(action.get("url"), str):
        out["url"] = action["url"]
    return out


def _search_sources(item: Mapping[str, Any]) -> list[str] | None:
    """The URLs a search drew on: `action.sources[] = {type:'url', url}`."""
    action = item.get("action")
    sources = action.get("sources") if isinstance(action, Mapping) else None
    if not isinstance(sources, list):
        return None
    urls = [
        entry["url"]
        for entry in sources
        if isinstance(entry, Mapping) and isinstance(entry.get("url"), str) and entry["url"]
    ]
    return urls or None


#: The four documented image-result fields, renamed to the library's camelCase.
_RESULT_RENAME = {
    "image_url": "imageUrl",
    "source_website_url": "sourceWebsiteUrl",
    "thumbnail_url": "thumbnailUrl",
    "caption": "caption",
}


def _search_results(item: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """`web_search_call.results[]`, present only when the request included them.

    Every other key is carried through untouched. Normalising only what is
    documented and dropping the rest would lose whatever OpenAI adds next, and
    this list exists precisely because the payload is richer than our type --
    measured 2026-09-30, a real result also carries `type`.
    """
    raw = item.get("results")
    if not isinstance(raw, list):
        return None
    out = [
        {_RESULT_RENAME.get(str(key), str(key)): value for key, value in entry.items()}
        for entry in raw
        if isinstance(entry, Mapping)
    ]
    return out or None


_RESPONSES_BUILTIN_ITEMS = frozenset(
    {
        "web_search_call",
        "code_interpreter_call",
        "shell_call",
        # Mapped so a BUFFERED container shell still reports its output;
        # `merge_shell_calls` then folds it into the call it belongs to, so a caller
        # sees one tool call rather than two halves. The streamed path does the same
        # join via its `openShell` state.
        "shell_call_output",
    }
)


def shell_environment_name(item: Mapping[str, Any]) -> str:
    """`environment` as the provider reports it on a `shell_call`.

    Measured 2026-10-02: OpenAI sends `null` for a local call and rewrites
    `container_auto` into `{"type": "container_reference", "container_id": ...}`;
    xAI omits the field entirely and only ever runs locally. All three absences
    mean the same thing -- nobody ran these commands yet -- so they normalise to
    `"local"`.
    """
    env = item.get("environment")
    kind = env.get("type") if isinstance(env, Mapping) else None
    return kind if isinstance(kind, str) else "local"


def shell_awaits_caller(item: Mapping[str, Any]) -> bool:
    """True when the provider is WAITING on the caller to run these commands.

    The whole reason this is a named predicate: the turn ends normally with empty
    text, so without asking this question a caller cannot tell a finished answer
    from a request for work.
    """
    return item.get("type") == "shell_call" and shell_environment_name(item) == "local"


def shell_commands(item: Mapping[str, Any]) -> str:
    """The commands a `shell_call` item asks for, as one newline-joined block.

    One item can carry several commands (measured: `["echo one", "ls /nonexistent"]`)
    and they run in order, so they read as a small script and are joined like one.
    """
    action = item.get("action")
    commands = action.get("commands") if isinstance(action, Mapping) else None
    if not isinstance(commands, list):
        return ""
    return "\n".join(c for c in commands if isinstance(c, str))


def shell_output_text(item: Mapping[str, Any]) -> str:
    """stdout and stderr of a `shell_call_output` item, in command order.

    Each entry is one command's `{outcome, stdout, stderr}`. stderr follows stdout
    for the same command rather than being dropped: a failing command's only output
    is usually on stderr, and `builtin_tool_end.output` is a caller's whole view of
    what happened. The exit code is NOT folded into this string -- it would have to
    be invented as text inside something callers read as program output, and it
    stays available on the raw item.
    """
    chunks = item.get("output")
    if not isinstance(chunks, list):
        return ""
    parts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        for key in ("stdout", "stderr"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
    return "".join(parts)


def builtin_call_from_responses_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = item.get("type")
    if kind not in _RESPONSES_BUILTIN_ITEMS:
        return None
    call: dict[str, Any] = {"tool": unified_builtin_tool(str(kind))}
    if isinstance(item.get("id"), str):
        call["id"] = item["id"]
    if kind == "code_interpreter_call":
        if isinstance(item.get("code"), str):
            call["code"] = item["code"]
        output = _code_output_from_responses_item(item)
        if output:
            call["output"] = output
    elif kind == "shell_call":
        commands = shell_commands(item)
        if commands:
            call["code"] = commands
        call["environment"] = shell_environment_name(item)
        # `call_id` (not the item `id`) is what the matching `shell_call_output`
        # item points back to, so it is what a merge has to key on.
        if isinstance(item.get("call_id"), str):
            call["callId"] = item["call_id"]
    elif kind == "shell_call_output":
        output = shell_output_text(item)
        if output:
            call["output"] = output
        if isinstance(item.get("call_id"), str):
            call["callId"] = item["call_id"]
    elif kind == "web_search_call":
        payload = _search_action_payload(item)
        if payload.get("query"):
            call["query"] = payload["query"]
        if payload.get("url"):
            call["url"] = payload["url"]
        sources = _search_sources(item)
        if sources:
            call["sources"] = sources
        results = _search_results(item)
        if results:
            call["results"] = results
    return call


_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "svg", "bmp"})


def _file_ext(name: str) -> str:
    i = name.rfind(".")
    return name[i + 1 :].lower() if i >= 0 else ""


def _is_display_artifact(c: Mapping[str, Any]) -> bool:
    """The auto-display twin OpenAI emits beside a saved image citation."""
    filename = c.get("filename")
    if not filename:
        return False
    ext = _file_ext(filename)
    return (
        filename == f"{c.get('fileId')}.{ext}"
        and ext in _IMAGE_EXTS
        and c["span"][0] == c["span"][1]
    )


def files_from_responses_output_item(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    kind = item.get("type")
    if kind == "message":
        citations: list[dict[str, Any]] = []
        for c in _rows(item.get("content")):
            if c.get("type") != "output_text":
                continue
            for a in _rows(c.get("annotations")):
                if a.get("type") == "container_file_citation" and isinstance(a.get("file_id"), str):
                    citations.append(
                        {
                            "fileId": a["file_id"],
                            "filename": a["filename"]
                            if isinstance(a.get("filename"), str)
                            else None,
                            "containerId": a["container_id"]
                            if isinstance(a.get("container_id"), str)
                            else None,
                            "span": [_num(a.get("start_index")), _num(a.get("end_index"))],
                        }
                    )
        # A saved (non-artifact) image citation -> its auto-display twin is a
        # duplicate.
        has_saved_image = any(
            not _is_display_artifact(c) and _file_ext(c.get("filename") or "") in _IMAGE_EXTS
            for c in citations
        )
        for c in citations:
            if has_saved_image and _is_display_artifact(c):
                continue  # drop the display duplicate
            f: dict[str, Any] = {"id": c["fileId"]}
            if c.get("filename"):
                f["name"] = c["filename"]
            if c.get("containerId"):
                f["ref"] = {"containerId": c["containerId"]}
            f["source"] = "code_execution"
            files.append(f)
    if kind == "code_interpreter_call":
        for out in _rows(item.get("outputs")):
            if out.get("type") == "image" and isinstance(out.get("url"), str):
                files.append({"url": out["url"], "source": "code_execution"})
            elif out.get("type") == "file" and isinstance(out.get("file_id"), str):
                # A saved file, as opposed to a rendered image. The TypeScript
                # reads only the image case, so a run that wrote a CSV reported
                # no files at all and `result.files` was empty for the one
                # scenario the feature exists for.
                produced: dict[str, Any] = {
                    "id": out["file_id"],
                    "source": "code_execution",
                }
                if isinstance(out.get("filename"), str):
                    produced["name"] = out["filename"]
                if isinstance(out.get("mime_type"), str):
                    produced["mimeType"] = out["mime_type"]
                if isinstance(out.get("container_id"), str):
                    produced["ref"] = {"containerId": out["container_id"]}
                files.append(produced)
    return files


def _num(v: Any) -> int:
    """`Number(x) || 0` -- a missing or unparseable index is 0."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n


__all__ = [
    "builtin_call_from_responses_item",
    "files_from_responses_output_item",
    "from_wire_caller",
    "openai_responses_usage",
    "openai_usage",
]
