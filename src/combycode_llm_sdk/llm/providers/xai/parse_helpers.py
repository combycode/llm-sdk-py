"""The one thing xAI parses differently from OpenAI Responses.

Transposed from the module-level helper in
`unified-library-ts/src/llm/providers/xai/responses.ts`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ....util.base64 import bytes_to_base64


def xai_code_exec_files(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """xAI returns code-execution output files INLINE inside the
    `code_interpreter_call` `logs` payload -- a JSON string carrying
    `{stdout, output_files:[{file_name, mime_type, data:[...bytes]}]}` -- not as
    OpenAI-style `container_file_citation` annotations. Requires the request to
    ask for them via `include: ['code_interpreter_call.outputs']`.
    """
    if item.get("type") != "code_interpreter_call":
        return []
    files: list[dict[str, Any]] = []
    outputs = item.get("outputs")
    for out in outputs if isinstance(outputs, list) else []:
        if not isinstance(out, Mapping):
            continue
        logs = out.get("logs")
        if out.get("type") != "logs" or not isinstance(logs, str):
            continue
        try:
            parsed = json.loads(logs)
        except (ValueError, TypeError):
            continue  # plain-text logs (not the xAI JSON envelope)
        if not isinstance(parsed, Mapping):
            continue
        for f in parsed.get("output_files") or []:
            if not isinstance(f, Mapping) or not isinstance(f.get("data"), list):
                continue
            entry: dict[str, Any] = {"data": bytes_to_base64(bytes(f["data"]))}
            if isinstance(f.get("file_name"), str):
                entry["name"] = f["file_name"]
            if isinstance(f.get("mime_type"), str):
                entry["mimeType"] = f["mime_type"]
            entry["source"] = "code_execution"
            files.append(entry)
    return files


__all__ = ["xai_code_exec_files"]
