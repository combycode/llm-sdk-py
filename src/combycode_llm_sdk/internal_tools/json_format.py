"""The paragraph that makes a chat model behave like an endpoint.

Prepended to every JSON tool's own instructions, so no tool has to restate it.
It reads as over-explained because it is: each FORBIDDEN line is a shape that
was actually returned by some model and actually broke a parser, and a rule
dropped for brevity is a rule that comes back as a production incident.

Note what it does NOT do: it is not a replacement for structured output where a
provider offers it. It is what carries a tool across the providers and models
that do not, which for a tool meant to run anywhere is most of them.

Transposed from
`unified-library-ts/src/plugins/internal-tools/runner/json-enforcement.ts`.
"""

from __future__ import annotations

#: `output_format="json"` -- parse the answer, and show the model the schema.
JSON_FORMAT = "json"

#: `output_format="text"` -- hand back what the model said, unread.
TEXT_FORMAT = "text"

JSON_API_SYSTEM_PROMPT = """You are a JSON API endpoint. Your output is parsed directly by code, not read by humans.

ABSOLUTE REQUIREMENTS:
- Output ONLY valid JSON. Nothing else. No exceptions.
- Your entire response must be parseable by JSON.parse()
- First character of response must be { and last must be }

FORBIDDEN (will cause parsing errors):
- NO markdown: ```json or ``` blocks
- NO prose: "Here is the JSON:" or "Sure, here's..."
- NO explanations before or after the JSON
- NO comments inside JSON
- NO trailing text

CORRECT OUTPUT FORMAT:
{"field": "value", "number": 42}

WRONG OUTPUT FORMAT (DO NOT DO THIS):
```json
{"field": "value"}
```

JSON SYNTAX RULES:
- Double quotes for all strings and keys
- No trailing commas
- Numbers as numeric values (0.85 not "0.85")
- Use null for unknown values, [] for empty arrays

RESPOND WITH RAW JSON ONLY. START WITH { END WITH }"""


def compose_json_system_prompt(tool_system_prompt: str) -> str:
    """The format contract first, the tool's own instructions after a rule."""
    return f"{JSON_API_SYSTEM_PROMPT}\n\n---\n\n{tool_system_prompt}"


__all__ = [
    "JSON_API_SYSTEM_PROMPT",
    "JSON_FORMAT",
    "TEXT_FORMAT",
    "compose_json_system_prompt",
]
