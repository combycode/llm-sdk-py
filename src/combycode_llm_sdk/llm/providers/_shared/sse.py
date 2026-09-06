"""The one piece of SSE handling every provider genuinely shares.

Transposed from `unified-library-ts/src/llm/providers/_shared/sse.ts`.

Report 035 proposed a shared "SSE -> StreamEvent parser" because the parsing
looked duplicated ~5x. Measured, it is not: the six parse bodies (76-149 lines
each) share ZERO runs of three or more identical lines, because they decode
different wire schemas -- Anthropic's `content_block_*` events, OpenAI chat's
`choices[].delta`, OpenAI Responses' typed events, and Google's
`candidates[].content.parts`. A "shared" parser would be a switch on provider
wearing a common signature.

What IS shared is exactly this line, repeated six times. Naming it gives the
three language ports one primitive to agree on instead of six independent
decisions, and one place to harden if malformed frames ever need handling (today
a bad frame raises out of the parser, which is the existing behaviour and
deliberately unchanged here).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def sse_json(event: Mapping[str, Any]) -> dict[str, Any]:
    """Decode an SSE frame's `data` payload as a JSON object."""
    decoded: dict[str, Any] = json.loads(event["data"])
    return decoded


__all__ = ["sse_json"]
