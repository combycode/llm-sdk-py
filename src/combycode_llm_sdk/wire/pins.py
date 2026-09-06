"""Which spec builds a model's request when the catalog has no pin for it.

Transposed from `unified-library-ts/src/wire/pins.ts`.

Every catalogued model carries an explicit `wireSpec`, so this only decides for
the models the catalog does not know: one released after this build, or an engine
running without a catalog at all. That case is not an edge -- it is how the SDK
works on the day a provider ships something new -- so it keeps a real answer
rather than a guess.

The rule tables are DATA, not code, for the same reason the specs are: this port
reads `pins/*.json` instead of re-implementing version arithmetic and drifting
from the TypeScript. Two versions of that arithmetic is exactly how the 2.2.1
regression happened. The two files are vendored byte-identical and are etalon.

Rules are ordered and the first match wins; `default` answers everything else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_PINS_DIR = Path(__file__).resolve().parent / "pins"

#: `interface PinRule` (pins.ts:19) -- `{match, spec, why?}`. `match` is a
#: regular expression matched against the model id; `why` is read by humans, not
#: by the resolver.
PinRule = dict[str, Any]

#: `interface ModelPins` (pins.ts:27) -- `{id, rules?, default}`.
ModelPins = dict[str, Any]

_compiled: dict[str, list[tuple[re.Pattern[str], str]]] = {}


def _load(name: str) -> ModelPins:
    with (_PINS_DIR / name).open(encoding="utf-8") as handle:
        data: ModelPins = json.load(handle)
        return data


def _rules_for(pins: ModelPins) -> list[tuple[re.Pattern[str], str]]:
    """Compile once per table id, then reuse -- as `compiled` does in TypeScript."""
    hit = _compiled.get(pins["id"])
    if hit is None:
        hit = [(re.compile(r["match"]), r["spec"]) for r in pins.get("rules") or []]
        _compiled[pins["id"]] = hit
    return hit


_ANTHROPIC_PREFIX = re.compile(r"^anthropic/")


def pin_for(model: str, pins: ModelPins) -> str:
    """The spec id for `model`, from an ordered rule table.

    The id is lower-cased and stripped of an `anthropic/` prefix first, because a
    caller may legitimately pass either form and a band must not depend on which.

    `re.search`, not `re.match`: JavaScript's `RegExp.test` scans the whole
    string, and one of the shipped rules (`gemini-2[.]5`) is deliberately
    unanchored. Anchoring it here would silently stop matching `gemini-2.5-flash`
    and route every 2.5 model to the 3.x node, which 400s on `thinkingLevel`.
    """
    model_id = _ANTHROPIC_PREFIX.sub("", model.lower())
    for pattern, spec in _rules_for(pins):
        if pattern.search(model_id):
            return spec
    return str(pins["default"])


ANTHROPIC_MESSAGE_PINS: ModelPins = _load("anthropic.messages.json")
GOOGLE_GENERATE_PINS: ModelPins = _load("google.generate.json")

__all__ = [
    "ANTHROPIC_MESSAGE_PINS",
    "GOOGLE_GENERATE_PINS",
    "ModelPins",
    "PinRule",
    "pin_for",
]
