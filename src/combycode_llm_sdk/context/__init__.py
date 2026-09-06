"""What the model is told, and what survives when the transcript is cut.

Three pieces answering three different questions:

- `registry.py` -- WHO contributed what. A system prompt assembled from layers
  (persona, scenario, learned facts) rather than one string several writers
  fight over.
- `guard.py` -- WHEN to compact, and how. A trigger ladder over a calibrated
  measurement, with strategies that differ in what they preserve.
- `facts.py` -- WHAT must not be lost. The account id, the deadline, the path:
  pulled out verbatim and carried forward on their own, because a summary that
  paraphrases them has lost them.
"""

from __future__ import annotations

from .facts import (
    FACT_CATEGORIES,
    FACTS_CLOSE,
    FACTS_HEADER,
    FACTS_OPEN,
    LAYER_CHAT_FACTS,
    ExtractedFact,
    merge_facts,
    parse_facts_block,
    parse_facts_lines,
    read_facts_layer,
    render_facts_block,
    render_facts_layer,
    render_prior_facts_for_extraction,
    write_facts_block,
)

__all__ = [
    "FACTS_CLOSE",
    "FACTS_HEADER",
    "FACTS_OPEN",
    "FACT_CATEGORIES",
    "LAYER_CHAT_FACTS",
    "ExtractedFact",
    "merge_facts",
    "parse_facts_block",
    "parse_facts_lines",
    "read_facts_layer",
    "render_facts_block",
    "render_facts_layer",
    "render_prior_facts_for_extraction",
    "write_facts_block",
]
