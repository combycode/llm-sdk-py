"""Sixty tools registered, one declared -- and the scope re-checked on the call.

Every tool in `tools=` is paid for on every turn, and a model reading sixty
declarations chooses worse for having read them. A `ToolCatalog` registers
everything and declares nothing: the model searches, and only what it matched
goes on the wire.

The other half is what makes that safe. `search(agent_id=...)` filtering by
scope is a courtesy to the model, not a boundary -- the agent id is optional, an
operator's own search passes none, and a tool name can be hallucinated without
any search at all. So `call()` re-checks the scope from scratch: a tool this
agent could never have discovered is still refused when it asks for it by name.

Deterministic: local tools, no key and no network.
"""

from typing import Any

from _check import check, report

from combycode_llm_sdk import ToolCatalog, tool
from combycode_llm_sdk.tool_catalog import (
    DEFAULT_SEARCH_LIMIT,
    AgentScope,
    NoToolAccess,
)

#: Enough lookalike tools that declaring the catalog outright is the expensive
#: option, and enough that the search has something to be wrong about.
NOISE_TOOLS = 57

SUPPORT_AGENT = "support"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"sunny in {city}"


@tool
def refund_order(order_id: str) -> str:
    """Refund an order that has already shipped."""
    return f"refunded {order_id}"


@tool
def send_email(to: str) -> str:
    """Send an email to a colleague."""
    return f"sent to {to}"


def noise_tool(index: int) -> Any:
    """One more plausible tool about orders, distinct in name only."""

    def handler(order_id: str) -> str:
        return f"report {index} for {order_id}"

    handler.__name__ = f"order_report_{index}"
    handler.__doc__ = "Produce a report about an order."
    return tool(handler)


catalog = ToolCatalog()
for one in [get_weather, refund_order, send_email, *(noise_tool(i) for i in range(NOISE_TOOLS))]:
    catalog.register(one)

registered = NOISE_TOOLS + 3

# The support agent may read the weather. It may not refund anything.
catalog.set_agent_scope(SUPPORT_AGENT, AgentScope(tool_names=("get_weather",)))

# A model asks in a sentence, and no description contains that phrase, so the
# query is matched word by word rather than whole.
matches = catalog.search("look up the weather")
check([t.name for t in matches] == ["get_weather"], "a sentence must find the tool it describes")
check(len(matches) < registered, "the saving is the point: one declaration, not sixty")

# The cap is the other half of the saving. A query that matched fifty-eight
# tools and declared them all would have spent more than declaring everything.
broad = catalog.search("order")
check(len(broad) == DEFAULT_SEARCH_LIMIT, "a broad query is capped, not answered in full")
check(len(catalog.search("order", limit=None)) > DEFAULT_SEARCH_LIMIT, "the cap is a default")

# Discovery filtered by scope: the model is never told about a tool it may not
# call. The operator's own search -- no agent id -- still sees everything.
check("refund_order" in [t.name for t in catalog.search("refund an order")],
      "an unscoped search is the whole catalog, which is what an operator wants")
check("refund_order" not in [t.name for t in catalog.search("refund an order",
                                                            agent_id=SUPPORT_AGENT)],
      "a scoped search must never mention a tool out of scope")

allowed = catalog.call("get_weather", source=SUPPORT_AGENT, arguments={"city": "Paris"})
check(allowed.output == "sunny in Paris", "a tool in scope runs and returns its own value")

# The assertion the whole arrangement rests on: the name was never discovered
# through a scoped search, and asking for it directly does not get around that.
try:
    catalog.call("refund_order", source=SUPPORT_AGENT, arguments={"order_id": "A1"})
    refused = False
except NoToolAccess:
    refused = True
check(refused, "scope must be enforced on call(), not only on discovery")

report(registered=registered, declared=len(matches), match=matches[0].name,
       capped=len(broad), refused="refund_order")
