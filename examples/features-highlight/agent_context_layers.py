"""Context as named layers the agent assembles, not one string several authors edit.

A conversation's system prompt has more than one author: the caller pins a
persona, the run supplies its scenario, the agent learns facts as it goes, the
context guard leaves a summary behind after compacting. Concatenating those by
hand is how the memory notes sit above the persona on Tuesday and below it on
Wednesday -- which moves the cache prefix and quietly stops every prompt-cache
hit.

So each contributor writes its own named `ContextLayer` and never touches
anyone else's, and precedence is priority, then age, then name -- never
insertion order. Stable layers get low numbers and render first; churny ones
land after the prefix the cache depends on. A registry can also have a PARENT,
which is how orchestrator-wide context reaches one conversation without being
copied into it: a child layer replaces the parent's by default, or merges with
it when it says so, because an override that silently appends is a layer you
cannot get rid of.

**What goes wrong without layers.** A fact stated once in message 3 dies the
moment message 3 is summarised away. Layers are not messages, so compaction --
which rewrites the message list -- has nothing here to drop, and the persona,
the run context and the pinned facts are still rendered afterwards.

Deterministic: no provider, no key, no network. The compaction below is done by
hand, because the point is not how the turns were destroyed but that the layer
was never in that list to destroy.
"""

from _check import check, report

from combycode_llm_sdk import ContextLayer, ContextRegistry

PERSONA = "You are a terse incident-response assistant."
RUN_CONTEXT = "Incident INC-4417, sev-2, opened 09:12 UTC."
FACT = "the on-call engineer is Dana"
ORG_PERSONA = "You work for ACME."
ORG_MEMORY = "ACME measures everything in metric units."
CHAT_MEMORY = "Dana prefers UTC timestamps."

#: Low is stable and renders first, so the cache prefix starts there; high is
#: rewritten turn to turn and must land after it. These are the numbers the
#: library's own well-known layers use.
PRIORITY_PERSONA = 10
PRIORITY_RUN = 100
PRIORITY_MEMORY = 200
PRIORITY_FACTS = 250

# -- precedence --------------------------------------------------------------

registry = ContextRegistry(id="conversation")

# Written churniest-first on purpose: if insertion order decided, the facts
# would render in front of the persona.
registry.set("chat.facts", FACT, priority=PRIORITY_FACTS)
registry.set("agentloop.context", RUN_CONTEXT, priority=PRIORITY_RUN)
persona: ContextLayer = registry.set("agentloop.system", PERSONA, priority=PRIORITY_PERSONA)

rendered = registry.render()
check(
    [p.name for p in rendered.parts] == ["agentloop.system", "agentloop.context", "chat.facts"],
    f"priority must decide the order, not the order written: {[p.name for p in rendered.parts]}",
)
check(rendered.flat.startswith(PERSONA), "the stable layer renders first or the cache prefix moves")
check(persona.version == 1, "a layer is versioned from its first write")

prefix = registry.flat()[: len(PERSONA)]
rewritten = registry.set("chat.facts", f"{FACT}, reachable on +49 30 0000")

# The quiet one: a writer that updates content without restating the priority
# would fall back to the default and jump above the run context it must follow.
check(rewritten.priority == PRIORITY_FACTS, f"the rewrite fell to {rewritten.priority}")
check(rewritten.version == 2, "every mutation bumps the version")
check(registry.flat()[: len(PERSONA)] == prefix, "rewriting a churny layer moved the prefix")
check(FACT in registry.flat(), "the rewritten layer must still be rendered")

# -- inheriting without copying ----------------------------------------------

org = ContextRegistry(id="orchestrator")
org.set("agentloop.system", ORG_PERSONA, priority=PRIORITY_PERSONA)
org.set("memory", ORG_MEMORY, priority=PRIORITY_MEMORY)

chat = ContextRegistry(id="conversation", parent=org)
chat.set("agentloop.system", PERSONA, priority=PRIORITY_PERSONA)
chat.set("memory", CHAT_MEMORY, priority=PRIORITY_MEMORY, merge_parent=True)

composed = chat.flat()
check(PERSONA in composed, "the child's own layer must be rendered")
check(ORG_PERSONA not in composed, "a child layer REPLACES the parent's by default")
check(ORG_MEMORY in composed, "merge_parent must keep the half that was inherited")
check(composed.index(ORG_MEMORY) < composed.index(CHAT_MEMORY), "the inherited half comes first")
check(chat.get("memory").content == CHAT_MEMORY, "merging is a render, not a write into the child")

# -- surviving the compaction that erases the turn ---------------------------

messages = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
messages[1] = {"role": "user", "content": f"by the way, {FACT}"}
chat.set("chat.facts", FACT, priority=PRIORITY_FACTS)

# What a compaction leaves behind: the oldest turns are gone for good.
del messages[:4]

transcript = " ".join(str(m["content"]) for m in messages)
check(FACT not in transcript, "the setup must really destroy the turn that stated the fact")
check(FACT in chat.flat(), "a fact kept in a layer must outlive the turn it was learned from")

report(
    order=[p.name for p in rendered.parts],
    version=rewritten.version,
    composed_layers=len(chat.render().parts),
    chars=chat.size_chars(),
    fact_in_transcript=FACT in transcript,
    fact_in_layers=FACT in chat.flat(),
)
