"""`make_registry` against the etalon wire specs.

NOT a translation: `unified-library-ts` has no `wire-transforms.test.ts`. Its
registry is exercised indirectly, by the `tests/unit/wire/*` differentials that
replay a corpus through real provider adapters -- none of which exist in Python
yet. Left at that, all 305 lines of `wire-transforms.ts` would land with no check
of any kind, which is the exact condition PORTING.md says produced the first
attempt's invisible losses.

So this asserts the one thing that CAN be checked without an adapter, against an
oracle this port did not author and cannot argue with: **every named rule the
vendored specs delegate to must exist in the registry, under exactly the name the
spec uses.** `specs/` is etalon -- byte-identical with the TypeScript tree and
read-only for the whole effort -- so a transform that was dropped, renamed while
being snake_cased, or misfiled as a builder fails here rather than at the first
real request, where the interpreter raises "unknown transform: ..." only if that
particular branch happens to be reached.

What this does NOT cover, stated so the gap is countable: the BEHAVIOUR of each
rule. That needs the provider adapters and the response corpus, and until they
land `wire-transforms.py` is `transposed-only` for everything except the handful
of rules reached by `test_strict_schema.py`, `test_service_tier.py` and
`test_moderation_native.py`.
"""

from __future__ import annotations

from typing import Any

from combycode_llm_sdk.llm.wire_transforms import make_registry
from combycode_llm_sdk.wire.registry import WIRE_SPECS


def _walk(node: Any, key: str, out: set[str]) -> None:
    """Every value of `key` anywhere in a spec, at any depth."""
    if isinstance(node, dict):
        v = node.get(key)
        if isinstance(v, str):
            out.add(v)
        for child in node.values():
            _walk(child, key, out)
    elif isinstance(node, list):
        for child in node:
            _walk(child, key, out)


def _referenced() -> dict[str, set[str]]:
    """What the shipped specs name, split by which registry table serves it.

    The split follows the interpreter, not a guess: `$call` and a field's `call`
    are looked up in `transforms`, a block's `call` in `builders`, `pred` in
    `predicates`, and a block's `effects` plus an overlay's `op: 'call'` in
    `effects`.
    """
    transforms: set[str] = set()
    builders: set[str] = set()
    predicates: set[str] = set()
    effects: set[str] = set()

    for spec in WIRE_SPECS.values():
        _walk(spec, "$call", transforms)
        _walk(spec, "pred", predicates)
        for f in spec.get("fields") or []:
            if f.get("call"):
                transforms.add(f["call"])
        for v in spec.get("variants") or []:
            if v.get("fn"):
                transforms.add(v["fn"])
        for b in spec.get("blocks") or []:
            if b.get("call"):
                builders.add(b["call"])
            for e in b.get("effects") or []:
                effects.add(e)
        for overlay in (spec.get("overlays") or {}).values():
            for op in overlay.get("ops") or []:
                if op.get("op") == "call" and op.get("call"):
                    effects.add(op["call"])

    return {
        "transforms": transforms,
        "builders": builders,
        "predicates": predicates,
        "effects": effects,
    }


#: The rules the shipped specs name that `wire-transforms.ts` deliberately does
#: NOT carry, with the file that does. Not an exemption invented to make this
#: pass -- `src/plugins/mcp/wire-rules.ts:1-17` states the reason in the
#: TypeScript itself: `llm -> plugins` is a forbidden edge that the layer test
#: enforces, and MCP is a plugin, so the MCP transport composes its own registry
#: from the shared one (`mcpWireRegistry(base)`) instead of the shared one
#: reaching down into MCP. They land with the MCP area.
#:
#: A THIRD name appearing here is a real failure and must not be added without
#: finding the TypeScript file that owns it.
_OWNED_BY_ANOTHER_REGISTRY: dict[str, set[str]] = {
    #: wire-rules.ts:41 -- the method's subject for the `Mcp-Name` routing header.
    "transforms": {"mcpNameHeader"},
    "builders": set(),
    #: wire-rules.ts:34 -- modern by negotiated era OR by declared version.
    "predicates": {"mcpModern"},
    "effects": set(),
}


class TestMakeRegistry:
    def test_builds_with_no_adapter_handles_at_all(self) -> None:
        # An adapter builds a registry to drive its OWN spec and carries only its
        # own handle; requiring the full set would force every adapter to import
        # every other adapter. `makeRegistry({})` is the shape the interpreter
        # tests already rely on -- wire-transforms.ts:34-47.
        reg = make_registry({})
        assert reg.transforms and reg.builders and reg.predicates and reg.effects

    def test_every_rule_the_etalon_specs_name_exists_in_the_registry(self) -> None:
        reg = make_registry({})
        available = {
            "transforms": set(reg.transforms),
            "builders": set(reg.builders),
            "predicates": set(reg.predicates),
            "effects": set(reg.effects),
        }
        missing = {
            table: sorted(names - available[table] - _OWNED_BY_ANOTHER_REGISTRY[table])
            for table, names in _referenced().items()
            if names - available[table] - _OWNED_BY_ANOTHER_REGISTRY[table]
        }
        assert not missing, f"the specs name rules the registry does not carry: {missing}"

    def test_the_rules_owned_by_another_registry_are_still_absent_from_this_one(self) -> None:
        # The exemption above is only honest while it stays an exemption: if
        # `mcpNameHeader` ever turns up in `make_registry`, the forbidden
        # `llm -> plugins` edge has been drawn and the exemption is hiding it.
        reg = make_registry({})
        assert "mcpNameHeader" not in reg.transforms
        assert "mcpModern" not in reg.predicates

    def test_the_predicate_the_interpreter_calls_by_name_is_present(self) -> None:
        # `{"isFunctionTool": true}`, `{"builtin": ...}` and `{"hasTool": ...}` do
        # not go through `pred`: interpreter.py reaches straight into
        # `reg.predicates["isFunctionTool"]`, so a KeyError there is a crash rather
        # than the interpreter's own "unknown predicate" message.
        assert "isFunctionTool" in make_registry({}).predicates
