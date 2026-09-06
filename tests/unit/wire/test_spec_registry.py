"""The shipped spec set is intact, and every spec still resolves.

Translated from the corresponding assertions in
`unified-library-ts/tests/unit/wire/spec-differential.test.ts` and
`chain-deltas.test.ts`. The specs are the etalon -- vendored byte-for-byte from
the TypeScript tree -- so a dropped file has to be a failure rather than a
silent gap, which is exactly what a count guard buys.
"""

from __future__ import annotations

import pytest

from combycode_llm_sdk.wire.inherit import resolve_spec
from combycode_llm_sdk.wire.registry import WIRE_SPECS, get_wire_spec
from combycode_llm_sdk.wire.service_specs import ABSTRACT as SERVICE_ABSTRACT


def test_it_ships_the_whole_set_so_a_dropped_file_is_a_failure() -> None:
    assert len(WIRE_SPECS) == 150


def test_every_spec_declares_the_id_it_is_registered_under() -> None:
    # The TypeScript map is written by hand, so a key can disagree with the
    # file it points at. Here the key IS the id, and this says so out loud.
    for spec_id, spec in WIRE_SPECS.items():
        assert spec["id"] == spec_id


@pytest.mark.parametrize("spec_id", sorted(WIRE_SPECS))
def test_every_shipped_spec_still_resolves(spec_id: str) -> None:
    # A chain delta that removes something the base no longer defines throws,
    # deliberately. Resolving every id is what catches that across the set.
    resolved = resolve_spec(spec_id, WIRE_SPECS)
    assert resolved["id"] == spec_id
    # A resolved spec must not carry variants: the pin already decided which
    # version applies.
    assert "variants" not in resolved


def test_get_wire_spec_names_an_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown wire spec: provider/does-not-exist"):
        get_wire_spec("provider/does-not-exist")


def test_every_abstract_id_is_actually_a_shipped_spec() -> None:
    # An abstract entry naming a spec that no longer exists would silently stop
    # guarding anything.
    for spec_id in SERVICE_ABSTRACT:
        assert spec_id in WIRE_SPECS
