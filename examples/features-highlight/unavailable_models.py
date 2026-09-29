"""Models the catalog has MEASURED as unreachable, and the one door left open.

A provider's model list is not a list of models you can call. `sora-2` is still
returned by OpenAI's `/v1/models` although `/v1/videos` shut down on 2026-09-24,
and Google's `imagen-4` family answers 404 on the Developer API because
`:predict` is Enterprise-only. A catalog built from a provider listing keeps
offering both, and the first thing a caller learns is a 404 with our own
spelling in it.

Two facts are kept apart on purpose:

    `unavailable_reason()` is the MEASUREMENT -- somebody called the endpoint and
    it was gone. It reports what was seen and when.

    `refuse_call()` is the DECISION built on it -- the sentence to fail with, or
    None to go ahead. Naming the provider's own id instead of our slug is the way
    through: we measured one account, not every account.

Deterministic: catalog only, no network.
"""

from _check import check, report

from combycode_llm_sdk import Engine, select_models

engine = Engine(
    catalog="defaults",
    api_keys={"openai": "k", "anthropic": "k", "google": "k", "xai": "k", "openrouter": "k"},
    register_as_default=False,
)
catalog = engine.catalog

# -- 1. the measurement, on the model itself ---------------------------------
imagen = catalog.get("google", "imagen-4")
check(imagen is not None and imagen.unavailable is not None,
      "imagen-4 should carry a measured `unavailable`")
measured = imagen.unavailable if imagen else None
check(bool(measured and measured.get("since")), "a measurement must say WHEN it was taken")

# Nothing serves it, so nothing should route to it by accident.
check(imagen is not None and imagen.active is False,
      "a measured-unavailable model must not be active")

# -- 2. the decision ---------------------------------------------------------
refusal = catalog.refuse_call("google", "imagen-4")
check(bool(refusal), "refuse_call must refuse our slug for a measured model")
# The refusal names the way through, so the message itself is the instruction.
check(bool(refusal and "imagen-4.0-generate-001" in refusal),
      "the refusal should name the provider id that still reaches the endpoint")

# Naming Google's own id is a deliberate request for THAT endpoint -- most
# plausibly from a deployment where it answers. Allowed.
forced = catalog.refuse_call("google", "imagen-4.0-generate-001")
check(forced is None, "the provider's own id is the force mode and must pass")

# The override lifts the DECISION, never the FACT.
check(catalog.unavailable_reason("google", "imagen-4.0-generate-001") is not None,
      "unavailable_reason must still report what was measured")

# -- 3. no door where there is nobody on the other side ----------------------
sora = catalog.refuse_call("openai", "sora-2")
check(bool(sora), "sora-2 should be refused")
check(not (sora and "name the provider" in sora), "sora-2 must not offer an override")

# -- 4. a model with nothing against it is untouched -------------------------
check(catalog.refuse_call("google", "gemini-3.1-flash-image") is None,
      "a working model must not be refused")
check(catalog.refuse_call("google", "a-model-nobody-has-heard-of") is None,
      "an unknown model must not be refused -- the catalog cannot be sure")

# -- 5. and select() never hands you one -------------------------------------
offered = select_models("type:image", engine=engine)
# `unavailable_reason` rather than the field: a catalog entry raises for a key
# it does not hold, which is what keeps a typo from reading as None.
dead = [m for m in offered if catalog.unavailable_reason(m.provider, m.model)]
check(not dead, f"select() offered {len(dead)} model(s) nobody can call")

report(
    measured_since=(measured or {}).get("since"),
    refused_our_slug=bool(refusal),
    forced_by_provider_id=forced is None,
    fact_survives_the_override=catalog.unavailable_reason(
        "google", "imagen-4.0-generate-001"
    ) is not None,
    sora_has_no_door=bool(sora) and "name the provider" not in (sora or ""),
    image_models_offered=len(offered),
    dead_ones_offered=len(dead),
)
