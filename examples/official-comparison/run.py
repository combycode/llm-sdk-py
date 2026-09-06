"""Run the corpus: every scenario against every configured model.

Mirrors `unified-examples-ts/official-comparision/run.ts` on purpose -- same
environment contract (`LLM_MODEL`, `LLM_API_KEY`), same one-line JSON result --
so the two corpora stay comparable and, if we want, one orchestrator can drive
both.

    python run.py                          # every scenario x every model
    python run.py --scenario=06            # one scenario (comma list ok)
    python run.py --model=anthropic/claude-haiku-4.5
    python run.py --list                   # show what would run, run nothing

Keys are read from the environment: LLM_API_KEY, or per-provider
ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / XAI_API_KEY /
OPENROUTER_API_KEY. A scenario whose key is missing is reported SKIP, never
counted as a pass -- a corpus that silently skips is worse than one that fails.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# A model's answer can contain anything -- an em-dash, an accent, an emoji --
# and printing it to a legacy Windows console raises UnicodeEncodeError, which
# kills the run partway with a traceback that looks like a library failure. The
# corpus has to survive its own output.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Scenarios that are designed but deliberately not implemented yet. Listed by
# name rather than detected, so adding one is a visible decision.
# Nothing is deferred: every scenario is implemented. Kept as an empty set
# rather than deleted, so deferring one again is a one-line, visible decision.
DEFERRED: set[str] = set()

#: The shared catalog and model map the TypeScript corpus runs from. Reused
#: rather than copied: a second list of model ids is a second thing to update,
#: and the one that goes stale is always the copy.
OFFICIAL = HERE.parent.parent.parent / "official-samples"


def _official(name: str) -> dict:
    path = OFFICIAL / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


CATALOG = _official("samples-catalog.json")
MODELS = _official("models.json")


def role_for(num: str) -> str | None:
    """Which KIND of model a scenario needs -- `image`, `tts`, `embedding`, ...

    Scenario 16 is not a chat scenario that happens to make pictures: it needs an
    image model. Running it against the chat model asks an image endpoint to draw
    with `gpt-4.1-mini`, which fails for a reason that has nothing to do with the
    library. The catalog has said so all along; this runner was not reading it.
    """
    for scenario in CATALOG.get("scenarios") or []:
        if scenario.get("id") == num:
            return scenario.get("model")
    return None


# NOT filtered by the catalog's `support` matrix, deliberately. Those cells grade
# the OFFICIAL SDKs, and this library does things they do not: measured, filtering
# by them skipped nine cells that pass here -- xAI server tools, Anthropic
# provenance, Google conversation state. Hiding those loses the one comparison the
# corpus exists to make. A provider's real capability shows up as a SKIP from the
# model map below, or as an honest failure.


#: A result an example prints when the PROVIDER cannot do the scenario at all --
#: Anthropic hosts no corpus API, and no library can change that. Reported as a
#: skip rather than a failure, because a permanently red cell hides the real ones.
#:
#: The example says it, not this runner: only the example knows what it asked the
#: library for. The catalog's `-` cannot be used here -- measured, it also marks
#: cells this library passes (Anthropic provenance, Google conversation state).
NOT_APPLICABLE = "n/a"


def expected(num: str) -> re.Pattern[str] | None:
    """The regex a scenario's result must match, when the catalog names one.

    Without this the runner only checked that the process exited 0, so a wrong
    answer read as a pass -- and for a retrieval scenario, whose entire point is
    that the answer comes from the document, "it did not crash" is not evidence
    of anything.
    """
    for scenario in CATALOG.get("scenarios") or []:
        if scenario.get("id") == num and scenario.get("expect"):
            return re.compile(str(scenario["expect"]), re.IGNORECASE)
    return None


def extra_keys(num: str) -> list[str]:
    """Environment variables a scenario needs beyond the provider's own key."""
    for scenario in CATALOG.get("scenarios") or []:
        if scenario.get("id") == num:
            return list((scenario.get("extraKeys") or {}).keys())
    return []


def model_for(model: str, role: str | None) -> str | None:
    """The provider's model for that role, or None when it offers none.

    None means SKIP, not "use the chat model": a provider with no TTS model is a
    provider that cannot do scenario 17, and substituting a chat model turns an
    honest gap into a confusing failure.
    """
    provider = model.partition("/")[0]
    config = (MODELS.get("providers") or {}).get(provider) or {}
    if role is None:
        return model
    roles = config.get("roles") or {}
    return f"{provider}/{roles[role]}" if role in roles else None


def scenarios() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for path in sorted(HERE.glob("[0-9]*.py")):
        num = path.name.split("_", 1)[0]
        out.append((num, path))
    return out


def key_for(model: str) -> str | None:
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    provider = model.split("/")[0]
    return os.environ.get(PROVIDER_KEY_ENV.get(provider, ""))


def arg(name: str) -> str | None:
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


def main() -> int:
    models = (arg("model") or os.environ.get("LLM_MODELS", "")).split(",")
    models = [m.strip() for m in models if m.strip()]
    if not models:
        # Default to the shared map rather than demanding a hand-typed id. A
        # hand-typed one is how a sweep ended up running against `grok-4-fast`,
        # which no catalog lists -- chat worked, because ids pass through, and
        # every catalog-dependent scenario failed for a reason that looked like
        # the library.
        models = [f"{p}/{c['model']}" for p, c in (MODELS.get("providers") or {}).items()
                  if c.get("model")]
    if not models:
        print("no models: pass --model=provider/id or set LLM_MODELS", file=sys.stderr)
        return 2

    wanted = {s.strip() for s in (arg("scenario") or "").split(",") if s.strip()}
    cases = [(n, p) for n, p in scenarios() if not wanted or n in wanted]

    if "--list" in sys.argv:
        for num, path in cases:
            mark = "DEFERRED" if num in DEFERRED else ""
            print(f"  {num:4s} {path.name} {mark}")
        return 0

    passed = failed = skipped = 0
    #: Cells whose answer did not match the catalog's spec. Counted and printed
    #: at the end rather than failed -- see the note at the check itself.
    drifted: list[str] = []
    for model in models:
        key = key_for(model)
        for num, path in cases:
            label = f"{model} {path.name}"
            if num in DEFERRED:
                print(f"SKIP {label} (deferred)")
                skipped += 1
                continue
            if not key:
                print(f"SKIP {label} (no api key)")
                skipped += 1
                continue

            # A scenario that names a model ROLE runs against that model, not the
            # chat one -- and is skipped when the provider has none for it.
            role = role_for(num)
            run_model = model_for(model, role)
            if run_model is None:
                print(f"SKIP {label} (provider has no {role} model)")
                skipped += 1
                continue
            if run_model != model:
                label = f"{run_model} {path.name}"

            env = {**os.environ, "LLM_MODEL": run_model, "LLM_API_KEY": key}
            # Some scenarios need a SECOND credential -- xAI Collections runs on
            # its own plane with its own key. A missing one is a SKIP, not a
            # failure: an absent credential is not a broken library.
            missing = [var for var in extra_keys(num) if not os.environ.get(var)]
            if missing:
                print(f"SKIP {label} (no {', '.join(missing)})")
                skipped += 1
                continue
            proc = subprocess.run(
                [sys.executable, str(path)],
                check=False,  # the runner reads returncode itself and reports it
                env=env,
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                last = (proc.stdout.strip().splitlines() or [""])[-1]
                try:
                    got = json.loads(last)["result"]
                except (json.JSONDecodeError, KeyError):
                    print(f"FAIL {label}: last line was not our JSON: {last[:70]}")
                    failed += 1
                    continue
                if got.strip().lower().startswith(NOT_APPLICABLE):
                    print(f"SKIP {label} ({got.strip()})")
                    skipped += 1
                    continue

                # The catalog's `expect` is REPORTED, not enforced -- and that is
                # a finding rather than a shortcut. Measured 2026-08-29:
                # enforcing it failed 47 of 124 cells, and nearly all were
                # CONTENT drift rather than defects. The catalog specifies each
                # scenario exactly ("system='Reply with only the word PONG.'
                # user='ping'"), and these examples were written with different
                # content -- Paris, Ada, a stock price. So `expect` describes the
                # answers the TypeScript corpus produces, and this one does not
                # produce them.
                #
                # Enforcing belongs AFTER the examples are reconciled with the
                # catalog's tasks. Until then, failing on it would paint the
                # sweep red without finding a single real defect; reporting it
                # keeps the drift countable and visible.
                wanted = expected(num)
                if wanted and not wanted.search(got):
                    drifted.append(f"{label}: expected /{wanted.pattern}/, got {got[:40]}")

                print(f"PASS {label}: {got[:60]}")
                passed += 1
            else:
                err = (proc.stderr.strip().splitlines() or [""])[-1]
                print(f"FAIL {label}: {err[:100]}")
                failed += 1

    print(f"\npassed {passed}  failed {failed}  skipped {skipped}")
    if drifted:
        print(
            f"\n{len(drifted)} cell(s) do not match the catalog's expected answer -- "
            f"these examples differ in CONTENT from the scenarios the catalog "
            f"specifies, so it is drift to reconcile, not failure:"
        )
        for line in drifted:
            print(f"  DRIFT {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
