/** Run the Python example corpus against live providers.
 *
 *  The twin of `unified-examples-ts/official-comparision/run.ts`, and it reuses
 *  that side's key path rather than growing a second one: `official-samples/keys.ts`
 *  reads the OS credential manager, and the sample only ever sees the resolved
 *  key through `LLM_API_KEY`. No keyring code reaches Python, so a Python example
 *  stays a statement about the library and not about this machine.
 *
 *  Usage:
 *    bun run examples/run.ts                      # every scenario, every provider
 *    bun run examples/run.ts --scenario=01,04     # some scenarios
 *    bun run examples/run.ts --provider=anthropic # one provider
 *    bun run examples/run.ts --list               # what would run
 */

import { spawnSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getApiKey } from '../../official-samples/keys.ts';
import { judge } from './judge.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const PY_ROOT = resolve(HERE, '..');
const CORPUS = join(HERE, 'official-comparison');
const MODELS = JSON.parse(
  readFileSync(resolve(HERE, '../../official-samples/models.json'), 'utf8'),
) as {
  providers: Record<
    string,
    { keyring: string; model: string; roles?: Record<string, string> }
  >;
};

/** Scenario metadata, shared with the TypeScript corpus. The role is read from
 *  here because a role is a property of the SCENARIO -- `15` needs a model that
 *  can hear, whichever corpus asks it. */
const CATALOG = JSON.parse(
  readFileSync(resolve(HERE, '../../official-samples/samples-catalog.json'), 'utf8'),
) as { scenarios: Array<{ id: string; model?: string; slow?: boolean; timeoutMs?: number }> };

/** What a correct answer looks like, for THIS corpus.
 *
 *  NOT the catalog's `expect`, which describes the TypeScript scenarios: those
 *  ask different questions, so judging these cells against them marks working
 *  cells as mismatches. Measured rather than assumed -- a live sweep reported
 *  02, 03, 09 and 10 as content mismatches while every one of them was right. */
const EXPECT = (
  JSON.parse(readFileSync(join(HERE, 'expectations.json'), 'utf8')) as {
    expect: Record<string, string>;
  }
).expect;

/** A scenario file's id: `29b_mcp_protocol.py` is `29b`, not `29`.
 *
 *  Taking the first two characters would give `29b` the role and expectation of
 *  a DIFFERENT scenario -- 29's `/stdio|http/` would then fail 29b, whose answer
 *  is a protocol summary with neither word in it. */
function idOf(file: string): string {
  return file.slice(0, file.indexOf('_'));
}

function scenarioOf(file: string): { model?: string } | undefined {
  return CATALOG.scenarios.find((s) => s.id === idOf(file));
}

/** The model a scenario should run on for this provider.
 *
 *  A scenario that needs a particular KIND of model says so as a role -- `audio`,
 *  `stt`, `image` -- and each provider maps its roles to its own model ids. The
 *  runner used to inject the default chat model for every cell, which is how
 *  `15_audio_in` and `18_speech_to_text` came to ask `gpt-5.4-nano` to listen: it
 *  has `capabilities.audio: false`, so OpenAI rejected the request and the cell
 *  looked like a library failure rather than the wrong model.
 *
 *  Falls back to the chat model when the provider has no such role, so a
 *  provider that cannot serve one still runs and fails for its own reason. */
function modelFor(file: string, config: { model: string; roles?: Record<string, string> }): string {
  const role = scenarioOf(file)?.model;
  return (role && config.roles?.[role]) || config.model;
}

/** The role this provider cannot serve, or null when the cell can run.
 *
 *  A scenario needing a KIND of model can only say something where the provider
 *  HAS one. Anthropic has no audio model, so `15_audio_in` there is not a
 *  failing cell -- it is a question that cannot be put. Running it anyway asked
 *  a text model to listen and then scored its polite refusal, which measures
 *  nothing about either the library or the provider.
 *
 *  Keyed on the provider's OWN role map, deliberately, and not on the catalog's
 *  support matrix. That matrix describes the OFFICIAL SDKs, and this library
 *  beats them in places -- it marks 05, 09 and 12 unsupported on xAI and
 *  OpenRouter where this corpus passes them live. Skipping on that basis would
 *  throw away real evidence.
 *
 *  This hides no gap of ours: every unported name still fails loudly on the
 *  providers that DO have a model for the role, which for each of them is at
 *  least two. */
function unservableRole(
  file: string,
  config: { roles?: Record<string, string> },
): string | null {
  const role = scenarioOf(file)?.model;
  if (!role) return null;
  return config.roles?.[role] ? null : role;
}

/** What a correct answer looks like, or null when nothing is recorded.
 *
 *  Case-insensitive: the patterns are content words (`sunny`, `paris`) and no
 *  scenario is asking about capitalisation. A scenario with no entry falls back
 *  to the non-empty check, which is honest about not knowing rather than
 *  inventing a criterion. */
function timeoutFor(file: string): number {
  const sc = scenarioOf(file);
  return sc?.timeoutMs ?? (sc?.slow ? SLOW_CELL_TIMEOUT_MS : CELL_TIMEOUT_MS);
}

function expectFor(file: string): RegExp | null {
  const pattern = EXPECT[idOf(file)];
  return pattern ? new RegExp(pattern, 'i') : null;
}

const CELL_TIMEOUT_MS = 180_000;
/** A batch job or a live session is slow by nature, not by fault. The catalog
 *  says which, and the TypeScript runner has always given those a longer
 *  budget; this one used a flat 180s, and Anthropic's batch cell -- 139s on one
 *  sweep -- crossed it on the next and was reported as a failure.
 *
 *  300s was still not enough. Measured 2026-09-04, all four batch cells green:
 *  xai 17s, google 114s, openai 141s, anthropic **876s**. A provider's queue
 *  depth is not something this corpus controls, and reporting a job that WORKED
 *  as a failure is the exact mistake the previous raise was for -- so the budget
 *  is set from the slowest MEASURED cell with room over it, not from a guess. */
const SLOW_CELL_TIMEOUT_MS = 1_200_000;

function arg(name: string): string | undefined {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit?.slice(name.length + 3);
}

const only = arg('scenario')?.split(',').map((s) => s.trim());
const onlyProvider = arg('provider');
const listOnly = process.argv.includes('--list');

/** Scenario files, by their numeric prefix. Deferred ones are skipped: they are
 *  written and reviewed but need features this port has not reached. */
const scenarios = readdirSync(CORPUS)
  .filter((f) => /^\d/.test(f) && f.endsWith('.py'))
  .filter((f) => !readFileSync(join(CORPUS, f), 'utf8').includes('DEFERRED'))
  .sort();

const providers = Object.entries(MODELS.providers).filter(
  ([name]) => !onlyProvider || name === onlyProvider,
);

const selected = scenarios.filter(
  (f) => !only || only.some((s) => f.startsWith(s.padStart(2, '0'))),
);

if (listOnly) {
  console.log(`scenarios (${selected.length}):`, selected.join(', '));
  console.log(`providers  (${providers.length}):`, providers.map(([p]) => p).join(', '));
  process.exit(0);
}

interface Cell {
  scenario: string;
  provider: string;
  ok: boolean;
  soft: boolean;
  ms: number;
  detail: string;
}

const results: Cell[] = [];
/** Cells that were never asked, and so are not failures. Kept out of `results`
 *  on purpose: counting them either way would be a lie -- as passes they
 *  inflate the number, as failures they blame a provider for not being a
 *  different one. */
const skipped: Array<{ scenario: string; provider: string; role: string }> = [];

for (const [provider, config] of providers) {
  let key: string;
  try {
    key = await getApiKey(config.keyring);
  } catch (e) {
    console.log(`SKIP ${provider}: ${(e as Error).message}`);
    continue;
  }

  // xAI's Collections API is a SECOND plane with its own credential, and a
  // scenario that manages a corpus needs it. Absent is not an error -- most
  // runs never touch it -- so this is offered, never required.
  let managementKey: string | undefined;
  if (provider === 'xai') {
    try {
      managementKey = await getApiKey('grokManagement');
    } catch {
      managementKey = undefined;
    }
  }

  for (const file of selected) {
    const missing = unservableRole(file, config);
    if (missing) {
      skipped.push({ scenario: file, provider, role: missing });
      console.log(
        `SKIP ${provider.padEnd(11)} ${file.padEnd(28)}         ` +
          `no ${missing} model for this provider`,
      );
      continue;
    }
    const started = Date.now();
    const res = spawnSync('python', [join(CORPUS, file)], {
      cwd: PY_ROOT,
      encoding: 'utf8',
      timeout: timeoutFor(file),
      env: {
        ...process.env,
        PYTHONPATH: join(PY_ROOT, 'src'),
        PYTHONIOENCODING: 'utf-8',
        LLM_API_KEY: key,
        LLM_MODEL: `${provider}/${modelFor(file, config)}`,
        ...(managementKey ? { XAI_MANAGEMENT_API_KEY: managementKey } : {}),
      },
    });
    const ms = Date.now() - started;
    const out = (res.stdout ?? '').trim();
    const err = (res.stderr ?? '').trim();
    // The last line is the sample's own JSON result; anything before it is noise.
    const last = out.split('\n').pop() ?? '';
    const verdict = judge(res.status, last, expectFor(file));
    const ok = verdict.ok;
    const detail =
      res.status === 0
        ? (verdict.ok ? last : `${verdict.why}: ${last}`)
        : (err.split('\n').pop() ?? `exit ${res.status}`);
    // SOFT is its own word on the line. A model that answered the wrong thing
    // and a call that never worked are both "not a pass", but they send you to
    // completely different places, and one label for both wastes that.
    const label = ok ? 'PASS' : verdict.soft ? 'SOFT' : 'FAIL';
    results.push({ scenario: file, provider, ok, soft: Boolean(verdict.soft), ms, detail });
    console.log(`${label} ${provider.padEnd(11)} ${file.padEnd(28)} ${String(ms).padStart(6)}ms  ${detail.slice(0, 120)}`);
  }
}

const passed = results.filter((r) => r.ok).length;
const soft = results.filter((r) => r.soft).length;
const failed = results.length - passed;
const notApplicable = skipped.length
  ? `, ${skipped.length} not applicable (no model for the role the scenario needs)`
  : '';
console.log(
  `\n${passed}/${results.length} cells passed${notApplicable}` +
    (soft
      ? ` — ${soft} of the ${failed} failure${failed === 1 ? '' : 's'} ` +
        `${soft === 1 ? 'is a content mismatch' : 'are content mismatches'}`
      : ''),
);
process.exit(passed === results.length ? 0 : 1);
