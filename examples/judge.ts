/** Did a corpus cell actually answer?
 *
 *  Its own module so it can be imported and tested. `run.ts` executes the whole
 *  corpus at import, so a predicate living there is one nothing can check.
 */

export interface Verdict {
  ok: boolean;
  /** Why not, for the report line. Empty when ok. */
  why: string;
  /** A CONTENT mismatch rather than a broken call: the SDK worked, the model
   *  answered, and the answer was not the one the scenario asked for. Worth
   *  distinguishing, because the two point in different directions -- a hard
   *  failure is usually ours, a soft one is usually the model's. */
  soft?: boolean;
}

/** Judge one cell from its exit status, its last stdout line, and what the
 *  scenario said to expect.
 *
 *  Exit status ALONE is not enough, and trusting it hid a real bug: with
 *  `structured=` broken, `11_structured_json.py` returned `''` -- it is written
 *  as `result.parsed.city if result.parsed else ''` -- and exited 0, so three of
 *  five providers reported PASS on a scenario that never once parsed an answer.
 *  A green run that cannot go red is not evidence of anything.
 *
 *  Nor is a non-empty answer enough, which is the same lesson one step on.
 *  Pointed at a model that cannot hear, `15_audio_in` came back with `I'm unable
 *  to process audio files` -- a real sentence, a passing cell, and no evidence
 *  whatsoever that the audio ever arrived. So the catalog's `expect` pattern is
 *  applied here, exactly as the TypeScript corpus runner applies it.
 *
 *  A scenario that is legitimately unavailable says so in words
 *  (`'n/a: openrouter has no batch API'`) and still passes, even against an
 *  `expect` it cannot match. That escape matters MORE here than in the
 *  TypeScript runner, which runs a scenario only on the providers its support
 *  matrix lists; this one runs every scenario everywhere, so "this provider has
 *  no such API" is a routine and honest result rather than a failure.
 */
export function judge(
  status: number | null,
  lastLine: string,
  expect?: RegExp | null,
): Verdict {
  if (status !== 0) return { ok: false, why: `exit ${status}` };

  let parsed: unknown;
  try {
    parsed = JSON.parse(lastLine);
  } catch {
    return { ok: false, why: 'no JSON result line' };
  }

  const result = (parsed as { result?: unknown })?.result;
  if (typeof result !== 'string') return { ok: false, why: 'no string result' };
  const answer = result.trim();
  if (answer === '') return { ok: false, why: 'empty result' };
  if (answer.toLowerCase().startsWith('n/a:')) return { ok: true, why: '' };
  if (expect && !expect.test(answer)) {
    return { ok: false, soft: true, why: `expected ${expect}` };
  }
  return { ok: true, why: '' };
}
