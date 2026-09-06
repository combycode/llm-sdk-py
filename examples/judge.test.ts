/** The runner's verdict, tested against known-bad lines.
 *
 *  `bun test examples/judge.test.ts`
 *
 *  These exist because the previous criterion -- exit status alone -- could not
 *  go red for the failure it was most likely to meet. The first case here is the
 *  exact line `11_structured_json.py` printed on Google and xAI while the
 *  structured path was completely broken, and the old runner called it PASS.
 */

import { describe, expect, test } from 'bun:test';

import { judge } from './judge.ts';

describe('judge', () => {
  test('an empty result is a failure even though the process exited 0', () => {
    // The recorded known-bad artifact.
    expect(judge(0, '{"result": "", "ms": 822}')).toEqual({ ok: false, why: 'empty result' });
  });

  test('whitespace is not an answer either', () => {
    expect(judge(0, '{"result": "   ", "ms": 5}').ok).toBe(false);
  });

  test('a real answer passes', () => {
    expect(judge(0, '{"result": "Paris", "ms": 822}')).toEqual({ ok: true, why: '' });
  });

  test('a documented n/a still passes', () => {
    // Answered-nothing is the distinction, not answered-badly: a provider
    // without a batch API says so, and that is a real report.
    expect(judge(0, '{"result": "n/a: openrouter has no batch API", "ms": 0}').ok).toBe(true);
  });

  test('a non-zero exit is a failure whatever it printed', () => {
    expect(judge(1, '{"result": "Paris"}')).toEqual({ ok: false, why: 'exit 1' });
  });

  test('a crash with no JSON line is a failure, not a pass', () => {
    expect(judge(0, 'Traceback (most recent call last):').ok).toBe(false);
  });

  test('no output at all is a failure', () => {
    expect(judge(0, '').ok).toBe(false);
  });

  test('a result that is not a string is a failure', () => {
    // `{"result": null}` is how a scenario reports it gave up.
    expect(judge(0, '{"result": null}')).toEqual({ ok: false, why: 'no string result' });
  });

  test('an answer that does not match what the scenario expects is SOFT', () => {
    // The recorded known-bad artifact: `15_audio_in` against a model that
    // cannot hear. A real sentence, exit 0, and no evidence the audio arrived.
    const verdict = judge(
      0,
      '{"result": "I am unable to process audio files.", "ms": 900}',
      /hello/i,
    );
    expect(verdict.ok).toBe(false);
    expect(verdict.soft).toBe(true);
  });

  test('a matching answer passes', () => {
    expect(judge(0, '{"result": "Hello.", "ms": 745}', /hello/i).ok).toBe(true);
  });

  test('the match is case-insensitive, as the pattern is built', () => {
    expect(judge(0, '{"result": "HELLO", "ms": 1}', /hello/i).ok).toBe(true);
  });

  test('a scenario with nothing to expect still only needs an answer', () => {
    expect(judge(0, '{"result": "anything at all", "ms": 1}', null).ok).toBe(true);
  });

  test('a documented n/a outranks an expect it cannot possibly match', () => {
    // This runner runs every scenario on every provider, so "no such API here"
    // is a routine answer -- and failing it would punish the honest report.
    expect(judge(0, '{"result": "n/a: openrouter has no batch API", "ms": 0}', /2/).ok).toBe(
      true,
    );
  });

  test('an empty answer is a hard failure, not a content mismatch', () => {
    // Nothing came back at all: that is ours, not the model answering badly,
    // and labelling it SOFT would send someone to look at the wrong thing.
    const verdict = judge(0, '{"result": "", "ms": 5}', /hello/i);
    expect(verdict.ok).toBe(false);
    expect(verdict.soft).toBeUndefined();
  });
});
