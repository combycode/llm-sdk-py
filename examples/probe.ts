/** Run one Python file with a live key. A debugging companion to run.ts --
 *  same key path, no corpus, full stdout. */
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { getApiKey } from '../../official-samples/keys.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const PY_ROOT = resolve(HERE, '..');
const [file, keyring = 'claude', model = 'anthropic/claude-haiku-4.5'] = process.argv.slice(2);

const res = spawnSync('python', [file], {
  cwd: PY_ROOT,
  encoding: 'utf8',
  env: {
    ...process.env,
    PYTHONPATH: resolve(PY_ROOT, 'src'),
    PYTHONIOENCODING: 'utf-8',
    LLM_API_KEY: await getApiKey(keyring),
    LLM_MODEL: model,
  },
});
if (res.stdout) console.log(res.stdout);
if (res.stderr) console.error(res.stderr);
process.exit(res.status ?? 1);
