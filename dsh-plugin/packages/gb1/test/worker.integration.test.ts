import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { access, copyFile, mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as wait } from 'node:timers/promises'
import test from 'node:test'
import { WorkbenchBackend } from '../src/server.ts'
import { DEFAULT_CONFIG } from '../src/settings.ts'
import { REFERENCE } from '../src/runs.ts'
import type { CredentialsPort, LlmPort } from '../src/model.ts'

// 复制公开科学代码和公开数据至临时目录，使用项目独立解释器完成真实协议测试。
const project = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..')
const python = join(project, '.venv/Scripts/python.exe')
let available = true
try { await access(python); await access(join(project, 'data/raw/elife-16965-supp1-v4.xlsx')) } catch { available = false }
test('real worker completes four configurable rounds across a saved pause boundary', { skip: !available, timeout: 120000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'gb1-real-host-'))
  for (const directory of ['gb1', 'configs', 'prompts', 'data/raw', '.venv/Scripts']) await mkdir(join(root, directory), { recursive: true })
  for (const name of await readdir(join(project, 'gb1'))) if (name.endsWith('.py')) await copyFile(join(project, 'gb1', name), join(root, 'gb1', name))
  for (const name of ['legacy-v1.txt', 'scientific-v2.txt', 'scientific-v3.txt']) await copyFile(join(project, 'prompts', name), join(root, 'prompts', name))
  for (const name of ['configs/knowledge.json', 'data/audit.json', 'data/raw/elife-16965-supp1-v4.xlsx']) await copyFile(join(project, name), join(root, name))
  await writeFile(join(root, '.venv/Scripts/python.exe'), '')
  let requests = 0
  const llm = { listProviders: () => [], listModels: async () => [], prepareCall: async () => { requests++; throw new Error('Offline methods must not dispatch model requests') } } as unknown as LlmPort
  const credentials = { resolve: async () => undefined, describe: async () => ({ configured: false, writable: true }), set: async () => {}, unset: async () => {} } as CredentialsPort
  const backend = new WorkbenchBackend({ llm, credentials, spawnWorker: (_executable, argv, options) => spawn(python, argv, options) as never }, { projectRoot: root })
  try {
    assert.equal((await backend.getInfo()).ready, true)
    const config = { ...structuredClone(DEFAULT_CONFIG), seed: 765, rounds: 4, k: 13, proposal_count: 17, score_budget: 17,
      train_size: 40, validation_size: 20, test_size: 20, pool_size: 100, model_families: ['ridge'], ridge_alphas: [1], save_predictors: false }
    const run = await backend.startRun({ operationId: 'real-worker-offline-config-001', reference: REFERENCE, seed: 765, rounds: 4, method: 'model', config })
    await backend.pauseRun(run.id)
    const settled = async () => {
      const deadline = Date.now() + 90000
      for (;;) {
        const detail = await backend.getRun(run.id)
        if ((await backend.getInfo()).activeRunId === null) return detail
        assert.ok(Date.now() < deadline, 'Real worker exceeded test duration')
        await wait(20)
      }
    }
    const paused = await settled()
    assert.equal(paused.summary.status, 'paused', paused.summary.error ?? '')
    assert.equal(paused.summary.completedRounds, 1); assert.equal(paused.knownCount, 53); assert.equal(paused.roundResults[0]!.top10.length, 13)
    await backend.resumeRun(run.id)
    const completed = await settled()
    assert.equal(completed.summary.status, 'completed', completed.summary.error ?? '')
    assert.equal(completed.summary.completedRounds, 4); assert.equal(completed.knownCount, 92); assert.equal(completed.summary.requestCount, 0)
    assert.deepEqual(completed.roundResults.map(row => row.top10.length), [13, 13, 13, 13])
    assert.equal(new Set(completed.roundResults.flatMap(row => row.top10.map(item => item.variant))).size, 52)
    assert.equal((await backend.getRunPage(run.id, 2, 2)).totalRounds, 4)
    assert.equal(requests, 0)
  } finally { await backend.dispose() }
})
