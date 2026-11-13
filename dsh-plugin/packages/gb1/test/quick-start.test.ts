import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { PassThrough } from 'node:stream'
import { setTimeout as wait } from 'node:timers/promises'
import test from 'node:test'
import type { ChildProcessWithoutNullStreams } from 'node:child_process'
import { QuickStartManager, type QuickStartHooks } from '../src/quick-start.ts'
import { ModelGateway, type CredentialsPort, type LlmPort } from '../src/model.ts'
import { WorkbenchBackend } from '../src/server.ts'
import { DEFAULT_CONFIG } from '../src/settings.ts'
import type { RunSummary } from '../src/contracts.ts'
import type { SpawnWorker } from '../src/runs.ts'
const SECRET = 'SYNTHETIC_PRIVATE_QUICK_START_VALUE'
class Child extends EventEmitter {
  stdin = new PassThrough(); stdout = new PassThrough(); stderr = new PassThrough(); kills = 0; closed = false
  frames: unknown[] = []
  constructor() { super(); this.stdin.on('data', chunk => String(chunk).trim().split('\n').filter(Boolean).forEach(line => this.frames.push(JSON.parse(line)))) }
  output(value: unknown) { this.stdout.write(JSON.stringify(value) + '\n') }
  close(code: number | null = 0) { if (!this.closed) { this.closed = true; queueMicrotask(() => this.emit('close', code)) } }
  kill() { this.kills++; this.close(null); return true }
}
async function until(check: () => boolean | Promise<boolean>) {
  const deadline = Date.now() + 5000
  while (Date.now() < deadline) { if (await check()) return; await wait(2) }
  assert.fail('Quick-start fake operation did not settle')
}
const input = (operationId = 'quick-operation-example-0001') => ({ operationId })
async function fixture(options: { invalidPython?: boolean; missingModel?: boolean; effort?: string | null } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'gb1-quick-start-'))
  for (const path of ['gb1/data.py', 'requirements.txt', 'configs/knowledge.json', 'tools/quick_start/bootstrap.py']) { await mkdir(dirname(join(root, path)), { recursive: true }); await writeFile(join(root, path), '') }
  await writeFile(join(root, 'configs/workbench-defaults.json'), JSON.stringify({ config: { ...DEFAULT_CONFIG, llm_reasoning_effort: options.effort === undefined ? 'xhigh' : options.effort }, promptProfile: 'v3', recommendedModel: { provider: 'fake', model: 'test-model' } }))
  const llm = { listProviders: () => [{ id: 'fake', name: 'Fake' }], listModels: async () => options.missingModel ? [] : [{ provider: 'fake', id: 'test-model', name: 'Test' }],
    resolveModelInfo: async () => ({ provider: 'fake', id: 'test-model', name: 'Test', reasoning: { efforts: [{ id: 'xhigh', name: 'Extra high' }] } }),
    prepareCall: async () => { throw new Error('A metadata preflight must not generate') } } as unknown as LlmPort
  const credentials = { describe: async () => ({ configured: false, writable: true }), resolve: async () => undefined, set: async () => {}, unset: async () => {} } as CredentialsPort
  const probes: Child[] = [], bootstrap: Child[] = [], starts: unknown[] = [], cancelled: string[] = [], spawnArguments: unknown[][] = []
  const spawnBootstrap: SpawnWorker = (executable, argv, settings) => {
    const child = new Child(); spawnArguments.push([executable, argv, settings])
    if (argv.includes('-c')) { probes.push(child); queueMicrotask(() => { child.stdout.write(options.invalidPython ? 'UNSUPPORTED\n' : 'GB1_PYTHON_3_12\n'); child.close() }) }
    else bootstrap.push(child)
    return child as unknown as ChildProcessWithoutNullStreams
  }
  let activeId: string | null = null
  let summary: RunSummary | null = null
  const hooks: QuickStartHooks = {
    activeRunId: () => activeId,
    startRun: async value => { starts.push(structuredClone(value)); activeId = 'gb1-00000000-0000-0000-0000-000000000001'; summary = { id: activeId, operationId: value.operationId, status: 'running', stage: 'starting', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), seed: value.seed, rounds: value.rounds, method: value.method,
      model: { source: 'dsh', provider: value.modelOverride!.provider, id: value.modelOverride!.model }, completedRounds: 0, requestCount: 0, error: null, config: value.config }; return summary },
    getRun: async id => { if (!summary || summary.id !== id) throw new Error(); return summary },
    cancelRun: async id => { cancelled.push(id); activeId = null; summary!.status = 'cancelled'; return summary! },
  }
  const gateway = new ModelGateway(llm, credentials)
  const manager = new QuickStartManager(root, gateway, hooks, { bootstrapPython: 'C:\\Python312\\python.exe', spawnBootstrap })
  await manager.initialize()
  return { root, llm, credentials, gateway, hooks, manager, probes, bootstrap, starts, cancelled, spawnArguments, spawnBootstrap,
    finishRun: () => { summary!.status = 'completed'; activeId = null }, setActive: (id: string | null) => { activeId = id } }
}
function success(child: Child) {
  child.output({ version: 1, type: 'progress', stage: 'dependencies', status: 'reused', message: SECRET, counts: { required_packages: 42, matched_packages: 42, dependencies_reused: true } })
  child.output({ version: 1, type: 'progress', stage: 'validate', status: 'completed', message: SECRET, counts: { variants: 149361, unique_variants: 149361, wt_is_one: true } })
  child.output({ version: 1, type: 'result', stage: 'ready', status: 'ready', code: 'READY', message: SECRET, counts: {} }); child.close()
}

test('quick start returns before preparation, deduplicates operation IDs and runs exactly the recommended three-round trajectory', async () => {
  const f = await fixture()
  const [first, repeated] = await Promise.all([f.manager.start(input()), f.manager.start(input())])
  assert.equal(first.id, repeated.id); assert.equal(first.status, 'preparing'); assert.equal(first.runId, null)
  await until(() => f.bootstrap.length === 1)
  await assert.rejects(f.manager.start(input('quick-operation-example-0002')), /已有环境准备/)
  await assert.rejects(f.manager.start({ ...input(), reasoningEffort: null }), /其他一键运行参数/)
  success(f.bootstrap[0]!)
  await until(async () => (await f.manager.get(first.id))?.status === 'running')
  const status = (await f.manager.get())!
  assert.equal(f.starts.length, 1); assert.equal(status.maxModelCalls, 6); assert.equal(status.requestedRounds, 3)
  const start = f.starts[0] as { rounds: number; method: string; promptProfile: string; config: typeof DEFAULT_CONFIG; modelOverride: unknown }
  assert.equal(start.rounds, 3); assert.equal(start.method, 'knowledge'); assert.equal(start.promptProfile, 'v3')
  assert.equal(start.config.max_model_calls, 6); assert.equal(start.config.llm_max_calls_per_round, 2); assert.equal(start.config.llm_reasoning_effort, 'xhigh')
  assert.deepEqual(start.modelOverride, { provider: 'fake', model: 'test-model' })
  assert.ok(status.steps.some(step => step.stage === 'dependencies' && step.status === 'completed'))
  assert.ok(!JSON.stringify(status).includes(SECRET))
  assert.equal((await f.manager.start(input())).runId, status.runId); assert.equal(f.starts.length, 1)
  f.finishRun(); assert.equal((await f.manager.get(first.id))?.status, 'completed')
  await f.manager.dispose()
})

test('model and reasoning preflight failures do not spawn Python, install dependencies or silently change models', async () => {
  for (const options of [{ missingModel: true }, { effort: 'unsupported' }]) {
    const f = await fixture(options), first = await f.manager.start(input())
    await until(async () => (await f.manager.get(first.id))?.status === 'failed')
    const status = (await f.manager.get(first.id))!
    assert.equal(status.errorCode, options.missingModel ? 'MODEL_UNAVAILABLE' : 'EFFORT_UNSUPPORTED')
    assert.equal(f.probes.length, 0); assert.equal(f.bootstrap.length, 0); assert.equal(f.starts.length, 0)
    await f.manager.dispose()
  }
})

test('unsupported Python candidates produce a fixed actionable error without launching the bootstrap', async () => {
  const f = await fixture({ invalidPython: true }), first = await f.manager.start(input())
  await until(async () => (await f.manager.get(first.id))?.status === 'failed')
  assert.equal((await f.manager.get(first.id))?.errorCode, 'PYTHON_NOT_FOUND'); assert.equal(f.bootstrap.length, 0)
  await f.manager.dispose()
})

test('bootstrap failures preserve allowed error codes and discard private messages and stderr', async () => {
  for (const code of ['DEPENDENCY_INSTALL_FAILED', 'DATA_DOWNLOAD_FAILED', 'INVALID_EXISTING_DATA']) {
    const f = await fixture(), first = await f.manager.start(input())
    await until(() => f.bootstrap.length === 1)
    f.bootstrap[0]!.stderr.write(SECRET)
    f.bootstrap[0]!.output({ version: 1, type: 'result', stage: 'dependencies', status: 'failed', code, message: SECRET, counts: {} }); f.bootstrap[0]!.close(2)
    await until(async () => (await f.manager.get(first.id))?.status === 'failed')
    const status = (await f.manager.get(first.id))!
    assert.equal(status.errorCode, code); assert.ok(!JSON.stringify(status).includes(SECRET)); assert.equal(f.starts.length, 0)
    await f.manager.dispose()
  }
})

test('cancel writes a private-path-free public state and owned cancel flag without starting a run', async () => {
  const f = await fixture(), first = await f.manager.start(input())
  await until(() => f.bootstrap.length === 1)
  const status = await f.manager.cancel(first.id)
  assert.equal(status.status, 'cancelled'); assert.equal(f.bootstrap[0]!.kills, 0)
  assert.match(await readFile(join(f.root, 'results/quick-start', first.id, 'cancel.flag'), 'utf8'), /cancel/)
  f.bootstrap[0]!.output({ version: 1, type: 'result', stage: 'dependencies', status: 'cancelled', code: 'CANCELLED', message: SECRET, counts: {} }); f.bootstrap[0]!.close(130)
  await until(() => !f.manager.preparing)
  assert.equal(f.starts.length, 0); assert.equal((await f.manager.start(input())).status, 'cancelled')
  await f.manager.dispose()
})

test('cancelQuickStart only cancels the run associated with that task', async () => {
  const f = await fixture(), first = await f.manager.start(input())
  await until(() => f.bootstrap.length === 1); success(f.bootstrap[0]!)
  await until(async () => Boolean((await f.manager.get(first.id))?.runId))
  const running = (await f.manager.get(first.id))!
  assert.equal((await f.manager.cancel(first.id)).status, 'cancelled')
  assert.deepEqual(f.cancelled, [running.runId]); await f.manager.dispose()
})

test('bootstrap argv and clean environment never contain model credentials or endpoint settings', async () => {
  const previous = process.env.OPENAI_API_KEY; process.env.OPENAI_API_KEY = SECRET
  const f = await fixture(), first = await f.manager.start(input())
  try {
    await until(() => f.bootstrap.length === 1)
    for (const [, argv, settings] of f.spawnArguments) {
      const env = (settings as { env: Record<string, string> }).env
      assert.equal(env.PYTHON_DOTENV_DISABLED, '1'); assert.equal('OPENAI_API_KEY' in env, false)
      assert.equal((settings as { shell: boolean }).shell, false)
      assert.ok(!JSON.stringify(argv).includes('test-model')); assert.ok(!JSON.stringify(settings).includes(SECRET))
    }
    await f.manager.cancel(first.id); f.bootstrap[0]!.close(130)
    await f.manager.dispose()
  } finally { if (previous === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = previous }
})

test('Host restart changes unfinished preparation to interrupted and never starts Python or a run on query or duplicate operation', async () => {
  const f = await fixture(), first = await f.manager.start(input())
  await until(() => f.bootstrap.length === 1)
  const restored = new QuickStartManager(f.root, f.gateway, f.hooks, { spawnBootstrap: () => { throw new Error('Read-only recovery must not spawn') } })
  await restored.initialize()
  assert.equal((await restored.get())?.status, 'interrupted')
  assert.equal((await restored.start(input())).status, 'interrupted'); assert.equal(f.starts.length, 0)
  await restored.dispose(); await f.manager.cancel(first.id); f.bootstrap[0]!.close(130); await f.manager.dispose()
})

test('active scientific runs prevent a new quick preparation before any installation', async () => {
  const f = await fixture(); f.setActive('gb1-other-running')
  await assert.rejects(f.manager.start(input()), /已有实验/)
  assert.equal(f.probes.length, 0); assert.equal(f.starts.length, 0); await f.manager.dispose()
})

test('missing environment and S1 do not block public metadata or quick preparation initialization', async () => {
  const f = await fixture(), workers: Child[] = []
  const backend = new WorkbenchBackend({ llm: f.llm, credentials: f.credentials, spawnBootstrap: f.spawnBootstrap, spawnWorker: () => {
    const child = new Child(); workers.push(child); return child as unknown as ChildProcessWithoutNullStreams
  } }, { projectRoot: f.root, bootstrapPython: 'C:\\Python312\\python.exe' })
  assert.equal((await backend.getInfo()).ready, false)
  const before = await backend.getInfo()
  assert.equal(before.reference.length, 56); assert.equal(before.recommendedModel!.model, 'test-model'); assert.equal(before.defaultConfig.llm_reasoning_effort, 'xhigh')
  assert.deepEqual((await backend.getModelCapabilities('fake', 'test-model')).reasoningEfforts, ['xhigh'])
  const first = await backend.startQuickRun(input())
  await until(() => f.bootstrap.length === 1)
  for (const path of ['.venv/Scripts/python.exe', 'data/raw/elife-16965-supp1-v4.xlsx']) { await mkdir(dirname(join(f.root, path)), { recursive: true }); await writeFile(join(f.root, path), '') }
  success(f.bootstrap[0]!)
  await until(async () => Boolean((await backend.getQuickStartStatus(first.id))?.runId))
  assert.equal(workers.length, 1)
  const frame = workers[0]!.frames[0] as { config: typeof DEFAULT_CONFIG; model: unknown }
  assert.deepEqual(frame.model, { source: 'dsh', provider: 'fake', id: 'test-model' }); assert.equal(frame.config.rounds, 3); assert.equal(frame.config.max_model_calls, 6)
  assert.equal((await backend.getInfo()).ready, true); assert.deepEqual((await backend.getInfo()).connection, before.connection)
  await backend.cancelQuickStart(first.id); await backend.dispose(); await f.manager.dispose()
})

test('unconfirmed run cancellation returns a safe state and can be checked and cancelled again without rerunning', async () => {
  const f = await fixture(), first = await f.manager.start(input())
  await until(() => f.bootstrap.length === 1); success(f.bootstrap[0]!)
  await until(() => !f.manager.preparing)
  const cancel = f.hooks.cancelRun
  f.hooks.cancelRun = async () => { throw new Error(SECRET) }
  const uncertain = await f.manager.cancel(first.id)
  assert.equal(uncertain.status, 'interrupted'); assert.equal(uncertain.errorCode, 'CANCEL_UNCONFIRMED')
  assert.ok(!JSON.stringify(uncertain).includes(SECRET)); assert.equal(f.starts.length, 1)
  assert.equal((await f.manager.get(first.id))?.status, 'running')
  f.hooks.cancelRun = cancel
  assert.equal((await f.manager.cancel(first.id)).status, 'cancelled'); assert.equal(f.starts.length, 1)
  await f.manager.dispose()
})

test('an unknown scientific-run start is persisted and duplicate quick operations do not resend', async () => {
  const f = await fixture(); let attempts = 0
  f.hooks.startRun = async () => { attempts++; throw new Error(SECRET) }
  const first = await f.manager.start(input())
  await until(() => f.bootstrap.length === 1); success(f.bootstrap[0]!)
  await until(() => !f.manager.preparing)
  const uncertain = (await f.manager.get(first.id))!
  assert.equal(uncertain.status, 'interrupted'); assert.equal(uncertain.errorCode, 'RUN_START_FAILED'); assert.equal(attempts, 1)
  assert.ok(!JSON.stringify(uncertain).includes(SECRET))
  assert.equal((await f.manager.start(input())).id, first.id); assert.equal(attempts, 1); assert.equal(f.bootstrap.length, 1)
  await f.manager.dispose()
})

function gate() { let release = () => {}; const promise = new Promise<void>(resolveGate => { release = resolveGate }); return { promise, release } }
async function readyBackend() {
  const f = await fixture(), workers: Child[] = []
  for (const path of ['.venv/Scripts/python.exe', 'data/raw/elife-16965-supp1-v4.xlsx']) { await mkdir(dirname(join(f.root, path)), { recursive: true }); await writeFile(join(f.root, path), '') }
  const backend = new WorkbenchBackend({ llm: f.llm, credentials: f.credentials, spawnBootstrap: f.spawnBootstrap, spawnWorker: () => {
    const child = new Child(); workers.push(child); return child as unknown as ChildProcessWithoutNullStreams
  } }, { projectRoot: f.root })
  const info = await backend.getInfo(); assert.equal(info.ready, true)
  const manual = { operationId: 'manual-admission-example-0001', reference: info.reference, seed: 11, rounds: 1, method: 'knowledge' as const,
    config: { ...DEFAULT_CONFIG, rounds: 1 }, modelOverride: { provider: 'fake', model: 'test-model' } }
  const batch = { operationId: 'batch-admission-example-0001', baseConfig: { ...DEFAULT_CONFIG, rounds: 1 }, promptProfile: 'v3', methods: ['knowledge' as const],
    targets: [{ provider: 'fake', model: 'test-model', reasoningEffort: null }], seeds: [11], repeats: 1, concurrency: 1, perProviderConcurrency: 1 }
  return { ...f, backend, workers, manual, batch }
}
async function cancelPrepared(f: Awaited<ReturnType<typeof readyBackend>>, id: string) {
  await f.backend.cancelQuickStart(id)
  for (const child of f.bootstrap) if (!child.closed) child.close(130)
  await f.backend.dispose(); await f.manager.dispose()
}

test('preparation is reserved before its initial disk await and a failed initial write releases the reservation', async () => {
  const f = await fixture(), storage = f.manager as unknown as { persist: (...args: unknown[]) => Promise<void> }
  const original = storage.persist.bind(f.manager), disk = gate(); let entered = false
  storage.persist = async (...args) => { if (!entered) { entered = true; await disk.promise }; await original(...args) }
  const pending = f.manager.start(input())
  await until(() => entered)
  assert.equal(f.manager.preparing, true); assert.equal(f.probes.length, 0)
  disk.release(); const first = await pending
  await f.manager.cancel(first.id)
  for (const child of f.bootstrap) if (!child.closed) child.close(130)
  await f.manager.dispose()
  const failed = await fixture(), failingStorage = failed.manager as unknown as { persist: () => Promise<void> }
  failingStorage.persist = async () => { throw new Error('Synthetic disk write failure') }
  await assert.rejects(failed.manager.start(input()), /尚未开始安装或实验/)
  assert.equal(failed.manager.preparing, false); assert.equal(failed.probes.length, 0); assert.equal(failed.starts.length, 0)
  await failed.manager.dispose()
})

test('manual creation and both resume routes cannot cross a quick preparation disk reservation', async () => {
  const f = await readyBackend(), quick = (f.backend as unknown as { quick: QuickStartManager }).quick
  const storage = quick as unknown as { persist: (...args: unknown[]) => Promise<void> }, original = storage.persist.bind(quick), disk = gate(); let entered = false
  storage.persist = async (...args) => { if (!entered) { entered = true; await disk.promise }; await original(...args) }
  const pending = f.backend.startQuickRun(input())
  await until(() => entered)
  const blocked = [
    assert.rejects(f.backend.startRun(f.manual), /准备进行中/),
    assert.rejects(f.backend.resumeRun('gb1-00000000-0000-0000-0000-000000000001'), /准备进行中/),
    assert.rejects(f.backend.startBatch(f.batch), /准备进行中/),
    assert.rejects(f.backend.resumeBatch('batch-00000000-0000-0000-0000-000000000001'), /准备进行中/),
  ]
  assert.equal(f.workers.length, 0); disk.release(); const first = await pending
  await Promise.all(blocked); assert.equal(f.workers.length, 0)
  await cancelPrepared(f, first.id)
})

test('an in-flight manual start is admitted before quick preparation and prevents installation after its metadata await', async () => {
  const f = await readyBackend(), metadata = gate(); let entered = false, calls = 0
  f.llm.listModels = async () => { if (++calls === 1) { entered = true; await metadata.promise }; return [{ provider: 'fake', id: 'test-model', name: 'Test' }] }
  const manual = f.backend.startRun(f.manual)
  await until(() => entered)
  const rejected = assert.rejects(f.backend.startQuickRun(input()), /已有实验/)
  assert.equal(f.probes.length, 0); assert.equal(f.bootstrap.length, 0)
  metadata.release(); await manual; await rejected
  assert.equal(f.workers.length, 1); assert.equal(f.bootstrap.length, 0)
  await f.backend.dispose(); await f.manager.dispose()
})

test('a queued batch awaiting dispatch blocks quick preparation even before its first active run exists', async () => {
  const f = await readyBackend(), metadata = gate(); let entered = false, calls = 0
  f.llm.listModels = async () => { if (++calls === 2) { entered = true; await metadata.promise }; return [{ provider: 'fake', id: 'test-model', name: 'Test' }] }
  const batch = await f.backend.startBatch(f.batch)
  await until(() => entered)
  assert.equal((await f.backend.getInfo()).activeRunId, null); assert.equal(f.workers.length, 0)
  await assert.rejects(f.backend.startQuickRun(input()), /已有实验/)
  assert.equal(f.probes.length, 0); assert.equal(f.bootstrap.length, 0)
  metadata.release(); await until(() => f.workers.length === 1)
  await f.backend.cancelBatch(batch.summary.id); await f.backend.dispose(); await f.manager.dispose()
})
