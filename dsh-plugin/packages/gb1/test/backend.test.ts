import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PassThrough } from 'node:stream'
import { setTimeout as pause } from 'node:timers/promises'
import test from 'node:test'
import type { ChildProcessWithoutNullStreams } from 'node:child_process'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import { Context } from '@deepseek-ai/cordis'
import { remoteMethods } from '@deepseek-ai/dsh-typert-protocol'
import Gb1Workbench, { WorkbenchBackend } from '../src/server.ts'
import { ConnectionStore, MAX_FRAME_BYTES, ModelGateway, normalizeDshUsage, type CredentialsPort, type LlmPort, type WorkerRequest } from '../src/model.ts'
import { REFERENCE, RunManager, validateStart, type SpawnWorker } from '../src/runs.ts'
import type { ConversationSnapshot, StartRunInput, StartBatchInput } from '../src/contracts.ts'
import { DEFAULT_CONFIG, historicalConfig, validateConfig } from '../src/settings.ts'

const SECRET = 'SYNTHETIC_GB1_TEST_KEY'
const prompt = { profile: 'v3', version: 'gb1-proposals-v3', system_prompt: 'Return one JSON object.' }
const request: WorkerRequest = { id: 'request-1', round: 1,
  payload: { model: 'test-model', messages: [{ role: 'system', content: prompt.system_prompt }, { role: 'user', content: '{"response_schema":{}}' }], max_completion_tokens: 8000,
    response_format: { type: 'json_schema', json_schema: { name: 'gb1_proposals', strict: true, schema: {} } } }, prompt_metadata: prompt }
const start = (operationId = 'operation-example-0001'): StartRunInput => ({ operationId, reference: REFERENCE, seed: 11, rounds: 1, method: 'knowledge', config: historicalConfig(11, 1) })
function credentials() {
  const values = new Map<string, string>(), calls = { resolve: 0, set: 0, describe: 0, unset: 0 }
  const port = {
    async resolve(ref: string) { calls.resolve++; const value = values.get(ref); return value ? { value, source: 'fake' } : undefined },
    async describe(ref: string) { calls.describe++; return { configured: values.has(ref), writable: true } },
    async set(ref: string, value: string) { calls.set++; values.set(ref, value) },
    async unset(ref: string) { calls.unset++; values.delete(ref) },
  } as CredentialsPort
  return { port, values, calls }
}
function llm(chunks: StreamChunk[] = [
  { type: 'block-start', index: 0, blockType: 'text' },
  { type: 'text-delta', index: 0, text: '{"ok":true}' },
  { type: 'block-end', index: 0, block: { type: 'text', text: '{"ok":true}' } },
  { type: 'usage', usage: { inputTokens: 10, outputTokens: 8, cacheReadTokens: 4, cacheWriteTokens: 2, reasoningTokens: 3, totalTokens: 24 } },
  { type: 'finish', reason: { kind: 'stop' } },
]) {
  const calls: unknown[] = [], preparations: unknown[] = []
  const port = {
    listProviders: () => [{ id: 'fake', name: 'Fake provider' }],
    async listModels() { return [{ provider: 'fake', id: 'test-model', name: 'Test model' }] },
    async prepareCall(config: unknown, signal: AbortSignal) {
      preparations.push(config)
      return { config, async *stream(options: unknown) { calls.push(options); signal.throwIfAborted(); yield* chunks } }
    },
  } as unknown as LlmPort
  return { port, calls, preparations }
}
class FakeWorker extends EventEmitter {
  stdin = new PassThrough(); stdout = new PassThrough(); stderr = new PassThrough()
  frames: Record<string, unknown>[] = []; kills = 0; closed = false
  constructor() {
    super()
    this.stdin.on('data', chunk => { for (const line of String(chunk).trim().split('\n')) this.frames.push(JSON.parse(line)) })
  }
  output(value: unknown) { this.stdout.write(JSON.stringify(value) + '\n') }
  kill() { this.kills++; this.close(null); return true }
  close(code: number | null = 0) { if (!this.closed) { this.closed = true; queueMicrotask(() => this.emit('close', code)) } }
}
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'gb1-host-test-'))
  for (const path of ['gb1/data.py', 'configs/knowledge.json', 'data/raw/elife-16965-supp1-v4.xlsx', '.venv/Scripts/python.exe']) {
    const file = join(root, path); await mkdir(dirname(file), { recursive: true }); await writeFile(file, '')
  }
  const creds = credentials(), model = llm(), children: FakeWorker[] = [], spawns: unknown[][] = []
  const spawnWorker: SpawnWorker = (executable, argv, options) => {
    spawns.push([executable, argv, options]); const child = new FakeWorker(); children.push(child)
    return child as unknown as ChildProcessWithoutNullStreams
  }
  const backend = new WorkbenchBackend({ llm: model.port, credentials: creds.port, spawnWorker }, { projectRoot: root })
  assert.equal((await backend.getInfo()).ready, true)
  await backend.saveConnection({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '' })
  return { root, creds, model, children, spawns, backend }
}
async function until(check: () => boolean | Promise<boolean>) {
  const deadline = Date.now() + 5000
  while (Date.now() < deadline) { if (await check()) return; await pause(2) }
  assert.fail('The fake operation did not settle')
}
function resultFrame(id: string, calls = 1) {
  const usage = { input_tokens: 16, output_tokens: 8, total_tokens: 24, missing_usage_count: 0 }
  const top10 = [...'ACDEFGHIKL'].map((amino, index) => {
    const variant = amino + 'DGV', sequence = REFERENCE.split(''); sequence[38] = amino
    return { rank: index + 1, variant, sequence: sequence.join(''), mutations: `V39${amino}`, mutation_count: 1,
      predicted_fitness: 1, actual_fitness: 0.5, hypothesis: 'Synthetic hypothesis', rationale: 'Synthetic rationale', knowledge_rule_ids: [] }
  })
  return { version: 1, type: 'result', status: 'completed', summary: { run_id: id, method: 'knowledge', seed: 11,
    requested_rounds: 1, completed_rounds: 1, model: { source: 'dsh', provider: 'fake', id: 'test-model' },
    paths: { hidden: '/private/hidden/oracle.csv' }, initial_known_count: 2000, known_count: 2010, calls, usage,
    rounds: [{ round: 1, status: 'evaluated', analysis: ['Synthetic summary'], known_count_before: 2000, known_count_after: 2010,
      valid_count: 40, scored_count: 40, calls, usage,
      metrics: { mean_fitness: 0.5, max_fitness: 0.5, median_fitness: 0.5, cumulative_best: 1 }, top10 }] } }
}

test('public DTO and credential references never include the key', async () => {
  const f = await fixture()
  const saved = await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://example.invalid/v1/', apiKey: SECRET })
  assert.equal(saved.hasKey, true); assert.equal(saved.baseUrl, 'https://example.invalid/v1')
  assert.equal(f.creds.calls.resolve, 0)
  const json = await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8')
  assert.ok(!json.includes(SECRET)); assert.match(json, /DSH_GB1_[A-F0-9]{32}/)
  assert.ok(!JSON.stringify(await f.backend.getInfo()).includes(SECRET))
  const oldRef = [...f.creds.values.keys()][0]!
  await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://other.invalid/v1', apiKey: 'NEW_SYNTHETIC_KEY' })
  assert.equal(f.creds.values.has(oldRef), false); assert.equal(f.creds.values.size, 1)
  await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://other.invalid/v1', clearKey: true })
  assert.equal(f.creds.values.size, 0)
})

test('custom key binding survives DSH switches and address changes require a new key', async () => {
  const f = await fixture()
  const custom = { source: 'custom' as const, provider: '', model: 'test-model', baseUrl: 'https://EXAMPLE.invalid:443/v1/' }
  const saved = await f.backend.saveConnection({ ...custom, apiKey: SECRET })
  assert.equal('credentialBaseUrl' in saved, false)
  let internal = JSON.parse(await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8'))
  const originalRef = internal.credentialRef
  assert.equal(internal.credentialBaseUrl, 'https://example.invalid/v1')
  await f.backend.saveConnection({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '' })
  internal = JSON.parse(await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8'))
  assert.equal(internal.baseUrl, ''); assert.equal(internal.credentialRef, originalRef)
  assert.equal(internal.credentialBaseUrl, 'https://example.invalid/v1')
  await assert.rejects(f.backend.saveConnection({ ...custom, baseUrl: 'https://other.invalid/v1' }), /更换 Base URL.*API Key/)
  assert.equal(f.creds.calls.set, 1); assert.equal(f.creds.calls.resolve, 0)
  await f.backend.saveConnection(custom)
  internal = JSON.parse(await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8'))
  assert.equal(internal.credentialRef, originalRef)
  await f.backend.saveConnection({ ...custom, baseUrl: 'https://other.invalid/v1', apiKey: 'NEW_SYNTHETIC_KEY' })
  internal = JSON.parse(await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8'))
  assert.equal(internal.credentialBaseUrl, 'https://other.invalid/v1')
  assert.notEqual(internal.credentialRef, originalRef); assert.equal(f.creds.values.has(originalRef), false)
})

test('custom destination binding is checked before resolving or sending the credential', async () => {
  const creds = credentials(), ref = 'DSH_GB1_' + 'A'.repeat(32); creds.values.set(ref, SECRET)
  let calls = 0
  const fetcher = (async () => { calls++; throw new Error('A mismatched binding must never fetch') }) as typeof fetch
  const gateway = new ModelGateway(llm().port, creds.port, fetcher)
  const connection = { source: 'custom' as const, provider: 'custom', model: 'test-model', baseUrl: 'https://other.invalid/v1',
    credentialRef: ref, credentialBaseUrl: 'https://example.invalid/v1' }
  await assert.rejects(gateway.generate(connection, request, new AbortController().signal), /Base URL 不匹配/)
  await assert.rejects(gateway.generate({ ...connection, credentialBaseUrl: '' }, request, new AbortController().signal), /Base URL 不匹配/)
  await assert.rejects(gateway.generate({ ...connection, credentialRef: '' }, request, new AbortController().signal), /保存 API Key/)
  assert.equal(creds.calls.resolve, 0); assert.equal(calls, 0)
})

test('initial key setup, explicit clearing and registered IDs with internal spaces are supported', async () => {
  const f = await fixture()
  const custom = { source: 'custom' as const, provider: '', model: '  custom model  ', baseUrl: 'https://example.invalid/v1/' }
  await assert.rejects(f.backend.saveConnection(custom), /填写 API Key/)
  assert.equal(f.creds.calls.set, 0)
  f.model.port.listProviders = () => [{ id: 'fake provider', name: 'Fake provider' }]
  f.model.port.listModels = async provider => {
    assert.equal(provider, 'fake provider')
    return [{ provider, id: 'Test Model 1', name: 'Test Model 1' }]
  }
  const dsh = { source: 'dsh' as const, provider: '  fake provider  ', model: ' Test Model 1 ', baseUrl: '' }
  const normalized = await f.backend.saveConnection(dsh)
  assert.equal(normalized.provider, 'fake provider'); assert.equal(normalized.model, 'Test Model 1')
  const configured = await f.backend.saveConnection({ ...custom, apiKey: SECRET })
  assert.equal(configured.model, 'custom model'); assert.equal(configured.hasKey, true)
  const cleared = await f.backend.saveConnection({ ...dsh, clearKey: true })
  assert.equal(cleared.hasKey, false); assert.equal(f.creds.values.size, 0)
  const internal = JSON.parse(await readFile(join(f.root, '.dsh-gb1/settings.json'), 'utf8'))
  assert.equal(internal.credentialRef, ''); assert.equal(internal.credentialBaseUrl, '')
  await assert.rejects(f.backend.saveConnection(custom), /填写 API Key/)
})

test('invalid URL, model and key reject before storing credentials', async () => {
  const f = await fixture()
  for (const baseUrl of ['file:///etc/passwd', 'https://user:pass@example.invalid', 'https://example.invalid?key=x']) {
    await assert.rejects(f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl, apiKey: SECRET }))
  }
  await assert.rejects(f.backend.saveConnection({ source: 'custom', provider: '', model: 'bad\nmodel', baseUrl: 'https://example.invalid', apiKey: SECRET }))
  assert.equal(f.creds.calls.set, 0)
})

test('provider failures retain healthy models and do not prevent a custom connection', async () => {
  const f = await fixture()
  f.model.port.listProviders = () => [{ id: 'broken', name: 'Broken' }, { id: 'fake', name: 'Healthy' }]
  f.model.port.listModels = async provider => {
    if (provider === 'broken') throw new Error(SECRET)
    return [{ provider: 'fake', id: 'test-model', name: 'Test model' }]
  }
  assert.deepEqual(await f.backend.listModels(), [{ provider: 'fake', id: 'test-model', name: 'Test model' }])
  f.model.port.listProviders = () => [{ id: 'broken', name: 'Broken' }]
  await assert.rejects(f.backend.listModels(), { message: '无法读取 DSH 模型目录' })
  const saved = await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://example.invalid/v1', apiKey: SECRET })
  assert.equal(saved.source, 'custom'); assert.equal(saved.hasKey, true)
  assert.ok(!JSON.stringify(saved).includes(SECRET))
})

test('missing project returns a clear not-ready DTO without resolving credentials', async () => {
  const creds = credentials(), model = llm()
  const backend = new WorkbenchBackend({ llm: model.port, credentials: creds.port }, { projectRoot: join(tmpdir(), 'gb1-project-does-not-exist') })
  const info = await backend.getInfo()
  assert.equal(info.ready, false); assert.ok(info.reason.length > 10); assert.equal(creds.calls.resolve, 0)
})

test('DSH visible text is assembled once and cache inputs are disjoint', async () => {
  const fake = llm([
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'not visible' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'not visible' } },
    { type: 'block-start', index: 1, blockType: 'text' }, { type: 'text-delta', index: 1, text: '{"ok":true}' },
    { type: 'block-end', index: 1, block: { type: 'text', text: '{"ok":true}' } },
    { type: 'usage', usage: { inputTokens: 10, outputTokens: 8, cacheReadTokens: 4, cacheWriteTokens: 2, reasoningTokens: 3, totalTokens: 99 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
  const gateway = new ModelGateway(fake.port, credentials().port)
  const response = await gateway.generate({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '', credentialRef: '', credentialBaseUrl: '' }, request, new AbortController().signal)
  assert.equal(response.text, '{"ok":true}'); assert.equal(response.complete, true)
  assert.equal(response.model_requested, 'test-model'); assert.equal(response.model_returned, null)
  assert.equal(response.usage!.input_tokens, 16); assert.equal(response.usage!.output_tokens, 8); assert.equal(response.usage!.total_tokens, 99)
  assert.equal((response.usage!.raw_dsh_usage as Record<string, unknown>).reasoningTokens, 3)
  const sent = fake.calls[0] as Record<string, unknown>
  assert.equal(sent.purpose, 'gb1-workbench'); assert.equal('response_format' in sent, false)
  assert.equal(fake.calls.length, 1); assert.equal(fake.preparations.length, 1)
})

test('missing accounting is marked and reasoning is not added to output', () => {
  assert.equal(normalizeDshUsage({ inputTokens: 10, outputTokens: 8, reasoningTokens: 4 })!.total_tokens, 18)
  const partial = normalizeDshUsage({ outputTokens: 8, totalTokens: 100 })!
  assert.equal(partial.input_tokens, null); assert.equal(partial.total_tokens, 100); assert.equal(partial.missing_usage, true)
  assert.equal(normalizeDshUsage(null), null)
})

test('non-stop DSH completion is explicitly incomplete and never retried', async () => {
  const fake = llm([{ type: 'finish', reason: { kind: 'max-tokens' } }])
  const response = await new ModelGateway(fake.port, credentials().port).generate({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '', credentialRef: '', credentialBaseUrl: '' }, request, new AbortController().signal)
  assert.equal(response.complete, false); assert.equal(fake.calls.length, 1)
})

test('custom fetch sends exactly one shared payload and never returns the key', async () => {
  const creds = credentials(), ref = 'DSH_GB1_' + 'A'.repeat(32); creds.values.set(ref, SECRET)
  const sent: unknown[][] = []
  const fetcher = (async (url: unknown, init: RequestInit) => {
    sent.push([url, init])
    return new Response(JSON.stringify({ model: 'test-model', id: 'response-1', choices: [{ finish_reason: 'stop', message: { content: '{"ok":true}' } }],
      usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 } }), { headers: { 'x-request-id': 'request-id' } })
  }) as typeof fetch
  const response = await new ModelGateway(llm().port, creds.port, fetcher).generate({ source: 'custom', provider: 'custom', model: 'test-model', baseUrl: 'https://example.invalid/v1', credentialRef: ref, credentialBaseUrl: 'https://example.invalid/v1' }, request, new AbortController().signal)
  assert.equal(sent.length, 1); assert.equal(sent[0]![0], 'https://example.invalid/v1/chat/completions')
  const init = sent[0]![1] as RequestInit
  assert.deepEqual(JSON.parse(String(init.body)), { ...request.payload, stream: true, stream_options: { include_usage: true } }); assert.equal(init.redirect, 'error')
  assert.equal((init.headers as Record<string, string>).Authorization, `Bearer ${SECRET}`)
  assert.equal(typeof (init.headers as Record<string, string>)['user-agent'], 'string')
  assert.equal('endpoint' in response.transport_request, false)
  assert.ok(!JSON.stringify(response).includes('Authorization')); assert.ok(!JSON.stringify(response).includes('user-agent'))
  assert.ok(!JSON.stringify(response).includes(SECRET)); assert.equal(response.api_mode, 'chat')
})

test('custom error, oversized response and credential echo never retry or leak', async () => {
  for (const variant of ['http', 'oversize', 'echo']) {
    const creds = credentials(), ref = 'DSH_GB1_' + 'A'.repeat(32); creds.values.set(ref, SECRET)
    let calls = 0
    const fetcher = (async () => {
      calls++
      if (variant === 'http') return new Response(SECRET, { status: 500 })
      if (variant === 'oversize') return new Response('X'.repeat(MAX_FRAME_BYTES))
      return new Response(JSON.stringify({ model: 'test-model', choices: [{ finish_reason: 'stop', message: { content: variant === 'echo' ? SECRET : '{}' } }] }))
    }) as typeof fetch
    const gateway = new ModelGateway(llm().port, creds.port, fetcher)
    await assert.rejects(gateway.generate({ source: 'custom', provider: 'custom', model: 'test-model', baseUrl: 'https://example.invalid/v1', credentialRef: ref, credentialBaseUrl: 'https://example.invalid/v1' }, request, new AbortController().signal), error => !String(error).includes(SECRET))
    assert.equal(calls, 1)
  }
})

test('custom returned aliases and missing model metadata are preserved without changing the request', async () => {
  for (const returned of ['resolved-model-version', null]) {
    const creds = credentials(), ref = 'DSH_GB1_' + 'A'.repeat(32); creds.values.set(ref, SECRET)
    const payloads: unknown[] = []
    const fetcher = (async (_url: unknown, init: RequestInit) => {
      payloads.push(JSON.parse(String(init.body)))
      return new Response(JSON.stringify({ model: returned, choices: [{ finish_reason: 'stop', message: { content: '{}' } }] }))
    }) as typeof fetch
    const response = await new ModelGateway(llm().port, creds.port, fetcher).generate(
      { source: 'custom', provider: 'custom', model: 'test-model', baseUrl: 'https://example.invalid/v1', credentialRef: ref, credentialBaseUrl: 'https://example.invalid/v1' },
      request, new AbortController().signal)
    assert.equal(response.model_requested, 'test-model'); assert.equal(response.model_returned, returned)
    assert.equal(payloads.length, 1); assert.equal((payloads[0] as Record<string, unknown>).model, 'test-model')
  }
})

test('custom request timeout is bounded and does not repeat the fetch', async () => {
  const creds = credentials(), ref = 'DSH_GB1_' + 'A'.repeat(32); creds.values.set(ref, SECRET)
  let calls = 0
  const fetcher = (async (_url: unknown, init: RequestInit) => {
    calls++
    return new Promise<Response>((_resolve, reject) => init.signal!.addEventListener('abort', () => reject(new Error(SECRET)), { once: true }))
  }) as typeof fetch
  const keepAlive = setTimeout(() => {}, 1000)
  try {
    await assert.rejects(new ModelGateway(llm().port, creds.port, fetcher, 10).generate(
      { source: 'custom', provider: 'custom', model: 'test-model', baseUrl: 'https://example.invalid/v1', credentialRef: ref, credentialBaseUrl: 'https://example.invalid/v1' },
      request, new AbortController().signal), /超时/)
  } finally { clearTimeout(keepAlive) }
  assert.equal(calls, 1)
})

test('credential provider error bodies stay out of public failures', async () => {
  const f = await fixture()
  await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://example.invalid/v1', apiKey: SECRET })
  f.creds.port.describe = async () => { throw new Error(SECRET) }
  const info = await f.backend.getInfo()
  assert.equal(info.ready, false); assert.ok(!JSON.stringify(info).includes(SECRET))
})

test('operationId is durable and concurrent repeated starts spawn one fixed worker', async () => {
  const f = await fixture()
  const [a, b] = await Promise.all([f.backend.startRun(start()), f.backend.startRun(start())])
  assert.equal(a.id, b.id); assert.equal(f.children.length, 1)
  const [executable, argv, options] = f.spawns[0] as [string, string[], Record<string, unknown>]
  assert.equal(executable, join(f.root, '.venv', 'Scripts', 'python.exe'))
  assert.equal(argv[1], resolve(dirname(fileURLToPath(import.meta.url)), '../python/worker.py'))
  assert.notEqual(argv[1], join(f.root, 'dsh-plugin', 'packages', 'gb1', 'python', 'worker.py'))
  assert.deepEqual(argv.slice(-4), ['--project-root', f.root, '--run-id', a.id]); assert.equal(options.shell, false)
  assert.ok(!JSON.stringify(f.children[0]!.frames[0]).includes('credential'))
  assert.deepEqual(Object.keys(options.env as object).sort().filter(key => /KEY|TOKEN|SECRET/.test(key)), [])
  await assert.rejects(f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://example.invalid', apiKey: SECRET }), /运行期间/)
  const parallel = await f.backend.startRun(start('operation-example-0002'))
  assert.notEqual(parallel.id, a.id)
  await f.backend.cancelRun(parallel.id)
  await until(() => f.children[1]!.closed)
  await assert.rejects(f.backend.getRun('../hidden/oracle.csv'))
  await f.backend.cancelRun(a.id)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const resumed = await f.backend.startRun(start())
  assert.equal(resumed.id, a.id); assert.equal(resumed.status, 'cancelled'); assert.equal(f.children.length, 2)
  await f.backend.dispose()
  const reopened = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port,
    spawnWorker: () => { throw new Error('A persisted operation must not start again') } }, { projectRoot: f.root })
  assert.equal((await reopened.startRun(start())).id, a.id)
})

test('underfilled genuine proposals retain completed Top10 with the actual count', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start())
  const worker = f.children[0]!
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => worker.frames.length === 2)
  const result = resultFrame(summary.id)
  result.summary.rounds[0]!.valid_count = 24
  result.summary.rounds[0]!.scored_count = 24
  worker.output(result); worker.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const detail = await f.backend.getRun(summary.id)
  assert.equal(detail.summary.status, 'completed'); assert.equal(detail.roundResults[0]!.validCount, 24)
  assert.equal(f.model.calls.length, 1)
})

test('per-round model request budget is enforced without a third dispatch', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start()), worker = f.children[0]!
  for (let index = 1; index <= 2; index++) {
    worker.output({ version: 1, type: 'llm_request', ...request, id: `request-${index}` })
    await until(() => worker.frames.length === index + 1)
  }
  worker.output({ version: 1, type: 'llm_request', ...request, id: 'request-3' })
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal(f.model.calls.length, 2); assert.equal((await f.backend.getRun(summary.id)).summary.status, 'failed')
})

test('cancellation aborts a model stream already awaiting its owned signal', async () => {
  const f = await fixture()
  let entered = false, aborted = false
  f.model.port.prepareCall = (async (config: Parameters<LlmPort['prepareCall']>[0], signal?: AbortSignal) => ({ config, async *stream() {
    entered = true
    await new Promise<void>((resolveWait, rejectWait) => signal!.addEventListener('abort', () => {
      aborted = true; rejectWait(new Error('synthetic cancellation'))
    }, { once: true }))
  } })) as unknown as LlmPort['prepareCall']
  const summary = await f.backend.startRun(start()), worker = f.children[0]!
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => entered)
  await f.backend.cancelRun(summary.id)
  await until(() => aborted)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal(worker.kills, 1); assert.equal(worker.frames.length, 1)
})

test('successful worker result exposes Top10 only and drops arbitrary paths', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start())
  const worker = f.children[0]!
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => worker.frames.length === 2)
  worker.output(resultFrame(summary.id)); worker.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const result = await f.backend.getRun(summary.id)
  assert.equal(result.summary.status, 'completed'); assert.equal(result.roundResults[0]!.top10.length, 10)
  assert.ok(!JSON.stringify(result).includes('/private/hidden'))
  const path = join(f.root, '.dsh-gb1/jobs', `${summary.id}.json`)
  const stored = JSON.parse(await readFile(path, 'utf8')); stored.hidden = 'must not return'; stored.summary.extra = 'must not return'
  await writeFile(path, JSON.stringify(stored))
  assert.ok(!JSON.stringify(await f.backend.getRun(summary.id)).includes('must not return'))
})

test('oversized worker frame terminates only the owned child', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start())
  f.children[0]!.stdout.write(Buffer.alloc(MAX_FRAME_BYTES + 1, 65))
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal(f.children[0]!.kills, 1); assert.equal((await f.backend.getRun(summary.id)).summary.status, 'failed')
})

test('dispose aborts owned model requests and cancels the owned worker', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start())
  await f.backend.dispose()
  assert.equal(f.children[0]!.kills, 1)
  assert.equal((await f.backend.getRun(summary.id)).summary.status, 'cancelled')
})

test('unknown prior active lease is not removed and does not start another worker', async () => {
  const f = await fixture()
  await writeFile(join(f.root, '.dsh-gb1/active.json'), JSON.stringify({ id: 'gb1-orphan' }))
  const manager = new RunManager(f.root, new ConnectionStore(f.root, f.creds.port), new ModelGateway(f.model.port, f.creds.port), () => { throw new Error('must not spawn') })
  await manager.initialize(); assert.equal(manager.activeRunId, 'gb1-orphan')
  await assert.rejects(manager.startRun(start()), /已有活动运行/)
  await manager.dispose()
  assert.equal(JSON.parse(await readFile(join(f.root, '.dsh-gb1/active.json'), 'utf8')).id, 'gb1-orphan')
})

test('run design accepts arbitrary positive rounds and seeds and rejects inconsistent configuration', () => {
  assert.throws(() => validateStart({ ...start(), reference: 'MT' + REFERENCE.slice(2) }))
  assert.equal(validateStart({ ...start(), seed: 765, rounds: 4, config: { ...historicalConfig(765, 4), k: 13, proposal_count: 17, score_budget: 17 } }).rounds, 4)
  assert.throws(() => validateStart({ ...start(), rounds: 4 }))
  assert.throws(() => validateStart({ ...start(), seed: -1 }))
  assert.throws(() => validateStart({ ...start(), operationId: '../escape' }))
})

test('conversation subscription disconnect never cancels, restarts or duplicates the run', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start())
  const controller = new AbortController(), stream = f.backend.watchConversation(summary.id, controller.signal)[Symbol.asyncIterator]()
  const first = (await stream.next()).value as ConversationSnapshot
  assert.equal(first.runId, summary.id); assert.equal(first.historyOnly, false)
  assert.ok(first.messages.some(row => row.id === 'run-start'))
  const waiting = stream.next()
  controller.abort(); assert.equal((await waiting).done, true)
  assert.equal(f.children[0]!.kills, 0)
  assert.equal((await f.backend.getRun(summary.id)).summary.status, 'running')
  const other = new AbortController(), reconnected = f.backend.watchConversation(summary.id, other.signal)[Symbol.asyncIterator]()
  assert.equal(((await reconnected.next()).value as ConversationSnapshot).revision, first.revision)
  other.abort(); await reconnected.return?.()
  assert.equal(f.children.length, 1)
  await f.backend.cancelRun(summary.id)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
})

test('public analysis and paired tools are delivered, persisted and replayed as real actors', async () => {
  const f = await fixture()
  const fake = llm([
    { type: 'block-start', index: 0, blockType: 'reasoning' },
    { type: 'reasoning-delta', index: 0, text: 'PRIVATE_VENDOR_REASONING' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'PRIVATE_VENDOR_REASONING' } },
    { type: 'block-start', index: 1, blockType: 'text' },
    { type: 'text-delta', index: 1, text: '{"analysis":"Visible public analysis","proposals":[]}' },
    { type: 'block-end', index: 1, block: { type: 'text', text: '{"analysis":"Visible public analysis","proposals":[]}' } },
    { type: 'finish', reason: { kind: 'stop' } },
  ])
  f.model.port.prepareCall = fake.port.prepareCall
  const summary = await f.backend.startRun(start()), worker = f.children[0]!
  assert.deepEqual(worker.frames[0]!.features, ['process-events-v1', 'round-control-v1'])
  worker.output({ version: 1, type: 'process_event', event: { id: 'r1-fit', kind: 'tool_start', tool: 'fit', round: 1,
    counts: { known_count: 2000, ignored_private: 123 }, apiKey: SECRET, prompt: 'PRIVATE_PROMPT' } })
  worker.output({ version: 1, type: 'process_event', event: { id: 'r1-fit', kind: 'tool_end', tool: 'fit', round: 1,
    counts: { known_count: 2000 }, ok: true } })
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => worker.frames.length === 2)
  worker.output(resultFrame(summary.id)); worker.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const controller = new AbortController(), stream = f.backend.watchConversation(summary.id, controller.signal)[Symbol.asyncIterator]()
  const snapshot = (await stream.next()).value as ConversationSnapshot
  assert.equal(snapshot.status, 'completed'); assert.equal((await stream.next()).done, true)
  const analysis = snapshot.messages.find(row => row.id === 'model-request-1')!
  assert.equal(analysis.actor, 'model'); assert.equal(analysis.text, 'Visible public analysis'); assert.equal(analysis.status, 'completed')
  const tool = snapshot.messages.filter(row => row.id === 'tool-r1-fit')
  assert.equal(tool.length, 1); assert.equal(tool[0]!.actor, 'tool'); assert.equal(tool[0]!.status, 'completed')
  assert.equal(tool[0]!.inputSummary, '已知记录：2000'); assert.equal(tool[0]!.outputSummary, '已知记录：2000')
  const serialized = JSON.stringify(snapshot)
  for (const forbidden of ['PRIVATE_VENDOR_REASONING', 'PRIVATE_PROMPT', SECRET, 'ignored_private', 'response_schema', 'max_completion_tokens']) assert.ok(!serialized.includes(forbidden))
  const file = JSON.parse(await readFile(join(f.root, '.dsh-gb1/conversations', `${summary.id}.json`), 'utf8'))
  assert.deepEqual(file, snapshot)
  const reopened = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port,
    spawnWorker: () => { throw new Error('Read-only restore must not spawn') } }, { projectRoot: f.root })
  const replay = reopened.watchConversation(summary.id, new AbortController().signal)[Symbol.asyncIterator]()
  assert.deepEqual((await replay.next()).value, snapshot)
  assert.equal((await replay.next()).done, true)
  await writeFile(join(f.root, '.dsh-gb1/conversations', `${summary.id}.json`), 'SYNTHETIC_CORRUPT_RECORD')
  const corrupted = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port,
    spawnWorker: () => { throw new Error('Read failure must not spawn') } }, { projectRoot: f.root })
  const invalid = corrupted.watchConversation(summary.id, new AbortController().signal)[Symbol.asyncIterator]()
  await assert.rejects(invalid.next(), { message: '无法恢复运行过程记录' })
  assert.equal(f.children.length, 1)
})

test('tool failure is explicit and unfinished tools are cancelled without fabricated completion', async () => {
  const f = await fixture(), summary = await f.backend.startRun(start()), worker = f.children[0]!
  worker.output({ version: 1, type: 'process_event', event: { id: 'r1-fit', kind: 'tool_start', tool: 'fit', round: 1, counts: { known_count: 2000 } } })
  worker.output({ version: 1, type: 'process_event', event: { id: 'r1-fit', kind: 'tool_end', tool: 'fit', round: 1, counts: {}, ok: false } })
  worker.output({ version: 1, type: 'process_event', event: { id: 'r1-predict', kind: 'tool_start', tool: 'predict', round: 1, counts: { candidate_count: 40 } } })
  await f.backend.cancelRun(summary.id)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const controller = new AbortController(), stream = f.backend.watchConversation(summary.id, controller.signal)[Symbol.asyncIterator]()
  const snapshot = (await stream.next()).value as ConversationSnapshot
  assert.equal(snapshot.status, 'cancelled')
  assert.equal(snapshot.messages.find(row => row.id === 'tool-r1-fit')!.status, 'failed')
  assert.equal(snapshot.messages.find(row => row.id === 'tool-r1-predict')!.status, 'cancelled')
})

test('process tool inherited names and unmatched completions are rejected safely', async () => {
  for (const event of [
    { id: 'invalid', kind: 'tool_start', tool: 'toString', round: 1, counts: {} },
    { id: 'missing', kind: 'tool_end', tool: 'fit', round: 1, counts: {} },
  ]) {
    const f = await fixture(), summary = await f.backend.startRun(start())
    f.children[0]!.output({ version: 1, type: 'process_event', event })
    await until(async () => (await f.backend.getInfo()).activeRunId === null)
    assert.equal((await f.backend.getRun(summary.id)).summary.status, 'failed')
  }
})

test('real Cordis service binds direct and stream Remote methods and disposes its own run', async () => {
  const f = await fixture(), ctx = new Context()
  ctx.provide('llm', f.model.port)
  ctx.provide('credentials', f.creds.port)
  const service = new Gb1Workbench(ctx, { projectRoot: f.root })
  assert.equal((await service.getInfo()).ready, true)
  assert.equal(service.typertRemote.namespace, 'gb1Workbench')
  assert.deepEqual(remoteMethods(service).map(method => method.method).sort(),
    ['cancelBatch', 'cancelQuickStart', 'cancelRun', 'getBatch', 'getInfo', 'getModelCapabilities', 'getQuickStartStatus', 'getRun', 'getRunPage', 'listBatches', 'listModels', 'listRuns', 'pauseBatch', 'pauseRun', 'resumeBatch', 'resumeRun', 'saveConnection', 'startBatch', 'startQuickRun', 'startRun', 'watchConversation'])
  await ctx.fiber.dispose()
})

function configuredResult(summary: Awaited<ReturnType<WorkbenchBackend['startRun']>>, completed: number, calls = 0, status = 'completed') {
  const config = summary.config!, base = resultFrame(summary.id, calls)
  const frame = { ...base, status, summary: { ...base.summary, seed: summary.seed, method: summary.method, model: summary.model,
    requested_rounds: summary.rounds, completed_rounds: completed, calls, initial_known_count: config.train_size,
    known_count: config.train_size + completed * config.k, stop_reason: status === 'stopped' ? 'max_assays' : null,
    rounds: Array.from({ length: completed }, (_, index) => ({ ...base.summary.rounds[0]!, round: index + 1,
      calls: calls ? 1 : 0, valid_count: config.proposal_count, scored_count: config.score_budget,
      known_count_before: config.train_size + index * config.k, known_count_after: config.train_size + (index + 1) * config.k,
      top10: [...'ACDEFGHIKLMNPQRSTVWY'].slice(0, config.k).map((amino, rank) => {
        const sequence = REFERENCE.split(''); sequence[38] = amino
        return { ...base.summary.rounds[0]!.top10[0]!, rank: rank + 1, variant: amino + 'DGV', sequence: sequence.join('') }
      }) })) } }
  return frame
}

test('workbench defaults and exact model capabilities expose safe configurable values', async () => {
  const f = await fixture()
  let info = await f.backend.getInfo()
  assert.equal(info.maxRounds, null); assert.equal(info.defaultConfig.llm_timeout, null); assert.equal(info.defaultPromptProfile, 'v3')
  f.model.port.resolveModelInfo = async (provider, model) => ({ provider, id: model, name: 'Test', context: { contextWindow: 123456 }, defaultMaxTokens: 65432,
    reasoning: { efforts: [{ id: 'high' as never, name: 'High' }] } })
  assert.deepEqual(await f.backend.getModelCapabilities('fake', 'test-model'), { provider: 'fake', model: 'test-model', reasoningEfforts: ['high'], contextWindow: 123456, defaultMaxOutputTokens: 65432 })
  await writeFile(join(f.root, 'configs/workbench-defaults.json'), JSON.stringify({ config: { ...DEFAULT_CONFIG, k: 13, proposal_count: 17, score_budget: 17 }, promptProfile: 'v2', recommendedModel: { provider: 'fake', model: 'test-model' } }))
  info = await f.backend.getInfo()
  assert.equal(info.topK, 13); assert.equal(info.scoreBudget, 17); assert.equal(info.defaultPromptProfile, 'v2')
  assert.equal(info.recommendedModel!.model, 'test-model')
  const run = await f.backend.startRun({ ...start(), config: undefined, seed: 765, rounds: 4 })
  assert.equal(run.config!.k, 13); assert.equal(run.seed, 765); assert.equal(run.config!.rounds, 4); assert.equal(run.promptProfile, 'v2')
  await f.backend.dispose()
})

test('configuration validation rejects invalid budgets, private paths and unsupported values', () => {
  for (const fields of [{ k: 41 }, { proposal_count: 41 }, { seed: 2 ** 32 }, { llm_timeout: 0 }, { n_jobs: 0 },
    { allowed_positions: [39, 42] }, { allowed_amino_acids: 'AA' }, { max_mutations: 5 }, { knowledge_path: '../.env' },
    { knowledge_path: 'configs/../../.env' }, { llm_format: 'invalid' }, { high_fitness_quantile: 1 }, { top_k_metric: 0 }, { min_improvement: null }, { secret: 'private' }]) {
    assert.throws(() => validateConfig({ ...DEFAULT_CONFIG, ...fields }))
  }
  assert.deepEqual(validateConfig({ ...DEFAULT_CONFIG, ridge_alphas: [0, 1], allowed_positions: [], min_mutations: 0 }).ridge_alphas, [0, 1])
})

test('offline methods start without a saved model and expose fixed local identity', async () => {
  const f = await fixture()
  await writeFile(join(f.root, '.dsh-gb1/settings.json'), JSON.stringify({ source: 'dsh', provider: '', model: '', baseUrl: '' }))
  const run = await f.backend.startRun({ ...start(), method: 'random', seed: 765, rounds: 4, config: { ...historicalConfig(765, 4), k: 13, proposal_count: 17, score_budget: 17 } })
  assert.deepEqual(run.model, { source: 'dsh', provider: 'local', id: 'offline' })
  f.children[0]!.output(configuredResult(run, 4)); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const detail = await f.backend.getRun(run.id)
  assert.equal(detail.summary.status, 'completed'); assert.equal(detail.summary.completedRounds, 4)
  assert.equal(detail.roundResults[3]!.top10.length, 13); assert.equal(f.model.calls.length, 0)
  assert.equal(detail.summary.config!.score_budget, 17)
})

test('concurrent runs freeze their own model, configuration and independent operation receipts', async () => {
  const f = await fixture()
  f.model.port.listModels = async provider => [{ provider, id: 'test-model', name: 'One' }, { provider, id: 'other-model', name: 'Two' }]
  const a = await f.backend.startRun(start())
  const config = { ...historicalConfig(123, 6), k: 3, proposal_count: 9, score_budget: 9 }
  const b = await f.backend.startRun({ ...start('operation-independent-002'), seed: 123, rounds: 6, config, modelOverride: { provider: 'fake', model: 'other-model' } })
  config.k = 1
  assert.equal(b.config!.k, 3); assert.equal(b.model.id, 'other-model'); assert.equal(a.model.id, 'test-model')
  assert.equal(f.children[0]!.frames[0]!.rounds, 1); assert.equal(f.children[1]!.frames[0]!.rounds, 6)
  assert.ok(!JSON.stringify(f.children.map(worker => worker.frames)).includes('credential'))
  await assert.rejects(f.backend.startRun({ ...start(), rounds: 2, config: historicalConfig(11, 2) }), /其他运行参数/)
  await f.backend.dispose()
})

test('default detail is bounded and pagination reports actual results including stopped runs', async () => {
  const f = await fixture()
  const run = await f.backend.startRun({ ...start(), method: 'model', rounds: 30, config: historicalConfig(11, 30) })
  f.children[0]!.output(configuredResult(run, 25, 0, 'stopped')); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const detail = await f.backend.getRun(run.id), page = await f.backend.getRunPage(run.id, 0, 4)
  assert.equal(detail.summary.status, 'stopped'); assert.equal(detail.summary.stopReason, 'max_assays'); assert.equal(detail.summary.completedRounds, 25)
  assert.equal(detail.totalRounds, 25); assert.equal(detail.roundResults.length, 20); assert.equal(detail.roundResults[0]!.round, 6)
  assert.equal(detail.roundMetrics!.length, 25); assert.equal('top10' in detail.roundMetrics![0]!, false)
  assert.equal(page.totalRounds, 25); assert.deepEqual(page.roundResults.map(row => row.round), [1, 2, 3, 4])
  assert.equal((await f.backend.getRunPage(run.id, 25, 10)).roundResults.length, 0)
  await assert.rejects(f.backend.getRunPage(run.id, -1, 10)); await assert.rejects(f.backend.getRunPage(run.id, 0, 101))
})

test('legacy history restores explicit historical configuration without new defaults', async () => {
  const f = await fixture(), run = await f.backend.startRun(start())
  f.children[0]!.output(configuredResult(run, 1)); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const file = join(f.root, '.dsh-gb1/jobs', run.id + '.json'), saved = JSON.parse(await readFile(file, 'utf8'))
  delete saved.summary.config; delete saved.summary.formatVersion; delete saved.summary.promptProfile
  await writeFile(file, JSON.stringify(saved))
  const legacy = await f.backend.getRun(run.id)
  assert.equal(legacy.summary.formatVersion, 1); assert.equal(legacy.summary.config!.llm_timeout, 180); assert.equal(legacy.summary.config!.llm_max_output_tokens, 8000)
  assert.equal(legacy.summary.config!.k, 10)
})

test('large summary result reads only its fixed public file and keeps white-listed DTOs', async () => {
  const f = await fixture(), run = await f.backend.startRun({ ...start(), method: 'random' })
  const directory = join(f.root, 'results/dsh-plugin', run.id); await mkdir(directory, { recursive: true })
  await writeFile(join(directory, 'public-summary.json'), JSON.stringify(configuredResult(run, 1).summary))
  f.children[0]!.output({ version: 1, type: 'result', status: 'completed', summary_file: true, arbitrary_file: '../hidden.csv' }); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const detail = await f.backend.getRun(run.id)
  assert.equal(detail.summary.status, 'completed'); assert.ok(!JSON.stringify(detail).includes('/private/hidden'))
})

test('pause waits for feedback boundary and resume uses session-local IDs and cumulative calls', async () => {
  const f = await fixture(), run = await f.backend.startRun({ ...start(), rounds: 2, config: historicalConfig(11, 2) })
  const worker = f.children[0]!
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => worker.frames.length === 2)
  const pending = await f.backend.pauseRun(run.id)
  assert.equal(pending.status, 'running'); assert.equal(pending.stage, 'pausing'); assert.equal(worker.kills, 0)
  worker.output({ version: 1, type: 'checkpoint', round: 1 })
  await until(() => worker.frames.length === 3)
  assert.deepEqual(worker.frames[2], { version: 1, type: 'round_control', round: 1, action: 'pause' })
  worker.output(configuredResult(run, 1, 1, 'paused')); worker.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal((await f.backend.getRun(run.id)).summary.status, 'paused')
  const resumed = await f.backend.resumeRun(run.id)
  assert.equal(resumed.config!.rounds, 2); assert.equal(resumed.requestCount, 1)
  const next = f.children[1]!
  assert.equal(next.frames[0]!.resume, true)
  next.output({ version: 1, type: 'llm_request', ...request, id: 'request-1', round: 2 })
  await until(() => next.frames.length === 2)
  next.output({ version: 1, type: 'checkpoint', round: 2 })
  await until(() => next.frames.length === 3)
  assert.equal(next.frames[2]!.action, 'continue')
  next.output(configuredResult(resumed, 2, 2)); next.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const detail = await f.backend.getRun(run.id)
  assert.equal(detail.summary.status, 'completed'); assert.equal(detail.summary.requestCount, 2); assert.equal(f.model.calls.length, 2)
})

test('unknown model results cannot dispatch again or be resumed', async () => {
  const f = await fixture(), run = await f.backend.startRun(start()), worker = f.children[0]!
  f.model.port.prepareCall = async () => { throw new Error(SECRET) }
  worker.output({ version: 1, type: 'llm_request', ...request })
  await until(() => worker.frames.length === 2)
  assert.equal(worker.frames[1]!.type, 'llm_error')
  worker.output({ version: 1, type: 'llm_request', ...request, id: 'request-2' })
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal((await f.backend.getRun(run.id)).summary.status, 'failed')
  await assert.rejects(f.backend.resumeRun(run.id), /仅可恢复/)
  assert.equal(f.model.preparations.length, 0); assert.equal(f.children.length, 1)
})

const batchInput = (): StartBatchInput => ({ operationId: 'batch-operation-example-001', baseConfig: historicalConfig(11, 1), promptProfile: 'v3', methods: ['knowledge'],
  targets: [{ provider: 'fake', model: 'test-model', reasoningEffort: null }], seeds: [11, 765, 987], repeats: 2, concurrency: 3, perProviderConcurrency: 2 })

test('batch grid creates independent operations and respects global and provider concurrency', async () => {
  const f = await fixture(), input = batchInput()
  const batch = await f.backend.startBatch(input)
  assert.equal(batch.runs.length, 6); assert.equal(new Set(batch.runs.map(row => row.operationId)).size, 6)
  const repeated = await f.backend.startBatch(input)
  assert.equal(repeated.summary.id, batch.summary.id)
  await until(() => f.children.length === 2)
  assert.equal((await f.backend.getBatch(batch.summary.id)).summary.activeRuns, 2)
  assert.deepEqual(f.children.slice(0, 2).map(child => child.frames[0]!.repeat_index), [0, 1])
  const a = (await f.backend.getBatch(batch.summary.id)).runs.find(row => row.id)!
  const summary = (await f.backend.getRun(a.id!)).summary
  f.children[0]!.output(configuredResult(summary, 1)); f.children[0]!.close()
  await until(() => f.children.length === 3)
  const detail = await f.backend.getBatch(batch.summary.id)
  assert.equal(detail.summary.activeRuns, 2); assert.equal(detail.summary.completedRuns, 1)
  assert.equal(summary.batchId, batch.summary.id)
  assert.equal(new Set(detail.runs.filter(row => row.id).map(row => row.id)).size, 3)
  await f.backend.cancelBatch(batch.summary.id)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal((await f.backend.getBatch(batch.summary.id)).runs.filter(row => row.status === 'queued').length, 0)
  assert.equal((await f.backend.listBatches()).length, 1)
  await f.backend.dispose()
})

test('batch pause prevents dispatch until resume and cancellation leaves finished results', async () => {
  const f = await fixture(), input = { ...batchInput(), repeats: 1, concurrency: 1, perProviderConcurrency: 1 }
  const batch = await f.backend.startBatch(input)
  await until(() => f.children.length === 1)
  await f.backend.pauseBatch(batch.summary.id)
  const active = (await f.backend.getBatch(batch.summary.id)).runs.find(row => row.id)!, run = (await f.backend.getRun(active.id!)).summary
  f.children[0]!.output({ version: 1, type: 'checkpoint', round: 1 })
  await until(() => f.children[0]!.frames.length === 2)
  assert.equal(f.children[0]!.frames[1]!.action, 'pause')
  f.children[0]!.output(configuredResult(run, 1, 0, 'paused')); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal(f.children.length, 1); assert.equal((await f.backend.getBatch(batch.summary.id)).summary.status, 'paused')
  await f.backend.resumeBatch(batch.summary.id)
  await until(() => f.children.length === 2)
  assert.equal(f.children[1]!.frames[0]!.resume, true)
  f.children[1]!.output(configuredResult(run, 1)); f.children[1]!.close()
  await until(() => f.children.length === 3)
  await f.backend.cancelBatch(batch.summary.id)
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const cancelled = await f.backend.getBatch(batch.summary.id)
  assert.equal(cancelled.summary.status, 'cancelled'); assert.equal(cancelled.summary.completedRuns, 1)
  await f.backend.dispose()
})

test('batch concurrency counts distinct providers and offline jobs within the global cap', async () => {
  const f = await fixture()
  const input = { ...batchInput(), methods: ['random' as const], seeds: [11, 22], repeats: 2,
    targets: [{ provider: 'fake', model: 'test-model', reasoningEffort: null }, { provider: 'other', model: 'test-model', reasoningEffort: null }], concurrency: 3, perProviderConcurrency: 2 }
  const batch = await f.backend.startBatch(input)
  await until(() => f.children.length === 3)
  const detail = await f.backend.getBatch(batch.summary.id), active = detail.runs.filter(row => row.status === 'running')
  assert.equal(active.length, 3)
  assert.equal(active.filter(row => row.target.provider === 'fake').length, 2)
  assert.equal(active.filter(row => row.target.provider === 'other').length, 1)
  assert.ok(f.children.every(child => (child.frames[0]!.model as { provider: string }).provider === 'local'))
  await f.backend.cancelBatch(batch.summary.id)
  await f.backend.dispose()
})

test('batch restart exposes sanitized history and never automatically dispatches unknown runs', async () => {
  const f = await fixture(), input = { ...batchInput(), seeds: [11, 22], repeats: 1, concurrency: 1, perProviderConcurrency: 1 }
  const batch = await f.backend.startBatch(input)
  await until(() => f.children.length === 1)
  const before = await f.backend.getBatch(batch.summary.id), old = before.runs.find(row => row.id)!
  await f.backend.dispose()
  const jobPath = join(f.root, '.dsh-gb1/jobs', old.id + '.json'), job = JSON.parse(await readFile(jobPath, 'utf8'))
  job.summary.status = 'running'; job.summary.requestCount = 1
  await writeFile(jobPath, JSON.stringify(job))
  const batchPath = join(f.root, '.dsh-gb1/batches', batch.summary.id + '.json'), persisted = JSON.parse(await readFile(batchPath, 'utf8'))
  persisted.private = 'SYNTHETIC_PRIVATE_BATCH_FIELD'; persisted.runs[0].private = 'SYNTHETIC_PRIVATE_BATCH_FIELD'; persisted.summary.private = 'SYNTHETIC_PRIVATE_BATCH_FIELD'
  await writeFile(batchPath, JSON.stringify(persisted))
  const children: FakeWorker[] = []
  const reopened = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port, spawnWorker: () => {
    const worker = new FakeWorker(); children.push(worker); return worker as unknown as ChildProcessWithoutNullStreams
  } }, { projectRoot: f.root })
  assert.equal((await reopened.getInfo()).ready, true)
  const restored = await reopened.getBatch(batch.summary.id)
  assert.equal(restored.summary.status, 'interrupted'); assert.equal(restored.runs[0]!.status, 'interrupted')
  assert.equal(children.length, 0); assert.ok(!JSON.stringify(restored).includes('SYNTHETIC_PRIVATE_BATCH_FIELD'))
  assert.equal((await reopened.startBatch(input)).summary.id, batch.summary.id); assert.equal(children.length, 0)
  await reopened.resumeBatch(batch.summary.id)
  await until(() => children.length === 1)
  const resumed = await reopened.getBatch(batch.summary.id)
  assert.equal(resumed.runs[0]!.id, old.id); assert.equal(resumed.runs[0]!.status, 'interrupted')
  assert.equal(children[0]!.frames[0]!.seed, 22)
  await reopened.cancelBatch(batch.summary.id); await reopened.dispose()
})

test('per-run DSH reasoning, sampling and nullable output controls reach prepareCall exactly', async () => {
  const fake = llm(), gateway = new ModelGateway(fake.port, credentials().port)
  const payload = { ...request.payload, reasoning_effort: 'high', temperature: 0.25 }
  delete payload.max_completion_tokens
  const config = { ...structuredClone(DEFAULT_CONFIG), llm_reasoning_effort: 'high', llm_temperature: 0.25 }
  const response = await gateway.generate({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '', credentialRef: '', credentialBaseUrl: '' },
    { ...request, payload }, new AbortController().signal, undefined, config)
  assert.equal(response.complete, true)
  assert.deepEqual(fake.preparations, [{ provider: 'fake', model: 'test-model', reasoningEffort: 'high', temperature: 0.25 }])
  await assert.rejects(gateway.generate({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '', credentialRef: '', credentialBaseUrl: '' },
    request, new AbortController().signal, undefined, config), /冻结配置不一致/)
  assert.equal(fake.preparations.length, 1)
})

async function pausedFixture(f: Awaited<ReturnType<typeof fixture>>, operationId = 'pause-boundary-review-0001') {
  const run = await f.backend.startRun({ ...start(operationId), rounds: 2, config: historicalConfig(11, 2) })
  await f.backend.pauseRun(run.id)
  const worker = f.children.at(-1)!
  worker.output({ version: 1, type: 'checkpoint', round: 1 })
  await until(() => worker.frames.length === 2)
  worker.output(configuredResult(run, 1, 0, 'paused')); worker.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  assert.equal((await f.backend.getRun(run.id)).summary.status, 'paused')
  return run
}

test('all interrupted batch rows remain interrupted after explicit batch resume', async () => {
  const f = await fixture(), input = { ...batchInput(), seeds: [11, 22], repeats: 1, concurrency: 2, perProviderConcurrency: 2 }
  const batch = await f.backend.startBatch(input)
  await until(() => f.children.length === 2)
  let spawns = 0
  const reopened = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port, spawnWorker: () => { spawns++; throw new Error('Unknown runs must not restart') } }, { projectRoot: f.root })
  assert.equal((await reopened.getInfo()).ready, true)
  const resumed = await reopened.resumeBatch(batch.summary.id)
  assert.equal(resumed.summary.status, 'interrupted'); assert.equal(resumed.summary.completedRuns, 0); assert.equal(resumed.summary.failedRuns, 2)
  assert.equal((await reopened.getBatch(batch.summary.id)).summary.status, 'interrupted'); assert.equal(spawns, 0)
  await reopened.dispose(); await f.backend.dispose()
})

test('individual pause in a batch is preserved until an explicit batch resume', async () => {
  const f = await fixture(), input = { ...batchInput(), seeds: [11], repeats: 1, concurrency: 1, perProviderConcurrency: 1 }
  const batch = await f.backend.startBatch(input)
  await until(() => f.children.length === 1)
  const row = (await f.backend.getBatch(batch.summary.id)).runs[0]!, run = (await f.backend.getRun(row.id!)).summary
  await f.backend.pauseRun(run.id)
  f.children[0]!.output({ version: 1, type: 'checkpoint', round: 1 })
  await until(() => f.children[0]!.frames.length === 2)
  assert.equal(f.children[0]!.frames[1]!.action, 'pause')
  f.children[0]!.output(configuredResult(run, 1, 0, 'paused')); f.children[0]!.close()
  await until(async () => (await f.backend.getInfo()).activeRunId === null)
  const paused = await f.backend.getBatch(batch.summary.id)
  assert.equal(paused.runs[0]!.status, 'paused'); assert.equal(paused.summary.status, 'paused'); assert.equal(f.children.length, 1)
  await f.backend.resumeBatch(batch.summary.id)
  await until(() => f.children.length === 2)
  assert.equal(f.children[1]!.frames[0]!.resume, true)
  await f.backend.cancelBatch(batch.summary.id); await f.backend.dispose()
})

test('connection changes protect custom paused history beyond the latest one hundred runs', async () => {
  const f = await fixture()
  await f.backend.saveConnection({ source: 'custom', provider: '', model: 'test-model', baseUrl: 'https://example.invalid/v1', apiKey: SECRET })
  const run = await pausedFixture(f)
  const path = join(f.root, '.dsh-gb1/jobs', run.id + '.json'), original = JSON.parse(await readFile(path, 'utf8'))
  original.summary.createdAt = '2020-01-01T00:00:00.000Z'; await writeFile(path, JSON.stringify(original))
  for (let index = 0; index < 101; index++) {
    const clone = structuredClone(original), id = `gb1-newer-history-${index}`
    clone.summary.id = id; clone.summary.operationId = `newer-operation-history-${index}`; clone.summary.status = 'completed'; clone.summary.createdAt = '2026-11-13T00:00:00.000Z'
    await writeFile(join(f.root, '.dsh-gb1/jobs', id + '.json'), JSON.stringify(clone))
  }
  const listed = await f.backend.listRuns()
  assert.equal(listed.length, 100); assert.ok(!listed.some(row => row.id === run.id))
  await assert.rejects(f.backend.saveConnection({ source: 'dsh', provider: 'fake', model: 'test-model', baseUrl: '', clearKey: true }), /暂停的自定义运行/)
  assert.equal(f.creds.calls.unset, 0); assert.equal(f.creds.values.size, 1)
  await f.backend.cancelRun(run.id); await f.backend.dispose()
})

test('synchronous resume startup failure preserves its saved pause and permits a later retry', async () => {
  const f = await fixture(), run = await pausedFixture(f)
  await f.backend.dispose()
  let fail = true, attempts = 0
  const children: FakeWorker[] = []
  const reopened = new WorkbenchBackend({ llm: f.model.port, credentials: f.creds.port, spawnWorker: () => {
    attempts++; if (fail) throw new Error('Synthetic spawn failure')
    const worker = new FakeWorker(); children.push(worker); return worker as unknown as ChildProcessWithoutNullStreams
  } }, { projectRoot: f.root })
  await assert.rejects(reopened.resumeRun(run.id), /重新恢复/)
  const paused = await reopened.getRun(run.id)
  assert.equal(paused.summary.status, 'paused'); assert.equal(paused.summary.completedRounds, 1); assert.equal(paused.summary.requestCount, 0)
  fail = false
  assert.equal((await reopened.resumeRun(run.id)).status, 'running'); assert.equal(attempts, 2); assert.equal(children.length, 1)
  assert.equal(children[0]!.frames[0]!.resume, true)
  await reopened.cancelRun(run.id); await reopened.dispose()
})

test('resume does not remove an existing lease and retains a recoverable pause', async () => {
  const f = await fixture(), run = await pausedFixture(f)
  const path = join(f.root, '.dsh-gb1/leases', run.id + '.json')
  await writeFile(path, JSON.stringify({ id: run.id, owner: 'existing-worker' }))
  await assert.rejects(f.backend.resumeRun(run.id), /重新恢复/)
  assert.equal((await f.backend.getRun(run.id)).summary.status, 'paused')
  assert.equal(JSON.parse(await readFile(path, 'utf8')).owner, 'existing-worker')
  assert.equal(f.children.length, 1)
  await f.backend.cancelRun(run.id); await f.backend.dispose()
})

test('model requests can only target the next round after the last saved checkpoint', async () => {
  for (const completed of [0, 1]) {
    const f = await fixture(), run = await f.backend.startRun({ ...start(), rounds: 2, config: historicalConfig(11, 2) }), worker = f.children[0]!
    if (completed) {
      worker.output({ version: 1, type: 'checkpoint', round: 1 })
      await until(() => worker.frames.length === 2)
    }
    worker.output({ version: 1, type: 'llm_request', ...request, round: completed ? 1 : 2 })
    await until(async () => (await f.backend.getInfo()).activeRunId === null)
    assert.equal((await f.backend.getRun(run.id)).summary.status, 'failed')
    assert.equal(f.model.calls.length, 0)
    await f.backend.dispose()
  }
})
