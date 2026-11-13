import assert from 'node:assert/strict'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setTimeout as pause } from 'node:timers/promises'
import test from 'node:test'
import { ConversationLog, MAX_CONVERSATION_BYTES, MAX_CONVERSATION_MESSAGES, type ConversationUpsert } from '../src/conversation.ts'
import type { ConversationSnapshot, RunDetail } from '../src/contracts.ts'

const id = 'gb1-conversation-test'
function detail(status: RunDetail['summary']['status'] = 'running'): RunDetail {
  return { summary: { id, operationId: 'operation-conversation-01', status, stage: 'run', createdAt: '2026-11-10T12:00:00.000Z',
    updatedAt: '2026-11-10T12:01:00.000Z', seed: 11, rounds: 1, method: 'knowledge', model: { source: 'dsh', provider: 'fake', id: 'test' },
    completedRounds: status === 'completed' ? 1 : 0, requestCount: 1, error: null }, reference: '', initialKnownCount: 2000,
    knownCount: 2000, roundResults: [], events: [], usage: { inputTokens: null, outputTokens: null, totalTokens: null, missingUsageCount: 0 } }
}
const message = (text: string, status: ConversationUpsert['status'] = 'running'): ConversationUpsert => ({
  id: 'model-request-1', actor: 'model', kind: 'analysis', status, round: 1, title: '公开分析', text,
  toolName: null, inputSummary: '', outputSummary: '',
})
async function root() { return mkdtemp(join(tmpdir(), 'gb1-conversation-')) }
async function current(log: ConversationLog): Promise<ConversationSnapshot> {
  const controller = new AbortController(), iterator = log.watch(controller.signal)[Symbol.asyncIterator]()
  const result = await iterator.next()
  controller.abort(); await iterator.return?.()
  assert.equal(result.done, false)
  return result.value!
}

test('full upserts preserve identity, order and initial time across restarts', async () => {
  const directory = await root(), log = await ConversationLog.open(directory, detail(), true)
  log.upsert(message('first')); await log.flush()
  const first = (await current(log)).messages.find(row => row.id === 'model-request-1')!
  log.upsert(message('first and final', 'completed')); await log.flush()
  const saved = await current(log)
  const second = saved.messages.find(row => row.id === first.id)!
  assert.equal(second.order, first.order); assert.equal(second.time, first.time)
  assert.equal(second.text, 'first and final'); assert.equal(second.status, 'completed')
  const restored = await ConversationLog.open(directory, detail())
  assert.deepEqual(await current(restored), saved)
  assert.equal(saved.historyOnly, false)
})

test('subscriber sees a new revision only after persistence commits', async () => {
  const directory = await root(), commits: ConversationSnapshot[] = []
  let hold = false, entered = false, release: (() => void) | undefined
  const log = await ConversationLog.open(directory, detail(), true, { intervalMs: 10000,
    persist: async (_path, snapshot) => { if (hold) { entered = true; await new Promise<void>(resolve => { release = resolve }) }; commits.push(structuredClone(snapshot)) } })
  const controller = new AbortController(), iterator = log.watch(controller.signal)[Symbol.asyncIterator]()
  const initial = (await iterator.next()).value!
  hold = true
  log.upsert(message('pending save'))
  const flushing = log.flush()
  await pause(1); assert.equal(entered, true)
  let resolved = false
  const next = iterator.next().then(value => { resolved = true; return value })
  await pause(2); assert.equal(resolved, false)
  release!(); await flushing
  const received = (await next).value!
  assert.equal(received.revision, initial.revision + 1)
  assert.deepEqual(received, commits.at(-1))
  controller.abort(); await iterator.return?.()
})

test('slow subscriptions retain only the latest committed full snapshot', async () => {
  const log = await ConversationLog.open(await root(), detail(), true)
  const controller = new AbortController(), iterator = log.watch(controller.signal)[Symbol.asyncIterator]()
  const initial = (await iterator.next()).value!
  for (let index = 0; index < 10; index++) { log.upsert(message(`analysis ${index}`)); await log.flush() }
  const latest = (await iterator.next()).value as ConversationSnapshot
  assert.equal(latest.revision, initial.revision + 10)
  assert.equal(latest.messages.find(row => row.id === 'model-request-1')!.text, 'analysis 9')
  controller.abort(); await iterator.return?.()
})

test('aborting one subscription leaves the log and other subscription active', async () => {
  const log = await ConversationLog.open(await root(), detail(), true)
  const a = new AbortController(), b = new AbortController()
  const left = log.watch(a.signal)[Symbol.asyncIterator](), right = log.watch(b.signal)[Symbol.asyncIterator]()
  await left.next(); await right.next()
  const pending = left.next(); a.abort()
  assert.equal((await pending).done, true)
  log.upsert(message('still running')); await log.flush()
  assert.equal((await right.next()).value!.status, 'running')
  b.abort(); await right.return?.()
})

test('terminal snapshot is flushed, closes the stream and marks unfinished cards', async () => {
  for (const status of ['completed', 'cancelled', 'failed'] as const) {
    const directory = await root(), log = await ConversationLog.open(directory, detail(), true)
    log.upsert(message('unfinished visible analysis'))
    await log.finish(status)
    const controller = new AbortController(), iterator = log.watch(controller.signal)[Symbol.asyncIterator]()
    const snapshot = (await iterator.next()).value as ConversationSnapshot
    assert.equal(snapshot.status, status)
    assert.equal(snapshot.messages.find(row => row.id === 'model-request-1')!.status, status === 'cancelled' ? 'cancelled' : 'interrupted')
    assert.equal((await iterator.next()).done, true)
    const persisted = JSON.parse(await readFile(join(directory, '.dsh-gb1/conversations', `${id}.json`), 'utf8'))
    assert.deepEqual(persisted, snapshot)
  }
})

test('text bursts coalesce and messages stay bounded by count and encoded size', async () => {
  const directory = await root(), log = await ConversationLog.open(directory, detail(), true, { intervalMs: 5 })
  const before = (await current(log)).revision
  for (let index = 0; index < 30; index++) log.upsert(message('x'.repeat(index)))
  await pause(30)
  assert.equal((await current(log)).revision, before + 1)
  for (let index = 0; index < 200; index++) log.upsert({ ...message('long '.repeat(3000)), id: `message-${index}` })
  await log.flush()
  const snapshot = await current(log)
  assert.ok(snapshot.messages.length <= MAX_CONVERSATION_MESSAGES)
  assert.ok(Buffer.byteLength(JSON.stringify(snapshot)) <= MAX_CONVERSATION_BYTES)
  assert.equal(snapshot.truncated, true)
  assert.ok(snapshot.messages.every(row => row.text.length <= 6000))
})

test('restored DTOs omit unknown private fields and redact explicitly known endpoints', async () => {
  const directory = await root(), endpoint = 'https://synthetic-endpoint.invalid/v1'
  const log = await ConversationLog.open(directory, detail(), true, { redactions: [endpoint] })
  log.upsert({ ...message(endpoint), apiKey: 'PRIVATE-FIELD', reasoning: 'PRIVATE-REASONING', rawPrompt: 'PRIVATE-PROMPT' } as ConversationUpsert)
  await log.flush()
  const file = join(directory, '.dsh-gb1/conversations', `${id}.json`)
  let saved = await readFile(file, 'utf8')
  assert.ok(!saved.includes(endpoint)); assert.ok(!saved.includes('PRIVATE-'))
  const parsed = JSON.parse(saved)
  parsed.rawPrompt = 'PRIVATE-TOP'; parsed.messages[0].authorization = 'PRIVATE-AUTH'
  await writeFile(file, JSON.stringify(parsed))
  const restored = await ConversationLog.open(directory, detail())
  saved = JSON.stringify(await current(restored))
  assert.ok(!saved.includes('PRIVATE-'))
})

test('legacy runs expose saved historical analysis without fabricated tool pairs', async () => {
  const old = detail('completed')
  old.events = [{ time: old.summary.createdAt, stage: 'prepare', message: 'Saved preparation' }]
  old.roundResults = [{ round: 1, analysis: ['Saved public analysis'], status: 'evaluated', knownCountBefore: 2000,
    knownCountAfter: 2010, validCount: 40, scoredCount: 40, calls: 1, usage: old.usage,
    meanFitness: 1, maxFitness: 2, medianFitness: 1, cumulativeBest: 2, top10: [] }]
  const log = await ConversationLog.open(await root(), old)
  const snapshot = await current(log)
  assert.equal(snapshot.historyOnly, true); assert.equal(snapshot.status, 'completed')
  assert.ok(snapshot.messages.every(row => row.kind === 'legacy' && row.actor !== 'tool'))
  assert.ok(snapshot.messages.some(row => row.text === 'Saved public analysis'))
  assert.ok(snapshot.messages.some(row => row.text === 'Saved preparation'))
})

test('recovered incomplete runs retain recorded analysis and become interrupted', async () => {
  const directory = await root(), log = await ConversationLog.open(directory, detail(), true)
  log.upsert(message('saved partial')); await log.flush()
  const restored = await ConversationLog.open(directory, detail('interrupted'))
  const snapshot = await current(restored)
  assert.equal(snapshot.status, 'interrupted'); assert.equal(snapshot.historyOnly, false)
  assert.equal(snapshot.messages.find(row => row.id === 'model-request-1')!.status, 'interrupted')
  assert.equal(snapshot.messages.find(row => row.id === 'model-request-1')!.text, 'saved partial')
})

test('invalid snapshots produce a safe error with no file text', async () => {
  const directory = await root()
  const log = await ConversationLog.open(directory, detail(), true)
  await log.finish('completed')
  await writeFile(join(directory, '.dsh-gb1/conversations', `${id}.json`), 'SYNTHETIC_PRIVATE_BAD_FILE')
  await assert.rejects(ConversationLog.open(directory, detail()), { message: '无法读取运行过程记录' })
})

test('persistence failure is a safe stream error and does not publish uncommitted text', async () => {
  let fail = false
  const log = await ConversationLog.open(await root(), detail(), true, { persist: async () => { if (fail) throw new Error('SYNTHETIC_PRIVATE_WRITE_ERROR') } })
  const controller = new AbortController(), iterator = log.watch(controller.signal)[Symbol.asyncIterator]()
  await iterator.next()
  fail = true
  const pending = iterator.next()
  const rejected = assert.rejects(pending, { message: '无法保存运行过程记录' })
  log.upsert(message('never committed'))
  await assert.rejects(log.flush(), { message: '无法保存运行过程记录' })
  await rejected
  controller.abort()
})

test('updates made during an atomic save remain pending for the next monotone revision', async () => {
  let release: (() => void) | undefined, hold = false, entered = false
  const log = await ConversationLog.open(await root(), detail(), true, { intervalMs: 10000, persist: async () => {
    if (hold) { hold = false; entered = true; await new Promise<void>(resolve => { release = resolve }) }
  } })
  const before = (await current(log)).revision
  hold = true; log.upsert(message('first commit'))
  const saving = log.flush()
  await pause(1); assert.equal(entered, true)
  log.upsert(message('new text while saving'))
  release!(); await saving
  assert.equal((await current(log)).revision, before + 1)
  assert.equal((await current(log)).messages.find(row => row.id === 'model-request-1')!.text, 'first commit')
  await log.flush()
  const latest = await current(log)
  assert.equal(latest.revision, before + 2)
  assert.equal(latest.messages.find(row => row.id === 'model-request-1')!.text, 'new text while saving')
})

test('byte truncation remains marked on subsequent complete snapshots', async () => {
  const log = await ConversationLog.open(await root(), detail(), true)
  for (let index = 0; index < 80; index++) log.upsert({ ...message('汉'.repeat(6000)), id: `wide-${index}` })
  await log.flush()
  assert.equal((await current(log)).truncated, true)
  log.upsert(message('small update')); await log.flush()
  assert.equal((await current(log)).truncated, true)
})

test('message identity and actual actor cannot be changed by a later upsert', async () => {
  const log = await ConversationLog.open(await root(), detail(), true)
  log.upsert(message('model'))
  assert.throws(() => log.upsert({ ...message('pretend tool'), actor: 'tool', kind: 'tool', toolName: 'fit' }))
  await log.finish('cancelled')
})
