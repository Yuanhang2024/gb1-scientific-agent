import assert from 'node:assert/strict'
import test from 'node:test'
import { AnalysisProgress, projectAnalysis, PublicTextGuard, type ModelProgress } from '../src/narrative.ts'
import { ModelGateway, type CredentialsPort, type LlmPort, type StoredConnection, type WorkerRequest } from '../src/model.ts'

const KEY = 'SYNTHETIC_SECRET_123'
const prompt = { profile: 'v3', version: 'gb1-proposals-v3', system_prompt: 'Public design summary only.' }
const request: WorkerRequest = { id: 'request-1', round: 1, prompt_metadata: prompt, payload: {
  model: 'test-model', messages: [{ role: 'system', content: prompt.system_prompt }, { role: 'user', content: '{}' }], max_completion_tokens: 8000 } }
const connection: StoredConnection = { source: 'custom', provider: 'custom', model: 'test-model', baseUrl: 'https://example.invalid/v1', credentialRef: 'DSH_GB1_' + 'A'.repeat(32), credentialBaseUrl: 'https://example.invalid/v1' }
const credentials = { resolve: async () => ({ value: KEY }) } as unknown as CredentialsPort
const emptyLlm = {} as LlmPort
function sse(body: unknown): string { return `data: ${JSON.stringify(body)}\n\n` }
function reply(parts: string[], incomplete = false): Response {
  let count = 0
  return new Response(new ReadableStream({ pull(controller) {
    if (count < parts.length) controller.enqueue(new TextEncoder().encode(parts[count++]!))
    else if (incomplete) controller.error(new Error('PRIVATE_PROVIDER_ERROR'))
    else controller.close()
  } }), { headers: { 'content-type': 'text/event-stream' } })
}
function gateway(response: () => Response) {
  let calls = 0
  let payload: Record<string, unknown> | undefined
  const fetcher = (async (_url: unknown, init: RequestInit) => { calls++; payload = JSON.parse(String(init.body)); return response() }) as typeof fetch
  return { model: new ModelGateway(emptyLlm, credentials, fetcher), calls: () => calls, payload: () => payload }
}

test('partial projection accepts only own public analysis strings and no prototype objects', () => {
  assert.deepEqual(projectAnalysis('{"analysis":"公开开头'), ['公开开头'])
  assert.deepEqual(projectAnalysis('{"analysis":["A","B'), ['A', 'B'])
  assert.deepEqual(projectAnalysis('{"proposals":[{"analysis":"hidden candidate"}]}'), [])
  assert.deepEqual(projectAnalysis('{"__proto__":{"analysis":"prototype value"}}'), [])
  assert.deepEqual(projectAnalysis('{"analysis":{"text":"not allowed"}}'), [])
  assert.equal(projectAnalysis('{"analysis":' + JSON.stringify('x'.repeat(7000)) + '}')![0]!.length, 6000)
})

test('invalid final JSON clears the draft instead of completing partial analysis', async () => {
  const values: ModelProgress[] = []
  const progress = new AnalysisProgress('request-1', 1, value => { values.push(value) })
  await progress.update('{"analysis":"draft')
  await progress.update('{"analysis":"draft', true)
  assert.equal(values.at(-1)!.text, '')
  assert.equal(values.at(-1)!.status, 'completed')
})

test('custom SSE streams safe full upserts, ignores reasoning and accepts empty-choice usage', async () => {
  const events: ModelProgress[] = []
  const full = '{"analysis":"中文公开总结","proposals":[]}'
  const frames = [sse({ model: 'model-alias', choices: [{ index: 0, delta: { reasoning_content: 'PRIVATE_REASONING', content: '{"analysis":"中文' } }] }),
    sse({ model: 'model-alias', choices: [{ index: 0, delta: { content: '公开总结","proposals":[]}' }, finish_reason: 'stop' }] }),
    sse({ model: 'model-alias', choices: [], usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 } }), 'data: [DONE]\n\n']
  const f = gateway(() => reply(frames))
  const result = await f.model.generate(connection, request, new AbortController().signal, value => { events.push(value) })
  assert.equal(f.calls(), 1); assert.equal(f.payload()!.stream, true)
  assert.deepEqual(f.payload()!.stream_options, { include_usage: true })
  assert.equal(result.text, full); assert.equal(result.model_returned, 'model-alias'); assert.equal(result.usage!.total_tokens, 30)
  assert.equal(events[0]!.text, '中文'); assert.equal(events.at(-1)!.text, '中文公开总结')
  assert.equal(events.at(-1)!.status, 'completed')
  assert.ok(!JSON.stringify([events, result]).includes('PRIVATE_REASONING'))
  assert.deepEqual((result.transport_request.payload as Record<string, unknown>).stream_options, { include_usage: true })
})

test('custom direct JSON fallback remains a single request and drops reasoning fields', async () => {
  const f = gateway(() => new Response(JSON.stringify({ choices: [{ message: { content: '{"analysis":"Safe"}', reasoning_content: 'PRIVATE_REASONING' }, finish_reason: 'stop' }], model: null })))
  const events: ModelProgress[] = []
  const value = await f.model.generate(connection, request, new AbortController().signal, event => { events.push(event) })
  assert.equal(value.text, '{"analysis":"Safe"}'); assert.equal(value.model_returned, null)
  assert.equal(events.at(-1)!.text, 'Safe'); assert.equal(f.calls(), 1)
  assert.ok(!JSON.stringify(value).includes('PRIVATE_REASONING'))
})

test('custom secret split over analysis chunks never reaches a public snapshot', async () => {
  for (let split = 1; split < KEY.length; split++) {
    const events: ModelProgress[] = []
    const f = gateway(() => reply([sse({ choices: [{ delta: { content: '{"analysis":"safe ' + KEY.slice(0, split) } }] }),
      sse({ choices: [{ delta: { content: KEY.slice(split) + '"}' }, finish_reason: 'stop' }] }), 'data: [DONE]\n\n']))
    await assert.rejects(f.model.generate(connection, request, new AbortController().signal, value => { events.push(value) }), error => !String(error).includes(KEY))
    assert.ok(!JSON.stringify(events).includes(KEY.slice(0, split)))
    assert.equal(f.calls(), 1)
  }
})

test('encoded key and endpoint never enter draft events', async () => {
  const endpoint = 'https://example.invalid/v1/chat/completions'
  for (const secret of [KEY, encodeURIComponent(endpoint), endpoint]) {
    const guard = new PublicTextGuard([KEY, endpoint])
    assert.throws(() => guard.safeDraft(secret))
  }
  const escaped = [...KEY].map(char => '\\u' + char.charCodeAt(0).toString(16).padStart(4, '0')).join('')
  const f = gateway(() => reply([sse({ choices: [{ delta: { content: '{"analysis":"' + escaped + '"}' }, finish_reason: 'stop' }] }), 'data: [DONE]\n\n']))
  await assert.rejects(f.model.generate(connection, request, new AbortController().signal))
})

test('disconnect, SSE error and missing DONE never resend and never publish provider errors', async () => {
  for (const response of [() => reply([sse({ choices: [{ delta: { content: '{"analysis":"draft' } }] })], true),
    () => reply([sse({ error: { message: 'PRIVATE_PROVIDER_ERROR' } })]),
    () => reply([sse({ choices: [{ delta: { content: '{"analysis":"Safe"}' }, finish_reason: 'stop' }] })])]) {
    const f = gateway(response), values: ModelProgress[] = []
    await assert.rejects(f.model.generate(connection, request, new AbortController().signal, value => { values.push(value) }), error => !String(error).includes('PRIVATE_PROVIDER_ERROR'))
    assert.equal(f.calls(), 1); assert.ok(!JSON.stringify(values).includes('PRIVATE_PROVIDER_ERROR'))
  }
})

test('DSH visible text projection excludes reasoning and block-end does not duplicate deltas', async () => {
  const chunks = [{ type: 'block-start', index: 0, blockType: 'reasoning' }, { type: 'reasoning-delta', index: 0, text: 'PRIVATE_REASONING' },
    { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'PRIVATE_REASONING' } },
    { type: 'block-start', index: 1, blockType: 'text' }, { type: 'text-delta', index: 1, text: '{"analysis":"公开' },
    { type: 'block-end', index: 1, block: { type: 'text', text: '{"analysis":"公开总结"}' } }, { type: 'finish', reason: { kind: 'stop' } }]
  const llm = { prepareCall: async () => ({ config: {}, async *stream() { yield* chunks } }) } as unknown as LlmPort
  const events: ModelProgress[] = []
  const result = await new ModelGateway(llm, credentials).generate({ ...connection, source: 'dsh' }, request, new AbortController().signal, value => { events.push(value) })
  assert.equal(result.text, '{"analysis":"公开总结"}'); assert.equal(events.at(-1)!.text, '公开总结')
  assert.ok(!JSON.stringify([result, events]).includes('PRIVATE_REASONING'))
})
