import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { credentialRef, type CredentialProvider } from '@deepseek-ai/dsh-credentials'
import { attributionHeaders, type LlmRuntime } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import type { ConnectionSettings, ExperimentConfiguration, ModelCapabilities, ModelIdentity, SaveConnectionInput } from './contracts.ts'
import { AnalysisProgress, PublicTextGuard, type ProgressCallback } from './narrative.ts'
export type { ModelProgress, ProgressCallback } from './narrative.ts'

declare module '@deepseek-ai/dsh-llm' { interface LlmRequestPurposeMap { 'gb1-workbench': 'gb1-workbench' } }

export type CredentialsPort = Pick<CredentialProvider, 'resolve' | 'describe' | 'set' | 'unset'>
export type LlmPort = Pick<LlmRuntime, 'listProviders' | 'listModels' | 'prepareCall'> & Partial<Pick<LlmRuntime, 'resolveModelInfo'>>
export interface StoredConnection { source: 'dsh' | 'custom'; provider: string; model: string; baseUrl: string; credentialRef: string; credentialBaseUrl: string }
export interface WorkerRequest { id: string; round: number; payload: Record<string, unknown>; prompt_metadata: Record<string, unknown> }
export interface ModelResponse {
  text: string; complete: boolean; usage: Record<string, unknown> | null
  model_requested: string; model_returned: string | null; api_mode: 'dsh' | 'chat'
  prompt_metadata: Record<string, unknown>; request_id?: string | null; response_id?: string | null
  transport_request: Record<string, unknown>
}
export const MAX_FRAME_BYTES = 16 * 1024 * 1024
export const REQUEST_TIMEOUT_MS = 180_000

export function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('数据必须为 JSON 对象')
  return value as Record<string, unknown>
}
export function identifier(value: unknown, name: string): string {
  if (typeof value !== 'string' || /[\x00-\x1f\x7f-\x9f]/.test(value)) throw new Error(`${name}格式无效`)
  const normalized = value.trim()
  if (!normalized || normalized.length > 200) throw new Error(`${name}格式无效`)
  return normalized
}
function normalizeBaseUrl(value: string): string {
  let url: URL
  try { url = new URL(value) } catch { throw new Error('Base URL 格式无效') }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('Base URL 仅允许不含凭据、查询或片段的 HTTP(S) 地址')
  return url.href.replace(/\/+$/, '')
}
export function validateConnection(input: SaveConnectionInput): Omit<StoredConnection, 'credentialRef' | 'credentialBaseUrl'> {
  if (input.source !== 'dsh' && input.source !== 'custom') throw new Error('请选择模型连接方式')
  const provider = input.source === 'custom' ? 'custom' : identifier(input.provider, 'Provider')
  const model = identifier(input.model, 'Model ID')
  const baseUrl = input.source === 'custom' ? normalizeBaseUrl(input.baseUrl) : ''
  if (input.apiKey !== undefined && (typeof input.apiKey !== 'string' || !input.apiKey.trim() || input.apiKey.length > 8192 || /[^\x21-\x7e]/.test(input.apiKey.trim()))) throw new Error('API Key 格式无效')
  if (input.apiKey && input.clearKey) throw new Error('不能同时保存和删除 API Key')
  if (input.apiKey && input.source !== 'custom') throw new Error('请在自定义连接中保存 API Key')
  return { source: input.source, provider, model, baseUrl }
}

/** 项目文件只保存公开设置与引用；密钥值只进入 Host 的凭据服务。 */
export class ConnectionStore {
  constructor(readonly root: string, readonly credentials: CredentialsPort) {}
  async load(): Promise<StoredConnection> {
    try {
      const raw = record(JSON.parse(await readFile(join(this.root, '.dsh-gb1', 'settings.json'), 'utf8')))
      const clean = validateConnection(raw as unknown as SaveConnectionInput)
      const ref = typeof raw.credentialRef === 'string' ? raw.credentialRef : ''
      if (ref && !/^DSH_GB1_[A-F0-9]{32}$/.test(ref)) throw new Error('凭据引用格式无效')
      const credentialBaseUrl = ref && typeof raw.credentialBaseUrl === 'string' && raw.credentialBaseUrl
        ? normalizeBaseUrl(raw.credentialBaseUrl) : ''
      return { ...clean, credentialRef: ref, credentialBaseUrl }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { source: 'dsh', provider: '', model: '', baseUrl: '', credentialRef: '', credentialBaseUrl: '' }
      throw new Error('无法读取项目连接设置，请重新检查设置文件')
    }
  }
  async publicSettings(): Promise<ConnectionSettings> {
    const stored = await this.load()
    let hasKey = false
    const bound = stored.credentialBaseUrl && (stored.source === 'dsh' || stored.credentialBaseUrl === stored.baseUrl)
    try { hasKey = stored.credentialRef && bound ? (await this.credentials.describe(credentialRef(stored.credentialRef))).configured : false }
    catch { throw new Error('无法查询项目凭据的配置状态') }
    return { source: stored.source, provider: stored.provider, model: stored.model, baseUrl: stored.baseUrl, hasKey }
  }
  async save(input: SaveConnectionInput): Promise<ConnectionSettings> {
    const clean = validateConnection(input)
    const previous = await this.load()
    let ref = previous.credentialRef
    let credentialBaseUrl = previous.credentialBaseUrl
    const replacing = Boolean(input.apiKey)
    if (clean.source === 'custom' && !input.apiKey && !input.clearKey) {
      if (!ref || !credentialBaseUrl) throw new Error('首次使用该自定义地址时请填写 API Key')
      if (credentialBaseUrl !== clean.baseUrl) throw new Error('更换 Base URL 时请填写新的 API Key')
      let configured = false
      try { configured = (await this.credentials.describe(credentialRef(ref))).configured }
      catch { throw new Error('无法查询项目凭据的配置状态') }
      if (!configured) throw new Error('请为该自定义地址填写 API Key')
    }
    if (input.apiKey) {
      ref = `DSH_GB1_${randomUUID().replaceAll('-', '').toUpperCase()}`
      credentialBaseUrl = clean.baseUrl
      try { await this.credentials.set(credentialRef(ref), input.apiKey.trim()) }
      catch { await this.credentials.unset(credentialRef(ref)).catch(() => {}); throw new Error('无法保存项目凭据') }
    } else if (input.clearKey) { ref = ''; credentialBaseUrl = '' }
    const directory = join(this.root, '.dsh-gb1')
    const temporary = join(directory, `settings-${randomUUID()}.tmp`)
    // 新值使用独立引用；公开设置提交后才移除旧引用，避免失败时改变旧连接的认证。
    try {
      await mkdir(directory, { recursive: true })
      await writeFile(temporary, JSON.stringify({ ...clean, credentialRef: ref, credentialBaseUrl }, null, 2), { flag: 'wx' })
      await rename(temporary, join(directory, 'settings.json'))
    } catch {
      if (replacing) await this.credentials.unset(credentialRef(ref)).catch(() => {})
      throw new Error('无法保存项目连接设置')
    }
    if (previous.credentialRef && previous.credentialRef !== ref) {
      try { await this.credentials.unset(credentialRef(previous.credentialRef)) }
      catch { throw new Error('连接设置已保存，旧凭据未能删除') }
    }
    return this.publicSettings()
  }
}

function count(value: unknown): number | null { return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null }
export function normalizeDshUsage(raw: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!raw) return null
  const uncached = count(raw.inputTokens), output = count(raw.outputTokens)
  const read = raw.cacheReadTokens === undefined ? 0 : count(raw.cacheReadTokens)
  const written = raw.cacheWriteTokens === undefined ? 0 : count(raw.cacheWriteTokens)
  const input = uncached === null || read === null || written === null ? null : uncached + read + written
  const total = count(raw.totalTokens) ?? (input !== null && output !== null ? input + output : null)
  return { input_tokens: input, output_tokens: output, total_tokens: total,
    missing_usage: input === null || output === null || total === null, raw_dsh_usage: structuredClone(raw) }
}
function normalizeCustomUsage(raw: unknown): Record<string, unknown> | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const usage = record(raw), input = count(usage.prompt_tokens), output = count(usage.completion_tokens)
  const total = count(usage.total_tokens) ?? (input !== null && output !== null ? input + output : null)
  return { input_tokens: input, output_tokens: output, total_tokens: total,
    missing_usage: input === null || output === null || total === null, raw_custom_usage: structuredClone(usage) }
}
function chatInput(request: WorkerRequest, model: ModelIdentity): { system: string; user: string; maxTokens?: number } {
  const payload = request.payload
  if (payload.model !== model.id || !Array.isArray(payload.messages) || payload.messages.length !== 2) throw new Error('Worker 模型请求与运行配置不一致')
  const system = record(payload.messages[0]), user = record(payload.messages[1])
  const maximum = payload.max_completion_tokens
  if (system.role !== 'system' || user.role !== 'user' || typeof system.content !== 'string' || typeof user.content !== 'string'
    || maximum != null && (!Number.isSafeInteger(maximum) || Number(maximum) < 1)) throw new Error('Worker 提示或输出预算无效')
  if (!['v1', 'v2', 'v3'].includes(String(request.prompt_metadata.profile)) || request.prompt_metadata.system_prompt !== system.content) throw new Error('Worker 提示版本不一致')
  return { system: system.content, user: user.content, ...(maximum == null ? {} : { maxTokens: Number(maximum) }) }
}

class ChatSse {
  text = ''
  usage: unknown = null
  finish: string | null = null
  model: string | null = null
  id: string | null = null
  refused = false
  done = false
  private buffer = ''
  private data: string[] = []
  private eventType = ''
  constructor(private readonly progress: AnalysisProgress, private readonly guard: PublicTextGuard) {}
  async feed(text: string, final = false): Promise<void> {
    this.guard.inspectChunk(text)
    this.buffer += text
    if (Buffer.byteLength(this.buffer) > MAX_FRAME_BYTES / 2) throw new Error('模型流事件超过允许大小')
    for (;;) {
      const found = /\r\n|\r|\n/.exec(this.buffer)
      if (!found || !final && found[0] === '\r' && found.index === this.buffer.length - 1) break
      const line = this.buffer.slice(0, found.index)
      this.buffer = this.buffer.slice(found.index + found[0].length)
      await this.line(line)
    }
    if (final) {
      if (this.buffer) { await this.line(this.buffer); this.buffer = '' }
      if (this.data.length) await this.dispatch()
      if (!this.done || this.finish === null) throw new Error('模型流未完整结束')
    }
  }
  private async line(line: string): Promise<void> {
    if (!line) { if (this.data.length) await this.dispatch(); this.eventType = ''; return }
    if (line.startsWith(':')) return
    const split = line.indexOf(':')
    const field = split < 0 ? line : line.slice(0, split)
    let value = split < 0 ? '' : line.slice(split + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'data') this.data.push(value)
    else if (field === 'event') this.eventType = value
    if (this.data.reduce((size, value) => size + Buffer.byteLength(value), 0) > MAX_FRAME_BYTES / 2) throw new Error('模型流事件超过允许大小')
  }
  private async dispatch(): Promise<void> {
    const data = this.data.join('\n'); this.data = []
    if (this.done) throw new Error('模型流结束后仍有数据')
    if (data === '[DONE]') { this.done = true; return }
    const body = record(JSON.parse(data))
    this.guard.assertJsonSafe(body)
    if (this.eventType === 'error' || Object.hasOwn(body, 'error')) throw new Error('模型流报告错误')
    if (body.model !== undefined && body.model !== null) {
      if (typeof body.model !== 'string' || !body.model.trim() || body.model.length > 200 || this.model !== null && this.model !== body.model) throw new Error('模型流标识无效')
      this.model = body.model
    }
    if (typeof body.id === 'string') { if (this.id !== null && this.id !== body.id) throw new Error('模型流响应标识变化'); this.id = body.id }
    if (Object.hasOwn(body, 'usage') && body.usage !== null) this.usage = structuredClone(body.usage)
    if (!Array.isArray(body.choices) || body.choices.length > 1) throw new Error('模型流响应数量无效')
    // include_usage允许choices=[]；reasoning_content/reasoning_details等字段不读取、不保存。
    if (!body.choices.length) return
    const choice = record(body.choices[0])
    if (choice.index !== undefined && choice.index !== 0 || this.finish !== null) throw new Error('模型流选择或结束顺序无效')
    const delta = record(choice.delta ?? {})
    if (delta.content !== undefined && delta.content !== null) {
      if (typeof delta.content !== 'string') throw new Error('模型流文本无效')
      this.text += delta.content
      if (Buffer.byteLength(this.text) > MAX_FRAME_BYTES / 2) throw new Error('模型文本超过允许大小')
      await this.progress.update(this.text)
    }
    if (delta.refusal) this.refused = true
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
      if (typeof choice.finish_reason !== 'string') throw new Error('模型流结束状态无效')
      this.finish = choice.finish_reason
    }
  }
}

export class ModelGateway {
  constructor(readonly llm: LlmPort, readonly credentials: CredentialsPort,
    readonly fetcher: typeof fetch = globalThis.fetch, readonly timeoutMs = REQUEST_TIMEOUT_MS) {}

  async capabilities(provider: string, model: string): Promise<ModelCapabilities> {
    provider = identifier(provider, 'Provider'); model = identifier(model, 'Model ID')
    try {
      if (!(await this.llm.listModels(provider)).some(item => item.id === model)) throw new Error('模型未注册')
      const info = await this.llm.resolveModelInfo?.(provider, model)
      return { provider, model, reasoningEfforts: info?.reasoning?.efforts.map(item => String(item.id)) ?? [],
        defaultMaxOutputTokens: info?.defaultMaxTokens ?? null, contextWindow: info?.context?.contextWindow ?? null }
    } catch { throw new Error('无法读取所选模型的能力') }
  }
  async generate(connection: StoredConnection, request: WorkerRequest, signal: AbortSignal, onProgress?: ProgressCallback, config?: ExperimentConfiguration): Promise<ModelResponse> {
    const model: ModelIdentity = { source: connection.source, provider: connection.provider, id: connection.model }
    const input = chatInput(request, model)
    if (config && ((request.payload.max_completion_tokens ?? null) !== config.llm_max_output_tokens
      || (request.payload.reasoning_effort ?? null) !== config.llm_reasoning_effort
      || (request.payload.temperature ?? null) !== config.llm_temperature)) throw new Error('Worker 请求与冻结配置不一致')
    const timeout = config ? config.llm_timeout === null ? null : config.llm_timeout * 1000 : this.timeoutMs
    const bounded = timeout === null ? signal : AbortSignal.any([signal, AbortSignal.timeout(Math.min(2147483647, Math.ceil(timeout)))])
    if (connection.source === 'dsh') return this.dsh(model, input, request, bounded, onProgress)
    return this.custom(connection, request, bounded, onProgress)
  }

  private async dsh(model: ModelIdentity, input: { system: string; user: string; maxTokens?: number }, request: WorkerRequest, signal: AbortSignal, onProgress?: ProgressCallback): Promise<ModelResponse> {
    try {
      const controls: Parameters<LlmPort['prepareCall']>[0] = { provider: model.provider, model: model.id }
      if (input.maxTokens !== undefined) controls.maxTokens = input.maxTokens
      if (request.payload.reasoning_effort != null) controls.reasoningEffort = request.payload.reasoning_effort as NonNullable<typeof controls.reasoningEffort>
      if (request.payload.temperature != null) controls.temperature = Number(request.payload.temperature)
      const prepared = await this.llm.prepareCall(controls, signal)
      const options: GenerateOptions<'gb1-workbench'> = { ...prepared.config,
        system: input.system, messages: [{ role: 'user', content: [{ type: 'text', text: input.user }] }],
        signal, purpose: 'gb1-workbench' }
      const blocks = new Map<number, { type: string; text: string }>()
      const publicText = () => [...blocks.entries()].sort(([a], [b]) => a - b).filter(([, block]) => block.type === 'text').map(([, block]) => block.text).join('')
      const progress = new AnalysisProgress(request.id, request.round, onProgress)
      let usage: Record<string, unknown> | null = null, finish: string | null = null
      // block-end 是该块的完整值，替换 delta 累积，避免双重拼接；reasoning 永不进入候选 JSON。
      for await (const value of prepared.stream(options) as AsyncIterable<StreamChunk>) {
        signal.throwIfAborted()
        if (finish !== null) throw new Error('模型在结束之后仍发送数据')
        if (value.type === 'block-start') blocks.set(value.index, { type: value.blockType, text: '' })
        else if (value.type === 'text-delta') {
          const current = blocks.get(value.index) ?? { type: 'text', text: '' }
          if (current.type === 'text') { current.text += value.text; blocks.set(value.index, current) }
        } else if (value.type === 'block-end') {
          blocks.set(value.index, { type: value.block.type, text: value.block.type === 'text' ? value.block.text : '' })
        } else if (value.type === 'usage') usage = record(structuredClone(value.usage))
        else if (value.type === 'finish') finish = value.reason.kind
        const visible = publicText()
        if (Buffer.byteLength(visible) > MAX_FRAME_BYTES / 2) throw new Error('模型文本超过允许大小')
        if (value.type === 'text-delta' || value.type === 'block-end' && value.block.type === 'text') await progress.update(visible)
      }
      signal.throwIfAborted()
      if (!finish) throw new Error('模型流缺少结束状态')
      if (finish === 'error' || finish === 'aborted') throw new Error('模型请求未完成')
      await progress.update(publicText(), true)
      // DSH 提供绑定的请求路由；服务端返回模型标识未知，按 null 保存。
      return { text: publicText(),
        complete: finish === 'stop', usage: normalizeDshUsage(usage), model_requested: model.id, model_returned: null,
        api_mode: 'dsh', prompt_metadata: structuredClone(request.prompt_metadata),
        transport_request: { source: 'dsh', ...prepared.config, system: input.system, messages: options.messages,
          purpose: 'gb1-workbench', finish_reason: finish } }
    } catch { throw new Error(signal.aborted ? '模型请求已取消或超时' : 'DSH 模型请求失败；未自动重试') }
  }

  private async custom(connection: StoredConnection, request: WorkerRequest, signal: AbortSignal, onProgress?: ProgressCallback): Promise<ModelResponse> {
    const baseUrl = normalizeBaseUrl(connection.baseUrl)
    if (!connection.credentialRef) throw new Error('请为该自定义地址保存 API Key')
    // 凭据只可用于保存时绑定的规范化地址；检查在 resolve 和网络请求之前完成。
    if (!connection.credentialBaseUrl || connection.credentialBaseUrl !== baseUrl) throw new Error('API Key 与当前 Base URL 不匹配，请重新填写 Key')
    let key = ''
    try {
      const resolved = connection.credentialRef ? await this.credentials.resolve(credentialRef(connection.credentialRef)) : undefined
      if (!resolved?.value) throw new Error('未配置项目 API Key')
      key = resolved.value.trim()
      if (!key || /[^\x21-\x7e]/.test(key)) throw new Error('项目 API Key 格式无效')
      const endpoint = `${baseUrl}/chat/completions`
      const guard = new PublicTextGuard([key, baseUrl, endpoint])
      const progress = new AnalysisProgress(request.id, request.round, onProgress, guard)
      const payload = { ...structuredClone(request.payload), stream: true, stream_options: { include_usage: true } }
      // 一次fetch同时支持SSE和兼容服务直接JSON；中途断线绝不重发。
      const response = await this.fetcher(endpoint, { method: 'POST', redirect: 'error', signal,
        headers: { ...attributionHeaders(), 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: JSON.stringify(payload) })
      if (!response.ok) { await response.body?.cancel(); throw new Error('自定义接口返回错误状态') }
      const requestId = response.headers.get('x-request-id')
      if (requestId) guard.assertSafe(requestId)
      const reader = response.body?.getReader()
      if (!reader) throw new Error('接口没有返回响应体')
      const contentType = response.headers.get('content-type') ?? ''
      let streaming: boolean | null = contentType.includes('text/event-stream') ? true : contentType.includes('application/json') ? false : null
      const decoder = new TextDecoder('utf-8', { fatal: true })
      const sse = new ChatSse(progress, guard)
      let raw = '', bytes = 0
      try {
        for (;;) {
          signal.throwIfAborted()
          const { value, done } = await reader.read()
          signal.throwIfAborted()
          const chunk = done ? decoder.decode() : decoder.decode(value, { stream: true })
          bytes += done ? 0 : value.byteLength
          if (bytes > MAX_FRAME_BYTES * 4) throw new Error('接口响应超过允许大小')
          if (streaming === null) {
            raw += chunk
            const first = raw.trimStart()
            if (first.startsWith('{')) streaming = false
            else if (/^(data:|event:|:)/.test(first)) streaming = true
            else if (first.length > 16 || done && first) throw new Error('接口响应格式无效')
            if (streaming === true) { await sse.feed(raw); raw = '' }
          } else if (streaming) await sse.feed(chunk)
          else { guard.inspectChunk(chunk); raw += chunk }
          if (streaming !== true && Buffer.byteLength(raw) > MAX_FRAME_BYTES / 2) throw new Error('接口响应超过允许大小')
          if (done) break
        }
        let text: string, complete: boolean, usage: unknown, returnedModel: unknown, responseId: unknown, finish: unknown
        if (streaming) {
          await sse.feed('', true)
          text = sse.text; complete = sse.finish === 'stop' && !sse.refused; usage = sse.usage
          returnedModel = sse.model; responseId = sse.id; finish = sse.finish
        } else {
          const body = record(JSON.parse(raw)); guard.assertJsonSafe(body)
          if (Object.hasOwn(body, 'error') || !Array.isArray(body.choices) || body.choices.length !== 1) throw new Error('接口响应数量无效')
          const choice = record(body.choices[0]), message = record(choice.message)
          if (typeof message.content !== 'string') throw new Error('接口没有返回文本')
          text = message.content; complete = choice.finish_reason === 'stop' && !message.refusal; usage = body.usage
          returnedModel = body.model ?? null; responseId = body.id; finish = choice.finish_reason ?? null
        }
        if (returnedModel !== null && (typeof returnedModel !== 'string' || !returnedModel.trim() || returnedModel.length > 200)) throw new Error('接口返回模型标识无效')
        guard.assertSafe(text)
        try { guard.assertJsonSafe(JSON.parse(text)) } catch (error) { if (!(error instanceof SyntaxError)) throw error }
        const normalizedUsage = normalizeCustomUsage(usage)
        guard.assertJsonSafe(normalizedUsage)
        await progress.update(text, true)
        return { text, complete, usage: normalizedUsage, model_requested: connection.model, model_returned: returnedModel as string | null,
          api_mode: 'chat', prompt_metadata: structuredClone(request.prompt_metadata), request_id: requestId,
          response_id: typeof responseId === 'string' ? responseId : null,
          transport_request: { source: 'custom', payload, finish_reason: finish } }
      } finally { await reader.cancel().catch(() => {}); reader.releaseLock() }
    } catch { throw new Error(signal.aborted ? '模型请求已取消或超时' : '自定义模型请求失败；未自动重试') }
    finally { key = '' }
  }
}
