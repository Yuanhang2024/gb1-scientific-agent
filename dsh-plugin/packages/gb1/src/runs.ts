import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access, mkdir, readFile, readdir, rename, unlink, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ConversationSnapshot, ExperimentConfiguration, ModelIdentity, Recommendation, RoundResult, RunDetail, RunPage, RunSummary, StartRunInput, UsageSummary } from './contracts.ts'
import { ConversationLog } from './conversation.ts'
import { ConnectionStore, MAX_FRAME_BYTES, ModelGateway, record, type StoredConnection, type WorkerRequest } from './model.ts'
import { historicalConfig, loadDefaults, modelOverride, promptProfile, validSeed, validateConfig } from './settings.ts'

export const REFERENCE = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
export type SpawnWorker = (executable: string, argv: string[], options: Parameters<typeof spawn>[2]) => ChildProcessWithoutNullStreams
const WORKER_SCRIPT = resolve(dirname(fileURLToPath(import.meta.url)), '../python/worker.py')
const SAFE_RUN = /^[a-z0-9][a-z0-9_-]{0,95}$/
export const SAFE_OPERATION = /^[a-zA-Z0-9][a-zA-Z0-9_-]{15,127}$/
const TERMINAL = new Set(['completed', 'stopped', 'failed', 'cancelled', 'interrupted'])
const emptyUsage = (): UsageSummary => ({ inputTokens: null, outputTokens: null, totalTokens: null, missingUsageCount: 0 })
export async function projectLocation(configured = ''): Promise<{ root: string; ready: boolean; reason: string }> {
  const root = configured ? resolve(configured) : resolve(dirname(fileURLToPath(import.meta.url)), '../../../..')
  try {
    await Promise.all([...['gb1/data.py', 'configs/knowledge.json', 'data/raw/elife-16965-supp1-v4.xlsx', '.venv/Scripts/python.exe'].map(path => access(join(root, path))), access(WORKER_SCRIPT)])
    return { root, ready: true, reason: '' }
  } catch { return { root, ready: false, reason: 'GB1 项目尚未就绪：请配置 projectRoot，并确认工程代码、数据、项目 Python 环境和插件自带 worker 已存在。' } }
}
export async function readJson(path: string): Promise<unknown> {
  const content = await readFile(path)
  if (content.length > 16 * MAX_FRAME_BYTES) throw new Error('运行记录超过允许大小')
  return JSON.parse(content.toString('utf8'))
}
export async function saveJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.${randomUUID()}.tmp`
  await writeFile(temporary, JSON.stringify(value, null, 2), { flag: 'wx' })
  await rename(temporary, path)
}
function numeric(value: unknown, nullable = false): number | null {
  if (value === null && nullable) return null
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Worker 数值字段无效')
  return value
}
function integer(value: unknown): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) throw new Error('Worker 计数字段无效')
  return Number(value)
}
function text(value: unknown, maximum = 5000): string {
  if (typeof value !== 'string' || value.length > maximum) throw new Error('Worker 文本字段无效')
  return value
}
function usageDto(value: unknown): UsageSummary {
  const raw = record(value)
  return { inputTokens: numeric(raw.input_tokens, true), outputTokens: numeric(raw.output_tokens, true), totalTokens: numeric(raw.total_tokens, true), missingUsageCount: integer(raw.missing_usage_count) }
}
function recommendationDto(value: unknown, config: ExperimentConfiguration): Recommendation {
  const row = record(value), variant = text(row.variant, 4), sequence = text(row.sequence, 56)
  if (!/^[ACDEFGHIKLMNPQRSTVWY]{4}$/.test(variant) || !/^[ACDEFGHIKLMNPQRSTVWY]{56}$/.test(sequence)) throw new Error('Worker 推荐序列格式无效')
  const expected = REFERENCE.split('')
  ;[39, 40, 41, 54].forEach((position, index) => { expected[position - 1] = variant[index]! })
  if (expected.join('') !== sequence || !Array.isArray(row.knowledge_rule_ids)) throw new Error('Worker 推荐序列映射无效')
  return { rank: integer(row.rank), variant, sequence, mutations: text(row.mutations, 100), mutationCount: integer(row.mutation_count), predictedFitness: numeric(row.predicted_fitness)!, actualFitness: numeric(row.actual_fitness, true),
    hypothesis: text(row.hypothesis, config.proposal_text_max_chars), rationale: text(row.rationale, config.proposal_text_max_chars), knowledgeRuleIds: row.knowledge_rule_ids.map(id => text(id, 200)) }
}
function roundDto(value: unknown, config: ExperimentConfiguration): RoundResult {
  const raw = record(value), metrics = record(raw.metrics)
  if (!Array.isArray(raw.analysis) || !Array.isArray(raw.top10) || raw.top10.length > config.k) throw new Error('Worker 轮次字段无效')
  const scored = integer(raw.scored_count), valid = integer(raw.valid_count), calls = integer(raw.calls)
  if (scored > config.score_budget || valid > config.proposal_count || calls > config.llm_max_calls_per_round) throw new Error('Worker 轮次超过预算')
  return { round: integer(raw.round), status: text(raw.status, 50), analysis: raw.analysis.map(item => text(item, config.analysis_max_chars)),
    knownCountBefore: integer(raw.known_count_before), knownCountAfter: integer(raw.known_count_after), validCount: valid, scoredCount: scored, calls, usage: usageDto(raw.usage),
    meanFitness: numeric(metrics.mean_fitness, true), maxFitness: numeric(metrics.max_fitness, true), medianFitness: numeric(metrics.median_fitness, true), cumulativeBest: numeric(metrics.cumulative_best, true), top10: raw.top10.map(value => recommendationDto(value, config)) }
}
export function validateStart(input: StartRunInput): StartRunInput {
  if (!SAFE_OPERATION.test(input.operationId) || input.reference !== REFERENCE || !validSeed(input.seed) || !Number.isSafeInteger(input.rounds) || input.rounds < 1
    || !['random', 'model', 'llm', 'knowledge'].includes(input.method)) throw new Error('运行参数无效，请检查参考序列、种子、轮次与方法')
  const clean: StartRunInput = { operationId: input.operationId, reference: REFERENCE, seed: input.seed, rounds: input.rounds, method: input.method }
  if (input.config !== undefined) {
    clean.config = validateConfig(input.config)
    if (clean.config.seed !== input.seed || clean.config.rounds !== input.rounds) throw new Error('配置seed和rounds必须与顶层一致')
  }
  if (input.promptProfile !== undefined) clean.promptProfile = promptProfile(input.promptProfile)
  if (input.modelOverride !== undefined) clean.modelOverride = modelOverride(input.modelOverride)
  if (input.repeatIndex !== undefined) clean.repeatIndex = integer(input.repeatIndex)
  return clean
}
function savedDetail(value: unknown): RunDetail {
  const raw = record(value), summary = record(raw.summary), model = record(summary.model)
  const input = validateStart({ operationId: text(summary.operationId, 128), reference: text(raw.reference, 56), seed: integer(summary.seed), rounds: integer(summary.rounds), method: text(summary.method, 20) as StartRunInput['method'] })
  const config = summary.formatVersion === 2 ? validateConfig(summary.config) : historicalConfig(input.seed, input.rounds)
  const status = text(summary.status, 30)
  if (!['queued', 'running', 'paused', ...TERMINAL].includes(status) || !['dsh', 'custom'].includes(String(model.source))) throw new Error('Host 运行记录状态无效')
  if (!Array.isArray(raw.roundResults) || !Array.isArray(raw.events)) throw new Error('Host 运行记录字段无效')
  const savedUsage = (value: unknown): UsageSummary => {
    const row = record(value)
    return usageDto({ input_tokens: row.inputTokens, output_tokens: row.outputTokens, total_tokens: row.totalTokens, missing_usage_count: row.missingUsageCount })
  }
  return { summary: { id: text(summary.id, 96), operationId: input.operationId, status: status as RunSummary['status'], stage: text(summary.stage, 50), createdAt: text(summary.createdAt, 50), updatedAt: text(summary.updatedAt, 50), seed: input.seed, rounds: input.rounds, method: input.method,
    model: { source: model.source as ModelIdentity['source'], provider: text(model.provider, 200), id: text(model.id, 200) }, completedRounds: integer(summary.completedRounds), requestCount: integer(summary.requestCount), error: summary.error === null ? null : text(summary.error),
    formatVersion: summary.formatVersion === 2 ? 2 : 1, config, promptProfile: summary.formatVersion === 2 ? promptProfile(summary.promptProfile) : 'v3', repeatIndex: summary.repeatIndex === undefined ? 0 : integer(summary.repeatIndex),
    batchId: summary.batchId == null ? null : text(summary.batchId, 96), stopReason: summary.stopReason == null ? null : text(summary.stopReason, 500) },
    reference: REFERENCE, initialKnownCount: integer(raw.initialKnownCount), knownCount: integer(raw.knownCount), usage: savedUsage(raw.usage), events: raw.events.slice(-100).map(value => {
      const event = record(value); return { time: text(event.time, 50), stage: text(event.stage, 50), message: text(event.message, 500) }
    }), roundResults: raw.roundResults.map(value => {
      const row = record(value), usage = savedUsage(row.usage)
      if (!Array.isArray(row.top10) || !Array.isArray(row.analysis)) throw new Error('Host 推荐记录无效')
      return roundDto({ round: row.round, status: row.status, analysis: row.analysis, known_count_before: row.knownCountBefore, known_count_after: row.knownCountAfter, valid_count: row.validCount, scored_count: row.scoredCount, calls: row.calls,
        usage: { input_tokens: usage.inputTokens, output_tokens: usage.outputTokens, total_tokens: usage.totalTokens, missing_usage_count: usage.missingUsageCount },
        metrics: { mean_fitness: row.meanFitness, max_fitness: row.maxFitness, median_fitness: row.medianFitness, cumulative_best: row.cumulativeBest },
        top10: row.top10.map(value => { const item = record(value); return { rank: item.rank, variant: item.variant, sequence: item.sequence, mutations: item.mutations, mutation_count: item.mutationCount, predicted_fitness: item.predictedFitness, actual_fitness: item.actualFitness, hypothesis: item.hypothesis, rationale: item.rationale, knowledge_rule_ids: item.knowledgeRuleIds } }) }, config)
    }) }
}
interface ActiveJob {
  detail: RunDetail; child: ChildProcessWithoutNullStreams; controller: AbortController; buffer: Buffer; pending: boolean; receivedResult: boolean
  callsByRound: Map<number, number>; writes: Promise<void>; frames: Promise<void>; done: Promise<void>; finish: () => void; connection: StoredConnection
  conversation: ConversationLog; tools: Map<string, { input: string; tool: string; round: number; ended: boolean }>; analyses: Map<string, string>
  sessionCalls: number; pauseRequested: boolean; lastCheckpoint: number; requestFailed: boolean
}
/** 每个运行拥有独立配置、连接和子进程；相同操作编号始终返回原结果。 */
export class RunManager {
  private active = new Map<string, ActiveJob>()
  private orphanId: string | null = null
  private serial: Promise<unknown> = Promise.resolve()
  private disposed = false
  private readonly jobDirectory: string
  private readonly conversations = new Map<string, ConversationLog>()
  private readonly conversationLoads = new Map<string, Promise<ConversationLog>>()
  private readonly listeners = new Set<() => void>()
  constructor(readonly root: string, readonly connections: ConnectionStore, readonly gateway: ModelGateway,
    readonly spawnWorker: SpawnWorker = (executable, argv, options) => spawn(executable, argv, options) as ChildProcessWithoutNullStreams) { this.jobDirectory = join(root, '.dsh-gb1', 'jobs') }
  get activeRunId(): string | null { return this.active.keys().next().value ?? this.orphanId }
  get activeSummaries(): RunSummary[] { return [...this.active.values()].map(job => structuredClone(job.detail.summary)) }
  onChange(listener: () => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener) }
  private notify(): void { for (const listener of this.listeners) listener() }
  private exclusive<T>(action: () => Promise<T>): Promise<T> { const next = this.serial.then(action); this.serial = next.catch(() => {}); return next }
  async initialize(): Promise<void> {
    for (const directory of ['jobs', 'operations', 'leases', 'frozen']) await mkdir(join(this.root, '.dsh-gb1', directory), { recursive: true })
    try {
      const lease = record(await readJson(join(this.root, '.dsh-gb1', 'active.json')))
      if (typeof lease.id === 'string' && SAFE_RUN.test(lease.id)) this.orphanId = lease.id
      else throw new Error('活动运行记录无效')
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
  }
  mutateConnection<T>(action: () => Promise<T>): Promise<T> { return this.exclusive(async () => {
    if (this.disposed || this.activeRunId) throw new Error('运行期间不能修改连接设置')
    // 已暂停的自定义运行仍引用其原凭据，避免保存连接时删除它。
    if ((await this.allRunSummaries()).some(run => run.status === 'paused' && run.model.source === 'custom')) throw new Error('存在暂停的自定义运行，请完成或取消后修改连接设置')
    return action()
  }) }
  async listRuns(): Promise<RunSummary[]> { return (await this.allRunSummaries()).slice(0, 100) }
  private async allRunSummaries(): Promise<RunSummary[]> {
    const files = await readdir(this.jobDirectory), summaries: RunSummary[] = []
    for (const file of files.filter(file => file.endsWith('.json') && SAFE_RUN.test(file.slice(0, -5)))) {
      try { summaries.push((await this.fullRun(file.slice(0, -5))).summary) } catch { /* 不公开损坏记录。 */ }
    }
    return summaries.sort((a, b) => b.createdAt.localeCompare(a.createdAt))
  }
  private async fullRun(id: string): Promise<RunDetail> {
    if (!SAFE_RUN.test(id)) throw new Error('运行 ID 无效')
    const active = this.active.get(id)
    if (active) return structuredClone(active.detail)
    const detail = savedDetail(await readJson(join(this.jobDirectory, `${id}.json`)))
    if (detail.summary.id !== id) throw new Error('运行记录无效')
    if (['running', 'queued'].includes(detail.summary.status)) { detail.summary.status = 'interrupted'; detail.summary.error = '此前 Host 运行状态未知；不会自动重复提交。' }
    return detail
  }
  async getRun(id: string): Promise<RunDetail> {
    const detail = await this.fullRun(id)
    detail.roundMetrics = detail.roundResults.map(({ top10: _top10, analysis: _analysis, ...metrics }) => metrics)
    detail.totalRounds = detail.roundResults.length
    detail.roundResults = detail.roundResults.slice(-20)
    return detail
  }
  async getRunPage(id: string, offset: number, limit: number): Promise<RunPage> {
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error('分页参数无效，limit范围为1至100')
    const detail = await this.fullRun(id)
    return { runId: id, offset, limit, totalRounds: detail.roundResults.length, roundResults: detail.roundResults.slice(offset, offset + limit) }
  }
  async *watchConversation(id: string, signal: AbortSignal): AsyncIterable<ConversationSnapshot> {
    if (!SAFE_RUN.test(id)) throw new Error('运行 ID 无效')
    if (signal.aborted) return
    let conversation = this.conversations.get(id)
    if (!conversation) {
      let pending = this.conversationLoads.get(id)
      if (!pending) { pending = this.fullRun(id).then(detail => ConversationLog.open(this.root, detail)); this.conversationLoads.set(id, pending) }
      try { conversation = await pending; this.conversations.set(id, conversation) }
      catch { throw new Error('无法恢复运行过程记录') }
      finally { if (this.conversationLoads.get(id) === pending) this.conversationLoads.delete(id) }
    }
    yield* conversation.watch(signal)
  }
  startRun(raw: StartRunInput, batchId: string | null = null): Promise<RunSummary> {
    return this.exclusive(async () => {
      const input = validateStart(raw)
      if (this.disposed) throw new Error('工作台已停止')
      const operationPath = join(this.root, '.dsh-gb1', 'operations', `${input.operationId}.json`)
      try {
        const receipt = record(await readJson(operationPath))
        if (JSON.stringify(receipt.input) !== JSON.stringify(input)) throw new Error('该操作编号已用于其他运行参数')
        return (await this.fullRun(String(receipt.id))).summary
      } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
      if (this.orphanId) throw new Error('已有活动运行状态未知，不会自动覆盖')
      const ready = await projectLocation(this.root)
      if (!ready.ready) throw new Error(ready.reason)
      const defaults = await loadDefaults(this.root)
      const config = input.config ?? validateConfig({ ...defaults.config, seed: input.seed, rounds: input.rounds })
      const connection: StoredConnection = !['llm', 'knowledge'].includes(input.method)
        ? { source: 'dsh', provider: 'local', model: 'offline', baseUrl: '', credentialRef: '', credentialBaseUrl: '' }
        : input.modelOverride ? { source: 'dsh', ...input.modelOverride, baseUrl: '', credentialRef: '', credentialBaseUrl: '' } : await this.connections.load()
      if (['llm', 'knowledge'].includes(input.method)) await this.validateModel(connection, config)
      const id = `gb1-${randomUUID()}`, now = new Date().toISOString()
      const model: ModelIdentity = { source: connection.source, provider: connection.provider, id: connection.model }
      const detail: RunDetail = { summary: { id, operationId: input.operationId, status: 'running', stage: 'starting', createdAt: now, updatedAt: now,
        seed: input.seed, rounds: input.rounds, method: input.method, model, completedRounds: 0, requestCount: 0, error: null, formatVersion: 2,
        config, promptProfile: input.promptProfile ?? defaults.promptProfile, repeatIndex: input.repeatIndex ?? 0, batchId, stopReason: null },
        reference: REFERENCE, initialKnownCount: 0, knownCount: 0, roundResults: [], events: [], usage: emptyUsage() }
      // 回执写入后才允许启动子进程；不自动补发未知的启动结果。
      await saveJson(join(this.jobDirectory, `${id}.json`), detail)
      await writeFile(operationPath, JSON.stringify({ id, input }), { flag: 'wx' })
      await saveJson(join(this.root, '.dsh-gb1', 'frozen', `${id}.json`), connection)
      return this.launch(detail, connection, false)
    })
  }
  private async validateModel(connection: StoredConnection, config: ExperimentConfiguration): Promise<void> {
    if (!connection.provider || !connection.model) throw new Error('请先保存模型连接')
    if (connection.source === 'custom' && !(await this.connections.publicSettings()).hasKey) throw new Error('请先保存自定义 API Key')
    if (connection.source === 'dsh') {
      let available = false
      try { available = (await this.gateway.llm.listModels(connection.provider)).some(model => model.id === connection.model) } catch { throw new Error('无法读取 DSH 模型目录') }
      if (!available) throw new Error('所选 DSH 模型当前不可用')
      if (config.llm_reasoning_effort !== null) {
        const capabilities = await this.gateway.capabilities(connection.provider, connection.model)
        if (!capabilities.reasoningEfforts.includes(config.llm_reasoning_effort)) throw new Error('所选模型不支持该思考强度')
      }
    }
  }
  private async launch(detail: RunDetail, connection: StoredConnection, resume: boolean): Promise<RunSummary> {
    const id = detail.summary.id, lease = join(this.root, '.dsh-gb1', 'leases', `${id}.json`)
    const pausedDetail = resume ? structuredClone(detail) : null
    let conversation: ConversationLog | undefined, leaseOwned = false, spawned = false, launchedJob: ActiveJob | undefined
    try {
      await writeFile(lease, JSON.stringify({ id }), { flag: 'wx' }); leaseOwned = true
      detail.summary.status = 'running'; detail.summary.stage = resume ? 'resuming' : 'starting'; detail.summary.error = null
      conversation = this.conversations.get(id) ?? await ConversationLog.open(this.root, detail, !resume, { redactions: [connection.baseUrl, connection.credentialBaseUrl] })
      this.conversations.set(id, conversation)
      await conversation.finish('running')
      const env: NodeJS.ProcessEnv = { PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONNOUSERSITE: '1' }
      for (const name of ['SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'PATH']) if (process.env[name]) env[name] = process.env[name]
      const child = this.spawnWorker(join(this.root, '.venv', 'Scripts', 'python.exe'), ['-u', WORKER_SCRIPT, '--project-root', this.root, '--run-id', id],
        { cwd: this.root, env, shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
      spawned = true
      let finish = () => {}; const done = new Promise<void>(resolveDone => { finish = resolveDone })
      const job: ActiveJob = { detail, child, controller: new AbortController(), buffer: Buffer.alloc(0), pending: false, receivedResult: false,
        callsByRound: new Map(detail.roundResults.map(row => [row.round, row.calls])), writes: Promise.resolve(), frames: Promise.resolve(), done, finish, connection, conversation,
        tools: new Map(), analyses: new Map(), sessionCalls: 0, pauseRequested: false, lastCheckpoint: detail.summary.completedRounds, requestFailed: false }
      launchedJob = job; this.active.set(id, job)
      child.stdout.on('data', (chunk: Buffer) => this.receive(job, chunk))
      child.stderr.on('data', () => {})
      child.stdin.on('error', () => this.fail(job, 'Worker 输入通道已关闭'))
      child.on('error', () => this.fail(job, '无法启动项目 Python worker'))
      child.once('close', (code: number | null) => { void this.closed(job, code) })
      this.send(job, { version: 1, type: 'start', reference: REFERENCE, seed: detail.summary.seed, rounds: detail.summary.rounds, method: detail.summary.method,
        model: detail.summary.model, config: detail.summary.config, prompt_profile: detail.summary.promptProfile, repeat_index: detail.summary.repeatIndex, resume,
        features: ['process-events-v1', 'round-control-v1'] })
      this.save(job); this.notify()
      return structuredClone(detail.summary)
    } catch {
      const recoverable = pausedDetail !== null && !spawned
      if (recoverable) {
        Object.assign(detail, pausedDetail)
        detail.summary.error = '恢复进程尚未启动，已保留暂停状态。'
      } else { detail.summary.status = 'failed'; detail.summary.error = '无法启动独立运行' }
      if (launchedJob) { launchedJob.controller.abort(); launchedJob.child.kill() }
      await conversation?.finish(detail.summary.status).catch(() => {})
      await saveJson(join(this.jobDirectory, `${id}.json`), detail)
      if (leaseOwned) await unlink(lease).catch(() => {})
      throw new Error(recoverable ? '恢复进程尚未启动，可以重新恢复原运行。' : '无法启动独立运行；请使用原操作编号检查结果')
    }
  }
  private save(job: ActiveJob): void {
    job.detail.summary.updatedAt = new Date().toISOString()
    job.detail.events = job.detail.events.slice(-100)
    const copy = structuredClone(job.detail)
    job.writes = job.writes.then(() => saveJson(join(this.jobDirectory, `${copy.summary.id}.json`), copy))
    job.writes.catch(() => { if (!job.controller.signal.aborted) this.fail(job, '无法保存 Host 运行状态') })
  }
  private send(job: ActiveJob, frame: unknown): void {
    const line = JSON.stringify(frame) + '\n'
    if (Buffer.byteLength(line) > MAX_FRAME_BYTES) throw new Error('Worker 输入帧超过允许大小')
    job.child.stdin.write(line)
  }
  private receive(job: ActiveJob, chunk: Buffer): void {
    if (job.controller.signal.aborted) return
    try {
      job.buffer = Buffer.concat([job.buffer, chunk])
      for (;;) {
        const newline = job.buffer.indexOf(10)
        if (newline < 0) break
        if (newline + 1 > MAX_FRAME_BYTES) throw new Error('Worker 输出帧超过允许大小')
        const line = job.buffer.subarray(0, newline).toString('utf8'); job.buffer = job.buffer.subarray(newline + 1)
        const frame = record(JSON.parse(line))
        job.frames = job.frames.then(async () => { if (!job.controller.signal.aborted) await this.frame(job, frame) })
        job.frames.catch(() => this.fail(job, 'Worker 输出协议无效'))
      }
      if (job.buffer.length >= MAX_FRAME_BYTES) throw new Error('Worker 输出帧超过允许大小')
    } catch { this.fail(job, 'Worker 输出协议无效') }
  }
  private async frame(job: ActiveJob, frame: Record<string, unknown>): Promise<void> {
    if (frame.version !== 1 || job.receivedResult) throw new Error('Worker 协议版本或状态无效')
    const config = job.detail.summary.config!
    if (frame.type === 'progress') {
      if (!['prepare', 'train', 'run'].includes(String(frame.stage))) throw new Error('Worker 阶段无效')
      job.detail.summary.stage = String(frame.stage)
      const message = frame.stage === 'prepare' ? '整理数据并建立本次划分' : frame.stage === 'train' ? '训练并选择评分模型' : '开始候选生成与反馈'
      job.detail.events.push({ time: new Date().toISOString(), stage: String(frame.stage), message })
      job.conversation.upsert({ id: `phase-${frame.stage}`, actor: 'coordinator', kind: 'status', status: 'completed', round: null, title: message, text: '', toolName: null, inputSummary: '', outputSummary: '' })
      this.save(job)
    } else if (frame.type === 'checkpoint') {
      const round = integer(frame.round)
      if (job.pending || round !== job.lastCheckpoint + 1 || round > job.detail.summary.rounds) throw new Error('Worker 检查点无效')
      job.lastCheckpoint = round; job.detail.summary.completedRounds = round
      this.save(job)
      await job.writes
      this.send(job, { version: 1, type: 'round_control', round, action: job.pauseRequested ? 'pause' : 'continue' })
    } else if (frame.type === 'llm_request') {
      if (job.pending || job.requestFailed || frame.id !== `request-${job.sessionCalls + 1}`) throw new Error('Worker 调用编号或并发状态无效')
      const round = integer(frame.round), used = job.callsByRound.get(round) ?? 0
      if (!['llm', 'knowledge'].includes(job.detail.summary.method) || round !== job.lastCheckpoint + 1 || round > job.detail.summary.rounds || used >= config.llm_max_calls_per_round
        || config.max_model_calls !== null && job.detail.summary.requestCount >= config.max_model_calls
        || config.max_reported_tokens !== null && (job.detail.usage.missingUsageCount > 0 || (job.detail.usage.totalTokens ?? 0) >= config.max_reported_tokens)) throw new Error('Worker 超过模型调用预算')
      if (record(frame.prompt_metadata).profile !== job.detail.summary.promptProfile) throw new Error('Worker 提示词与运行不一致')
      job.callsByRound.set(round, used + 1); job.detail.summary.requestCount++; job.sessionCalls++; job.pending = true
      const request: WorkerRequest = { id: String(frame.id), round, payload: record(frame.payload), prompt_metadata: record(frame.prompt_metadata) }
      job.detail.summary.stage = `round-${round}`
      job.detail.events.push({ time: new Date().toISOString(), stage: `round-${round}`, message: `第${round}轮：第${used + 1}次生成请求` })
      const messageId = `request-${job.detail.summary.requestCount}`
      job.conversation.upsert({ id: `request-${messageId}`, actor: 'coordinator', kind: 'status', status: 'completed', round,
        title: used ? '纠正候选请求' : '候选生成请求', text: used ? '根据上一批候选的检查结果补充方案。' : '根据当前已知结果提出候选，随后进行检查和评分。', toolName: null, inputSummary: `第${round}轮；第${used + 1}次请求`, outputSummary: '' })
      job.conversation.upsert({ id: `model-${messageId}`, actor: 'model', kind: 'analysis', status: 'running', round, title: '公开分析', text: '', toolName: null, inputSummary: '', outputSummary: '' })
      this.save(job)
      // 请求计数先保存，再派发网络调用；Host重新启动时不会自动恢复未知请求。
      await job.writes
      if (!job.controller.signal.aborted) void this.generate(job, request, messageId)
    } else if (frame.type === 'process_event') this.processEvent(job, record(frame.event))
    else if (frame.type === 'result') {
      if (job.pending || !['completed', 'failed', 'paused', 'stopped'].includes(String(frame.status))) throw new Error('Worker 终态无效')
      const result = record(frame.summary_file === true ? await readJson(join(this.root, 'results', 'dsh-plugin', job.detail.summary.id, 'public-summary.json')) : frame.summary)
      const summary = job.detail.summary, model = record(result.model)
      if (result.run_id !== summary.id || result.seed !== summary.seed || result.method !== summary.method || result.requested_rounds !== summary.rounds
        || model.source !== summary.model.source || model.provider !== summary.model.provider || model.id !== summary.model.id || result.calls !== summary.requestCount
        || !Array.isArray(result.rounds) || result.rounds.length > summary.rounds) throw new Error('Worker 终态与本次运行不一致')
      const rounds = result.rounds.map(value => roundDto(value, config))
      if (new Set(rounds.map(row => row.round)).size !== rounds.length || rounds.some((row, i) => row.round !== i + 1)) throw new Error('Worker 轮次编号无效')
      const completed = integer(result.completed_rounds), evaluated = rounds.filter(row => row.top10.length === config.k && row.validCount === row.scoredCount && row.scoredCount >= config.k && row.top10.every(item => item.actualFitness !== null))
      if (completed > summary.rounds || evaluated.length !== completed || frame.status === 'completed' && completed !== summary.rounds
        || frame.status === 'paused' && (!job.pauseRequested || job.requestFailed || completed !== job.lastCheckpoint)
        || frame.status === 'stopped' && (typeof result.stop_reason !== 'string' || !result.stop_reason)) throw new Error('Worker 完成轮次不完整')
      job.detail.initialKnownCount = integer(result.initial_known_count); job.detail.knownCount = integer(result.known_count)
      job.detail.roundResults = rounds; job.detail.usage = usageDto(result.usage)
      summary.completedRounds = completed; summary.status = frame.status as RunSummary['status']; summary.stage = frame.status === 'paused' ? 'paused' : 'finished'
      summary.stopReason = result.stop_reason == null ? null : text(result.stop_reason, 500)
      summary.error = frame.status === 'failed' ? '实验未完整完成，请查看该运行的独立记录。' : null
      job.receivedResult = true; this.save(job)
    } else if (frame.type === 'error') this.fail(job, 'Python worker 未能完成运行')
    else throw new Error('未知 Worker 消息')
  }
  private processEvent(job: ActiveJob, event: Record<string, unknown>): void {
    const tools: Record<string, string> = { prepare: '整理实验数据', model_selection: '选择评分模型', fit: '训练评分模型', validate: '校验候选', predict: '预测候选分数', select: '选择推荐', oracle: '查询所选变体测量', feedback: '更新已知结果' }
    const id = text(event.id, 100), tool = text(event.tool, 40), round = integer(event.round)
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$/.test(id) || !Object.hasOwn(tools, tool) || round > job.detail.summary.rounds || !['tool_start', 'tool_end'].includes(String(event.kind)) || (event.ok !== undefined && typeof event.ok !== 'boolean')) throw new Error('工具过程消息无效')
    const counts = record(event.counts), labels: Record<string, string> = { known_count: '已知记录', candidate_count: '候选数量', requested_count: '请求数量', valid_count: '有效候选', rejected_count: '未通过候选', scored_count: '已评分', selected_count: '已选推荐', feedback_count: '新增反馈', validation_count: '验证样本', configurations: '模型配置' }
    const summary = Object.keys(labels).filter(key => counts[key] !== undefined).map(key => `${labels[key]}：${integer(counts[key])}`).join('；')
    const ended = event.kind === 'tool_end', previous = job.tools.get(id)
    if ((!ended && previous) || (ended && (!previous || previous.ended || previous.tool !== tool || previous.round !== round))) throw new Error('工具过程开始与结束不匹配')
    job.tools.set(id, { input: previous?.input ?? summary, tool, round, ended })
    job.conversation.upsert({ id: `tool-${id}`, actor: 'tool', kind: 'tool', status: ended ? event.ok === false ? 'failed' : 'completed' : 'running', round: round || null, title: tools[tool]!, text: '', toolName: tool, inputSummary: previous?.input ?? summary, outputSummary: ended ? summary : '' })
  }
  private async generate(job: ActiveJob, request: WorkerRequest, messageId: string): Promise<void> {
    try {
      const response = await this.gateway.generate(job.connection, request, job.controller.signal, progress => {
        if (job.controller.signal.aborted || progress.requestId !== request.id || progress.round !== request.round || progress.kind !== 'analysis') return
        job.analyses.set(request.id, progress.text)
        job.conversation.upsert({ id: `model-${messageId}`, actor: 'model', kind: 'analysis', status: progress.status === 'completed' ? 'completed' : 'running', round: request.round, title: '公开分析', text: progress.text, toolName: null, inputSummary: '', outputSummary: '' })
      }, job.detail.summary.config)
      if (job.controller.signal.aborted) return
      job.pending = false
      const usage = response.usage, total = usage?.total_tokens
      if (typeof total === 'number') job.detail.usage.totalTokens = (job.detail.usage.totalTokens ?? 0) + total
      if (!usage || usage.missing_usage) job.detail.usage.missingUsageCount++
      job.conversation.upsert({ id: `model-${messageId}`, actor: 'model', kind: 'analysis', status: 'completed', round: request.round, title: '公开分析', text: job.analyses.get(request.id) ?? '', toolName: null, inputSummary: '', outputSummary: '模型输出已接收' })
      this.save(job); await job.writes
      if (!job.controller.signal.aborted) this.send(job, { version: 1, type: 'llm_response', id: request.id, response })
    } catch {
      if (job.controller.signal.aborted) return
      job.pending = false; job.requestFailed = true; job.detail.usage.missingUsageCount++
      job.conversation.upsert({ id: `model-${messageId}`, actor: 'model', kind: 'analysis', status: 'failed', round: request.round, title: '公开分析', text: job.analyses.get(request.id) ?? '', toolName: null, inputSummary: '', outputSummary: '请求失败或结果未知' })
      try { this.send(job, { version: 1, type: 'llm_error', id: request.id, error: { code: 'MODEL_REQUEST_FAILED', message: '模型请求失败或结果未知；未自动重试。' } }) }
      catch { this.fail(job, '无法返回模型调用结果') }
    }
  }
  private fail(job: ActiveJob, message: string): void {
    if (job.controller.signal.aborted) return
    job.detail.summary.status = 'failed'; job.detail.summary.error = message; job.controller.abort(); job.child.kill(); this.save(job)
  }
  private async closed(job: ActiveJob, code: number | null): Promise<void> {
    await job.frames.catch(() => {})
    if (job.detail.summary.status !== 'cancelled' && (!job.receivedResult || code !== 0 || job.buffer.length !== 0)) { job.detail.summary.status = 'failed'; job.detail.summary.error ||= 'Worker 退出时未返回完整结果' }
    job.controller.abort(); job.detail.summary.stage = job.detail.summary.status === 'paused' ? 'paused' : 'finished'; this.save(job)
    const terminal = job.detail.summary.status
    job.conversation.upsert({ id: `run-terminal-${job.detail.summary.requestCount}-${terminal}`, actor: 'coordinator', kind: terminal === 'failed' ? 'error' : 'status', status: terminal, round: null,
      title: terminal === 'completed' ? '运行已完成' : terminal === 'paused' ? '运行已暂停' : terminal === 'stopped' ? '运行已按条件停止' : terminal === 'cancelled' ? '运行已取消' : '运行未完成', text: job.detail.summary.error ?? job.detail.summary.stopReason ?? '', toolName: null, inputSummary: '', outputSummary: `已完成${job.detail.summary.completedRounds}轮` })
    await job.conversation.finish(terminal).catch(() => {})
    try { await job.writes } catch {}
    await unlink(join(this.root, '.dsh-gb1', 'leases', `${job.detail.summary.id}.json`)).catch(() => {})
    this.active.delete(job.detail.summary.id); job.finish(); this.notify()
  }
  async pauseRun(id: string): Promise<RunSummary> {
    const job = this.active.get(id)
    if (!job) return (await this.fullRun(id)).summary
    job.pauseRequested = true; job.detail.summary.stage = 'pausing'; this.save(job)
    return structuredClone(job.detail.summary)
  }
  resumeRun(id: string): Promise<RunSummary> {
    return this.exclusive(async () => {
      if (this.disposed) throw new Error('工作台已停止')
      if (this.active.has(id)) return structuredClone(this.active.get(id)!.detail.summary)
      const detail = await this.fullRun(id)
      if (detail.summary.status !== 'paused' || detail.summary.formatVersion !== 2) throw new Error('仅可恢复已确认在轮次边界暂停的运行')
      const connection = record(await readJson(join(this.root, '.dsh-gb1', 'frozen', `${id}.json`))) as unknown as StoredConnection
      if (connection.source !== detail.summary.model.source || connection.provider !== detail.summary.model.provider || connection.model !== detail.summary.model.id) throw new Error('原运行连接记录无效')
      return this.launch(detail, connection, true)
    })
  }
  async cancelRun(id: string): Promise<RunSummary> {
    if (!SAFE_RUN.test(id)) throw new Error('运行 ID 无效')
    const job = this.active.get(id)
    if (!job) {
      const detail = await this.fullRun(id)
      if (detail.summary.status === 'paused') { detail.summary.status = 'cancelled'; detail.summary.stage = 'finished'; detail.summary.updatedAt = new Date().toISOString(); await saveJson(join(this.jobDirectory, `${id}.json`), detail); this.notify() }
      return detail.summary
    }
    await job.frames.catch(() => {})
    if (TERMINAL.has(job.detail.summary.status)) return structuredClone(job.detail.summary)
    job.detail.summary.status = 'cancelled'; job.detail.summary.error = null; job.detail.summary.stage = 'cancelling'
    job.controller.abort(); job.child.kill(); this.save(job)
    return structuredClone(job.detail.summary)
  }
  async dispose(): Promise<void> {
    this.disposed = true; await this.serial.catch(() => {})
    const jobs = [...this.active.values()]
    await Promise.all(jobs.map(async job => {
      await this.cancelRun(job.detail.summary.id)
      let timer: NodeJS.Timeout | undefined
      try { await Promise.race([job.done, new Promise<void>(resolveDone => { timer = setTimeout(() => { job.child.kill('SIGKILL'); resolveDone() }, 5000); timer.unref() })]) }
      finally { if (timer) clearTimeout(timer) }
    }))
  }
}
