import { randomUUID } from 'node:crypto'
import { mkdir, readdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { BatchDetail, BatchSummary, StartBatchInput } from './contracts.ts'
import { record } from './model.ts'
import { readJson, REFERENCE, RunManager, SAFE_OPERATION, saveJson } from './runs.ts'
import { modelOverride, promptProfile, validSeed, validateConfig } from './settings.ts'
const SAFE_BATCH = /^batch-[a-f0-9-]{36}$/
const FINISHED = new Set(['completed', 'stopped', 'failed', 'cancelled', 'interrupted'])
export function validateBatch(value: StartBatchInput): StartBatchInput {
  if (!SAFE_OPERATION.test(value.operationId)) throw new Error('批量操作编号无效')
  const baseConfig = validateConfig(value.baseConfig)
  if (!Array.isArray(value.methods) || !value.methods.length || value.methods.some(x => !['random', 'model', 'llm', 'knowledge'].includes(x))
    || new Set(value.methods).size !== value.methods.length) throw new Error('批量方法无效')
  if (!Array.isArray(value.seeds) || !value.seeds.length || value.seeds.some(x => !validSeed(x)) || new Set(value.seeds).size !== value.seeds.length) throw new Error('批量种子无效')
  if (!Array.isArray(value.targets) || !value.targets.length) throw new Error('请选择批量模型')
  const targets = value.targets.map(target => {
    const model = modelOverride(target)
    const effort = target.reasoningEffort
    if (effort !== null && (typeof effort !== 'string' || !effort.trim() || effort.length > 200)) throw new Error('批量思考强度无效')
    return { ...model, reasoningEffort: effort }
  })
  for (const field of ['repeats', 'concurrency', 'perProviderConcurrency'] as const) if (!Number.isSafeInteger(value[field]) || value[field] < 1) throw new Error(`${field}必须为正整数`)
  const total = value.methods.length * value.seeds.length * targets.length * value.repeats
  if (!Number.isSafeInteger(total) || total > 10000) throw new Error('单批最多10000个独立运行')
  return { operationId: value.operationId, baseConfig, promptProfile: promptProfile(value.promptProfile), methods: [...value.methods], targets, seeds: [...value.seeds], repeats: value.repeats, concurrency: value.concurrency, perProviderConcurrency: value.perProviderConcurrency }
}
/** 批量队列只派发保存的操作编号，恢复不会重发未知调用。 */
export class BatchManager {
  private batches = new Map<string, BatchDetail>()
  private serial: Promise<unknown> = Promise.resolve()
  private disposed = false
  private readonly resumeAuthorized = new Set<string>()
  private unsubscribe: () => void
  constructor(readonly root: string, readonly runs: RunManager) { this.unsubscribe = runs.onChange(() => this.schedule()) }
  get hasRunnableWork(): boolean { return [...this.batches.values()].some(detail => ['queued', 'running'].includes(detail.summary.status)) }
  private exclusive<T>(action: () => Promise<T>): Promise<T> { const next = this.serial.then(action); this.serial = next.catch(() => {}); return next }
  private path(id: string): string { if (!SAFE_BATCH.test(id)) throw new Error('批量任务ID无效'); return join(this.root, '.dsh-gb1', 'batches', `${id}.json`) }
  async initialize(): Promise<void> {
    await mkdir(join(this.root, '.dsh-gb1', 'batches'), { recursive: true })
    for (const name of await readdir(join(this.root, '.dsh-gb1', 'batches'))) {
      if (!name.endsWith('.json') || !SAFE_BATCH.test(name.slice(0, -5))) continue
      try {
        const raw = record(await readJson(this.path(name.slice(0, -5))))
        const input = validateBatch(raw.input as StartBatchInput)
        const summary = record(raw.summary)
        if (summary.id !== name.slice(0, -5) || summary.operationId !== input.operationId || !Array.isArray(raw.runs)
          || !['queued', 'running', 'paused', 'completed', 'cancelled', 'interrupted'].includes(String(summary.status))
          || typeof summary.createdAt !== 'string' || !Number.isFinite(Date.parse(summary.createdAt))) continue
        const rows = raw.runs.map(value => {
          const row = record(value), target = record(row.target)
          if (row.id !== null && (typeof row.id !== 'string' || !/^gb1-[a-f0-9-]{36}$/.test(row.id))
            || typeof row.operationId !== 'string' || !SAFE_OPERATION.test(row.operationId) || !input.seeds.includes(Number(row.seed))
            || !input.methods.includes(row.method as never) || !Number.isSafeInteger(row.repeatIndex) || Number(row.repeatIndex) < 0 || Number(row.repeatIndex) >= input.repeats
            || !['queued', 'running', 'paused', ...FINISHED].includes(String(row.status))
            || row.error !== null && (typeof row.error !== 'string' || row.error.length > 5000)) throw new Error('批量运行记录无效')
          const original = input.targets.find(item => item.provider === target.provider && item.model === target.model && item.reasoningEffort === target.reasoningEffort)
          if (!original) throw new Error('批量模型记录无效')
          return { id: row.id as string | null, operationId: row.operationId, seed: Number(row.seed), method: row.method as StartBatchInput['methods'][number],
            target: structuredClone(original), repeatIndex: Number(row.repeatIndex), status: row.status as BatchDetail['runs'][number]['status'], error: row.error as string | null }
        })
        const expected = input.targets.length * input.seeds.length * input.methods.length * input.repeats
        if (rows.length !== expected || new Set(rows.map(row => row.operationId)).size !== rows.length) continue
        const detail: BatchDetail = { summary: { id: String(summary.id), operationId: input.operationId, status: summary.status as BatchSummary['status'], createdAt: summary.createdAt,
          updatedAt: summary.createdAt, totalRuns: rows.length, completedRuns: 0, failedRuns: 0, activeRuns: 0, concurrency: input.concurrency, perProviderConcurrency: input.perProviderConcurrency }, input, runs: rows }
        this.summarize(detail)
        if (['running', 'queued'].includes(detail.summary.status)) detail.summary.status = 'interrupted'
        for (const row of detail.runs) if (row.status === 'running') row.status = 'interrupted'
        this.batches.set(detail.summary.id, detail)
      } catch { /* 损坏记录不进入队列。 */ }
    }
  }
  private summarize(detail: BatchDetail): void {
    const rows = detail.runs
    detail.summary.completedRuns = rows.filter(row => ['completed', 'stopped'].includes(row.status)).length
    detail.summary.failedRuns = rows.filter(row => ['failed', 'interrupted'].includes(row.status)).length
    detail.summary.activeRuns = rows.filter(row => row.status === 'running').length
    detail.summary.updatedAt = new Date().toISOString()
    if (!['paused', 'cancelled', 'interrupted'].includes(detail.summary.status)) {
      detail.summary.status = rows.every(row => FINISHED.has(row.status))
        ? rows.some(row => row.status === 'interrupted') ? 'interrupted' : 'completed'
        : rows.some(row => row.status === 'running') ? 'running'
          : rows.some(row => row.status === 'queued' || row.status === 'paused' && this.resumeAuthorized.has(row.operationId)) ? 'queued' : 'paused'
    }
  }
  private async refresh(detail: BatchDetail): Promise<void> {
    for (const row of detail.runs) if (row.id) {
      try { const run = (await this.runs.getRun(row.id)).summary; row.status = run.status; row.error = run.error }
      catch { row.status = 'interrupted'; row.error = '无法读取独立运行状态' }
    }
    this.summarize(detail)
  }
  private async persist(detail: BatchDetail): Promise<void> { await saveJson(this.path(detail.summary.id), detail) }
  async startBatch(raw: StartBatchInput): Promise<BatchDetail> {
    return this.exclusive(async () => {
      if (this.disposed) throw new Error('批量队列已停止')
      const input = validateBatch(raw)
      const existing = [...this.batches.values()].find(batch => batch.summary.operationId === input.operationId)
      if (existing) {
        if (JSON.stringify(existing.input) !== JSON.stringify(input)) throw new Error('该操作编号已用于其他批量参数')
        await this.refresh(existing); return structuredClone(existing)
      }
      // 能力检查属于只读准备，确保所有目标均能表达所选思考档位。
      for (const target of input.targets) {
        const capabilities = await this.runs.gateway.capabilities(target.provider, target.model)
        if (target.reasoningEffort !== null && !capabilities.reasoningEfforts.includes(target.reasoningEffort)) throw new Error('批量目标不支持指定思考强度')
      }
      const id = `batch-${randomUUID()}`, now = new Date().toISOString()
      const detail: BatchDetail = { summary: { id, operationId: input.operationId, status: 'queued', createdAt: now, updatedAt: now,
        totalRuns: 0, completedRuns: 0, failedRuns: 0, activeRuns: 0, concurrency: input.concurrency, perProviderConcurrency: input.perProviderConcurrency }, input, runs: [] }
      for (const target of input.targets) for (const seed of input.seeds) for (const method of input.methods) for (let repeatIndex = 0; repeatIndex < input.repeats; repeatIndex++) {
        detail.runs.push({ id: null, operationId: `batch-run-${randomUUID()}`, seed, method, target: structuredClone(target), repeatIndex, status: 'queued', error: null })
      }
      detail.summary.totalRuns = detail.runs.length
      // 与单次运行共用操作编号目录，防止跨接口复用操作编号。
      await writeFile(join(this.root, '.dsh-gb1', 'operations', `${input.operationId}.json`), JSON.stringify({ id, input, kind: 'batch' }), { flag: 'wx' })
      await this.persist(detail); this.batches.set(id, detail); this.schedule()
      return structuredClone(detail)
    })
  }
  private providerFor(id: string, fallback: string): string {
    for (const batch of this.batches.values()) { const row = batch.runs.find(row => row.id === id); if (row) return row.target.provider }
    return fallback
  }
  private schedule(): void {
    if (this.disposed) return
    void this.exclusive(async () => {
      if (this.disposed) return
      for (const detail of this.batches.values()) {
        await this.refresh(detail)
        if (!['queued', 'running'].includes(detail.summary.status)) { await this.persist(detail); continue }
        for (const row of detail.runs) {
          if (row.status !== 'queued' && !(row.status === 'paused' && this.resumeAuthorized.has(row.operationId))) continue
          const active = this.runs.activeSummaries
          if (active.length >= detail.input.concurrency) break
          if (active.filter(run => this.providerFor(run.id, run.model.provider) === row.target.provider).length >= detail.input.perProviderConcurrency) continue
          // 先保存派发状态；进程退出后不会自动重新派发该行。
          row.status = 'running'; await this.persist(detail)
          this.resumeAuthorized.delete(row.operationId)
          try {
            const config = validateConfig({ ...detail.input.baseConfig, seed: row.seed, llm_reasoning_effort: row.target.reasoningEffort })
            const run = row.id ? await this.runs.resumeRun(row.id) : await this.runs.startRun({ operationId: row.operationId, reference: REFERENCE, seed: row.seed, rounds: config.rounds, method: row.method,
              config, promptProfile: detail.input.promptProfile, repeatIndex: row.repeatIndex, modelOverride: { provider: row.target.provider, model: row.target.model } }, detail.summary.id)
            row.id = run.id; row.status = run.status; row.error = run.error
          } catch {
            row.status = 'failed'; row.error = '无法启动独立运行，请检查该操作编号的记录。'
            if (row.id) { try { row.status = (await this.runs.getRun(row.id)).summary.status } catch {} }
          }
          await this.persist(detail)
        }
        this.summarize(detail); await this.persist(detail)
      }
    }).catch(() => {})
  }
  async getBatch(id: string): Promise<BatchDetail> { return this.exclusive(async () => {
    const detail = this.batches.get(id); if (!detail) throw new Error('批量任务不存在')
    await this.refresh(detail); return structuredClone(detail)
  }) }
  async listBatches(): Promise<BatchSummary[]> { return this.exclusive(async () => {
    for (const detail of this.batches.values()) await this.refresh(detail)
    return [...this.batches.values()].map(detail => structuredClone(detail.summary)).sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 100)
  }) }
  async pauseBatch(id: string): Promise<BatchDetail> { return this.exclusive(async () => {
    const detail = this.batches.get(id); if (!detail) throw new Error('批量任务不存在')
    if (['completed', 'cancelled'].includes(detail.summary.status)) return structuredClone(detail)
    detail.summary.status = 'paused'
    for (const row of detail.runs) this.resumeAuthorized.delete(row.operationId)
    await this.persist(detail)
    for (const row of detail.runs) if (row.id && row.status === 'running') await this.runs.pauseRun(row.id)
    await this.refresh(detail); await this.persist(detail); return structuredClone(detail)
  }) }
  async resumeBatch(id: string): Promise<BatchDetail> { return this.exclusive(async () => {
    const detail = this.batches.get(id); if (!detail) throw new Error('批量任务不存在')
    await this.refresh(detail)
    if (['completed', 'cancelled'].includes(detail.summary.status)) return structuredClone(detail)
    // 显式批量恢复授权当前已暂停或正在等待轮次边界的运行。
    for (const row of detail.runs) if (row.id) {
      const summary = (await this.runs.getRun(row.id)).summary
      if (summary.status === 'paused' || summary.status === 'running' && summary.stage === 'pausing') this.resumeAuthorized.add(row.operationId)
    }
    detail.summary.status = 'queued'; this.summarize(detail)
    await this.persist(detail); this.schedule(); return structuredClone(detail)
  }) }
  async cancelBatch(id: string): Promise<BatchDetail> { return this.exclusive(async () => {
    const detail = this.batches.get(id); if (!detail) throw new Error('批量任务不存在')
    detail.summary.status = 'cancelled'
    for (const row of detail.runs) this.resumeAuthorized.delete(row.operationId)
    await this.persist(detail)
    for (const row of detail.runs) {
      if (row.id && ['running', 'paused'].includes(row.status)) row.status = (await this.runs.cancelRun(row.id)).status
      else if (row.status === 'queued') row.status = 'cancelled'
    }
    this.summarize(detail); await this.persist(detail); return structuredClone(detail)
  }) }
  async dispose(): Promise<void> { this.disposed = true; this.unsubscribe(); await this.serial.catch(() => {}) }
}
