import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { ConversationMessage, ConversationSnapshot, RunDetail, RunStatus } from './contracts.ts'

export const MAX_CONVERSATION_MESSAGES = 160
export const MAX_CONVERSATION_BYTES = 512 * 1024
export const CONVERSATION_INTERVAL_MS = 150
const RUN_ID = /^[a-z0-9][a-z0-9_-]{0,95}$/
const MESSAGE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$/
const RUN_STATUSES: readonly RunStatus[] = ['queued', 'running', 'paused', 'stopped', 'completed', 'failed', 'cancelled', 'interrupted']
const TOOLS = ['prepare', 'model_selection', 'fit', 'validate', 'predict', 'select', 'oracle', 'feedback']
export type ConversationUpsert = Omit<ConversationMessage, 'order' | 'time'> & { time?: string }
interface Subscriber { latest: ConversationSnapshot | null; wake: (() => void) | null }
interface Options {
  intervalMs?: number
  persist?: (path: string, snapshot: ConversationSnapshot) => Promise<void>
  redactions?: readonly string[]
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('过程记录格式无效')
  return value as Record<string, unknown>
}
function publicText(value: unknown, maximum: number, redactions: readonly string[]): { text: string; truncated: boolean } {
  if (typeof value !== 'string') throw new Error('过程记录文本格式无效')
  let text = value.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/g, '')
  for (const secret of redactions.filter(Boolean)) {
    text = text.replaceAll(secret, '[redacted]').replaceAll(encodeURIComponent(secret), '[redacted]')
  }
  return { text: text.slice(0, maximum), truncated: text.length > maximum }
}
function cleanMessage(raw: ConversationUpsert, order: number, time: string, redactions: readonly string[]): { message: ConversationMessage; truncated: boolean } {
  if (!MESSAGE_ID.test(raw.id) || !Number.isSafeInteger(order) || order < 1 || !Number.isFinite(Date.parse(time))
    || time.length > 40 || (raw.round !== null && (!Number.isSafeInteger(raw.round) || raw.round < 1))
    || !['coordinator', 'model', 'tool'].includes(raw.actor) || !['status', 'analysis', 'tool', 'error', 'legacy'].includes(raw.kind)
    || !RUN_STATUSES.includes(raw.status)) throw new Error('过程消息字段无效')
  if ((raw.actor === 'tool' && (raw.kind !== 'tool' || !TOOLS.includes(raw.toolName ?? '')))
    || (raw.actor !== 'tool' && raw.toolName !== null)
    || (raw.actor === 'model' && !['analysis', 'legacy'].includes(raw.kind))
    || (raw.actor === 'coordinator' && !['status', 'error', 'legacy'].includes(raw.kind))) throw new Error('过程消息角色无效')
  const title = publicText(raw.title, 160, redactions), text = publicText(raw.text, 6000, redactions)
  const input = publicText(raw.inputSummary, 1200, redactions), output = publicText(raw.outputSummary, 1200, redactions)
  return { message: { id: raw.id, order, time, round: raw.round, actor: raw.actor, kind: raw.kind, status: raw.status,
    title: title.text, text: text.text, toolName: raw.toolName, inputSummary: input.text, outputSummary: output.text },
    truncated: title.truncated || text.truncated || input.truncated || output.truncated }
}
async function atomicSnapshot(path: string, snapshot: ConversationSnapshot): Promise<void> {
  const temporary = `${path}.${randomUUID()}.tmp`
  await writeFile(temporary, JSON.stringify(snapshot), { flag: 'wx' })
  await rename(temporary, path)
}

/** 每次更新保存完整消息，先提交当前快照，再通知观察者。 */
export class ConversationLog {
  private committed: ConversationSnapshot
  private messages = new Map<string, ConversationMessage>()
  private nextOrder = 1
  private draftStatus: RunStatus
  private truncated: boolean
  private dirty = false
  private timer: NodeJS.Timeout | undefined
  private writes: Promise<void> = Promise.resolve()
  private failure: Error | null = null
  private subscribers = new Set<Subscriber>()
  private readonly intervalMs: number
  private readonly persist: NonNullable<Options['persist']>
  private readonly redactions: readonly string[]
  private constructor(readonly path: string, initial: ConversationSnapshot, options: Options) {
    this.committed = initial
    this.draftStatus = initial.status
    this.truncated = initial.truncated
    for (const message of initial.messages) this.messages.set(message.id, structuredClone(message))
    this.nextOrder = Math.max(0, ...initial.messages.map(message => message.order)) + 1
    this.intervalMs = options.intervalMs ?? CONVERSATION_INTERVAL_MS
    this.persist = options.persist ?? atomicSnapshot
    this.redactions = options.redactions ?? []
  }
  static async open(root: string, detail: RunDetail, newRun = false, options: Options = {}): Promise<ConversationLog> {
    const id = detail.summary.id
    if (!RUN_ID.test(id)) throw new Error('运行 ID 无效')
    const directory = join(root, '.dsh-gb1', 'conversations')
    await mkdir(directory, { recursive: true })
    const path = join(directory, `${id}.json`)
    let restored: ConversationSnapshot | null = null
    try {
      if ((await stat(path)).size > MAX_CONVERSATION_BYTES) throw new Error('过程记录超过允许大小')
      const raw = object(JSON.parse(await readFile(path, 'utf8')))
      if (raw.runId !== id || !Number.isSafeInteger(raw.revision) || Number(raw.revision) < 1
        || !RUN_STATUSES.includes(raw.status as RunStatus) || typeof raw.historyOnly !== 'boolean'
        || typeof raw.truncated !== 'boolean' || !Array.isArray(raw.messages)) throw new Error('过程快照格式无效')
      const cleaned = raw.messages.slice(-MAX_CONVERSATION_MESSAGES).map(value => {
        const row = object(value)
        return cleanMessage(row as unknown as ConversationUpsert, Number(row.order), String(row.time), options.redactions ?? [])
      })
      const messages = cleaned.map(item => item.message)
      if (new Set(messages.map(message => message.id)).size !== messages.length
        || messages.some((message, index) => index > 0 && message.order <= messages[index - 1]!.order)) throw new Error('过程消息顺序无效')
      restored = { runId: id, revision: Number(raw.revision), status: raw.status as RunStatus, historyOnly: raw.historyOnly,
        truncated: raw.truncated || raw.messages.length > messages.length || cleaned.some(item => item.truncated), messages }
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new Error('无法读取运行过程记录') }
    const log = new ConversationLog(path, restored ?? { runId: id, revision: 0, status: detail.summary.status,
      historyOnly: !newRun, truncated: false, messages: [] }, options)
    if (!restored) {
      if (newRun) log.upsert({ id: 'run-start', actor: 'coordinator', kind: 'status', status: 'completed', round: null,
        title: '运行已启动', text: '', toolName: null, inputSummary: `种子：${detail.summary.seed}；计划：${detail.summary.rounds}轮；每轮最多${detail.summary.config?.proposal_count ?? 40}个候选，推荐${detail.summary.config?.k ?? 10}条`, outputSummary: '' })
      else log.legacy(detail)
      await log.flush()
    } else if (restored.status === 'running' && detail.summary.status !== 'running') {
      log.upsert({ id: 'host-recovery', actor: 'coordinator', kind: 'status', status: detail.summary.status, round: null,
        title: '已恢复保存的过程记录', text: '显示最后保存的运行状态。', toolName: null, inputSummary: '', outputSummary: '' })
      await log.finish(detail.summary.status)
    }
    return log
  }
  private legacy(detail: RunDetail): void {
    this.upsert({ id: 'legacy-summary', actor: 'coordinator', kind: 'legacy', status: 'completed', round: null,
      time: detail.summary.createdAt, title: '历史摘要', text: '此运行仅保留已保存的阶段和轮次摘要。', toolName: null, inputSummary: '', outputSummary: '' })
    detail.events.forEach((event, index) => this.upsert({ id: `legacy-stage-${index + 1}`, actor: 'coordinator', kind: 'legacy',
      status: 'completed', round: null, time: event.time, title: event.stage, text: event.message, toolName: null, inputSummary: '', outputSummary: '' }))
    for (const round of detail.roundResults) {
      round.analysis.forEach((analysis, index) => this.upsert({ id: `legacy-analysis-${round.round}-${index + 1}`,
        actor: 'model', kind: 'legacy', status: 'completed', round: round.round, time: detail.summary.updatedAt,
        title: '历史公开分析', text: analysis, toolName: null, inputSummary: '', outputSummary: '' }))
    }
  }
  upsert(update: ConversationUpsert): void {
    if (this.failure) return
    const previous = this.messages.get(update.id)
    if (previous && (previous.actor !== update.actor || previous.kind !== update.kind || previous.round !== update.round || previous.toolName !== update.toolName)) throw new Error('同一过程消息的身份不可改变')
    const { message, truncated } = cleanMessage(update, previous?.order ?? this.nextOrder,
      previous?.time ?? update.time ?? new Date().toISOString(), this.redactions)
    if (!previous) this.nextOrder++
    this.messages.set(message.id, message)
    if (this.messages.size > MAX_CONVERSATION_MESSAGES) { this.messages.delete(this.messages.keys().next().value!); this.truncated = true }
    this.truncated ||= truncated
    this.dirty = true
    this.schedule()
  }
  private schedule(): void {
    if (this.timer || this.failure) return
    this.timer = setTimeout(() => { this.timer = undefined; void this.flush().catch(() => {}) }, this.intervalMs)
    this.timer.unref()
  }
  private boundedSnapshot(): ConversationSnapshot {
    let messages = [...this.messages.values()].sort((left, right) => left.order - right.order)
    let truncated = this.truncated
    if (messages.length > MAX_CONVERSATION_MESSAGES) { messages = messages.slice(-MAX_CONVERSATION_MESSAGES); truncated = true }
    const snapshot: ConversationSnapshot = { runId: this.committed.runId, revision: this.committed.revision + 1,
      status: this.draftStatus, historyOnly: this.committed.historyOnly, truncated, messages: structuredClone(messages) }
    while (Buffer.byteLength(JSON.stringify(snapshot)) > MAX_CONVERSATION_BYTES && snapshot.messages.length > 1) {
      snapshot.messages.shift(); snapshot.truncated = true
    }
    this.messages = new Map(snapshot.messages.map(message => [message.id, structuredClone(message)]))
    this.truncated = snapshot.truncated
    return snapshot
  }
  async flush(): Promise<void> {
    if (this.timer) { clearTimeout(this.timer); this.timer = undefined }
    const task = this.writes.then(async () => {
      if (this.failure) throw this.failure
      if (!this.dirty) return
      const snapshot = this.boundedSnapshot()
      this.dirty = false
      try { await this.persist(this.path, snapshot) }
      catch {
        this.failure = new Error('无法保存运行过程记录')
        for (const subscriber of this.subscribers) subscriber.wake?.()
        throw this.failure
      }
      this.committed = snapshot
      for (const subscriber of this.subscribers) { subscriber.latest = snapshot; subscriber.wake?.() }
    })
    this.writes = task.catch(() => {})
    await task
    if (this.dirty) this.schedule()
  }
  async finish(status: RunStatus): Promise<void> {
    if (this.failure) return
    this.draftStatus = status
    for (const [id, message] of this.messages) {
      if (message.status === 'running') this.messages.set(id, { ...message, status: status === 'cancelled' ? 'cancelled' : 'interrupted' })
    }
    this.dirty = true
    await this.flush()
  }
  async *watch(signal: AbortSignal): AsyncIterable<ConversationSnapshot> {
    if (signal.aborted) return
    const subscriber: Subscriber = { latest: null, wake: null }
    // 先加入监听，再读取当前快照，避免恢复与下一次提交之间遗漏一次更新。
    this.subscribers.add(subscriber)
    subscriber.latest = this.committed
    const abort = () => subscriber.wake?.()
    signal.addEventListener('abort', abort, { once: true })
    let revision = -1
    try {
      while (!signal.aborted) {
        if (this.failure) throw this.failure
        if (!subscriber.latest) {
          await new Promise<void>(resolveWake => { subscriber.wake = resolveWake })
          subscriber.wake = null
          continue
        }
        const snapshot = subscriber.latest
        subscriber.latest = null
        if (snapshot.revision <= revision) continue
        revision = snapshot.revision
        yield structuredClone(snapshot)
        if (snapshot.status !== 'running') return
      }
    } finally {
      signal.removeEventListener('abort', abort)
      this.subscribers.delete(subscriber)
    }
  }
}
