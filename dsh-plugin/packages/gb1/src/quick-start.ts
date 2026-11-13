import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { homedir } from 'node:os'
import type { ExperimentConfiguration, QuickStartInput, QuickStartStatus, RunSummary, StartRunInput } from './contracts.ts'
import { identifier, ModelGateway, record } from './model.ts'
import { readJson, REFERENCE, SAFE_OPERATION, saveJson, type SpawnWorker } from './runs.ts'
import { loadDefaults, modelOverride, validateConfig } from './settings.ts'
const SAFE_ID = /^quick-[a-f0-9-]{36}$/
const PUBLIC_STATUS = ['preparing', 'running', 'completed', 'failed', 'cancelled', 'interrupted']
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted'])
const MAX_FRAME = 64 * 1024
const MESSAGES: Record<string, string> = {
  queued: '已提交，正在准备本次实验。', model: '检查所选模型与思考强度。', python: '检查可用的 Python 3.12。',
  venv: '准备项目独立 Python 环境。', dependencies: '检查并安装项目固定依赖。', download: '获取官方 GB1 S1 数据。',
  validate: '检查数据表、序列和实测分数。', ready: '环境与官方数据已准备好。', 'starting-run': '使用推荐参数启动三轮知识辅助实验。',
  running: '已启动实验，继续查看工作对话与结果。', completed: '三轮实验已完成。', cancelled: '本次准备或实验已取消。',
  interrupted: '此前操作结果尚未确认；仅恢复已保存状态，不会自动重发。', failed: '本次准备未完成。',
}
const BOOTSTRAP_CODES = ['PROJECT_PATH_INVALID', 'PROJECT_LAYOUT_INVALID', 'REPORT_PATH_INVALID', 'PREPARATION_BUSY', 'PYTHON_UNAVAILABLE', 'INVALID_EXISTING_ENV', 'VENV_CREATE_FAILED', 'REQUIREMENTS_INVALID', 'DEPENDENCIES_UNAVAILABLE_OFFLINE', 'DEPENDENCY_INSTALL_FAILED', 'DEPENDENCY_VALIDATION_FAILED', 'DATA_UNAVAILABLE_OFFLINE', 'DATA_DOWNLOAD_FAILED', 'DATA_VALIDATION_FAILED', 'INVALID_EXISTING_DATA', 'DATA_COMMIT_FAILED', 'DATA_AUDIT_INVALID', 'CANCELLED', 'PREPARATION_FAILED', 'INDEX_URL_INVALID']
const ERRORS: Record<string, string> = {
  PROJECT_NOT_READY: '项目缺少公开核心代码、准备脚本或固定依赖清单，请检查项目目录。',
  DEFAULTS_INVALID: '工作台默认配置无法读取，请检查项目公开配置。', MODEL_UNAVAILABLE: '推荐模型未注册或当前不可用，请选择已注册模型。',
  EFFORT_UNSUPPORTED: '所选模型不支持该思考强度，请明确选择支持的档位。', MODEL_NOT_SELECTED: '未配置推荐模型，请明确选择已注册模型。',
  PYTHON_NOT_FOUND: '未找到可用的 Python 3.12，请安装 Python 3.12 或设置准备解释器路径。',
  BOOTSTRAP_FAILED: '环境或官方数据准备失败，请检查 Python、网络以及项目依赖安装条件。',
  BOOTSTRAP_PROTOCOL: '准备进程没有返回有效的公开状态。', RUN_ACTIVE: '已有实验正在运行，请先等待它完成或取消。',
  RUN_START_FAILED: '准备已完成，但实验启动结果尚未确认；请查看本次任务和运行历史。',
  PROJECT_PATH_INVALID: '项目路径无效，请检查插件项目目录。', PROJECT_LAYOUT_INVALID: '项目缺少必要的公开源码或依赖清单。',
  REPORT_PATH_INVALID: '准备报告路径无效，请检查项目目录。', PREPARATION_BUSY: '另一个准备操作仍在使用项目环境，请稍后重试。',
  PYTHON_UNAVAILABLE: '未找到可用的 Python 3.12，请安装或配置解释器。', INVALID_EXISTING_ENV: '已有项目虚拟环境无效，已保留现有文件，请检查环境。',
  VENV_CREATE_FAILED: '无法创建项目虚拟环境，请检查 Python 和项目写入权限。', REQUIREMENTS_INVALID: '项目固定依赖清单无效。',
  DEPENDENCIES_UNAVAILABLE_OFFLINE: '离线模式下项目依赖尚未准备好。', DEPENDENCY_INSTALL_FAILED: '项目依赖安装失败，请检查网络；需要代理或镜像时请按机器要求配置后重试。',
  DEPENDENCY_VALIDATION_FAILED: '项目依赖版本或导入检查未通过。', DATA_UNAVAILABLE_OFFLINE: '离线模式下缺少通过校验的 GB1 S1 数据。',
  DATA_DOWNLOAD_FAILED: '官方 GB1 S1 下载失败，请检查网络后重试。', DATA_VALIDATION_FAILED: 'GB1 数据的表结构、序列或实测分数未通过检查。',
  INVALID_EXISTING_DATA: '已有 GB1 数据未通过检查，已保留原文件，请检查数据来源。', DATA_COMMIT_FAILED: '无法保存已校验的官方数据，请检查项目目录。',
  DATA_AUDIT_INVALID: '项目公开数据说明未通过检查。', PREPARATION_FAILED: '环境或数据准备失败，请查看准备步骤。', INDEX_URL_INVALID: '依赖安装源设置无效。',
  CANCEL_UNCONFIRMED: '取消请求的结果尚未确认，请查看本次运行的实际状态后重试取消。',
  CANCELLED: MESSAGES.cancelled!, HOST_INTERRUPTED: MESSAGES.interrupted!,
}
interface StoredTask { input: QuickStartInput; public: QuickStartStatus; runOperationId: string; runInput: StartRunInput | null }
interface ActivePreparation {
  task: StoredTask; controller: AbortController; child: ChildProcessWithoutNullStreams | null; cancelled: boolean
  writes: Promise<void>; done: Promise<void>; finish: () => void; cancelTimer?: NodeJS.Timeout
}
export interface QuickStartHooks {
  activeRunId(): string | null
  startRun(input: StartRunInput): Promise<RunSummary>
  getRun(id: string): Promise<RunSummary>
  cancelRun(id: string): Promise<RunSummary>
}
export interface QuickStartOptions { bootstrapPython?: string; spawnBootstrap?: SpawnWorker }
export function cleanPreparationEnvironment(): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONNOUSERSITE: '1', PYTHON_DOTENV_DISABLED: '1' }
  for (const key of ['SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'PATH']) if (process.env[key]) env[key] = process.env[key]
  return env
}
export function validateQuickStart(input: QuickStartInput): QuickStartInput {
  if (!input || !SAFE_OPERATION.test(input.operationId)) throw new Error('一键运行操作编号无效')
  const result: QuickStartInput = { operationId: input.operationId }
  if (input.model !== undefined) result.model = modelOverride(input.model)
  if (input.reasoningEffort !== undefined) result.reasoningEffort = input.reasoningEffort === null ? null : identifier(input.reasoningEffort, '思考强度')
  return result
}
function waitWithCancellation<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(new Error('Cancelled'))
  return new Promise<T>((resolveValue, rejectValue) => {
    const abort = () => rejectValue(new Error('Cancelled'))
    signal.addEventListener('abort', abort, { once: true })
    operation.then(value => { signal.removeEventListener('abort', abort); resolveValue(value) }, error => { signal.removeEventListener('abort', abort); rejectValue(error) })
  })
}
function publicTask(value: unknown): QuickStartStatus {
  const raw = record(value)
  const string = (value: unknown, max = 500): string => { if (typeof value !== 'string' || value.length > max) throw new Error('准备状态记录无效'); return value }
  if (!SAFE_ID.test(String(raw.id)) || !SAFE_OPERATION.test(String(raw.operationId)) || !PUBLIC_STATUS.includes(String(raw.status)) || !Array.isArray(raw.steps)
    || raw.requestedRounds !== 3 || raw.maxModelCalls !== 6) throw new Error('准备状态记录无效')
  const model = raw.model === null ? null : modelOverride(raw.model)
  return { id: String(raw.id), operationId: String(raw.operationId), status: raw.status as QuickStartStatus['status'], stage: string(raw.stage, 100), message: string(raw.message),
    createdAt: string(raw.createdAt, 50), updatedAt: string(raw.updatedAt, 50), runId: raw.runId === null ? null : string(raw.runId, 96), errorCode: raw.errorCode === null ? null : string(raw.errorCode, 100),
    model, reasoningEffort: raw.reasoningEffort === null ? null : string(raw.reasoningEffort, 200), requestedRounds: 3, maxModelCalls: 6,
    steps: raw.steps.slice(-40).map(value => { const step = record(value); if (!['running', 'completed', 'failed', 'cancelled'].includes(String(step.status))) throw new Error('准备步骤记录无效')
      return { stage: string(step.stage, 100), status: step.status as QuickStartStatus['steps'][number]['status'], message: string(step.message), time: string(step.time, 50) } }) }
}
/** 准备过程单独拥有操作编号和子进程，查询与页面刷新都不会再次提交。 */
export class QuickStartManager {
  private tasks = new Map<string, StoredTask>()
  private active: ActivePreparation | null = null
  private serial: Promise<unknown> = Promise.resolve()
  private disposed = false
  private readonly directory: string
  private readonly spawnBootstrap: SpawnWorker
  private readonly cancellations = new Map<string, Promise<boolean>>()
  constructor(readonly root: string, readonly gateway: ModelGateway, readonly hooks: QuickStartHooks, readonly options: QuickStartOptions = {}) {
    this.directory = join(root, '.dsh-gb1', 'quick-start'); this.spawnBootstrap = options.spawnBootstrap ?? ((executable, argv, settings) => spawn(executable, argv, settings) as ChildProcessWithoutNullStreams)
  }
  get preparing(): boolean { return this.active !== null }
  private exclusive<T>(action: () => Promise<T>): Promise<T> { const next = this.serial.then(action); this.serial = next.catch(() => {}); return next }
  async initialize(): Promise<void> {
    await mkdir(this.directory, { recursive: true })
    for (const name of await readdir(this.directory)) {
      if (!name.endsWith('.json') || !SAFE_ID.test(name.slice(0, -5))) continue
      try {
        const raw = record(await readJson(join(this.directory, name))), input = validateQuickStart(raw.input as QuickStartInput), summary = publicTask(raw.public)
        if (summary.id !== name.slice(0, -5) || input.operationId !== summary.operationId || !SAFE_OPERATION.test(String(raw.runOperationId))) continue
        const task: StoredTask = { input, public: summary, runOperationId: String(raw.runOperationId), runInput: null }
        if (summary.status === 'preparing') { summary.status = 'interrupted'; summary.stage = 'interrupted'; summary.errorCode = 'HOST_INTERRUPTED'; summary.message = MESSAGES.interrupted!; await this.persist(task) }
        this.tasks.set(summary.id, task)
      } catch { /* 损坏的本地状态不触发安装或实验。 */ }
    }
  }
  private async persist(task: StoredTask): Promise<void> { task.public.updatedAt = new Date().toISOString(); await saveJson(join(this.directory, `${task.public.id}.json`), task) }
  private update(job: ActivePreparation, stage: string, status: 'running' | 'completed' | 'failed' | 'cancelled' = 'running', reused = false): void {
    if (job.cancelled && status !== 'cancelled') return
    const reusedMessages: Record<string, string> = { python: '已确认可复用的 Python 3.12。', venv: '项目独立 Python 环境已复用。', dependencies: '项目固定依赖检查通过，已复用。', download: '已有官方 GB1 S1 数据检查通过，已复用。', validate: '已有 GB1 数据通过检查。', ready: MESSAGES.ready! }
    const message = reused ? reusedMessages[stage] ?? MESSAGES[stage] ?? MESSAGES.failed! : MESSAGES[stage] ?? MESSAGES.failed!
    job.task.public.stage = stage; job.task.public.message = message
    const step = { stage, status, message, time: new Date().toISOString() }
    const steps = job.task.public.steps, previous = steps.at(-1)
    if (previous?.stage === stage) steps[steps.length - 1] = step
    else steps.push(step)
    job.task.public.steps = steps.slice(-40)
    job.task.public.updatedAt = new Date().toISOString()
    const snapshot = structuredClone(job.task)
    job.writes = job.writes.then(() => this.persist(snapshot))
    void job.writes.catch(() => { job.controller.abort(); job.child?.kill() })
  }
  start(raw: QuickStartInput): Promise<QuickStartStatus> {
    return this.exclusive(async () => {
      if (this.disposed) throw new Error('工作台已停止')
      const input = validateQuickStart(raw), existing = [...this.tasks.values()].find(task => task.input.operationId === input.operationId)
      if (existing) {
        if (JSON.stringify(existing.input) !== JSON.stringify(input)) throw new Error('该操作编号已用于其他一键运行参数')
        return (await this.get(existing.public.id))!
      }
      if (this.active) throw new Error('已有环境准备操作，请查看其进度或取消后重试')
      if (this.hooks.activeRunId()) throw new Error(ERRORS.RUN_ACTIVE)
      const id = `quick-${randomUUID()}`, now = new Date().toISOString()
      const task: StoredTask = { input, public: { id, operationId: input.operationId, status: 'preparing', stage: 'queued', message: MESSAGES.queued!, createdAt: now, updatedAt: now,
        runId: null, errorCode: null, model: input.model ?? null, reasoningEffort: input.reasoningEffort ?? null, requestedRounds: 3, maxModelCalls: 6, steps: [] }, runOperationId: `quick-run-${randomUUID()}`, runInput: null }
      let finish = () => {}; const done = new Promise<void>(resolveDone => { finish = resolveDone })
      const job: ActivePreparation = { task, controller: new AbortController(), child: null, cancelled: false, writes: Promise.resolve(), done, finish }
      this.active = job
      try {
        await this.persist(task); this.tasks.set(id, task)
        const initial = structuredClone(task.public)
        void this.execute(job)
        return initial
      } catch {
        if (this.active === job) this.active = null
        job.finish()
        throw new Error('无法保存准备任务，尚未开始安装或实验。')
      }
    })
  }
  private async execute(job: ActivePreparation): Promise<void> {
    let code = 'BOOTSTRAP_FAILED'
    try {
      this.update(job, 'model'); await job.writes
      let defaults
      try { defaults = await loadDefaults(this.root) } catch { code = 'DEFAULTS_INVALID'; throw new Error() }
      const model = job.task.input.model ?? defaults.recommendedModel
      if (!model) { code = 'MODEL_NOT_SELECTED'; throw new Error() }
      const effort = job.task.input.reasoningEffort === undefined ? defaults.config.llm_reasoning_effort : job.task.input.reasoningEffort
      let capabilities
      try { capabilities = await this.gateway.capabilities(model.provider, model.model) } catch { code = 'MODEL_UNAVAILABLE'; throw new Error() }
      if (effort !== null && !capabilities.reasoningEfforts.includes(effort)) { code = 'EFFORT_UNSUPPORTED'; throw new Error() }
      job.controller.signal.throwIfAborted()
      this.update(job, 'model', 'completed')
      const config: ExperimentConfiguration = validateConfig({ ...defaults.config, rounds: 3, llm_reasoning_effort: effort, llm_max_calls_per_round: 2, max_model_calls: 6 })
      job.task.public.model = model; job.task.public.reasoningEffort = effort
      job.task.runInput = { operationId: job.task.runOperationId, reference: REFERENCE, seed: config.seed, rounds: 3, method: 'knowledge', promptProfile: 'v3', config, modelOverride: model }
      code = 'PROJECT_NOT_READY'
      await Promise.all(['gb1/data.py', 'requirements.txt', 'configs/knowledge.json', 'tools/quick_start/bootstrap.py'].map(path => access(join(this.root, path))))
      const folder = join(this.root, 'results', 'quick-start', job.task.public.id)
      await mkdir(folder, { recursive: true })
      this.update(job, 'python'); await job.writes
      code = 'PYTHON_NOT_FOUND'
      const python = await this.findPython(job)
      job.controller.signal.throwIfAborted()
      this.update(job, 'python', 'completed')
      code = 'BOOTSTRAP_FAILED'
      await this.prepare(job, python, folder)
      job.controller.signal.throwIfAborted()
      if (this.hooks.activeRunId()) { code = 'RUN_ACTIVE'; throw new Error() }
      this.update(job, 'starting-run'); await job.writes
      code = 'RUN_START_FAILED'
      const run = await this.hooks.startRun(job.task.runInput)
      job.task.public.runId = run.id
      this.update(job, 'starting-run', 'completed')
      if (job.cancelled) { if (await this.cancelOwnedRun(job.task)) this.update(job, 'cancelled', 'cancelled') }
      else { job.task.public.status = 'running'; this.update(job, 'running', 'completed') }
      await job.writes; await this.persist(job.task)
    } catch (error) {
      const reported = error instanceof Error && 'code' in error ? String(error.code) : ''
      if (BOOTSTRAP_CODES.includes(reported) || reported === 'BOOTSTRAP_PROTOCOL') code = reported
      if (code === 'CANCELLED') job.cancelled = true
      job.task.public.status = job.cancelled ? 'cancelled' : code === 'RUN_START_FAILED' ? 'interrupted' : 'failed'
      job.task.public.errorCode = job.cancelled ? 'CANCELLED' : code
      job.task.public.stage = job.task.public.status
      job.task.public.message = ERRORS[job.task.public.errorCode] ?? ERRORS.BOOTSTRAP_FAILED!
      const unfinished = job.task.public.steps.filter(step => step.status === 'running')
      for (const step of unfinished) { step.status = job.cancelled ? 'cancelled' : 'failed'; step.message = job.task.public.message; step.time = new Date().toISOString() }
      if (!unfinished.length) job.task.public.steps.push({ stage: job.task.public.stage, status: job.cancelled ? 'cancelled' : 'failed', message: job.task.public.message, time: new Date().toISOString() })
      job.task.public.steps = job.task.public.steps.slice(-40)
      await job.writes.catch(() => {}); await this.persist(job.task).catch(() => {})
    } finally {
      if (job.cancelTimer) clearTimeout(job.cancelTimer)
      if (this.active === job) this.active = null
      job.finish()
    }
  }
  private async findPython(job: ActivePreparation): Promise<{ executable: string; prefix: string[] }> {
    const candidates = [
      { executable: join(this.root, '.venv', 'Scripts', 'python.exe'), prefix: [] },
      ...(this.options.bootstrapPython ? [{ executable: this.options.bootstrapPython, prefix: [] }] : []),
      { executable: 'py', prefix: ['-3.12'] }, { executable: 'python3.12', prefix: [] }, { executable: 'python', prefix: [] },
      { executable: resolve(dirname(process.execPath), '../../python/python.exe'), prefix: [] },
      { executable: resolve(dirname(process.execPath), 'resources/runtime/primary-runtime/dependencies/python/python.exe'), prefix: [] },
      { executable: join(homedir(), '.dsh', 'dsh-runtimes', 'dsh-primary-runtime', 'dependencies', 'python', 'python.exe'), prefix: [] },
    ]
    for (const candidate of candidates) {
      job.controller.signal.throwIfAborted()
      const valid = await new Promise<boolean>(resolveValid => {
        let child: ChildProcessWithoutNullStreams, output = '', settled = false
        const finish = (value: boolean) => { if (settled) return; settled = true; clearTimeout(timer); job.controller.signal.removeEventListener('abort', abort); if (job.child === child) job.child = null; resolveValid(value) }
        const abort = () => { child?.kill(); finish(false) }
        let timer: NodeJS.Timeout
        try { child = this.spawnBootstrap(candidate.executable, [...candidate.prefix, '-I', '-c', 'import sys; print("GB1_PYTHON_3_12" if sys.version_info[:2] == (3,12) else "UNSUPPORTED")'], { cwd: this.root, env: cleanPreparationEnvironment(), shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] }) }
        catch { resolveValid(false); return }
        job.child = child
        timer = setTimeout(() => { child.kill(); finish(false) }, 5000); timer.unref()
        job.controller.signal.addEventListener('abort', abort, { once: true })
        child.stdout.on('data', chunk => { output += String(chunk); if (output.length > 1024) { child.kill(); finish(false) } })
        child.stderr.on('data', () => {})
        child.on('error', () => finish(false)); child.once('close', code => finish(code === 0 && output.trim() === 'GB1_PYTHON_3_12'))
      })
      if (valid) return candidate
    }
    throw new Error('No supported bootstrap interpreter')
  }
  private async prepare(job: ActivePreparation, python: { executable: string; prefix: string[] }, folder: string): Promise<void> {
    return new Promise<void>((resolveReady, rejectReady) => {
      let child: ChildProcessWithoutNullStreams, buffer = Buffer.alloc(0), terminal: string | null = null, terminalCode = 'BOOTSTRAP_FAILED', invalid = false
      try { child = this.spawnBootstrap(python.executable, [...python.prefix, join(this.root, 'tools', 'quick_start', 'bootstrap.py'), '--project-root', this.root, '--prepare-only',
        '--report', join(folder, 'prepare.json'), '--cancel-file', join(folder, 'cancel.flag')], { cwd: this.root, env: cleanPreparationEnvironment(), shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] }) }
      catch { rejectReady(new Error()); return }
      job.child = child
      child.stdout.on('data', (chunk: Buffer) => {
        try {
          buffer = Buffer.concat([buffer, chunk])
          for (;;) {
            const end = buffer.indexOf(10); if (end < 0) break
            if (end >= MAX_FRAME || terminal !== null) throw new Error()
            const frame = record(JSON.parse(buffer.subarray(0, end).toString('utf8'))); buffer = buffer.subarray(end + 1)
            if (frame.version !== 1 || !['progress', 'result'].includes(String(frame.type))) throw new Error()
            if (frame.type === 'result') {
              if (!['ready', 'failed', 'cancelled'].includes(String(frame.status))) throw new Error()
              terminal = String(frame.status)
              if (terminal === 'failed' && BOOTSTRAP_CODES.includes(String(frame.code))) terminalCode = String(frame.code)
              if (terminal === 'cancelled') terminalCode = 'CANCELLED'
            } else {
              const stage = String(frame.stage)
              if (!['python', 'venv', 'dependencies', 'download', 'validate', 'ready'].includes(stage) || !['running', 'completed', 'reused'].includes(String(frame.status))) throw new Error()
              const counts = record(frame.counts ?? {})
              const keys = ['python_312', 'environment_created', 'environment_reused', 'required_packages', 'matched_packages', 'imports_ok', 'dependencies_reused', 'bytes_downloaded', 'data_reused', 'variants', 'unique_variants', 'zero_fitness', 'wt_is_one']
              if (Object.entries(counts).some(([key, value]) => !keys.includes(key) || !(typeof value === 'boolean' || typeof value === 'number' && Number.isFinite(value) && value >= 0))) throw new Error()
              this.update(job, stage, frame.status === 'running' ? 'running' : 'completed', frame.status === 'reused')
            }
          }
          if (buffer.length > MAX_FRAME) throw new Error()
        } catch { invalid = true; child.kill() }
      })
      child.stderr.on('data', () => {})
      child.stdin.on('error', () => {})
      child.on('error', () => rejectReady(new Error()))
      child.once('close', code => {
        if (job.child === child) job.child = null
        if (!invalid && code === 0 && terminal === 'ready' && buffer.length === 0) resolveReady()
        else rejectReady(Object.assign(new Error(), { code: invalid || buffer.length ? 'BOOTSTRAP_PROTOCOL' : terminalCode }))
      })
    })
  }
  async get(id?: string): Promise<QuickStartStatus | null> {
    let task: StoredTask | undefined
    if (!id) task = [...this.tasks.values()].sort((a, b) => b.public.createdAt.localeCompare(a.public.createdAt))[0]
    else { if (!SAFE_ID.test(id)) throw new Error('准备任务编号无效'); task = this.tasks.get(id) }
    if (!task) return null
    if (task.public.runId === null && task.public.status === 'interrupted') {
      try { const receipt = record(await readJson(join(this.root, '.dsh-gb1', 'operations', `${task.runOperationId}.json`))); if (typeof receipt.id === 'string' && /^gb1-[a-f0-9-]{36}$/.test(receipt.id)) task.public.runId = receipt.id } catch {}
    }
    if (task.public.runId && !['cancelled', 'failed'].includes(task.public.status)) {
      try {
        const run = await this.hooks.getRun(task.public.runId)
        if (run.status === 'running' || run.status === 'paused' || run.status === 'queued') { task.public.status = 'running'; task.public.stage = 'running'; task.public.message = MESSAGES.running! }
        else { task.public.status = run.status === 'completed' ? 'completed' : run.status === 'cancelled' ? 'cancelled' : run.status === 'interrupted' ? 'interrupted' : 'failed'; task.public.stage = task.public.status; task.public.message = MESSAGES[task.public.status]! }
        await this.persist(task)
      } catch { task.public.status = 'interrupted'; task.public.stage = 'interrupted'; task.public.message = MESSAGES.interrupted! }
    }
    return structuredClone(task.public)
  }
  private async cancelOwnedRun(task: StoredTask): Promise<boolean> {
    const id = task.public.runId
    if (!id) return true
    let pending = this.cancellations.get(id)
    if (!pending) {
      pending = Promise.resolve().then(() => this.hooks.cancelRun(id)).then(() => true, () => false)
      this.cancellations.set(id, pending)
    }
    const confirmed = await pending
    if (this.cancellations.get(id) === pending) this.cancellations.delete(id)
    task.public.status = confirmed ? 'cancelled' : 'interrupted'
    task.public.stage = task.public.status; task.public.errorCode = confirmed ? 'CANCELLED' : 'CANCEL_UNCONFIRMED'
    task.public.message = confirmed ? MESSAGES.cancelled! : ERRORS.CANCEL_UNCONFIRMED!
    return confirmed
  }
  private terminateOwnedChild(job: ActivePreparation, child: ChildProcessWithoutNullStreams): void {
    if (job.child !== child) return
    if (process.platform !== 'win32' || !child.pid) { child.kill(); return }
    try {
      const killer = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { shell: false, windowsHide: true, stdio: 'ignore', env: cleanPreparationEnvironment() })
      const timer = setTimeout(() => { killer.kill(); if (job.child === child) child.kill() }, 2000); timer.unref()
      killer.once('error', () => { clearTimeout(timer); if (job.child === child) child.kill() })
      killer.once('close', code => { clearTimeout(timer); if (code !== 0 && job.child === child) child.kill() })
    } catch { child.kill() }
  }
  async cancel(id: string): Promise<QuickStartStatus> {
    if (!SAFE_ID.test(id)) throw new Error('准备任务编号无效')
    const task = this.tasks.get(id); if (!task) throw new Error('准备任务不存在')
    const job = this.active?.task.public.id === id ? this.active : null
    if (job) {
      job.cancelled = true; job.controller.abort()
      const folder = join(this.root, 'results', 'quick-start', id); await mkdir(folder, { recursive: true }); await writeFile(join(folder, 'cancel.flag'), 'cancel\n')
      if (job.child) { const child = job.child; job.cancelTimer = setTimeout(() => this.terminateOwnedChild(job, child), 10000); job.cancelTimer.unref() }
      for (const step of task.public.steps) if (step.status === 'running') { step.status = 'cancelled'; step.message = MESSAGES.cancelled!; step.time = new Date().toISOString() }
      task.public.status = 'cancelled'; task.public.errorCode = 'CANCELLED'; this.update(job, 'cancelled', 'cancelled')
      await job.writes
    }
    if (task.public.runId && (job || !TERMINAL.has(task.public.status) || task.public.errorCode === 'CANCEL_UNCONFIRMED')) await this.cancelOwnedRun(task)
    await this.persist(task)
    return structuredClone(task.public)
  }
  async dispose(): Promise<void> {
    this.disposed = true; await this.serial.catch(() => {})
    if (this.active) { const job = this.active; await this.cancel(job.task.public.id); await job.done }
  }
}
