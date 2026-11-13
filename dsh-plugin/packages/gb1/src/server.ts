import { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { Remote, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'
import type { BatchDetail, BatchSummary, ConversationSnapshot, ConnectionSettings, ModelCapabilities, ModelChoice, RunDetail, RunPage, RunSummary, QuickStartInput, QuickStartStatus, SaveConnectionInput, StartBatchInput, StartRunInput, WorkbenchInfo } from './contracts.ts'
import { BatchManager } from './batches.ts'
import { DEFAULT_CONFIG, loadDefaults } from './settings.ts'
import { ConnectionStore, ModelGateway, validateConnection, type CredentialsPort, type LlmPort } from './model.ts'
import { projectLocation, REFERENCE, RunManager, type SpawnWorker } from './runs.ts'
import { QuickStartManager } from './quick-start.ts'

export interface Config { projectRoot?: string; bootstrapPython?: string }
export const Config = z.object({ projectRoot: z.string().default(''), bootstrapPython: z.string().default('') })
interface BackendPorts { llm: LlmPort; credentials: CredentialsPort; fetcher?: typeof fetch; spawnWorker?: SpawnWorker; spawnBootstrap?: SpawnWorker }

/** Host 业务类可使用模拟服务测试，RPC 类仅负责显式类型边界。 */
export class WorkbenchBackend {
  private location: { root: string; ready: boolean; reason: string } | null = null
  private manager: RunManager | null = null
  private connections: ConnectionStore | null = null
  private batches: BatchManager | null = null
  private initialization: Promise<void>
  private managerInitialization: Promise<RunManager> | null = null
  private quick: QuickStartManager | null = null
  private readonly gateway: ModelGateway
  private workAdmission: Promise<unknown> = Promise.resolve()
  constructor(readonly ports: BackendPorts, config: Config = {}) {
    this.gateway = new ModelGateway(ports.llm, ports.credentials, ports.fetcher)
    this.initialization = this.initialize(config.projectRoot ?? '', config.bootstrapPython ?? '')
    void this.initialization.catch(() => {})
  }
  private async initialize(root: string, bootstrapPython: string): Promise<void> {
    this.location = await projectLocation(root)
    this.connections = new ConnectionStore(this.location.root, this.ports.credentials)
    if (this.location.ready) await this.ensureManager()
    this.quick = new QuickStartManager(this.location.root, this.gateway, {
      activeRunId: () => this.manager?.activeRunId ?? (this.batches?.hasRunnableWork ? 'batch-queue' : null),
      startRun: input => this.admitWork(async () => {
        const manager = await this.ensureManager()
        if (manager.activeRunId || this.batches?.hasRunnableWork) throw new Error('已有实验或批次等待运行，请查看运行状态。')
        return manager.startRun(input)
      }),
      getRun: async id => (await this.ensureManager()).getRun(id).then(detail => detail.summary),
      cancelRun: async id => (await this.ensureManager()).cancelRun(id),
    }, { bootstrapPython, spawnBootstrap: this.ports.spawnBootstrap })
    await this.quick.initialize()
  }
  private ensureManager(): Promise<RunManager> {
    if (this.manager) return Promise.resolve(this.manager)
    if (this.managerInitialization) return this.managerInitialization
    const pending = (async () => {
      if (!this.location || !this.connections) throw new Error('工作台尚未初始化')
      this.location = await projectLocation(this.location.root)
      if (!this.location.ready) throw new Error(this.location.reason)
      const manager = new RunManager(this.location.root, this.connections, this.gateway, this.ports.spawnWorker)
      await manager.initialize()
      const batches = new BatchManager(this.location.root, manager)
      await batches.initialize()
      this.manager = manager; this.batches = batches
      return manager
    })()
    this.managerInitialization = pending
    void pending.finally(() => { if (this.managerInitialization === pending) this.managerInitialization = null }).catch(() => {})
    return pending
  }
  private async ready(): Promise<RunManager> { await this.initialization; return this.ensureManager() }
  private assertNoPreparation(): void { if (this.quick?.preparing) throw new Error('环境准备进行中，请查看一键运行进度。') }
  private admitWork<T>(action: () => Promise<T>): Promise<T> {
    const next = this.workAdmission.then(action)
    this.workAdmission = next.catch(() => {})
    return next
  }
  async getInfo(): Promise<WorkbenchInfo> {
    try { await this.initialization } catch { return this.notReady('无法初始化工作台运行记录，请检查项目目录。') }
    if (!this.location || !this.connections) return this.notReady(this.location?.reason ?? '工作台尚未就绪')
    try {
      const defaults = await loadDefaults(this.location.root)
      return { ready: this.location.ready, reason: this.location.reason, reference: REFERENCE, positions: [39, 40, 41, 54],
        connection: await this.connections.publicSettings(), activeRunId: this.manager?.activeRunId ?? null,
        scoreBudget: defaults.config.score_budget, topK: defaults.config.k, maxRounds: null,
        defaultConfig: defaults.config, defaultPromptProfile: defaults.promptProfile, recommendedModel: defaults.recommendedModel }
    } catch { return this.notReady('项目连接设置无法读取，请检查项目设置记录。') }
  }
  private notReady(reason: string): WorkbenchInfo {
    return { ready: false, reason, reference: REFERENCE, positions: [39, 40, 41, 54],
      connection: { source: 'dsh', provider: '', model: '', baseUrl: '', hasKey: false },
      activeRunId: null, scoreBudget: 40, topK: 10, maxRounds: null, defaultConfig: structuredClone(DEFAULT_CONFIG), defaultPromptProfile: 'v3', recommendedModel: null }
  }
  async listModels(): Promise<ModelChoice[]> {
    try {
      const result: ModelChoice[] = []
      const providers = this.ports.llm.listProviders()
      let failures = 0
      for (const provider of providers) {
        try {
          const models = await this.ports.llm.listModels(provider.id)
          for (const model of models) result.push({ provider: provider.id, id: model.id, name: model.name })
        } catch { failures++ }
      }
      if (providers.length > 0 && failures === providers.length) throw new Error('模型目录当前不可用')
      return result
    } catch { throw new Error('无法读取 DSH 模型目录') }
  }
  async saveConnection(input: SaveConnectionInput): Promise<ConnectionSettings> {
    const manager = await this.ready()
    return manager.mutateConnection(async () => {
      const clean = validateConnection(input)
      if (clean.source === 'dsh' && !(await this.listModels()).some(model => model.provider === clean.provider && model.id === clean.model)) throw new Error('请选择当前已注册的 DSH 模型')
      return this.connections!.save(input)
    })
  }
  async listRuns(): Promise<RunSummary[]> { return (await this.ready()).listRuns() }
  async getRun(id: string): Promise<RunDetail> { return (await this.ready()).getRun(id) }
  async startRun(input: StartRunInput): Promise<RunSummary> { return this.admitWork(async () => { const manager = await this.ready(); this.assertNoPreparation(); return manager.startRun(input) }) }
  async cancelRun(id: string): Promise<RunSummary> { return (await this.ready()).cancelRun(id) }
  async getModelCapabilities(provider: string, model: string): Promise<ModelCapabilities> { return this.gateway.capabilities(provider, model) }
  async getRunPage(id: string, offset: number, limit: number): Promise<RunPage> { return (await this.ready()).getRunPage(id, offset, limit) }
  async pauseRun(id: string): Promise<RunSummary> { return (await this.ready()).pauseRun(id) }
  async resumeRun(id: string): Promise<RunSummary> { return this.admitWork(async () => { const manager = await this.ready(); this.assertNoPreparation(); return manager.resumeRun(id) }) }
  async startBatch(input: StartBatchInput): Promise<BatchDetail> { return this.admitWork(async () => { await this.ready(); this.assertNoPreparation(); return this.batches!.startBatch(input) }) }
  async getBatch(id: string): Promise<BatchDetail> { await this.ready(); return this.batches!.getBatch(id) }
  async listBatches(): Promise<BatchSummary[]> { await this.ready(); return this.batches!.listBatches() }
  async pauseBatch(id: string): Promise<BatchDetail> { await this.ready(); return this.batches!.pauseBatch(id) }
  async resumeBatch(id: string): Promise<BatchDetail> { return this.admitWork(async () => { await this.ready(); this.assertNoPreparation(); return this.batches!.resumeBatch(id) }) }
  async cancelBatch(id: string): Promise<BatchDetail> { await this.ready(); return this.batches!.cancelBatch(id) }
  async startQuickRun(input: QuickStartInput): Promise<QuickStartStatus> { return this.admitWork(async () => { await this.initialization; return this.quick!.start(input) }) }
  async getQuickStartStatus(id?: string): Promise<QuickStartStatus | null> { await this.initialization; return this.quick!.get(id) }
  async cancelQuickStart(id: string): Promise<QuickStartStatus> { await this.initialization; return this.quick!.cancel(id) }
  async *watchConversation(id: string, signal: AbortSignal): AsyncIterable<ConversationSnapshot> {
    yield* (await this.ready()).watchConversation(id, signal)
  }
  async dispose(): Promise<void> {
    try { await this.initialization } catch { return }
    await this.quick?.dispose()
    await this.batches?.dispose()
    await this.manager?.dispose()
  }
}

declare module '@deepseek-ai/cordis' { interface Context { gb1Workbench: Gb1Workbench } }

export default class Gb1Workbench extends TypertRemoteService {
  static Config = Config
  static inject = ['llm', 'credentials']
  private backend: WorkbenchBackend
  constructor(ctx: Context, config: Config = {}) {
    super(ctx, 'gb1Workbench', { namespace: 'gb1Workbench' })
    this.backend = new WorkbenchBackend({ llm: ctx.llm, credentials: ctx.credentials }, config)
    ctx.effect(() => () => this.backend.dispose(), 'gb1Workbench:owned-run-cleanup')
  }
  @Remote
  async getInfo(): Promise<WorkbenchInfo> { return this.backend.getInfo() }
  @Remote
  async listModels(): Promise<ModelChoice[]> { return this.backend.listModels() }
  @Remote
  async saveConnection(input: SaveConnectionInput): Promise<ConnectionSettings> { return this.backend.saveConnection(input) }
  @Remote
  async listRuns(): Promise<RunSummary[]> { return this.backend.listRuns() }
  @Remote
  async getRun(id: string): Promise<RunDetail> { return this.backend.getRun(id) }
  @Remote
  async startRun(input: StartRunInput): Promise<RunSummary> { return this.backend.startRun(input) }
  @Remote
  async cancelRun(id: string): Promise<RunSummary> { return this.backend.cancelRun(id) }
  @Remote
  async getModelCapabilities(provider: string, model: string): Promise<ModelCapabilities> { return this.backend.getModelCapabilities(provider, model) }
  @Remote
  async getRunPage(id: string, offset: number, limit: number): Promise<RunPage> { return this.backend.getRunPage(id, offset, limit) }
  @Remote
  async pauseRun(id: string): Promise<RunSummary> { return this.backend.pauseRun(id) }
  @Remote
  async resumeRun(id: string): Promise<RunSummary> { return this.backend.resumeRun(id) }
  @Remote
  async startBatch(input: StartBatchInput): Promise<BatchDetail> { return this.backend.startBatch(input) }
  @Remote
  async getBatch(id: string): Promise<BatchDetail> { return this.backend.getBatch(id) }
  @Remote
  async listBatches(): Promise<BatchSummary[]> { return this.backend.listBatches() }
  @Remote
  async pauseBatch(id: string): Promise<BatchDetail> { return this.backend.pauseBatch(id) }
  @Remote
  async resumeBatch(id: string): Promise<BatchDetail> { return this.backend.resumeBatch(id) }
  @Remote
  async cancelBatch(id: string): Promise<BatchDetail> { return this.backend.cancelBatch(id) }
  @Remote
  async startQuickRun(input: QuickStartInput): Promise<QuickStartStatus> { return this.backend.startQuickRun(input) }
  @Remote
  async getQuickStartStatus(id?: string): Promise<QuickStartStatus | null> { return this.backend.getQuickStartStatus(id) }
  @Remote
  async cancelQuickStart(id: string): Promise<QuickStartStatus> { return this.backend.cancelQuickStart(id) }
  @Remote({ mode: 'stream' })
  async *watchConversation(id: string, signal: AbortSignal): AsyncIterable<ConversationSnapshot> {
    yield* this.backend.watchConversation(id, signal)
  }
}
