import type { RemoteResult } from '@deepseek-ai/dsh-typert-protocol'
import type { ClientRemote, RemoteStream } from '@deepseek-ai/dsh-api-gateway/client'
import type {} from '../../lib/typert.remote-client.js'
import type { BatchDetail, BatchSummary, ConnectionSettings, ConversationSnapshot, ModelCapabilities, ModelChoice, QuickStartInput, QuickStartStatus, RunDetail, RunPage, RunSummary, SaveConnectionInput, StartBatchInput, StartRunInput, WorkbenchInfo } from '../contracts.ts'

export interface WorkbenchApi {
  getInfo(): Promise<WorkbenchInfo>
  startQuickRun(input: QuickStartInput): Promise<QuickStartStatus>
  getQuickStartStatus(id?: string): Promise<QuickStartStatus | null>
  cancelQuickStart(id: string): Promise<QuickStartStatus>
  listModels(): Promise<ModelChoice[]>
  getModelCapabilities(provider: string, model: string): Promise<ModelCapabilities>
  getRunPage(id: string, offset: number, limit: number): Promise<RunPage>
  pauseRun(id: string): Promise<RunSummary>
  resumeRun(id: string): Promise<RunSummary>
  startBatch(input: StartBatchInput): Promise<BatchDetail>
  getBatch(id: string): Promise<BatchDetail>
  listBatches(): Promise<BatchSummary[]>
  pauseBatch(id: string): Promise<BatchDetail>
  resumeBatch(id: string): Promise<BatchDetail>
  cancelBatch(id: string): Promise<BatchDetail>
  saveConnection(input: SaveConnectionInput): Promise<ConnectionSettings>
  listRuns(): Promise<RunSummary[]>
  getRun(id: string): Promise<RunDetail>
  startRun(input: StartRunInput): Promise<RunSummary>
  cancelRun(id: string): Promise<RunSummary>
  followConversation(id: string, reconnecting: () => void): RemoteStream<ConversationSnapshot>
}

// 只在这里拆解DSH的RemoteResult，让页面处理业务数据和可读错误。
async function value<T>(result: Promise<RemoteResult<T>>): Promise<T> {
  const resolved = await result
  if (!resolved.ok) throw new Error(resolved.error.message || '操作未完成，请刷新后重试。')
  return resolved.value
}

export function createApi(root: ClientRemote): WorkbenchApi {
  const remote = root.gb1Workbench
  return {
    getInfo: () => value(remote.getInfo()),
    startQuickRun: input => value(remote.startQuickRun(input)),
    getQuickStartStatus: id => value(remote.getQuickStartStatus(id)),
    cancelQuickStart: id => value(remote.cancelQuickStart(id)),
    listModels: () => value(remote.listModels()),
    getModelCapabilities: (provider, model) => value(remote.getModelCapabilities(provider, model)),
    getRunPage: (id, offset, limit) => value(remote.getRunPage(id, offset, limit)),
    pauseRun: id => value(remote.pauseRun(id)),
    resumeRun: id => value(remote.resumeRun(id)),
    startBatch: input => value(remote.startBatch(input)),
    getBatch: id => value(remote.getBatch(id)),
    listBatches: () => value(remote.listBatches()),
    pauseBatch: id => value(remote.pauseBatch(id)),
    resumeBatch: id => value(remote.resumeBatch(id)),
    cancelBatch: id => value(remote.cancelBatch(id)),
    saveConnection: input => value(remote.saveConnection(input)),
    listRuns: () => value(remote.listRuns()),
    getRun: id => value(remote.getRun(id)),
    startRun: input => value(remote.startRun(input)),
    cancelRun: id => value(remote.cancelRun(id)),
    // 重连仅打开只读订阅，完整快照替换旧画面，不会再次调用startRun。
    followConversation: (id, reconnecting) => root.$stream({
      name: 'gb1-workbench-conversation',
      open: signal => remote.watchConversation(id, signal),
      ended: () => new Error('过程连接已结束，请重新连接查看。'),
      carrierFailed: reconnecting,
    }),
  }
}
