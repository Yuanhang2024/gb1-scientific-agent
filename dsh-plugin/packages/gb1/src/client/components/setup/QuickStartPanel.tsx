import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ModelCapabilities, ModelChoice, QuickStartInput, QuickStartStatus, WorkbenchInfo } from '../../../contracts.ts'
import type { WorkbenchApi } from '../../api.ts'

const PENDING_KEY = 'gb1.quick-start.pending.v1'
const DEFAULT_MODEL = { provider: 'openai', model: 'gpt-6.1-sol' }
const identity = (provider: string, model: string) => JSON.stringify([provider, model])
const message = (cause: unknown) => cause instanceof Error ? cause.message : '操作暂时未完成，请查询当前状态。'
const STATUS = { preparing: '正在准备', running: '实验运行中', completed: '已完成', failed: '未完成', cancelled: '已取消', interrupted: '已中断' }
const STEP_STATUS = { running: '进行中', completed: '已完成', failed: '未完成', cancelled: '已取消' }
const STAGES: Record<string, string> = {
  queued: '等待准备', model: '检查模型与思考强度', 'starting-run': '启动三轮实验',
  starting: '准备启动', python: '检查 Python', environment: '准备环境', venv: '准备 Python 环境',
  dependencies: '检查与安装依赖', install: '安装依赖', download: '获取 GB1 数据', data: '准备 GB1 数据',
  validate: '校验 GB1 数据', validation: '校验 GB1 数据', ready: '准备就绪', training: '训练预测模型',
  run: '运行实验', running: '运行实验', completed: '已完成', cancelled: '已取消', failed: '任务未完成', interrupted: '已中断',
}

function readPending(): QuickStartInput | null {
  try {
    const text = sessionStorage.getItem(PENDING_KEY)
    if (!text || text.length > 2048) return null
    const input: unknown = JSON.parse(text)
    if (!input || typeof input !== 'object' || !('operationId' in input) || typeof input.operationId !== 'string'
      || !/^[\da-f-]{36}$/i.test(input.operationId)) return null
    const value = input as QuickStartInput
    if (value.model && (typeof value.model.provider !== 'string' || typeof value.model.model !== 'string')) return null
    if (value.reasoningEffort !== undefined && value.reasoningEffort !== null && typeof value.reasoningEffort !== 'string') return null
    return value
  } catch { return null }
}

export function QuickStartPanel({ api, info, models, modelsError, disabled, onRefreshModels, onRunReady, onActivityChange }: {
  api: WorkbenchApi; info: WorkbenchInfo | null; models: ModelChoice[]; modelsError: string; disabled: boolean
  onRefreshModels: () => void; onRunReady: (id: string) => Promise<void>; onActivityChange: (active: boolean) => void
}) {
  const [selected, setSelected] = useState(identity(DEFAULT_MODEL.provider, DEFAULT_MODEL.model))
  const [effort, setEffort] = useState<string | null>('xhigh')
  const [capabilities, setCapabilities] = useState<ModelCapabilities | null>(null)
  const [capabilitiesError, setCapabilitiesError] = useState('')
  const [capabilityRetry, setCapabilityRetry] = useState(0)
  const [status, setStatus] = useState<QuickStartStatus | null>(null)
  const [pending, setPending] = useState<QuickStartInput | null>(readPending)
  const [busy, setBusy] = useState(false)
  const [checking, setChecking] = useState(true)
  const [checked, setChecked] = useState(false)
  const [queryError, setQueryError] = useState('')
  const [actionError, setActionError] = useState('')
  const pendingRef = useRef(pending)
  const submitting = useRef(false)
  const touched = useRef(false)
  const initialized = useRef(false)
  const openedRun = useRef<string | null>(null)
  const mounted = useRef(true)
  const model = models.find(item => identity(item.provider, item.id) === selected)
  const recommended = info?.recommendedModel ?? DEFAULT_MODEL
  const active = status?.status === 'preparing' || status?.status === 'running'
  const cancelUnconfirmed = status?.status === 'interrupted' && status.errorCode === 'CANCEL_UNCONFIRMED'
  const locked = disabled || busy || active || cancelUnconfirmed || Boolean(pending) || checking || !checked

  const remember = useCallback((input: QuickStartInput | null) => {
    pendingRef.current = input
    setPending(input)
    try {
      if (input) sessionStorage.setItem(PENDING_KEY, JSON.stringify(input))
      else sessionStorage.removeItem(PENDING_KEY)
    } catch { /* The Host also persists the operation identifier. */ }
  }, [])

  const accept = useCallback((value: QuickStartStatus | null) => {
    if (!mounted.current) return
    setStatus(value); setChecked(true); setQueryError('')
    if (value && pendingRef.current?.operationId === value.operationId) {
      remember(null)
      setActionError('')
    }
  }, [remember])

  const query = useCallback(async (id?: string) => {
    const value = await api.getQuickStartStatus(id)
    accept(value)
  }, [api, accept])

  const refreshStatus = useCallback(async () => {
    setChecking(true)
    try { await query() }
    catch (cause) { if (mounted.current) setQueryError(message(cause)) }
    finally { if (mounted.current) setChecking(false) }
  }, [query])

  useEffect(() => {
    mounted.current = true
    void refreshStatus()
    return () => { mounted.current = false }
  }, [refreshStatus])

  useEffect(() => {
    if (!info || initialized.current || touched.current) return
    initialized.current = true
    const initial = pendingRef.current?.model ?? info.recommendedModel ?? DEFAULT_MODEL
    setSelected(identity(initial.provider, initial.model))
    setEffort(pendingRef.current ? pendingRef.current.reasoningEffort ?? null : info.defaultConfig.llm_reasoning_effort)
  }, [info])

  useEffect(() => {
    setCapabilities(null); setCapabilitiesError('')
    if (!model) return
    let disposed = false
    void api.getModelCapabilities(model.provider, model.id).then(value => {
      if (!disposed) setCapabilities(value)
    }).catch(cause => { if (!disposed) setCapabilitiesError(message(cause)) })
    return () => { disposed = true }
  }, [api, model?.provider, model?.id, capabilityRetry])

  useEffect(() => {
    onActivityChange(checking || !checked || busy || active || cancelUnconfirmed || Boolean(pending))
  }, [onActivityChange, checking, checked, busy, active, cancelUnconfirmed, pending])

  useEffect(() => {
    if (!active || !status) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try { await query(status.id) }
      catch (cause) { if (!disposed && mounted.current) setQueryError(message(cause)) }
      if (!disposed) timer = setTimeout(poll, 1200)
    }
    timer = setTimeout(poll, 800)
    return () => { disposed = true; clearTimeout(timer) }
  }, [active, status?.id, query])

  useEffect(() => {
    const runId = status?.runId
    if (!runId || openedRun.current === runId || (pending && status.operationId !== pending.operationId)) return
    openedRun.current = runId
    void onRunReady(runId).catch(cause => { if (mounted.current) setActionError(message(cause)) })
  }, [status?.runId, status?.operationId, pending, onRunReady])

  const supported = effort === null || Boolean(capabilities?.reasoningEfforts.includes(effort))
  const modelProblem = modelsError ? '模型列表暂时不可用，请刷新列表。'
    : !model ? '当前 DSH 未启用所选模型，请启用推荐模型或明确选择其他已注册模型。'
      : capabilitiesError || (!capabilities ? '正在读取模型支持的思考档位…'
        : !supported ? `当前模型不支持 ${effort}，请展开模型选择并指定受支持档位。` : '')

  const start = async () => {
    if (submitting.current || disabled || active || cancelUnconfirmed || !checked) return
    if (!pendingRef.current && (!model || !capabilities || !supported || modelProblem)) return
    submitting.current = true; setBusy(true); setActionError('')
    try {
      const input = pendingRef.current ?? {
        operationId: crypto.randomUUID(), model: { provider: model!.provider, model: model!.id }, reasoningEffort: effort,
      }
      remember(input)
      setStatus(null)
      accept(await api.startQuickRun(input))
    } catch (cause) {
      setActionError(`${message(cause)} 提交结果尚未确认；可查询状态，或使用同一操作重试。`)
      try { await query() } catch (failure) { if (mounted.current) setQueryError(message(failure)) }
    } finally { submitting.current = false; if (mounted.current) setBusy(false) }
  }

  const cancel = async () => {
    if (!status || submitting.current) return
    submitting.current = true; setBusy(true); setActionError('')
    try { accept(await api.cancelQuickStart(status.id)) }
    catch (cause) { setActionError(message(cause)); try { await query(status.id) } catch (failure) { setQueryError(message(failure)) } }
    finally { submitting.current = false; if (mounted.current) setBusy(false) }
  }

  const visibleModel = active && status?.model ? status.model.model : model?.id ?? (JSON.parse(selected) as string[])[1]
  const visibleEffort = active && status ? status.reasoningEffort : effort
  const rounds = active && status ? status.requestedRounds : 3
  const calls = active && status ? status.maxModelCalls : 6
  return <section className="gb1-quick-start" aria-labelledby="gb1-quick-heading" data-testid="gb1-quick-start" data-quick-status={status?.status ?? 'idle'}>
    <div className="gb1-quick-heading"><div><h2 id="gb1-quick-heading">一键准备并运行 GB1</h2>
      <p>一键预设 · {visibleModel} · {visibleEffort ?? '模型默认'} · {rounds} 轮 · 每轮 {info?.defaultConfig.proposal_count ?? 40} 个候选 / {info?.defaultConfig.k ?? 10} 条推荐</p>
      <p>自动准备环境与官方数据，复用已就绪内容；每轮最多 2 次生成，本次最多 {calls} 次请求。</p>
      <p>使用本次推荐预设；下方手动实验设置不影响一键运行，已保存连接保持不变。</p>
    </div></div>
    <div className="gb1-quick-actions">
      <Button variant="primary" disabled={disabled || busy || active || cancelUnconfirmed || checking || !checked || (!pending && Boolean(modelProblem))} onClick={() => void start()}>
        {busy ? '处理中…' : checking ? '正在恢复状态…' : pending ? '重试同一准备任务' : '一键准备并运行'}
      </Button>
      {(active || cancelUnconfirmed) && <Button variant="outline" disabled={busy} onClick={() => void cancel()}>{cancelUnconfirmed ? '再次请求取消' : '取消本次任务'}</Button>}
      {status?.runId && <Button variant="outline" disabled={busy} onClick={() => void onRunReady(status.runId!).catch(cause => setActionError(message(cause)))}>查看工作对话</Button>}
      {(pending || queryError || actionError || status?.status === 'interrupted') && <Button size="sm" disabled={busy || checking} onClick={() => void refreshStatus()}>查询准备状态</Button>}
    </div>
    <details className="gb1-quick-model"><summary>模型与思考档位 · {identity(recommended.provider, recommended.model) === selected ? '推荐模型' : '自选模型'}</summary>
      <div className="gb1-quick-model-fields"><div><label htmlFor="gb1-quick-model">已注册的候选设计模型</label>
        <select id="gb1-quick-model" value={selected} disabled={locked} onChange={event => { touched.current = true; setSelected(event.target.value) }}>
          {!model && <option value={selected}>{visibleModel}（当前不可用）</option>}
          {models.map(item => <option key={identity(item.provider, item.id)} value={identity(item.provider, item.id)}>{item.name || item.id} · {item.provider}</option>)}
        </select></div><div><label htmlFor="gb1-quick-effort">思考强度</label>
        <select id="gb1-quick-effort" value={effort ?? ''} disabled={locked || !capabilities} onChange={event => { touched.current = true; setEffort(event.target.value || null) }}>
          <option value="">模型默认</option>{effort && !capabilities?.reasoningEfforts.includes(effort) && <option value={effort}>{effort}（{capabilities ? '当前模型不支持' : '等待支持确认'}）</option>}
          {capabilities?.reasoningEfforts.map(value => <option key={value} value={value}>{value}</option>)}
        </select></div>
        <p className="gb1-help">此选择仅用于本次一键实验。推荐模型与档位不可用时，需要明确选择其他受支持设置。</p>
      </div>
      <Button size="sm" disabled={busy || active} onClick={() => { onRefreshModels(); setCapabilityRetry(value => value + 1) }}>刷新模型能力</Button>
    </details>
    {!active && modelProblem && <p className="gb1-help" role="status">{modelProblem}</p>}
    {disabled && !active && <p className="gb1-help">已有实验或其他操作正在执行，结束后可开始一键任务。</p>}
    {(actionError || queryError) && <p className="gb1-run-error" role="alert">{actionError || queryError}</p>}
    {pending && status && pending.operationId !== status.operationId && <p className="gb1-help" role="status">提交尚未确认，以下显示最近一次任务的实际状态。</p>}
    {status && <div className="gb1-quick-state" role="status" aria-live="polite">
      {active && <span className="gb1-progress-dot"/>}<strong>{STATUS[status.status]}</strong>
      <span className="gb1-subtle">{STAGES[status.stage] ?? '任务进度'}</span><p>{status.message}</p>
    </div>}
    {status && (status.steps.length > 0 || status.errorCode) && <details className="gb1-quick-details"><summary>查看准备详情 · {status.steps.length} 条实际记录</summary>
      {status.errorCode && <p className="gb1-help">错误代码：{status.errorCode}</p>}
      <ol className="gb1-quick-steps">{status.steps.map((step, index) => <li key={`${step.time}-${index}`}>
        <time dateTime={step.time}>{new Date(step.time).toLocaleTimeString('zh-CN')}</time>
        <span>{STEP_STATUS[step.status]} · {step.message}</span>
      </li>)}</ol>
    </details>}
  </section>
}
