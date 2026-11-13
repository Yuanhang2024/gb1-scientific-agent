import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@deepseek-ai/dsh-client-ui-primitives'
import type { BatchDetail, BatchSummary, BatchTarget, ConnectionSettings, ExperimentConfiguration, ModelCapabilities, ModelChoice, RunDetail, RunMethod, StartBatchInput } from '../../../contracts.ts'
import type { WorkbenchApi } from '../../api.ts'
import { METHOD_LABELS } from './ExperimentSetupPanel.tsx'
import { NumberField, NumberListField } from './ConfigurationFields.tsx'

const STATUS: Record<string, string> = { queued: '等待运行', running: '运行中', paused: '已暂停', stopped: '提前结束', completed: '已完成', failed: '未完成', cancelled: '已停止', interrupted: '已中断' }
const modelKey = (provider: string, model: string) => JSON.stringify([provider, model])
const targetKey = (target: BatchTarget) => JSON.stringify([target.provider, target.model, target.reasoningEffort])
const message = (cause: unknown) => cause instanceof Error ? cause.message : '批量操作未完成。'

export function BatchPanel({ api, config, promptProfile, models, connection, ready, onOpenRun, onChanged }: {
  api: WorkbenchApi; config: ExperimentConfiguration | null; promptProfile: string; models: ModelChoice[]; connection: ConnectionSettings; ready: boolean
  onOpenRun: (id: string) => void; onChanged: () => void
}) {
  const [targets, setTargets] = useState<BatchTarget[]>([])
  const [methods, setMethods] = useState<RunMethod[]>(['knowledge'])
  const [seeds, setSeeds] = useState([11, 22, 33])
  const [repeats, setRepeats] = useState(1)
  const [concurrency, setConcurrency] = useState(2)
  const [perProvider, setPerProvider] = useState(1)
  const [capabilities, setCapabilities] = useState<Record<string, ModelCapabilities>>({})
  const [capErrors, setCapErrors] = useState<Record<string, string>>({})
  const [batches, setBatches] = useState<BatchSummary[]>([])
  const [detail, setDetail] = useState<BatchDetail | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [page, setPage] = useState(0)
  const [results, setResults] = useState<Record<string, RunDetail>>({})
  const [resultErrors, setResultErrors] = useState<Record<string, string>>({})
  const selected = useRef<string | null>(null)
  const pending = useRef<{ signature: string; input: StartBatchInput } | null>(null)
  const requestGeneration = useRef(0)
  const refresh = useCallback(async () => {
    const values = await api.listBatches()
    setBatches(values)
    const recovered = pending.current && values.find(batch => batch.operationId === pending.current!.input.operationId)
    if (recovered) { selected.current = recovered.id; pending.current = null }
    const id = selected.current ?? values[0]?.id
    if (id) { const value = await api.getBatch(id); if (selected.current === null || selected.current === id) { selected.current = id; setDetail(value) } }
  }, [api])
  useEffect(() => { let disposed = false; void refresh().catch(cause => { if (!disposed) setError(message(cause)) }); return () => { disposed = true } }, [refresh])
  const active = batches.some(batch => batch.status === 'running' || batch.status === 'queued' || batch.activeRuns > 0)
  useEffect(() => {
    if (!active) return
    let disposed = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => { try { await refresh(); onChanged() } catch (cause) { if (!disposed) setError(message(cause)) } if (!disposed) timer = setTimeout(poll, 2000) }
    timer = setTimeout(poll, 1500)
    return () => { disposed = true; clearTimeout(timer) }
  }, [active, refresh, onChanged])
  const loadCapabilities = async (model: ModelChoice) => {
    const key = modelKey(model.provider, model.id)
    if (capabilities[key]) return
    try { const value = await api.getModelCapabilities(model.provider, model.id); setCapabilities(previous => ({ ...previous, [key]: value })); setCapErrors(previous => ({ ...previous, [key]: '' })) }
    catch (cause) { setCapErrors(previous => ({ ...previous, [key]: message(cause) })) }
  }
  const toggleTarget = (target: BatchTarget, checked: boolean) => setTargets(previous => checked ? [...previous.filter(value => targetKey(value) !== targetKey(target)), target] : previous.filter(value => targetKey(value) !== targetKey(target)))
  const addCurrent = () => {
    const model = models.find(value => value.provider === connection.provider && value.id === connection.model)
    if (connection.source === 'dsh' && model) { toggleTarget({ provider: model.provider, model: model.id, reasoningEffort: null }, true); void loadCapabilities(model) }
  }
  const count = targets.length * seeds.length * repeats * methods.length
  const usesModelCount = targets.length * seeds.length * repeats * methods.filter(method => method === 'llm' || method === 'knowledge').length
  const maxRequests = config ? usesModelCount * Math.min(config.rounds * config.llm_max_calls_per_round, config.max_model_calls ?? Infinity) : 0
  const maxAssays = config ? count * Math.min(config.rounds * config.k, config.max_assays === null ? Infinity : Math.floor(config.max_assays / config.k) * config.k) : 0
  const start = async () => {
    if (!config || !count) return
    setBusy(true); setError(''); setNotice('')
    try {
      const specification = { baseConfig: config, promptProfile, methods, targets, seeds, repeats, concurrency, perProviderConcurrency: perProvider }
      const signature = JSON.stringify(specification)
      if (pending.current?.signature !== signature) pending.current = { signature, input: { ...specification, operationId: crypto.randomUUID() } }
      const value = await api.startBatch(pending.current.input)
      pending.current = null; selected.current = value.summary.id; setDetail(value); setPage(0)
      await refresh(); onChanged(); setNotice('批量任务已创建，每项运行使用独立数据与反馈。')
    } catch (cause) { setError(message(cause)); void refresh().catch(() => {}) }
    finally { setBusy(false) }
  }
  const control = async (action: 'pause' | 'resume' | 'cancel') => {
    if (!detail) return
    setBusy(true); setError(''); setNotice('')
    try {
      const value = await (action === 'pause' ? api.pauseBatch(detail.summary.id) : action === 'resume' ? api.resumeBatch(detail.summary.id) : api.cancelBatch(detail.summary.id))
      setDetail(value); await refresh(); onChanged()
      setNotice(action === 'pause' ? '已请求暂停，正在运行的实验会在完整反馈轮次后暂停。' : action === 'resume' ? '已请求恢复，继续使用各项运行原有设置。' : '已请求停止批量任务，已完成结果会保留。')
    } catch (cause) { setError(message(cause)) } finally { setBusy(false) }
  }
  const pageRuns = useMemo(() => detail?.runs.slice(page * 10, page * 10 + 10) ?? [], [detail, page])
  useEffect(() => {
    const generation = ++requestGeneration.current
    let disposed = false
    void Promise.all(pageRuns.filter(run => run.id).map(async run => {
      try {
        const value = await api.getRun(run.id!)
        if (!disposed && requestGeneration.current === generation) {
          setResults(previous => ({ ...previous, [run.operationId]: value }))
          setResultErrors(previous => ({ ...previous, [run.operationId]: '' }))
        }
      } catch (cause) {
        if (!disposed && requestGeneration.current === generation) setResultErrors(previous => ({ ...previous, [run.operationId]: message(cause) }))
      }
    }))
    return () => { disposed = true }
  }, [api, pageRuns])
  return <section className="gb1-batch" aria-labelledby="gb1-batch-heading">
    <div className="gb1-panel-heading"><div><h2 id="gb1-batch-heading">批量比较</h2><p>模型与思考档位 × 数据种子 × 重复次数。基础与高级设置应用于每项运行。</p></div></div>
    <details className="gb1-batch-settings" open={!detail}><summary>配置新的批量任务</summary>
      <div className="gb1-batch-grid"><section><div className="gb1-section-heading"><h3>模型与思考档位</h3><Button size="sm" disabled={busy || connection.source !== 'dsh' || !connection.model} onClick={addCurrent}>加入当前模型</Button></div>
        <p className="gb1-help">勾选模型后可选择其支持的档位；每个勾选项独立运行。</p>
        <div className="gb1-batch-models">{models.map(model => {
          const key = modelKey(model.provider, model.id), selectedTargets = targets.filter(target => target.provider === model.provider && target.model === model.id)
          const checked = selectedTargets.length > 0
          return <div key={key} className="gb1-batch-model"><label className="gb1-checkbox"><input type="checkbox" checked={checked} disabled={busy} onChange={event => {
            if (event.target.checked) { toggleTarget({ provider: model.provider, model: model.id, reasoningEffort: null }, true); void loadCapabilities(model) }
            else setTargets(previous => previous.filter(target => target.provider !== model.provider || target.model !== model.id))
          }}/><span>{model.name || model.id}<small className="gb1-block gb1-subtle">{model.provider} · {model.id}</small></span></label>
            {checked && <div className="gb1-batch-efforts">{Boolean(capabilities[key]?.reasoningEfforts.length) && <Button size="sm" disabled={busy} onClick={() => setTargets(previous => [...previous.filter(target => target.provider !== model.provider || target.model !== model.id), ...capabilities[key].reasoningEfforts.map(reasoningEffort => ({ provider: model.provider, model: model.id, reasoningEffort }))])}>选择全部档位</Button>}{[null, ...(capabilities[key]?.reasoningEfforts ?? [])].map(effort => <label className="gb1-checkbox" key={effort ?? 'default'}><input type="checkbox" disabled={busy} checked={selectedTargets.some(target => target.reasoningEffort === effort)} onChange={event => toggleTarget({ provider: model.provider, model: model.id, reasoningEffort: effort }, event.target.checked)}/><span>{effort ?? '模型默认'}</span></label>)}
              {capErrors[key] && <p className="gb1-help" role="status">{capErrors[key]} <Button size="sm" onClick={() => void loadCapabilities(model)}>重新读取</Button></p>}
            </div>}
          </div>
        })}</div>
      </section><section><h3>数据与调度</h3><NumberListField label="数据随机种子" value={seeds} min={0} max={4294967295} disabled={busy} onChange={setSeeds}/>
        <div className="gb1-field-pair"><NumberField label="每种设置重复次数" value={repeats} min={1} disabled={busy} onChange={value => { if (value !== null) setRepeats(value) }}/><NumberField label="全局并发" value={concurrency} min={1} disabled={busy} onChange={value => { if (value !== null) setConcurrency(value) }}/></div>
        <NumberField label="每提供方并发" value={perProvider} min={1} max={concurrency} disabled={busy} onChange={value => { if (value !== null) setPerProvider(value) }}/>
        <label>推荐方式</label>{Object.entries(METHOD_LABELS).map(([value, label]) => <label key={value} className="gb1-checkbox"><input type="checkbox" checked={methods.includes(value as RunMethod)} disabled={busy} onChange={event => setMethods(previous => event.target.checked ? [...previous, value as RunMethod] : previous.filter(method => method !== value))}/><span>{label}</span></label>)}
        <div className="gb1-budget" aria-live="polite"><strong>整批预算预览</strong><span>每项 {config?.rounds ?? '—'} 轮 · C {config?.proposal_count ?? '—'} / k {config?.k ?? '—'} · {promptProfile}</span><span>{targets.length} 个模型/档位 × {seeds.length} 个种子 × {repeats} 次重复 × {methods.length} 种方法 = {count} 次运行</span><span>最多 {maxRequests.toLocaleString()} 次生成请求 · {maxAssays.toLocaleString()} 条实测查询</span></div>
        <Button variant="primary" disabled={busy || !ready || !config || !count || perProvider > concurrency} onClick={() => void start()}>{busy ? '处理中…' : pending.current ? '重试同一批量提交' : '开始批量比较'}</Button>
      </section></div>
    </details>
    {(error || notice) && <p className={error ? 'gb1-run-error' : 'gb1-help'} role={error ? 'alert' : 'status'}>{error || notice}</p>}
    <div className="gb1-batch-toolbar"><div><label htmlFor="gb1-batch-history">已保存批量任务</label><select id="gb1-batch-history" value={detail?.summary.id ?? ''} onChange={event => {
      const id = event.target.value; selected.current = id; setPage(0)
      void api.getBatch(id).then(value => { if (selected.current === id) setDetail(value) }).catch(cause => setError(message(cause)))
    }}><option value="" disabled>选择批量任务</option>{batches.map(batch => <option key={batch.id} value={batch.id}>{new Date(batch.createdAt).toLocaleString('zh-CN')} · {STATUS[batch.status]} · {batch.completedRuns}/{batch.totalRuns}</option>)}</select></div><Button size="sm" onClick={() => void refresh().catch(cause => setError(message(cause)))}>刷新批量任务</Button></div>
    {detail && <><div className="gb1-active-run"><div className="gb1-active-run-copy"><strong>{STATUS[detail.summary.status]}</strong><span className="gb1-subtle">完成 {detail.summary.completedRuns}/{detail.summary.totalRuns} · 失败 {detail.summary.failedRuns} · 正在运行 {detail.summary.activeRuns}</span></div>
      <div className="gb1-run-actions">{['queued', 'running'].includes(detail.summary.status) && <Button size="sm" variant="outline" disabled={busy} onClick={() => void control('pause')}>暂停整批</Button>}
        {['paused', 'interrupted'].includes(detail.summary.status) && <Button size="sm" variant="outline" disabled={busy} onClick={() => void control('resume')}>恢复整批</Button>}
        {['queued', 'running', 'paused', 'interrupted'].includes(detail.summary.status) && <Button size="sm" disabled={busy} onClick={() => void control('cancel')}>停止整批</Button>}</div></div>
      <div className="gb1-batch-table-scroll"><table className="gb1-batch-table"><caption>各项独立运行与已保存结果</caption><thead><tr><th>模型 / 强度</th><th>设置</th><th>状态</th><th>实测均值</th><th>新增最高</th><th>已报告tokens</th><th>结果</th></tr></thead><tbody>{pageRuns.map(run => {
        const result = results[run.operationId], metrics = result?.roundMetrics ?? result?.roundResults ?? []
        const measured = metrics.filter(round => round.meanFitness !== null && round.knownCountAfter > round.knownCountBefore)
        const total = measured.reduce((sum, round) => sum + round.knownCountAfter - round.knownCountBefore, 0)
        const mean = total ? measured.reduce((sum, round) => sum + round.meanFitness! * (round.knownCountAfter - round.knownCountBefore), 0) / total : null
        const maxima = metrics.map(round => round.maxFitness).filter((value): value is number => value !== null)
        return <tr key={run.operationId}><td><strong>{run.target.model}</strong><small>{run.target.provider} · {run.target.reasoningEffort ?? '默认'}</small></td><td>{METHOD_LABELS[run.method]}<small>seed {run.seed} · 重复 {run.repeatIndex + 1}</small></td><td>{STATUS[run.status]}{resultErrors[run.operationId] && <small>结果读取失败：{resultErrors[run.operationId]}</small>}{run.error && <small title={run.error}>{run.error}</small>}{result?.summary.stopReason && <small>{result.summary.stopReason}</small>}</td><td>{mean?.toFixed(3) ?? '—'}</td><td>{maxima.length ? Math.max(...maxima).toFixed(3) : '—'}</td><td>{result?.usage.totalTokens?.toLocaleString() ?? '—'}{Boolean(result?.usage.missingUsageCount) && <small>部分用量未知</small>}</td><td><Button size="sm" variant="outline" disabled={!run.id} onClick={() => { if (run.id) onOpenRun(run.id) }}>查看</Button></td></tr>
      })}</tbody></table></div><div className="gb1-pagination"><span>第 {page + 1}/{Math.max(1, Math.ceil(detail.runs.length / 10))} 页 · 共 {detail.runs.length} 次运行</span><div><Button size="sm" disabled={page === 0} onClick={() => setPage(value => value - 1)}>上一页</Button><Button size="sm" disabled={(page + 1) * 10 >= detail.runs.length} onClick={() => setPage(value => value + 1)}>下一页</Button></div></div>
    </>}
  </section>
}
