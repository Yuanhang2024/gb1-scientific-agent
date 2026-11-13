import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, SegmentedControl } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ConnectionSettings, ExperimentConfiguration, ModelCapabilities, ModelChoice, RunDetail, RunMethod, RunSummary, StartRunInput, WorkbenchInfo } from '../contracts.ts'
import type { WorkbenchApi } from './api.ts'
import { ConversationPanel } from './ConversationPanel.tsx'
import { AnalysisPanel } from './components/analysis/AnalysisPanel.tsx'
import { HistoryPanel } from './components/history/HistoryPanel.tsx'
import { ProjectPanel } from './components/project/ProjectPanel.tsx'
import { ResultsPanel } from './components/results/ResultsPanel.tsx'
import { ExperimentSetupPanel, METHOD_LABELS } from './components/setup/ExperimentSetupPanel.tsx'
import { BatchPanel } from './components/setup/BatchPanel.tsx'
import { ModelConnectionPanel } from './components/setup/ModelConnectionPanel.tsx'
import { RunStatusPanel } from './components/setup/RunStatusPanel.tsx'
import { QuickStartPanel } from './components/setup/QuickStartPanel.tsx'

type View = 'process' | 'results' | 'analysis' | 'history' | 'batch' | 'project'
const EMPTY_CONNECTION: ConnectionSettings = { source: 'dsh', provider: '', model: '', baseUrl: '', hasKey: false }
const errorText = (error: unknown) => error instanceof Error ? error.message : '操作未完成，请刷新后重试。'

export function FlaskIcon({ size = 18 }: { size?: number; active?: boolean }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M9 3h6M10 3v6L4.9 18a2 2 0 0 0 1.8 3h10.6a2 2 0 0 0 1.8-3L14 9V3M8 14h8"/>
    <path d="M9 17h.01M14 18h.01"/>
  </svg>
}

// 字符串单元格防止被表格软件当公式执行；数值仍按数值导出。
function exportCsv(detail: RunDetail) {
  const cell = (value: string | number | null) => {
    let text = value === null ? '' : String(value)
    if (typeof value === 'string' && /^[=+\-@\t\r]/.test(text)) text = `'${text}`
    return `"${text.replaceAll('"', '""')}"`
  }
  const rows: Array<Array<string | number | null>> = [['轮次', '排名', '变体', '完整序列', '突变', '预测分数', '实验数据分数', '推荐理由', '设计假设']]
  for (const round of detail.roundResults) {
    for (const row of round.top10) rows.push([round.round, row.rank, row.variant, row.sequence, row.mutations, row.predictedFitness, row.actualFitness, row.rationale, row.hypothesis])
  }
  const url = URL.createObjectURL(new Blob(['\uFEFF' + rows.map(row => row.map(cell).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `gb1-${detail.summary.id}.csv`
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function Workbench({ api }: { api: WorkbenchApi }) {
  const [info, setInfo] = useState<WorkbenchInfo | null>(null)
  const [connection, setConnection] = useState<ConnectionSettings>(EMPTY_CONNECTION)
  const [apiKey, setApiKey] = useState('')
  const [models, setModels] = useState<ModelChoice[]>([])
  const [modelsError, setModelsError] = useState('')
  const [reference, setReference] = useState('')
  const [config, setConfig] = useState<ExperimentConfiguration | null>(null)
  const [promptProfile, setPromptProfile] = useState('v3')
  const [capabilities, setCapabilities] = useState<ModelCapabilities | null>(null)
  const [capabilitiesError, setCapabilitiesError] = useState('')
  const [method, setMethod] = useState<RunMethod>('knowledge')
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [tab, setTab] = useState<View>('process')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [quickActive, setQuickActive] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const mounted = useRef(true)
  const selectedRef = useRef<string | null>(null)
  const pendingStart = useRef<{ signature: string; input: StartRunInput } | null>(null)
  const activeId = info?.activeRunId ?? runs.find(run => run.status === 'running' || run.status === 'queued')?.id ?? null
  const active = Boolean(activeId)

  const showRun = useCallback(async (id: string) => {
    selectedRef.current = id
    const value = await api.getRun(id)
    if (!mounted.current || selectedRef.current !== id) return
    setDetail(value)
  }, [api])

  const refresh = useCallback(async (initial = false) => {
    const value = await api.getInfo()
    if (!mounted.current) return
    setInfo(value)
    if (initial) {
      setConnection(value.connection)
      setReference(value.reference)
      setConfig(structuredClone(value.defaultConfig))
      setPromptProfile(value.defaultPromptProfile)
      setSettingsOpen(false)
    }
    if (!value.ready) { setRuns([]); return }
    const history = await api.listRuns()
    if (!mounted.current) return
    setRuns(history)
    const recovered = pendingStart.current && history.find(run => run.operationId === pendingStart.current!.input.operationId)
    if (recovered) pendingStart.current = null
    const id = recovered?.id ?? selectedRef.current ?? value.activeRunId ?? history[0]?.id
    if (id) {
      if (selectedRef.current !== id) await showRun(id)
      else {
        const current = await api.getRun(id)
        if (mounted.current && selectedRef.current === id) setDetail(current)
      }
    }
  }, [api, showRun])

  const openQuickRun = useCallback(async (id: string) => {
    await showRun(id)
    if (!mounted.current) return
    setTab('process')
    setSettingsOpen(false)
    await refresh()
  }, [showRun, refresh])

  const refreshModels = useCallback(async () => {
    try {
      const values = await api.listModels()
      if (mounted.current) { setModels(values); setModelsError('') }
    } catch (cause) {
      if (mounted.current) setModelsError(errorText(cause))
    }
  }, [api])

  useEffect(() => {
    mounted.current = true
    void refresh(true)
      .catch(cause => { if (mounted.current) setError(errorText(cause)) })
      .finally(() => { if (mounted.current) setLoading(false) })
    void refreshModels()
    return () => { mounted.current = false }
  }, [refresh, refreshModels])

  useEffect(() => {
    setCapabilities(null); setCapabilitiesError('')
    if (connection.source !== 'dsh' || !connection.provider || !connection.model) return
    let disposed = false
    void api.getModelCapabilities(connection.provider, connection.model).then(value => {
      if (disposed) return
      setCapabilities(value)
      setConfig(previous => previous && previous.llm_reasoning_effort && !value.reasoningEfforts.includes(previous.llm_reasoning_effort)
        ? { ...previous, llm_reasoning_effort: null } : previous)
    }).catch(cause => { if (!disposed) setCapabilitiesError(errorText(cause)) })
    return () => { disposed = true }
  }, [api, connection.source, connection.provider, connection.model])

  // 页面刷新只查询状态，不会再次提交实验；只有用户按开始按钮才创建运行。
  useEffect(() => {
    if (!active) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try { await refresh() } catch (cause) {
        if (!cancelled && mounted.current) setError(errorText(cause))
      }
      if (!cancelled) timer = setTimeout(poll, 1500)
    }
    timer = setTimeout(poll, 1000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [active, refresh])

  const recommendedModel = info?.recommendedModel ?? null
  const recommendationAvailable = Boolean(recommendedModel && models.some(model => model.provider === recommendedModel.provider && model.id === recommendedModel.model))
  const recommendationProblem = recommendedModel && !recommendationAvailable
    ? modelsError ? '模型列表读取失败，暂时无法确认推荐模型是否可用。请刷新模型列表。'
      : '尚未在当前模型列表中找到推荐模型，请检查模型是否已启用并刷新列表。'
    : ''
  const applyRecommendation = () => {
    if (!info || !recommendedModel || busy || active || quickActive) return
    if (!recommendationAvailable) { setError(recommendationProblem); return }
    setConfig(structuredClone(info.defaultConfig))
    setPromptProfile(info.defaultPromptProfile)
    setConnection(previous => ({ ...previous, source: 'dsh', provider: recommendedModel.provider, model: recommendedModel.model }))
    setError(''); setNotice('推荐模型、思考强度与实验参数已填入表单。连接尚未保存，实验尚未开始。')
  }

  const saveConnection = async (clearKey = false) => {
    const saved = await api.saveConnection({
      source: connection.source, provider: connection.provider,
      model: connection.model, baseUrl: connection.baseUrl,
      ...(apiKey ? { apiKey } : {}), ...(clearKey ? { clearKey: true } : {}),
    })
    setConnection(saved)
    setApiKey('')
    setInfo(previous => previous ? { ...previous, connection: saved } : previous)
    return saved
  }
  const save = async (clearKey = false) => {
    setBusy(true); setError(''); setNotice('')
    try {
      await saveConnection(clearKey)
      setNotice(clearKey ? '独立接口密钥已移除。' : '模型连接设置已保存。')
    } catch (cause) { setError(errorText(cause)) }
    finally { setBusy(false) }
  }
  const start = async () => {
    const normalized = reference.replace(/\s/g, '').toUpperCase()
    if (!info?.ready || !config) { setError(info?.reason || '项目尚未就绪。'); return }
    if (!config.allowed_amino_acids || !config.knowledge_path.trim()) { setError('请填写允许氨基酸和项目内知识规则文件。'); setSettingsOpen(true); return }
    if (normalized !== info.reference) { setError('当前数据对应固定的 GB1 参考序列，请使用预填的 56 个氨基酸。'); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const saved = method === 'random' || method === 'model' ? connection : await saveConnection()
      const specification = { reference: normalized, seed: config.seed, rounds: config.rounds, method, promptProfile,
        config: { ...config, llm_reasoning_effort: saved.source === 'custom' ? null : config.llm_reasoning_effort } }
      const signature = JSON.stringify({ ...specification, connection: saved })
      // 同一提交的回复不明确时复用 operationId，Host 会返回已有运行，避免重复启动。
      if (pendingStart.current?.signature !== signature) {
        pendingStart.current = { signature, input: { ...specification, operationId: crypto.randomUUID() } }
      }
      const started = await api.startRun(pendingStart.current.input)
      pendingStart.current = null
      setTab('process')
      setSettingsOpen(false)
      await showRun(started.id)
      await refresh()
    } catch (cause) {
      setError(errorText(cause))
      void refresh().catch(() => {})
    } finally { setBusy(false) }
  }
  const cancel = async (id = activeId) => {
    if (!id) return
    setBusy(true); setError('')
    try {
      await api.cancelRun(id)
      await refresh()
      setNotice('已请求停止，已完成的结果会保留。')
    } catch (cause) { setError(errorText(cause)) }
    finally { setBusy(false) }
  }
  const controlRun = async (action: 'pause' | 'resume') => {
    const id = action === 'resume' ? detail?.summary.id : activeId
    if (!id) return
    setBusy(true); setError(''); setNotice('')
    try {
      await (action === 'pause' ? api.pauseRun(id) : api.resumeRun(id))
      await refresh()
      setNotice(action === 'pause' ? '已请求暂停，将在本轮完整反馈保存后暂停。' : '已请求恢复，将继续使用本次运行保存的参数。')
    } catch (cause) { setError(errorText(cause)) } finally { setBusy(false) }
  }
  const exportAll = async () => {
    if (!detail) return
    const selectedDetail = detail
    setBusy(true); setError('')
    try {
      const total = selectedDetail.totalRounds ?? selectedDetail.roundMetrics?.length ?? selectedDetail.roundResults.length
      const roundResults: RunDetail['roundResults'] = []
      for (let offset = 0; offset < total; offset += 20) {
        const page = await api.getRunPage(selectedDetail.summary.id, offset, 20)
        roundResults.push(...page.roundResults)
      }
      exportCsv({ ...selectedDetail, roundResults })
    } catch (cause) { setError(errorText(cause)) } finally { setBusy(false) }
  }
  const copy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); setNotice('序列已复制。') }
    catch { setError('浏览器未允许复制，请直接选中序列复制。') }
  }
  const reload = useCallback(() => { void refresh().catch(cause => setError(errorText(cause))) }, [refresh])
  const openHistory = (id: string) => {
    void showRun(id).then(() => { if (selectedRef.current === id) setTab('results') }).catch(cause => setError(errorText(cause)))
  }
  const openActive = () => {
    if (!activeId) return
    void showRun(activeId).then(() => setTab('process')).catch(cause => setError(errorText(cause)))
  }
  const canRun = Boolean(info?.ready && config && ((method === 'random' || method === 'model') || (connection.model && (connection.source === 'custom'
    ? connection.baseUrl && (connection.hasKey || apiKey) : connection.provider && (!config.llm_reasoning_effort || capabilities)))))
  const activeRun = runs.find(run => run.id === activeId) ?? (detail?.summary.id === activeId ? detail.summary : null)
  const localMethod = method === 'random' || method === 'model'
  const setupTitle = localMethod ? `${METHOD_LABELS[method]}（不调用 LLM）` : connection.model || '请选择候选设计模型'

  return <main className="gb1-page" data-testid="gb1-workbench">
    <div className="gb1-shell">
      <header className="gb1-header">
        <div>
          <div className="gb1-eyebrow"><FlaskIcon size={16}/>GB1 · 四个位点</div>
          <h1>蛋白质进化</h1>
          <p>设计候选，预测评分，查看实验反馈。</p>
        </div>
        <span className="gb1-status-pill">{loading ? '正在读取项目' : info?.ready ? '项目已就绪' : '项目待设置'}</span>
      </header>
      {(error || notice) && <div className={`gb1-message ${error ? 'gb1-message-error' : ''}`} role={error ? 'alert' : 'status'}>
        <span>{error || notice}</span>
        <Button size="sm" onClick={() => { setError(''); setNotice(''); reload() }}>刷新状态</Button>
      </div>}
      <QuickStartPanel api={api} info={info} models={models} modelsError={modelsError} disabled={loading || busy || active}
        onRefreshModels={() => void refreshModels()} onRunReady={openQuickRun} onActivityChange={setQuickActive}/>
      {!loading && info && !info.ready && <details className="gb1-environment-note">
        <summary>项目状态与准备说明</summary><p>{info.reason}</p>
        <p>一键入口会检查并准备项目环境与官方数据。若项目位置设置有误，可在插件设置中调整 projectRoot。</p>
      </details>}
      <div className="gb1-setup-bar">
        <div className="gb1-setup-summary">
          <strong title={setupTitle}>{setupTitle}</strong>
          <span className="gb1-subtle">{localMethod ? '本地计算' : connection.source === 'dsh' ? 'DSH 模型' : '独立接口'} · {METHOD_LABELS[method]} · {config?.rounds ?? '—'} 轮 · 种子 {config?.seed ?? '—'} · C {config?.proposal_count ?? '—'} / k {config?.k ?? '—'}</span>
        </div>
        <div className="gb1-run-actions">
          <Button variant="outline" aria-expanded={settingsOpen} aria-controls="gb1-settings-panel" onClick={() => setSettingsOpen(value => !value)}>{settingsOpen ? '收起设置' : '实验设置'}</Button>
          <Button variant="primary" disabled={loading || busy || active || quickActive || !canRun} onClick={() => void start()}>
            {busy ? '处理中…' : pendingStart.current ? '重试同一提交' : '开始实验'}
          </Button>
          {active && <><Button variant="outline" disabled={busy} onClick={() => void controlRun('pause')}>暂停</Button><Button variant="outline" disabled={busy} onClick={() => void cancel()}>停止</Button></>}
          {detail && ['paused', 'interrupted'].includes(detail.summary.status) && <><Button variant="outline" disabled={busy || active} onClick={() => void controlRun('resume')}>恢复所选实验</Button><Button variant="outline" disabled={busy} onClick={() => void cancel(detail.summary.id)}>停止所选实验</Button></>}
        </div>
      </div>
      <RunStatusPanel run={activeRun} isSelected={detail?.summary.id === activeId && tab === 'process'} onOpen={openActive}/>
      <div className={`gb1-layout${settingsOpen ? ' gb1-layout-settings' : ''}`}>
        {settingsOpen && <aside id="gb1-settings-panel" className="gb1-controls" aria-label="实验设置">
          <ModelConnectionPanel
            connection={connection} apiKey={apiKey} models={models} modelsError={modelsError}
            disabled={busy || active || quickActive} loading={loading} ready={Boolean(info?.ready)}
            onChange={value => { setConnection(value); setError(''); setNotice('') }}
            onApiKeyChange={setApiKey} onRefreshModels={() => void refreshModels()} onSave={clearKey => void save(clearKey)}
          />
          {config && info && <ExperimentSetupPanel
            reference={reference} method={method} config={config} defaults={info.defaultConfig} promptProfile={promptProfile}
            defaultPromptProfile={info.defaultPromptProfile} capabilities={capabilities} capabilitiesError={capabilitiesError} customModel={connection.source === 'custom'}
            recommendedModel={recommendedModel} recommendationProblem={recommendationProblem} onApplyRecommendation={applyRecommendation}
            disabled={busy || active || quickActive} loading={loading} onReferenceChange={setReference} onMethodChange={setMethod} onConfigChange={setConfig} onProfileChange={setPromptProfile}
          />}
        </aside>}
        <section className="gb1-results" aria-label="实验工作台">
          <div className="gb1-results-toolbar">
            <div className="gb1-view-navigation">
              <SegmentedControl id="gb1-view" label="工作台视图" value={tab} options={[
                { value: 'process', label: '工作对话' }, { value: 'results', label: '推荐结果' },
                { value: 'analysis', label: '实验分析' }, { value: 'history', label: '运行记录' }, { value: 'batch', label: '批量比较' },
                { value: 'project', label: '项目说明' },
              ]} onChange={setTab}/>
            </div>
            <Button size="sm" disabled={loading} onClick={reload}>刷新</Button>
          </div>
          <div id={`gb1-view-${tab}-panel`} role="tabpanel" aria-labelledby={`gb1-view-${tab}`}>
            {tab === 'process' && <ConversationPanel key={detail?.summary.id ?? 'empty'} api={api} detail={detail} onViewResults={() => setTab('results')}/>}
            {tab === 'results' && <ResultsPanel api={api} detail={detail} onCopy={(text: string) => void copy(text)} onExport={() => void exportAll()} exporting={busy}/>}
            {tab === 'batch' && <BatchPanel api={api} config={config} promptProfile={promptProfile} models={models} connection={connection} ready={Boolean(info?.ready) && !quickActive} onOpenRun={openHistory} onChanged={reload}/>}
            {tab === 'analysis' && <AnalysisPanel detail={detail}/>}
            {tab === 'history' && <HistoryPanel runs={runs} onOpen={openHistory}/>}
            {tab === 'project' && <ProjectPanel info={info}/>}
          </div>
          <p className="gb1-footnote">GB1 四个位点虚拟实验 · 反馈来自 Wu 等人的公开实验记录（S1）。</p>
        </section>
      </div>
    </div>
  </main>
}
