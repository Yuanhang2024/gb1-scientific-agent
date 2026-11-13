import { Fragment, useEffect, useId, useState } from 'react'
import { Button, Input } from '@deepseek-ai/dsh-client-ui-primitives'
import type { Recommendation, RoundResult, RunDetail, RunStatus } from '../../../contracts.ts'
import type { WorkbenchApi } from '../../api.ts'
import { METHOD_LABELS } from '../setup/ExperimentSetupPanel.tsx'
import { SequenceComparison } from './SequenceComparison.tsx'
import { finiteValue, residual, scoreText, visibleRecommendations, type ResultSort } from './result-data.ts'
import css from './results.css'

export interface ResultsPanelProps {
  detail: RunDetail | null
  api: WorkbenchApi
  exporting?: boolean
  onCopy: (sequence: string) => void
  onExport: () => void
}
const STATUS: Record<RunStatus, string> = { queued: '等待运行', running: '运行中', paused: '已暂停', stopped: '提前结束', completed: '已完成', failed: '未完成', cancelled: '已停止', interrupted: '已中断' }
const SORTS: Array<{ value: ResultSort; label: string }> = [
  { value: 'rank', label: '原始排名' }, { value: 'actual', label: '实测分数 · 从高到低' },
  { value: 'predicted', label: '预测分数 · 从高到低' }, { value: 'residual', label: '残差绝对值 · 从大到小' },
  { value: 'mutations', label: '突变数量 · 从少到多' },
]
function countText(value: number | null | undefined) { return finiteValue(value)?.toLocaleString('zh-CN') ?? '—' }

function RecommendationDetails({ row, reference, onCopy }: { row: Recommendation; reference: string; onCopy: (sequence: string) => void }) {
  return <div className="gb1-r-detail">
    <div className="gb1-r-reasons">
      <section><h4>推荐理由</h4><p>{row.rationale || '未提供推荐理由。'}</p></section>
      <section><h4>设计假设</h4><p>{row.hypothesis || '未提供设计假设。'}</p></section>
    </div>
    <div className="gb1-r-rules"><h4>引用规则 ID</h4>
      {row.knowledgeRuleIds.length ? <ul aria-label="本候选引用的规则编号">{row.knowledgeRuleIds.map((id, index) => <li key={`${id}-${index}`}><code>{id}</code></li>)}</ul>
        : <p className="gb1-subtle">本候选未引用规则。</p>}
    </div>
    <SequenceComparison reference={reference} sequence={row.sequence} variant={row.variant} onCopy={onCopy}/>
  </div>
}

function ResultsView({ detail, api, exporting, onCopy, onExport }: Omit<ResultsPanelProps, 'detail'> & { detail: RunDetail }) {
  const [roundNumber, setRoundNumber] = useState<number | null>(null)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<ResultSort>('rank')
  const [expanded, setExpanded] = useState<string | null>(null)
  const localId = useId()
  const total = detail.totalRounds ?? detail.roundMetrics?.length ?? detail.roundResults.length
  const [pageIndex, setPageIndex] = useState(() => Math.max(0, Math.ceil(total / 10) - 1))
  const [pageRounds, setPageRounds] = useState<RoundResult[]>(detail.roundResults.slice(-10))
  const [pageError, setPageError] = useState('')
  const [pageLoading, setPageLoading] = useState(false)
  const [rowPage, setRowPage] = useState(0)
  useEffect(() => {
    let disposed = false
    setPageLoading(true); setPageError('')
    void api.getRunPage(detail.summary.id, pageIndex * 10, 10).then(value => {
      if (!disposed) setPageRounds(value.roundResults)
    }).catch(cause => { if (!disposed) setPageError(cause instanceof Error ? cause.message : '读取结果页失败。') })
      .finally(() => { if (!disposed) setPageLoading(false) })
    return () => { disposed = true }
  }, [api, detail.summary.id, detail.summary.updatedAt, pageIndex])
  const round = pageRounds.find(item => item.round === roundNumber) ?? pageRounds.at(-1)
  const visible = visibleRecommendations(round?.top10 ?? [], query, sort)
  const displayed = visible.slice(rowPage * 50, rowPage * 50 + 50)
  useEffect(() => { setRowPage(0) }, [round?.round, query, sort])
  const hasRows = total > 0
  const config = detail.summary.config
  const candidateCount = config?.proposal_count ?? 40
  const k = config?.k ?? 10
  const usesModel = detail.summary.method === 'llm' || detail.summary.method === 'knowledge'
  const totalRequests = usesModel ? Math.min(detail.summary.rounds * (config?.llm_max_calls_per_round ?? 2), config?.max_model_calls ?? Infinity) : 0
  return <>
    <div className="gb1-run-header" data-run-id={detail.summary.id} data-run-status={detail.summary.status}>
      <div><span className="gb1-status-pill">{STATUS[detail.summary.status]}</span>
        <h2>推荐结果</h2><p className="gb1-subtle">{METHOD_LABELS[detail.summary.method]} · {detail.summary.model.id} · 种子 {detail.summary.seed} · C {candidateCount} / k {k}</p>
      </div>
      <Button size="sm" variant="outline" disabled={!hasRows || exporting} onClick={onExport}>{exporting ? '正在导出…' : '导出所有轮次CSV'}</Button>
    </div>
    {detail.summary.error && <div className="gb1-run-error" role="alert">{detail.summary.error}</div>}
    {detail.summary.stopReason && <p className="gb1-run-error">停止原因：{detail.summary.stopReason} · 实际完成 {detail.summary.completedRounds}/{detail.summary.rounds} 轮</p>}
    <details className="gb1-saved-config"><summary>查看本次保存的实验参数</summary>{config ? <><p className="gb1-help">提示版本 {detail.summary.promptProfile ?? '未提供'} · 思考强度 {config.llm_reasoning_effort ?? '模型默认'} · 重复编号 {(detail.summary.repeatIndex ?? 0) + 1}</p><pre>{JSON.stringify(config, null, 2)}</pre></> : <p className="gb1-help">历史运行配置：C=40、k=10、每轮最多2次生成，单次输出8000 tokens、超时180秒。其余字段以当时运行记录为准。</p>}</details>
    {total > 10 && <div className="gb1-pagination"><span>轮次页 {pageIndex + 1}/{Math.ceil(total / 10)} · 共 {total} 个已保存轮次</span><div><Button size="sm" disabled={pageLoading || pageIndex === 0} onClick={() => { setPageIndex(value => value - 1); setRoundNumber(null) }}>上一页</Button><Button size="sm" disabled={pageLoading || (pageIndex + 1) * 10 >= total} onClick={() => { setPageIndex(value => value + 1); setRoundNumber(null) }}>下一页</Button></div></div>}
    {pageError && <p className="gb1-run-error" role="alert">{pageError}</p>}
    {pageLoading && <p className="gb1-help" role="status">正在读取本页结果…</p>}
    <div className="gb1-summary-strip" aria-label="本次运行摘要">
      <div><span>已知记录</span><strong>{detail.initialKnownCount > 0 ? `${countText(detail.initialKnownCount)} → ${countText(detail.knownCount)}` : '—'}</strong></div>
      <div><span>生成请求</span><strong>{countText(detail.summary.requestCount)} / {totalRequests}</strong></div>
      <div><span>已报告tokens</span><strong>{countText(detail.usage.totalTokens)}{detail.usage.missingUsageCount > 0 && <small> · 部分用量未知</small>}</strong></div>
    </div>
    {!round ? <div className="gb1-empty gb1-r-waiting"><h3>{detail.summary.status === 'running' ? '正在准备本轮候选' : '暂无推荐结果'}</h3>
      <p>{detail.summary.status === 'running' ? '本地评分和实验反馈完成后，可在这里查看已保存的推荐。' : '本次运行尚未保存候选推荐，可在工作对话中查看执行状态。'}</p></div> : <>
      <div className="gb1-r-rounds" aria-label="选择推荐轮次">
        <div className="gb1-r-round-buttons">{pageRounds.map(item => <Button key={item.round} size="sm"
          variant={round.round === item.round ? 'outline' : 'ghost'} aria-pressed={round.round === item.round}
          onClick={() => { setRoundNumber(item.round); setExpanded(null) }}>第{item.round}轮</Button>)}</div>
        <span className="gb1-subtle">可用候选 {round.validCount}/{candidateCount} · 已评分 {round.scoredCount}</span>
      </div>
      <div className="gb1-r-round-metrics" aria-label="当前轮次指标">
        <div><span>本轮实测平均</span><strong>{scoreText(round.meanFitness)}</strong></div>
        <div><span>本轮实测最高</span><strong>{scoreText(round.maxFitness)}</strong></div>
        <div><span>本轮实测中位数</span><strong>{scoreText(round.medianFitness)}</strong></div>
        <div><span>累计最佳（含原有已知）</span><strong>{scoreText(round.cumulativeBest)}</strong></div>
      </div>
      <p className="gb1-r-method-note">分数来自 S1，野生型为 1。残差 = 实测 − 预测；正值表示模型低估，负值表示模型高估。</p>
      {round.validCount >= k && round.validCount < candidateCount && <p className="gb1-help">本轮可用候选未达到{candidateCount}个，已按实际数量完成筛选。</p>}
      <div className="gb1-r-filters">
        <div><label htmlFor={`${localId}-search`}>搜索本轮推荐</label><Input id={`${localId}-search`} type="search" value={query}
          placeholder="变体、突变、理由或规则 ID" onChange={event => setQuery(event.target.value)}/></div>
        <div><label htmlFor={`${localId}-sort`}>显示顺序</label><select id={`${localId}-sort`} value={sort}
          onChange={event => setSort(event.target.value as ResultSort)}>{SORTS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></div>
      </div>
      <div className="gb1-r-list-meta"><span role="status">显示 {visible.length} / {round.top10.length} 条推荐</span><span>排名保留原记录 · CSV 导出所有轮次</span></div>
      {round.top10.length === 0 ? <div className="gb1-empty gb1-r-waiting"><h3>本轮还没有已选候选</h3><p>候选尚未保存，分数不会显示为零。</p></div>
        : visible.length === 0 ? <div className="gb1-empty gb1-r-no-match"><h3>没有符合条件的推荐</h3><p>尝试其他变体编号、突变或关键词。</p>
          <Button size="sm" variant="outline" onClick={() => setQuery('')}>清除搜索</Button></div>
          : <div className="gb1-r-table-scroll" tabIndex={0} role="region" aria-label={`第${round.round}轮推荐表，可横向滚动`}>
            <table className="gb1-recommendations"><caption>第{round.round}轮 Top-{round.top10.length} 推荐，分数单位为相对野生型 fitness</caption>
              <thead><tr><th scope="col">原排名</th><th scope="col">变体 / 替换</th><th scope="col" className="gb1-r-number">预测分数</th>
                <th scope="col" className="gb1-r-number">实验数据分数</th><th scope="col" className="gb1-r-number">残差</th><th scope="col" className="gb1-r-number">突变数量</th><th scope="col">详情</th></tr></thead>
              <tbody>{displayed.map(row => {
                const open = expanded === row.variant
                const sectionId = `${localId}-round-${round.round}-rank-${row.rank}`
                const error = residual(row)
                return <Fragment key={row.variant}><tr className={open ? 'gb1-r-selected-row' : undefined}>
                  <td className="gb1-rank">{row.rank}</td><td><span className="gb1-variant">{row.variant}</span><span className="gb1-r-mutation-label">{row.mutations || '未提供'}</span></td>
                  <td className="gb1-r-number">{scoreText(row.predictedFitness)}</td><td className="gb1-r-number">{scoreText(row.actualFitness)}</td>
                  <td className={`gb1-r-number${error === null ? '' : error > 0 ? ' gb1-r-residual-positive' : error < 0 ? ' gb1-r-residual-negative' : ''}`}>{scoreText(error, true)}</td>
                  <td className="gb1-r-number">{countText(row.mutationCount)}</td><td><Button size="sm" variant="ghost" aria-expanded={open} aria-controls={sectionId}
                    aria-label={`${open ? '收起' : '展开'} ${row.variant} 的序列与设计详情`} onClick={() => setExpanded(open ? null : row.variant)}>{open ? '收起' : '详情'}</Button></td>
                </tr>{open && <tr className="gb1-r-expanded-row"><td colSpan={7}><div id={sectionId}>
                  <RecommendationDetails row={row} reference={detail.reference} onCopy={onCopy}/>
                </div></td></tr>}</Fragment>
              })}</tbody>
            </table>
          </div>}
      {visible.length > 50 && <div className="gb1-pagination"><span>推荐页 {rowPage + 1}/{Math.ceil(visible.length / 50)} · 每页50条</span><div><Button size="sm" disabled={rowPage === 0} onClick={() => setRowPage(value => value - 1)}>上一页</Button><Button size="sm" disabled={(rowPage + 1) * 50 >= visible.length} onClick={() => setRowPage(value => value + 1)}>下一页</Button></div></div>}
      <details className="gb1-r-round-analysis"><summary>查看本轮公开分析 <span className="gb1-subtle">{round.analysis.length} 条模型响应摘要</span></summary>
        {round.analysis.length ? round.analysis.map((text, index) => <p key={index}>{text}</p>) : <p className="gb1-subtle">本轮没有已保存的公开分析。</p>}
      </details>
    </>}
  </>
}

export function ResultsPanel(props: ResultsPanelProps) {
  return <section className="gb1-results-panel" aria-label="推荐结果与序列比较"><style>{css}</style>
    {props.detail ? <ResultsView key={props.detail.summary.id} detail={props.detail} api={props.api} exporting={props.exporting} onCopy={props.onCopy} onExport={props.onExport}/>
      : <div className="gb1-empty"><h3>查看候选的预测与实验反馈</h3><p>开始一次实验，或从运行记录中打开已有结果，即可比较推荐序列。</p></div>}
  </section>
}
