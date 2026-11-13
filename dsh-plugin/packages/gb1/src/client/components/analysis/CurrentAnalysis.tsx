import React, { useId, useState } from 'react'
import type { RunDetail } from '../../../contracts.ts'
import { CHART_COLORS, LineChart, PredictionChart, type ChartSeries } from './Charts.tsx'
import { chooseCases, finite, format, hammingGroups, measuredRecommendations, selectedError, signed, siteDistribution, type MeasuredRecommendation } from './analysisData.ts'

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="ga-metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
}
function DataDetails({ children, label = '查看数值' }: { children: React.ReactNode; label?: string }) {
  return <details className="ga-data-details"><summary>{label}</summary><div className="ga-table-wrap">{children}</div></details>
}
function CaseCard({ row, overestimated }: { row: MeasuredRecommendation; overestimated?: boolean }) {
  return <article className="ga-case"><header><code>{row.variant}</code><span>第{row.round}轮 · {row.mutationCount} 处替换</span></header>
    <div className="ga-case-values"><span>实测 <strong>{format(row.actualFitness)}</strong></span><span>预测 {format(row.predictedFitness)}</span>
      <span>残差 {signed(row.residual)}</span></div>
    <p className="ga-case-mutations">{row.mutations}</p>
    <div className="ga-case-text"><span>原假设</span><p>{row.hypothesis || '未记录'}</p></div>
    <details><summary>推荐理由与规则引用</summary><p>{row.rationale || '未记录'}</p>
      <p className="ga-note">规则 ID：{row.knowledgeRuleIds.length ? row.knowledgeRuleIds.join('、') : '未引用'}</p></details>
    {overestimated && <p className="ga-note">预测高于实测 {format(-row.residual)}。该差异记录本次预测误差。</p>}
  </article>
}

export function CurrentAnalysis({ detail }: { detail: RunDetail }) {
  const [roundFilter, setRoundFilter] = useState('all')
  const [plot, setPlot] = useState<'prediction' | 'residual'>('prediction')
  const selectId = useId(), plotId = useId()
  const allRows = measuredRecommendations(detail)
  const rows = roundFilter === 'all' ? allRows : allRows.filter(row => row.round === Number(roundFilter))
  const rounds = [...(detail.roundMetrics ?? detail.roundResults)].sort((a, b) => a.round - b.round)
  const loadedRounds = detail.roundResults.map(round => round.round)
  const partial = rounds.length > loadedRounds.length
  const k = detail.summary.config?.k ?? 10
  const candidateCount = detail.summary.config?.proposal_count ?? 40
  const method = { random: '随机推荐', model: '预测模型推荐', llm: '基础 Agent', knowledge: '知识辅助 Agent' }[detail.summary.method]
  const error = selectedError(rows), sites = siteDistribution(rows), groups = hammingGroups(rows), cases = chooseCases(rows)
  const actual = rows.map(row => row.actualFitness)
  const mean = actual.length ? actual.reduce((sum, value) => sum + value, 0) / actual.length : null
  const best = actual.length ? Math.max(...actual) : null
  const series: ChartSeries[] = [
    { label: '本轮推荐均值', color: CHART_COLORS[0], values: rounds.map(round => finite(round.meanFitness) ? round.meanFitness : null) },
    { label: '本轮推荐最高', color: CHART_COLORS[1], values: rounds.map(round => finite(round.maxFitness) ? round.maxFitness : null) },
    { label: '累计最佳（含初始已知）', color: CHART_COLORS[2], dash: true, values: rounds.map(round => finite(round.cumulativeBest) ? round.cumulativeBest : null) },
  ]
  return <section className="ga-current" aria-label="当前运行分析">
    <div className="ga-section-heading"><div><span className="ga-kicker">当前运行 · {detail.summary.promptProfile ?? '历史提示版本'}</span><h2>从推荐到实测</h2>
      <p>{method} · 种子 {detail.summary.seed} · 已完成 {detail.summary.completedRounds}/{detail.summary.rounds} 轮</p></div>
      <span className="ga-tag">只统计当前运行</span></div>
    {partial && <p className="ga-note">轮次趋势包含全部 {rounds.length} 轮。推荐细看仅统计当前加载的 {loadedRounds.length} 轮；其他推荐可在推荐结果中分页查看，CSV包含全部已保存轮次。</p>}
    <div className="ga-current-metrics">
      <Metric label="已测推荐" value={`${allRows.length}`} note={`每轮最多 ${k} 条 · 已知记录 ${detail.knownCount}`} />
      <Metric label="新增推荐最高" value={format(allRows.length ? Math.max(...allRows.map(row => row.actualFitness)) : null)} note={partial ? '当前已加载轮次的新测推荐' : '仅本次新测推荐，不含初始数据'} />
      <Metric label="推荐均值" value={format(allRows.length ? allRows.reduce((sum, row) => sum + row.actualFitness, 0) / allRows.length : null)} note={partial ? '当前已加载轮次的推荐合并' : '所有已评估轮次的推荐合并'} />
    </div>

    <article className="ga-card ga-trend-card"><div className="ga-card-heading"><div><h3>轮次表现</h3><p>本轮质量与累计最佳分开展示</p></div><span className="ga-small-label">fitness 越高越好</span></div>
      <LineChart labels={rounds.map(round => `第${round.round}轮`)} series={series} title="当前运行各轮推荐均值、最高及含初始已知结果的累计最佳" />
      <p className="ga-note">累计最佳包含初始已知实验结果。未完成或缺失的数值留空；单轮结果显示为点，未推算初始值。</p>
      <DataDetails label="查看每轮数值"><table><caption>当前运行的已保存轮次指标</caption><thead><tr><th>轮次</th><th>有效候选</th><th>推荐均值</th><th>推荐最高</th><th>累计最佳</th></tr></thead>
        <tbody>{rounds.map(round => <tr key={round.round}><th>第{round.round}轮</th><td>{round.validCount}/{candidateCount}</td><td>{format(round.meanFitness)}</td><td>{format(round.maxFitness)}</td><td>{format(round.cumulativeBest)}</td></tr>)}</tbody></table></DataDetails>
    </article>

    <div className="ga-analysis-controls"><div><h3>推荐细看</h3><p>统计范围为已加载且取得实测结果的 Top-{k} 推荐</p></div>
      <div className="ga-control"><label htmlFor={selectId}>轮次</label><select id={selectId} value={roundFilter} onChange={event => setRoundFilter(event.target.value)}>
        <option value="all">全部已加载轮次</option>{detail.roundResults.map(round => <option key={round.round} value={round.round}>第{round.round}轮</option>)}</select></div></div>
    {!rows.length ? <div className="ga-empty-inline"><strong>等待实测反馈</strong><p>收到本轮推荐结果后，在这里展示预测偏差、位点分布和案例。</p></div> : <>
      <article className="ga-card"><div className="ga-card-heading"><div><h3>预测与实测</h3><p>{rows.length} 条配对 · 均值 {format(mean)} · 最高 {format(best)}</p></div>
        <div className="ga-control"><label htmlFor={plotId}>图形</label><select id={plotId} value={plot} onChange={event => setPlot(event.target.value as 'prediction' | 'residual')}>
          <option value="prediction">预测 vs 实测</option><option value="residual">预测残差</option></select></div></div>
        <PredictionChart rows={rows} residual={plot === 'residual'} />
        <p className="ga-note">{plot === 'residual' ? '残差 = 实测 − 预测。零线下方为高估，上方为低估。' : '虚线为预测 = 实测。点在线下方时，模型高估该变体。'} 悬停点可查看变体和分数。</p>
        <div className="ga-error-strip"><span>平均绝对误差 <strong>{format(error.meanAbsoluteError)}</strong></span><span>平均残差 <strong>{format(error.meanResidual)}</strong></span><span>高估 <strong>{error.overestimated}/{rows.length}</strong></span></div>
        <p className="ga-note">误差只描述模型已选出的推荐，不作为独立测试集表现。</p>
        <DataDetails label="查看配对分数与残差"><table><caption>当前筛选范围的推荐预测与实测</caption><thead><tr><th>变体</th><th>轮次</th><th>预测</th><th>实测</th><th>实测 − 预测</th></tr></thead>
          <tbody>{rows.map(row => <tr key={`${row.round}-${row.variant}`}><th><code>{row.variant}</code></th><td>{row.round}</td><td>{format(row.predictedFitness)}</td><td>{format(row.actualFitness)}</td><td>{signed(row.residual)}</td></tr>)}</tbody></table></DataDetails>
      </article>

      <div className="ga-distribution-grid"><article className="ga-card"><div className="ga-card-heading"><div><h3>四个位点的替换频率</h3><p>相对参考 VDGV · 分母为 {rows.length} 条已测推荐</p></div></div>
        <div className="ga-site-list">{sites.map(site => <div key={site.position} className="ga-site-row"><div><strong>{site.reference}{site.position}</strong><span>{site.replacements}/{site.total} · {format(site.replacements / site.total * 100, 1)}%</span></div>
          <div className="ga-track" role="img" aria-label={`${site.position}位点 ${site.replacements}/${site.total} 条推荐发生替换`}><i style={{ width: `${site.replacements / site.total * 100}%` }} /></div>
          <p>{site.residues.slice(0, 5).map(item => `${item.amino} × ${item.count}`).join(' · ')}{site.residues.length > 5 ? ' · …' : ''}</p></div>)}</div>
        <DataDetails label="查看所有残基计数"><table><caption>各位点在推荐中的残基计数，含参考残基</caption><thead><tr><th>位点</th><th>残基与数量</th></tr></thead><tbody>{sites.map(site => <tr key={site.position}><th>{site.position}</th><td>{site.residues.map(item => `${item.amino}: ${item.count}`).join('；')}</td></tr>)}</tbody></table></DataDetails>
      </article>
      <article className="ga-card"><div className="ga-card-heading"><div><h3>替换数量分组</h3><p>HD 为四个位点相对 VDGV 的差异数</p></div></div>
        <div className="ga-hd-chart" role="img" aria-label="已测推荐按零至四处氨基酸替换分组的数量">{groups.map(group => <div key={group.distance} className="ga-hd-column"><strong>{group.count}</strong><div><i style={{ height: `${group.count / Math.max(1, ...groups.map(item => item.count)) * 100}%` }} /></div><span>{group.distance === 0 ? '参考' : `${group.distance}处`}</span></div>)}</div>
        <table className="ga-hd-table"><caption>各组推荐数量及平均实测 fitness</caption><thead><tr><th>HD</th><th>数量</th><th>实测均值</th></tr></thead><tbody>{groups.map(group => <tr key={group.distance}><th>{group.distance}</th><td>{group.count}</td><td>{format(group.mean)}</td></tr>)}</tbody></table>
        <p className="ga-note">这些分组来自本次所选推荐。组间数目和序列背景不同，不解释为替换数量的因果效应。</p>
      </article></div>

      <section className="ga-cases"><div className="ga-card-heading"><div><h3>表现较好的推荐</h3><p>按实测 fitness 降序选前两条；同分时按轮次、变体排序。</p></div></div>
        <div className="ga-case-grid">{cases.high.map(row => <CaseCard key={`${row.round}-${row.variant}`} row={row} />)}</div></section>
      <section className="ga-cases"><div className="ga-card-heading"><div><h3>高估案例</h3><p>仅取实测低于预测的推荐，按残差从小到大选前两条。</p></div></div>
        {cases.overestimated.length ? <div className="ga-case-grid">{cases.overestimated.map(row => <CaseCard key={`${row.round}-${row.variant}`} row={row} overestimated />)}</div>
          : <p className="ga-empty-caption">当前筛选范围内没有实测低于预测的推荐。</p>}</section>
    </>}
  </section>
}
