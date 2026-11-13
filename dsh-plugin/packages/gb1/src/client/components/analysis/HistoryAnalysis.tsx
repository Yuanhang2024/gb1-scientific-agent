import React from 'react'
import { CHART_COLORS, LineChart } from './Charts.tsx'
import { format, SITES } from './analysisData.ts'
import { HISTORY_V1 } from './historyData.ts'

export function HistoryAnalysis() {
  const methods = HISTORY_V1.methods
  const largestMean = Math.max(...methods.map(method => method.mean.mean))
  return <section className="ga-history" aria-label="已完成的历史 v1 四方法对照">
    <div className="ga-section-heading"><div><span className="ga-kicker">独立历史实验 · 固定数据</span><h2>四种推荐方法的首次对照</h2>
      <p>提示版本 v1 · 种子 11 / 22 / 33 · 各 3 轮 · 每轮 Top-10</p></div><span className="ga-tag">历史 v1</span></div>
    <p className="ga-history-scope">来自已完成报告的 {HISTORY_V1.totalRecommendations} 条实测推荐，与上方当前 v3 运行分别展示。下列均值和样本标准差均按三个种子计算。</p>
    <article className="ga-card"><div className="ga-card-heading"><div><h3>新增推荐的平均质量</h3><p>每个种子合并三轮 30 条推荐，再汇总三个种子</p></div></div>
      <div className="ga-method-bars">{methods.map((method, index) => <div key={method.id} className="ga-method-bar"><div><strong>{method.label}</strong><span>{format(method.mean.mean)} <small>± {format(method.mean.sd)}</small></span></div>
        <div className="ga-track" role="img" aria-label={`${method.label}新增推荐平均fitness ${format(method.mean.mean)}，种子样本标准差 ${format(method.mean.sd)}`}><i style={{ width: `${method.mean.mean / largestMean * 100}%`, background: CHART_COLORS[index] }} /></div></div>)}</div>
      <p className="ga-note">横条表示均值，右侧为均值 ± 种子样本标准差。每种方法各 90 条推荐。</p>
    </article>
    <article className="ga-card"><div className="ga-card-heading"><div><h3>各轮推荐均值</h3><p>每条曲线均为三个种子的该轮平均值</p></div></div>
      <LineChart title="历史v1四方法在三个轮次的平均推荐fitness" labels={['第1轮', '第2轮', '第3轮']}
        series={methods.map((method, index) => ({ label: method.label, color: CHART_COLORS[index]!, values: method.rounds.map(round => round.mean) }))} />
      <details className="ga-data-details"><summary>查看四方法逐轮数值</summary><div className="ga-table-wrap"><table><caption>历史 v1 轮次推荐均值 ± 种子样本标准差，n = 3</caption>
        <thead><tr><th>方法</th>{[1, 2, 3].map(round => <th key={round}>第{round}轮</th>)}</tr></thead>
        <tbody>{methods.map(method => <tr key={method.id}><th>{method.label}</th>{method.rounds.map((round, index) => <td key={index}>{format(round.mean)} ± {format(round.sd)}</td>)}</tr>)}</tbody></table></div></details>
    </article>
    <article className="ga-card"><div className="ga-card-heading"><div><h3>最高新增与位点分布</h3><p>最高值与平均质量使用各自定义，不混为同一指标</p></div></div>
      <div className="ga-table-wrap"><table><caption>历史 v1 新增最高 fitness：每个种子取三轮最高，再计算均值与样本标准差</caption>
        <thead><tr><th>方法</th><th>最高新增<br />均值 ± SD</th><th>有效候选 / 轮</th><th>推荐数</th></tr></thead><tbody>{methods.map(method => <tr key={method.id}><th>{method.label}</th><td>{format(method.bestNew.mean)} ± {format(method.bestNew.sd)}</td><td>{method.candidateMin === method.candidateMax ? method.candidateMax : `${method.candidateMin}–${method.candidateMax}`}</td><td>{method.selectedCount}</td></tr>)}</tbody></table></div>
      <details className="ga-data-details"><summary>查看四位点的替换频率</summary><div className="ga-table-wrap"><table><caption>相对 VDGV，各方法 90 条推荐中的位点替换比例</caption>
        <thead><tr><th>方法</th>{SITES.map(position => <th key={position}>{position}</th>)}</tr></thead><tbody>{methods.map(method => <tr key={method.id}><th>{method.label}</th>{method.siteFractions.map((value, index) => <td key={index}>{format(value * 100, 1)}%</td>)}</tr>)}</tbody></table></div></details>
    </article>
    <aside className="ga-method-note"><strong>原实验的候选预算</strong>
      <p>随机与模型直接推荐每轮随机抽取 40 个候选；两种 Agent 自主设计最多 40 个候选，随后选取 Top-10。Agent 的实际有效数量保留为 10–40，未自动补齐。</p>
      <p>该对照比较固定评分上限下的候选生成方式，模型方法未穷举候选池。三个种子的均值与波动用于描述本次实验。</p>
    </aside>
    <footer className="ga-source-note">公开报告来源：<code>reports/summary.json</code>、<code>reports/all_recommendations.csv</code>、<code>reports/site_statistics.csv</code>。历史摘要随插件构建保存，不由当前运行更新。</footer>
  </section>
}
