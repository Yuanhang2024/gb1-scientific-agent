import React, { useId } from 'react'
import { finite, format, signed, type MeasuredRecommendation } from './analysisData.ts'

export interface ChartSeries { label: string; color: string; values: (number | null)[]; dash?: boolean }
export const CHART_COLORS = ['var(--ga-blue)', 'var(--ga-teal)', 'var(--ga-orange)', 'var(--ga-purple)'] as const
const WIDTH = 640, HEIGHT = 290, LEFT = 58, RIGHT = 20, TOP = 25, BOTTOM = 50
function yRange(values: number[], includeZero = true): [number, number] {
  const lo = includeZero ? Math.min(0, ...values) : Math.min(...values)
  const hi = includeZero ? Math.max(0, ...values) : Math.max(...values)
  const span = hi - lo || Math.max(1, Math.abs(hi) * 0.2)
  return [lo < 0 ? lo - span * 0.08 : 0, hi + span * 0.12]
}
function axesY(domain: [number, number], y: (value: number) => number) {
  return Array.from({ length: 5 }, (_, index) => {
    const value = domain[0] + (domain[1] - domain[0]) * index / 4
    return <g key={index}><line x1={LEFT} x2={WIDTH - RIGHT} y1={y(value)} y2={y(value)} className="ga-grid-line" />
      <text x={LEFT - 10} y={y(value) + 4} textAnchor="end" className="ga-axis-text">{format(value, 2)}</text></g>
  })
}
export function Legend({ series }: { series: readonly Pick<ChartSeries, 'label' | 'color' | 'dash'>[] }) {
  return <div className="ga-legend">{series.map(item => <span key={item.label}><i style={{ borderColor: item.color, borderTopStyle: item.dash ? 'dashed' : 'solid' }} />{item.label}</span>)}</div>
}

/** 缺失值切断曲线，单个轮次只画点，不把 null 转成零。 */
export function LineChart({ labels, series, title }: { labels: string[]; series: readonly ChartSeries[]; title: string }) {
  const id = useId()
  const values = series.flatMap(item => item.values.filter(finite))
  if (!values.length) return <div className="ga-chart-empty">尚无已评估轮次，完成后显示趋势。</div>
  const domain = yRange(values)
  const x = (index: number) => labels.length === 1 ? (LEFT + WIDTH - RIGHT) / 2 : LEFT + index * (WIDTH - LEFT - RIGHT) / Math.max(1, labels.length - 1)
  const y = (value: number) => HEIGHT - BOTTOM - (value - domain[0]) * (HEIGHT - TOP - BOTTOM) / (domain[1] - domain[0])
  return <div className="ga-chart"><svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={id}>
    <title id={id}>{title}</title>
    {axesY(domain, y)}
    <text x={LEFT} y={13} className="ga-axis-caption">fitness</text>
    {labels.map((label, index) => (index === 0 || index === labels.length - 1 || index % Math.max(1, Math.ceil(labels.length / 8)) === 0) ? <text key={index} x={x(index)} y={HEIGHT - BOTTOM + 25} textAnchor="middle" className="ga-axis-text">{label}</text> : null)}
    {series.map(item => {
      const segments: { index: number; value: number }[][] = []
      let current: { index: number; value: number }[] = []
      item.values.forEach((value, index) => {
        if (finite(value)) current.push({ index, value })
        else if (current.length) { segments.push(current); current = [] }
      })
      if (current.length) segments.push(current)
      return <g key={item.label} style={{ color: item.color }}>
        {segments.filter(segment => segment.length > 1).map((segment, index) => <polyline key={index} points={segment.map(point => `${x(point.index)},${y(point.value)}`).join(' ')} fill="none" stroke="currentColor" strokeWidth={2.3} strokeDasharray={item.dash ? '6 5' : undefined} vectorEffect="non-scaling-stroke" />)}
        {item.values.map((value, index) => finite(value) ? <circle key={index} cx={x(index)} cy={y(value)} r={4.5} fill="var(--ga-surface)" stroke="currentColor" strokeWidth={2.3}>
          <title>{labels[index]} · {item.label}：{format(value)}</title></circle> : null)}
      </g>
    })}
  </svg><Legend series={series} /></div>
}

export function PredictionChart({ rows, residual = false }: { rows: readonly MeasuredRecommendation[]; residual?: boolean }) {
  const id = useId()
  if (!rows.length) return <div className="ga-chart-empty">尚无预测与实测配对。</div>
  const xDomain = yRange(rows.flatMap(row => residual ? [row.predictedFitness] : [row.predictedFitness, row.actualFitness]))
  const yDomain = residual ? yRange(rows.map(row => row.residual)) : xDomain
  const x = (value: number) => LEFT + (value - xDomain[0]) * (WIDTH - LEFT - RIGHT) / (xDomain[1] - xDomain[0])
  const y = (value: number) => HEIGHT - BOTTOM - (value - yDomain[0]) * (HEIGHT - TOP - BOTTOM) / (yDomain[1] - yDomain[0])
  return <div className="ga-chart"><svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={id}>
    <title id={id}>{residual ? '所选推荐的预测残差，实测减预测' : '所选推荐的预测 fitness 与实测 fitness'}</title>
    {axesY(yDomain, y)}
    {Array.from({ length: 5 }, (_, index) => {
      const value = xDomain[0] + (xDomain[1] - xDomain[0]) * index / 4
      return <text key={index} x={x(value)} y={HEIGHT - BOTTOM + 25} className="ga-axis-text" textAnchor="middle">{format(value, 2)}</text>
    })}
    <text x={LEFT} y={13} className="ga-axis-caption">{residual ? '实测 − 预测' : '实测 fitness'}</text>
    <text x={(LEFT + WIDTH - RIGHT) / 2} y={HEIGHT - 5} className="ga-axis-caption" textAnchor="middle">预测 fitness</text>
    {residual ? <line x1={LEFT} x2={WIDTH - RIGHT} y1={y(0)} y2={y(0)} className="ga-reference-line" />
      : <line x1={x(xDomain[0])} x2={x(xDomain[1])} y1={y(xDomain[0])} y2={y(xDomain[1])} className="ga-reference-line" />}
    {rows.map(row => <circle key={`${row.round}-${row.variant}`} cx={x(row.predictedFitness)} cy={y(residual ? row.residual : row.actualFitness)} r={5}
      fill={CHART_COLORS[(row.round - 1) % 3]} stroke="var(--ga-surface)" strokeWidth={1.1} opacity={0.8}>
      <title>{`第${row.round}轮 ${row.variant} · 预测 ${format(row.predictedFitness)} · 实测 ${format(row.actualFitness)} · 残差 ${signed(row.residual)}`}</title>
    </circle>)}
  </svg><Legend series={[...new Set(rows.map(row => row.round))].sort((a, b) => a - b).map(round => ({ label: `第${round}轮`, color: CHART_COLORS[(round - 1) % 3]! }))} /></div>
}
