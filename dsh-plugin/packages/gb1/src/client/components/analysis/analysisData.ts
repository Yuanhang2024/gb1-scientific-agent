import type { Recommendation, RunDetail } from '../../../contracts.ts'

export const SITES = [39, 40, 41, 54] as const
export const REFERENCE_VARIANT = 'VDGV'
export function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
export function format(value: number | null | undefined, digits = 3): string {
  return finite(value) ? value.toLocaleString('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '未测'
}
export function signed(value: number): string { return `${value > 0 ? '+' : ''}${format(value)}` }
export interface MeasuredRecommendation extends Recommendation { actualFitness: number; round: number; residual: number }
export function measuredRecommendations(detail: RunDetail | null): MeasuredRecommendation[] {
  return (detail?.roundResults ?? []).flatMap(round => round.top10.filter(row => finite(row.actualFitness) && finite(row.predictedFitness))
    .map(row => ({ ...row, actualFitness: row.actualFitness!, round: round.round, residual: row.actualFitness! - row.predictedFitness })))
}
export function siteDistribution(rows: readonly MeasuredRecommendation[]) {
  return SITES.map((position, index) => {
    const counts = new Map<string, number>()
    for (const row of rows) {
      const amino = row.variant[index]
      if (amino) counts.set(amino, (counts.get(amino) ?? 0) + 1)
    }
    const replacements = rows.filter(row => row.variant[index] !== REFERENCE_VARIANT[index]).length
    return { position, reference: REFERENCE_VARIANT[index]!, replacements, total: rows.length,
      residues: [...counts].map(([amino, count]) => ({ amino, count })).sort((a, b) => b.count - a.count || a.amino.localeCompare(b.amino)) }
  })
}
export function hammingGroups(rows: readonly MeasuredRecommendation[]) {
  return [0, 1, 2, 3, 4].map(distance => {
    const selected = rows.filter(row => [...row.variant].filter((amino, index) => amino !== REFERENCE_VARIANT[index]).length === distance)
    return { distance, count: selected.length, mean: selected.length ? selected.reduce((sum, row) => sum + row.actualFitness, 0) / selected.length : null }
  })
}
export function selectedError(rows: readonly MeasuredRecommendation[]) {
  if (!rows.length) return { meanAbsoluteError: null, meanResidual: null, overestimated: 0 }
  return { meanAbsoluteError: rows.reduce((sum, row) => sum + Math.abs(row.residual), 0) / rows.length,
    meanResidual: rows.reduce((sum, row) => sum + row.residual, 0) / rows.length,
    overestimated: rows.filter(row => row.residual < 0).length }
}
export function chooseCases(rows: readonly MeasuredRecommendation[]) {
  const stable = (a: MeasuredRecommendation, b: MeasuredRecommendation) => a.round - b.round || a.variant.localeCompare(b.variant)
  return {
    high: [...rows].sort((a, b) => b.actualFitness - a.actualFitness || stable(a, b)).slice(0, 2),
    overestimated: rows.filter(row => row.residual < 0).sort((a, b) => a.residual - b.residual || stable(a, b)).slice(0, 2),
  }
}
