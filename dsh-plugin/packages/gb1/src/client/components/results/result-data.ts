import type { Recommendation } from '../../../contracts.ts'

export const EXPERIMENT_POSITIONS = [39, 40, 41, 54] as const
export type ResultSort = 'rank' | 'actual' | 'predicted' | 'residual' | 'mutations'

export function finiteValue(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
export function scoreText(value: number | null | undefined, signed = false): string {
  const finite = finiteValue(value)
  return finite === null ? '—' : `${signed && finite > 0 ? '+' : ''}${finite.toFixed(3)}`
}
export function residual(row: Recommendation): number | null {
  const actual = finiteValue(row.actualFitness), predicted = finiteValue(row.predictedFitness)
  return actual === null || predicted === null ? null : actual - predicted
}

// 只创建显示用副本；过滤和排序不修改保存的排名、候选或整次运行导出。
export function visibleRecommendations(rows: Recommendation[], query: string, sort: ResultSort): Recommendation[] {
  const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  const result = rows.filter(row => {
    const text = [row.variant, row.mutations, row.sequence, row.rationale, row.hypothesis, ...row.knowledgeRuleIds].join(' ').toLocaleLowerCase()
    return words.every(word => text.includes(word))
  })
  const value = (row: Recommendation) => sort === 'actual' ? finiteValue(row.actualFitness)
    : sort === 'predicted' ? finiteValue(row.predictedFitness)
      : sort === 'residual' ? (residual(row) === null ? null : Math.abs(residual(row)!))
        : sort === 'mutations' ? finiteValue(row.mutationCount) : finiteValue(row.rank)
  return result.sort((a, b) => {
    const av = value(a), bv = value(b)
    if (av === null && bv !== null) return 1
    if (av !== null && bv === null) return -1
    const comparison = av === null || bv === null ? 0 : sort === 'rank' || sort === 'mutations' ? av - bv : bv - av
    return comparison || a.rank - b.rank || a.variant.localeCompare(b.variant)
  })
}

export function sequenceCells(reference: string, sequence: string) {
  if (reference.length !== 56 || sequence.length !== 56 || !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(reference + sequence)) return null
  return Array.from(sequence, (residue, index) => ({ position: index + 1, residue, reference: reference[index]!,
    allowed: EXPERIMENT_POSITIONS.some(position => position === index + 1), changed: residue !== reference[index] }))
}
