/** 来自已完成的公开报告；构建时固定，独立于当前运行。 */
export interface HistoryEstimate { mean: number; sd: number }
export interface HistoryMethod {
  id: 'random' | 'model' | 'llm' | 'knowledge'
  label: string
  mean: HistoryEstimate
  bestNew: HistoryEstimate
  rounds: HistoryEstimate[]
  selectedCount: number
  candidateMin: number
  candidateMax: number
  siteFractions: number[]
}
export const HISTORY_V1 = {
  prompt: 'v1', seeds: [11, 22, 33], rounds: 3, totalRecommendations: 360,
  sources: ['reports/summary.json', 'reports/all_recommendations.csv', 'reports/site_statistics.csv'],
  methods: [
    { id: 'random', label: '随机推荐', mean: { mean: 0.17412084965083666, sd: 0.07144710305607177 },
      bestNew: { mean: 3.2412451791666665, sd: 1.9309242630011976 },
      rounds: [{ mean: 0.13399446941484233, sd: 0.1412967282123295 }, { mean: 0.03653541427101767, sd: 0.030205380536323795 }, { mean: 0.35183266526665, sd: 0.16937227540249933 }],
      selectedCount: 90, candidateMin: 40, candidateMax: 40, siteFractions: [0.9777777777777777, 0.9666666666666667, 0.9444444444444444, 0.9555555555555556] },
    { id: 'model', label: '模型直接推荐', mean: { mean: 0.38612226034570185, sd: 0.24021137058106717 },
      bestNew: { mean: 4.02937168646, sd: 2.5210021001105454 },
      rounds: [{ mean: 0.312303215439285, sd: 0.3383838222018925 }, { mean: 0.4011647753045131, sd: 0.5199959558178326 }, { mean: 0.4448987902933073, sd: 0.36864879331689543 }],
      selectedCount: 90, candidateMin: 40, candidateMax: 40, siteFractions: [0.9666666666666667, 0.9444444444444444, 0.8444444444444444, 0.9333333333333333] },
    { id: 'llm', label: '基础 Agent', mean: { mean: 2.0839732867194773, sd: 1.7949266691344299 },
      bestNew: { mean: 5.123271606186667, sd: 0.8179817283330078 },
      rounds: [{ mean: 2.6899815844740123, sd: 1.7245940138362468 }, { mean: 1.8162939144259478, sd: 2.5904058706838597 }, { mean: 1.7456443612584724, sd: 1.8818242401251108 }],
      selectedCount: 90, candidateMin: 39, candidateMax: 40, siteFractions: [0.9777777777777777, 0.9666666666666667, 0.9111111111111111, 0.9666666666666667] },
    { id: 'knowledge', label: '知识辅助 Agent', mean: { mean: 1.514581416260635, sd: 1.017660995741691 },
      bestNew: { mean: 5.010068138786667, sd: 0.1514440769710222 },
      rounds: [{ mean: 0.6581892641046666, sd: 0.45124187091877993 }, { mean: 2.5579525187658696, sd: 1.9778531940558604 }, { mean: 1.3276024659113697, sd: 1.5241754053372547 }],
      selectedCount: 90, candidateMin: 10, candidateMax: 40, siteFractions: [0.9555555555555556, 0.9888888888888889, 0.8777777777777778, 0.9444444444444444] },
  ] satisfies HistoryMethod[],
} as const
