import React from 'react'
import type { RunDetail } from '../../../contracts.ts'
import { CurrentAnalysis } from './CurrentAnalysis.tsx'
import { HistoryAnalysis } from './HistoryAnalysis.tsx'
import styles from './analysis.css'

export function AnalysisPanel({ detail }: { detail: RunDetail | null }) {
  return <div className="ga-panel" data-testid="gb1-analysis-panel"><style>{styles}</style>
    {detail ? <CurrentAnalysis key={detail.summary.id} detail={detail} />
      : <section className="ga-current ga-start-state"><span className="ga-kicker">当前运行</span><h2>让每一轮结果都能比较</h2>
        <p>选择已有运行，即可查看轮次变化、预测偏差与推荐位点；也可以先浏览下方已完成的四方法历史实验。</p>
        <div className="ga-start-labels"><span>轮次表现</span><span>预测与实测</span><span>替换分布</span><span>案例回看</span></div></section>}
    <div className="ga-experiment-divider"><span>独立的历史实验</span></div>
    <HistoryAnalysis />
  </div>
}
