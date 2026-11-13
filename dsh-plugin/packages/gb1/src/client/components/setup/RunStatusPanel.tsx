import { Button } from '@deepseek-ai/dsh-client-ui-primitives'
import type { RunSummary } from '../../../contracts.ts'

const STAGE: Record<string, string> = {
  starting: '准备启动', prepare: '准备数据', preparing: '准备数据',
  train: '训练评分模型', training: '训练评分模型', run: '运行实验',
  generating: '生成候选', evaluating: '核对实验分数', cancelling: '正在停止',
}

function stageText(stage: string) {
  return STAGE[stage] ?? (/^round-(\d+)$/.test(stage) ? `第 ${stage.slice(6)} 轮生成候选` : '实验运行中')
}

export function RunStatusPanel({ run, isSelected, onOpen }: {
  run: RunSummary | null
  isSelected: boolean
  onOpen: () => void
}) {
  if (!run || !['running', 'queued'].includes(run.status)) return null

  return <div className="gb1-active-run" role="status" aria-live="polite" data-active-run-id={run.id}>
    <div className="gb1-active-run-copy">
      <span className="gb1-progress-dot"/>
      <strong>{run.status === 'queued' ? '等待运行' : stageText(run.stage)}</strong>
      <span className="gb1-subtle">已完成 {run.completedRounds}/{run.rounds} 轮 · {run.model.id}</span>
    </div>
    {!isSelected && <Button size="sm" variant="outline" onClick={onOpen}>查看当前实验</Button>}
  </div>
}
