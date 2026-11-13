import { useId } from 'react'
import { Button } from '@deepseek-ai/dsh-client-ui-primitives'
import { sequenceCells } from './result-data.ts'

export function SequenceComparison({ reference, sequence, variant, onCopy }: {
  reference: string; sequence: string; variant: string; onCopy: (sequence: string) => void
}) {
  const heading = useId()
  const cells = sequenceCells(reference, sequence)
  return <section className="gb1-r-sequence" aria-labelledby={heading}>
    <div className="gb1-r-detail-heading"><h4 id={heading}>完整域序列比较</h4>
      <Button size="sm" variant="outline" onClick={() => onCopy(sequence)} aria-label={`复制 ${variant} 的完整序列`}>复制序列</Button>
    </div>
    {cells ? <>
      <p className="gb1-r-legend"><span><i className="gb1-r-site-key" aria-hidden="true"/>实验位点 39、40、41、54</span>
        <span><i className="gb1-r-change-key" aria-hidden="true"/>相对参考的实际替换</span><span>位置从 1 开始编号 · 56 aa</span></p>
      <div className="gb1-r-sequence-blocks">
        {[0, 14, 28, 42].map(start => <div className="gb1-r-sequence-block" key={start}>
          <div className="gb1-r-sequence-row gb1-r-position-row" aria-hidden="true"><span>位置</span>
            {cells.slice(start, start + 14).map(cell => <span key={cell.position}>{cell.position}</span>)}
          </div>
          <div className="gb1-r-sequence-row"><span className="gb1-r-sequence-label">参考</span>
            {cells.slice(start, start + 14).map(cell => <span key={cell.position} className={`gb1-r-residue${cell.allowed ? ' gb1-r-site' : ''}`}
              title={`参考位置 ${cell.position}：${cell.reference}`}>{cell.reference}</span>)}
          </div>
          <div className="gb1-r-sequence-row"><span className="gb1-r-sequence-label">候选</span>
            {cells.slice(start, start + 14).map(cell => <span key={cell.position}
              className={`gb1-r-residue${cell.allowed ? ' gb1-r-site' : ''}${cell.changed ? ' gb1-r-changed' : ''}`}
              title={`位置 ${cell.position}：${cell.reference}${cell.changed ? ` → ${cell.residue}` : '（未变）'}`}>{cell.residue}</span>)}
          </div>
        </div>)}
      </div>
      <p className="gb1-r-site-summary">{cells.filter(cell => cell.allowed).map(cell => <span key={cell.position}>
        <span className="gb1-r-site-number">{cell.position}</span><code>{cell.reference} → {cell.residue}</code>
        <span className="gb1-subtle">{cell.changed ? '已替换' : '未变'}</span>
      </span>)}</p>
      {cells.some(cell => cell.changed && !cell.allowed) && <p className="gb1-help">记录中含实验位点之外的差异，请检查保存的序列。</p>}
    </> : <p className="gb1-help">参考或候选序列不是完整的 56 aa，暂不显示逐位比较。</p>}
    <details className="gb1-r-plain-sequence"><summary>查看连续序列文本</summary>
      <p className="gb1-subtle">参考序列</p><code className="gb1-sequence-value">{reference || '未提供'}</code>
      <p className="gb1-subtle">候选序列</p><code className="gb1-sequence-value">{sequence || '未提供'}</code>
    </details>
  </section>
}
