import { useMemo, useState } from 'react'
import { Button, Input } from '@deepseek-ai/dsh-client-ui-primitives'
import type { RunMethod, RunStatus, RunSummary } from '../../../contracts.ts'

const STATUS: Record<RunStatus, string> = {
  queued: '等待运行', running: '运行中', paused: '已暂停', stopped: '提前结束', completed: '已完成', failed: '未完成', cancelled: '已停止', interrupted: '已中断',
}
const METHOD: Record<RunMethod, string> = { random: '随机推荐', model: '预测模型推荐', llm: '基础 Agent', knowledge: '知识辅助 Agent' }
const dateFormat = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
})

function timeText(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : dateFormat.format(date)
}

export function HistoryPanel({ runs, onOpen }: { runs: RunSummary[]; onOpen: (id: string) => void }) {
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState<RunStatus | 'all'>('all')
  const [method, setMethod] = useState<RunMethod | 'all'>('all')
  const [page, setPage] = useState(0)
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase()
    return runs.filter(run => (status === 'all' || run.status === status)
      && (method === 'all' || run.method === method)
      && (!query || [run.id, run.model.id, run.model.provider, METHOD[run.method], STATUS[run.status], timeText(run.createdAt), String(run.seed)]
        .join(' ').toLocaleLowerCase().includes(query)))
  }, [runs, search, status, method])

  const safePage = Math.min(page, Math.max(0, Math.ceil(filtered.length / 20) - 1))
  return <section className="gb1-history" aria-labelledby="gb1-history-heading">
    <div className="gb1-panel-heading">
      <div><h2 id="gb1-history-heading">运行记录</h2><p>重新查看已保存的推荐与工作过程。</p></div>
      <span className="gb1-subtle">共 {runs.length} 次运行</span>
    </div>
    {runs.length === 0 ? <div className="gb1-empty">
      <h3>还没有运行记录</h3><p>开始一次实验后，结果会保存在这里。</p>
    </div> : <>
      <div className="gb1-history-filters">
        <div className="gb1-history-search">
          <label htmlFor="gb1-history-search">搜索记录</label>
          <Input id="gb1-history-search" type="search" value={search} placeholder="模型、时间或运行编号" onChange={event => setSearch(event.target.value)}/>
        </div>
        <div>
          <label htmlFor="gb1-history-status">运行状态</label>
          <select id="gb1-history-status" value={status} onChange={event => setStatus(event.target.value as RunStatus | 'all')}>
            <option value="all">全部状态</option>
            {Object.entries(STATUS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor="gb1-history-method">推荐方式</label>
          <select id="gb1-history-method" value={method} onChange={event => setMethod(event.target.value as RunMethod | 'all')}>
            <option value="all">全部方式</option>
            {Object.entries(METHOD).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
      </div>
      <p className="gb1-history-count gb1-subtle" role="status">显示 {filtered.length} 条记录</p>
      {filtered.length === 0 ? <div className="gb1-empty gb1-history-no-match">
        <h3>没有符合条件的记录</h3><p>调整关键词或筛选条件后重试。</p>
        <Button size="sm" variant="outline" onClick={() => { setSearch(''); setStatus('all'); setMethod('all') }}>清除筛选</Button>
      </div> : <ul className="gb1-history-list">
        {filtered.slice(safePage * 20, safePage * 20 + 20).map(run => <li key={run.id} data-run-id={run.id} data-run-status={run.status}>
          <div className="gb1-history-record">
            <div className="gb1-history-record-heading">
              <time dateTime={run.createdAt}>{timeText(run.createdAt)}</time>
              <span className="gb1-status-pill">{STATUS[run.status]}</span>
            </div>
            <strong className="gb1-history-model" title={`${run.model.provider} / ${run.model.id}`}>{run.model.id}</strong>
            <div className="gb1-history-meta">
              <span>{METHOD[run.method]}</span><span>{run.model.source === 'dsh' ? 'DSH 模型' : '独立接口'}</span>
              <span>{run.completedRounds}/{run.rounds} 轮</span><span>种子 {run.seed}</span>
            </div>
          </div>
          <Button size="sm" variant="outline" aria-label={`查看 ${timeText(run.createdAt)} 的${METHOD[run.method]}运行`} onClick={() => onOpen(run.id)}>查看</Button>
        </li>)}
      </ul>}
      {filtered.length > 20 && <div className="gb1-pagination"><span>第 {safePage + 1}/{Math.ceil(filtered.length / 20)} 页</span><div><Button size="sm" disabled={safePage === 0} onClick={() => setPage(safePage - 1)}>上一页</Button><Button size="sm" disabled={(safePage + 1) * 20 >= filtered.length} onClick={() => setPage(safePage + 1)}>下一页</Button></div></div>}
    </>}
  </section>
}
