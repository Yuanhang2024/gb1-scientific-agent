import { memo, useEffect, useRef, useState } from 'react'
import { Button, DisclosureRow, MarkdownText, StateDot, UserMessageBubble } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ConversationMessage, ConversationSnapshot, RunDetail } from '../contracts.ts'
import type { WorkbenchApi } from './api.ts'

const LABELS = { code: { copyLabel: '复制', copiedLabel: '已复制' }, footnotes: '参考' }
const ROLES = { coordinator: '实验调度器', model: '候选设计模型', tool: '实验工具' }
const STATUS = { queued: '等待运行', running: '进行中', paused: '已暂停', stopped: '提前结束', completed: '已完成', failed: '未完成', cancelled: '已停止', interrupted: '已中断' }
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted', 'paused', 'stopped'])
// 公开分析按文字展示；图片引用保留为可点击文字，避免自动请求外部图片。
const safeMarkdown = (text: string) => text.replace(/!\[/g, '[')
const timeText = (time: string) => new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date(time))

function RoleIcon({ actor }: { actor: ConversationMessage['actor'] }) {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {actor === 'model' ? <path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3Z"/> : actor === 'tool' ? <><path d="M14 6a5 5 0 0 0-6 6L3 17a2.8 2.8 0 0 0 4 4l5-5a5 5 0 0 0 6-6l-3 3-4-4 3-3Z"/></> : <><rect x="3" y="3" width="6" height="5" rx="1.5"/><rect x="15" y="16" width="6" height="5" rx="1.5"/><path d="M6 8v8h9m-3-3 3 3-3 3"/></>}
  </svg>
}

function stateOf(message: ConversationMessage) {
  return message.status === 'running' ? 'ongoing' : message.status === 'completed' ? 'done' : message.status === 'failed' ? 'error' : 'warning'
}

const MessageCard = memo(function MessageCard({ message }: { message: ConversationMessage }) {
  const [choice, setChoice] = useState<boolean | null>(null)
  const running = message.status === 'running'
  const expanded = choice ?? (message.kind === 'analysis' || message.kind === 'error')
  const toggle = () => setChoice(!expanded)
  const heading = <div className="gb1-message-heading"><span className="gb1-role-icon"><RoleIcon actor={message.actor}/></span><strong>{ROLES[message.actor]}</strong>{message.round !== null && message.round > 0 && <span>第{message.round}轮</span>}<time>{timeText(message.time)}</time></div>
  if (message.actor === 'coordinator') return <article className="gb1-process-message gb1-coordinator-message" data-message-id={message.id} data-message-status={message.status} data-message-kind={message.kind} data-message-role={message.actor} data-tool-name={message.toolName ?? ''}>
    {heading}<UserMessageBubble className="gb1-coordinator-bubble"><strong>{message.title}</strong>{message.text && <p>{message.text}</p>}{message.inputSummary && <p className="gb1-subtle">{message.inputSummary}</p>}{message.outputSummary && <p>{message.outputSummary}</p>}</UserMessageBubble>
  </article>
  if (message.kind === 'tool') return <article className="gb1-process-message gb1-tool-message" data-message-id={message.id} data-message-status={message.status} data-message-kind={message.kind} data-message-role={message.actor} data-tool-name={message.toolName ?? ''}>
    {heading}<DisclosureRow icon={<StateDot state={stateOf(message)} appearance="step"/>} title={message.title} open={expanded} expandable={true} onToggle={toggle} running={running} expandOnRowClick={true} previewChevron={true}
      collapsedContent={<span className="gb1-process-summary">{message.outputSummary || message.text || STATUS[message.status]}</span>}>
      <div className="gb1-tool-detail">{message.toolName && <code>{message.toolName}</code>}<dl>
        {message.inputSummary && <><dt>输入</dt><dd>{message.inputSummary}</dd></>}
        {message.outputSummary && <><dt>输出</dt><dd>{message.outputSummary}</dd></>}
        <dt>状态</dt><dd>{STATUS[message.status]}</dd>
      </dl>{message.text && <p>{message.text}</p>}</div>
    </DisclosureRow>
  </article>
  return <article className="gb1-process-message gb1-model-message" data-message-id={message.id} data-message-status={message.status} data-message-kind={message.kind} data-message-role={message.actor} data-tool-name={message.toolName ?? ''}>
    {heading}<DisclosureRow icon={<StateDot state={stateOf(message)}/>} title={message.kind === 'analysis' ? '公开分析与设计说明' : message.title} open={expanded} expandable={true} onToggle={toggle} running={running} expandOnRowClick={true}
      collapsedContent={<span className="gb1-process-summary">{message.kind === 'analysis' ? message.title : STATUS[message.status]}</span>}>
      <div className="gb1-analysis-body">{message.text ? <MarkdownText text={safeMarkdown(message.text)} streaming={running} labels={LABELS} variant="compact"/> : <p className="gb1-subtle">{running ? '等待模型返回公开分析内容…' : '本次没有可展示的公开分析。'}</p>}
        {!running && <span className="gb1-message-state">{message.kind === 'analysis' && message.status === 'completed' ? '公开输出已接收；候选检查结果见后续工具记录' : STATUS[message.status]}</span>}
      </div>
    </DisclosureRow>
  </article>
}, (left, right) => {
  const a = left.message, b = right.message
  return a.id === b.id && a.status === b.status && a.text === b.text && a.title === b.title && a.inputSummary === b.inputSummary && a.outputSummary === b.outputSummary && a.time === b.time && a.round === b.round
})

export function ConversationPanel({ api, detail, onViewResults }: { api: WorkbenchApi; detail: RunDetail | null; onViewResults: () => void }) {
  const id = detail?.summary.id ?? null
  const [snapshot, setSnapshot] = useState<ConversationSnapshot | null>(null)
  const [connection, setConnection] = useState<'connecting' | 'live' | 'reconnecting' | 'finished' | 'error'>('connecting')
  const [error, setError] = useState('')
  const [restart, setRestart] = useState(0)
  const [following, setFollowing] = useState(true)
  const scroller = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)

  useEffect(() => {
    setSnapshot(null); setError(''); setConnection('connecting')
    followRef.current = true; setFollowing(true)
    if (!id) return
    let disposed = false, generation = -1, revision = -1
    const stream = api.followConversation(id, () => { if (!disposed) setConnection('reconnecting') })
    void (async () => {
      try {
        for await (const item of stream) {
          if (disposed || item.signal.aborted) continue
          const value = item.value
          if (value.runId !== id || !Number.isSafeInteger(value.revision) || value.revision < 0) throw new Error('运行过程记录与当前实验不一致。')
          if (item.generation !== generation) { generation = item.generation; revision = -1 }
          if (value.revision >= revision) { revision = value.revision; setSnapshot(value) }
          item.accept()
          setConnection(TERMINAL.has(value.status) ? 'finished' : 'live')
          if (TERMINAL.has(value.status)) break
        }
      } catch (cause) {
        if (!disposed) { setConnection('error'); setError(cause instanceof Error ? cause.message : '过程连接暂时不可用。') }
      } finally { await stream.dispose().catch(() => {}) }
    })()
    // 离开页面只关闭这条订阅，后台实验仍由独立的停止按钮管理。
    return () => { disposed = true; void stream.dispose().catch(() => {}) }
  }, [api, id, restart, detail?.summary.status])

  useEffect(() => {
    if (!followRef.current || !scroller.current) return
    const frame = requestAnimationFrame(() => { if (followRef.current && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight })
    return () => cancelAnimationFrame(frame)
  }, [snapshot?.revision])

  const onScroll = () => {
    const element = scroller.current
    if (!element) return
    const close = element.scrollHeight - element.clientHeight - element.scrollTop < 48
    followRef.current = close; setFollowing(close)
  }
  const follow = () => {
    followRef.current = true; setFollowing(true)
    if (scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
  }

  if (!detail) return <div className="gb1-empty"><RoleIcon actor="coordinator"/><h3>实验过程会显示在这里</h3><p>开始实验后，按实际顺序查看调度、模型公开分析和工具调用。</p></div>
  return <div className="gb1-conversation" data-testid="gb1-conversation" data-conversation-revision={snapshot?.revision ?? -1}>
    <div className="gb1-run-header" data-run-id={detail.summary.id} data-run-status={detail.summary.status}>
      <div><span className="gb1-status-pill">{STATUS[detail.summary.status]}</span><h2>工作对话 · {detail.summary.rounds}轮</h2><p className="gb1-subtle">{detail.summary.model.id} · 实验调度器与工具协作</p></div>
      <Button size="sm" variant="outline" onClick={onViewResults}>查看推荐结果</Button>
    </div>
    <div className="gb1-conversation-status" role="status" aria-live="polite"><span>{connection === 'live' ? '实时更新' : connection === 'connecting' ? '正在读取过程' : connection === 'reconnecting' ? '正在恢复连接，保留已有内容' : connection === 'error' ? '连接暂时不可用' : '过程已保存'}</span><span className="gb1-subtle">仅展示模型公开输出和实际操作</span></div>
    {error && <div className="gb1-message gb1-message-error" role="alert"><span>{error}</span><Button size="sm" onClick={() => setRestart(value => value + 1)}>重新连接</Button></div>}
    {snapshot?.historyOnly && <p className="gb1-process-note">该运行保存的是阶段与结果摘要，以下按已有记录展示。</p>}
    {snapshot?.truncated && <p className="gb1-process-note">部分较长内容已按展示上限保留摘要。</p>}
    <div className="gb1-process-scroll" ref={scroller} onScroll={onScroll} tabIndex={0} aria-label="实验工作过程" role="log" aria-live="off" data-testid="gb1-process-scroll">
      {!snapshot?.messages.length && <p className="gb1-process-placeholder">正在等待本次运行的过程记录…</p>}
      {snapshot?.messages.map(message => <MessageCard key={message.id} message={message}/>)}
    </div>
    <div className="gb1-process-footer"><span className="gb1-subtle">{following ? '正在跟随最新内容' : '已暂停跟随，可向上阅读'}</span>{!following && <Button size="sm" variant="outline" onClick={follow}>回到最新</Button>}</div>
  </div>
}
