import { Allow, parse } from 'partial-json'

export interface ModelProgress {
  kind: 'analysis'
  requestId: string
  round: number
  text: string
  status: 'streaming' | 'completed'
}
export type ProgressCallback = (event: ModelProgress) => void | Promise<void>
const PARTIAL_DISPLAY = Allow.OBJ | Allow.ARR | Allow.STR
const MAX_ANALYSIS_CHARACTERS = 6000
const MAX_ANALYSIS_ITEMS = 8

/** 只读取own analysis的字符串；解析对象及其原型绝不进入公开事件。 */
export function projectAnalysis(text: string): string[] | null {
  let value: unknown
  try { value = parse(text, PARTIAL_DISPLAY) } catch { return null }
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Object.hasOwn(value, 'analysis')) return []
  const analysis = (value as Record<string, unknown>).analysis
  if (typeof analysis === 'string') return [analysis.slice(0, MAX_ANALYSIS_CHARACTERS)]
  if (!Array.isArray(analysis)) return []
  const result: string[] = []
  for (let index = 0, length = 0; index < analysis.length && result.length < MAX_ANALYSIS_ITEMS; index++) {
    if (!Object.hasOwn(analysis, index) || typeof analysis[index] !== 'string') continue
    const part = (analysis[index] as string).slice(0, MAX_ANALYSIS_CHARACTERS - length)
    result.push(part); length += part.length
    if (length >= MAX_ANALYSIS_CHARACTERS) break
  }
  return result
}

function protectedForms(values: string[]): string[] {
  const forms = new Set<string>()
  for (const value of values.filter(Boolean)) {
    for (const variant of [value, JSON.stringify(value).slice(1, -1), value.replaceAll('/', '\\/'), encodeURI(value), encodeURIComponent(value)]) {
      forms.add(variant)
      forms.add(variant.replace(/%[0-9A-F]{2}/g, match => match.toLowerCase()))
    }
  }
  return [...forms].filter(Boolean)
}

/** 完整秘密立即拒绝；未完成流中可能组成秘密的尾部先扣留，不能先落盘再撤回。 */
export class PublicTextGuard {
  private readonly forms: string[]
  private rawTail = ''
  constructor(values: string[] = []) { this.forms = protectedForms(values) }
  inspectChunk(text: string): void {
    const combined = this.rawTail + text
    this.assertSafe(combined)
    const maximum = Math.max(0, ...this.forms.map(value => value.length - 1))
    this.rawTail = maximum ? combined.slice(-maximum) : ''
  }
  assertSafe(text: string): void {
    if (this.forms.some(value => text.includes(value))) throw new Error('模型输出包含不可公开的连接内容')
  }
  safeDraft(text: string, final = false): string {
    this.assertSafe(text)
    if (final) return text
    let withheld = 0
    for (const value of this.forms) {
      const maximum = Math.min(value.length - 1, text.length)
      for (let count = maximum; count > withheld; count--) {
        if (text.endsWith(value.slice(0, count))) { withheld = count; break }
      }
    }
    return withheld ? text.slice(0, -withheld) : text
  }
  assertJsonSafe(value: unknown): void {
    if (typeof value === 'string') this.assertSafe(value)
    else if (Array.isArray(value)) for (const item of value) this.assertJsonSafe(item)
    else if (value && typeof value === 'object') {
      for (const key of Object.keys(value)) { this.assertSafe(key); this.assertJsonSafe((value as Record<string, unknown>)[key]) }
    }
  }
}

export class AnalysisProgress {
  private lastText: string | undefined
  private lastStatus: ModelProgress['status'] | undefined
  constructor(private readonly requestId: string, private readonly round: number,
    private readonly callback?: ProgressCallback, private readonly guard = new PublicTextGuard()) {}
  async update(visibleText: string, final = false): Promise<void> {
    this.guard.assertSafe(visibleText)
    let projection = projectAnalysis(visibleText)
    if (final) {
      try {
        const complete: unknown = JSON.parse(visibleText)
        if (!complete || typeof complete !== 'object' || Array.isArray(complete) || !Object.hasOwn(complete, 'analysis')) projection = []
      } catch { projection = [] }
    }
    if (projection === null && !final) return
    // JSON解码后的公开字符串再检查一次；不能仅扫描HTTP原文。
    if (projection) for (const value of projection) this.guard.assertSafe(value)
    let text = projection?.join('\n\n') ?? this.lastText ?? ''
    text = this.guard.safeDraft(text, final)
    if (!final && !text) return
    const status = final ? 'completed' : 'streaming'
    if (text === this.lastText && status === this.lastStatus) return
    this.lastText = text; this.lastStatus = status
    await this.callback?.({ kind: 'analysis', requestId: this.requestId, round: this.round, text, status })
  }
}
