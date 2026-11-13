import { useEffect, useId, useState } from 'react'
import { Input } from '@deepseek-ai/dsh-client-ui-primitives'

export function NumberField({ label, value, onChange, disabled, min, max, integer = true, nullable = false, hint }: {
  label: string; value: number | null; onChange: (value: number | null) => void
  disabled?: boolean; min?: number; max?: number; integer?: boolean; nullable?: boolean; hint?: string
}) {
  const id = useId()
  const [text, setText] = useState(value === null ? '' : String(value))
  useEffect(() => { setText(value === null ? '' : String(value)) }, [value])
  const valid = text.trim() === '' ? nullable : Number.isFinite(Number(text)) && (!integer || Number.isSafeInteger(Number(text)))
    && (min === undefined || Number(text) >= min) && (max === undefined || Number(text) <= max)
  return <div className="gb1-config-field"><label htmlFor={id}>{label}</label>
    <Input id={id} type="number" value={text} min={min !== undefined && Number.isFinite(min) ? min : undefined} max={max} step={integer ? 1 : 'any'} placeholder={nullable ? '留空不限制 / 沿用默认' : undefined}
      disabled={disabled} aria-invalid={!valid} aria-describedby={hint || !valid ? `${id}-hint` : undefined}
      onChange={event => {
        const next = event.target.value
        setText(next)
        if (!next.trim()) { if (nullable) onChange(null); return }
        const number = Number(next)
        if (Number.isFinite(number) && (!integer || Number.isSafeInteger(number)) && (min === undefined || number >= min) && (max === undefined || number <= max)) onChange(number)
      }} onBlur={() => { if (!valid) setText(value === null ? '' : String(value)) }}/>
    {(hint || !valid) && <p id={`${id}-hint`} className="gb1-help">{!valid ? '请输入允许范围内的数值。' : hint}</p>}
  </div>
}

export function NumberListField({ label, value, onChange, disabled, min = 0, max, integer = true }: {
  label: string; value: number[]; onChange: (value: number[]) => void; disabled?: boolean; min?: number; max?: number; integer?: boolean
}) {
  const id = useId()
  const [text, setText] = useState(value.join(', '))
  useEffect(() => { setText(value.join(', ')) }, [value.join(',')])
  const parse = (input: string) => {
    const parts = input.trim().split(/[,，\s]+/).filter(Boolean)
    const numbers = parts.map(Number)
    return parts.length && numbers.every(number => Number.isFinite(number) && (!integer || Number.isSafeInteger(number)) && number >= min && (max === undefined || number <= max)) ? [...new Set(numbers)] : null
  }
  const valid = Boolean(parse(text))
  return <div className="gb1-config-field"><label htmlFor={id}>{label}</label>
    <Input id={id} value={text} disabled={disabled} aria-invalid={!valid} onChange={event => setText(event.target.value)}
      onBlur={() => { const values = parse(text); if (values) onChange(values); else setText(value.join(', ')) }}/>
    <p className="gb1-help">{valid ? '多个数值用逗号分隔，离开输入框后保存。' : '请输入允许范围内的数值列表。'}</p>
  </div>
}
