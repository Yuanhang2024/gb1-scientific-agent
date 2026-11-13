import { Button, Input, SegmentedControl } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ConnectionSettings, ModelChoice } from '../../../contracts.ts'

export interface ModelConnectionPanelProps {
  connection: ConnectionSettings
  apiKey: string
  models: ModelChoice[]
  modelsError: string
  disabled: boolean
  loading: boolean
  ready: boolean
  onChange: (connection: ConnectionSettings) => void
  onApiKeyChange: (value: string) => void
  onRefreshModels: () => void
  onSave: (clearKey?: boolean) => void
}

export function ModelConnectionPanel({
  connection, apiKey, models, modelsError, disabled, loading, ready,
  onChange, onApiKeyChange, onRefreshModels, onSave,
}: ModelConnectionPanelProps) {
  const modelValue = JSON.stringify([connection.provider, connection.model])
  const listed = models.some(model => JSON.stringify([model.provider, model.id]) === modelValue)

  return <section className="gb1-control-section" aria-labelledby="gb1-connection-heading">
    <div className="gb1-section-heading">
      <h2 id="gb1-connection-heading">模型连接</h2>
      <span className="gb1-subtle">提出候选方案</span>
    </div>
    <SegmentedControl
      id="gb1-source"
      label="模型连接来源"
      value={connection.source}
      options={[{ value: 'dsh', label: 'DSH 模型' }, { value: 'custom', label: '独立接口' }]}
      disabled={disabled || loading}
      onChange={source => onChange({ ...connection, source })}
    />
    {connection.source === 'dsh' ? <div id="gb1-source-dsh-panel" role="tabpanel" aria-labelledby="gb1-source-dsh" className="gb1-fields">
      <label htmlFor="gb1-model">已注册模型</label>
      <select id="gb1-model" value={modelValue} disabled={disabled || loading} onChange={event => {
        const found = models.find(model => JSON.stringify([model.provider, model.id]) === event.target.value)
        if (found) onChange({ ...connection, provider: found.provider, model: found.id })
      }}>
        <option value={JSON.stringify(['', ''])}>请选择 DSH 中的模型</option>
        {connection.model && !listed && <option value={modelValue}>{connection.model}（已保存，等待列表确认）</option>}
        {models.map(model => <option key={JSON.stringify([model.provider, model.id])} value={JSON.stringify([model.provider, model.id])}>
          {model.name || model.id} · {model.provider}
        </option>)}
      </select>
      <div className="gb1-inline-note">
        <span>沿用 DSH 中的连接设置。</span>
        <Button size="sm" disabled={disabled || loading} onClick={onRefreshModels}>刷新列表</Button>
      </div>
      {modelsError && <p className="gb1-field-error" role="status">{modelsError}</p>}
    </div> : <div id="gb1-source-custom-panel" role="tabpanel" aria-labelledby="gb1-source-custom" className="gb1-fields">
      <label htmlFor="gb1-base-url">Base URL</label>
      <Input id="gb1-base-url" type="url" placeholder="https://api.example.com/v1" value={connection.baseUrl} disabled={disabled || loading} onChange={event => onChange({ ...connection, baseUrl: event.target.value })}/>
      <label htmlFor="gb1-custom-model">Model ID</label>
      <Input id="gb1-custom-model" placeholder="服务提供的模型名称" value={connection.model} disabled={disabled || loading} onChange={event => onChange({ ...connection, model: event.target.value })}/>
      <label htmlFor="gb1-api-key">API Key <span className="gb1-subtle">{connection.hasKey ? '已保存' : '未设置'}</span></label>
      <Input id="gb1-api-key" type="password" autoComplete="new-password" placeholder={connection.hasKey ? '留空沿用已保存密钥' : '填写独立接口的密钥'} value={apiKey} disabled={disabled || loading} onChange={event => onApiKeyChange(event.target.value)}/>
      <p className="gb1-help">支持 OpenAI 兼容接口。密钥由 DSH 凭据服务保存；更换地址后需重新填写密钥。</p>
    </div>}
    <div className="gb1-connection-actions">
      <Button size="sm" variant="outline" disabled={disabled || loading || !ready || !connection.model} onClick={() => onSave()}>保存连接</Button>
      {connection.source === 'custom' && connection.hasKey && <Button size="sm" disabled={disabled || loading || Boolean(apiKey)} onClick={() => onSave(true)}>移除密钥</Button>}
    </div>
  </section>
}
