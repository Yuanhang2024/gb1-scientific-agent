import { useState } from 'react'
import { Button, Input, SegmentedControl } from '@deepseek-ai/dsh-client-ui-primitives'
import type { ExperimentConfiguration, ModelCapabilities, ModelOverride, RunMethod } from '../../../contracts.ts'
import { NumberField, NumberListField } from './ConfigurationFields.tsx'

export const METHOD_LABELS: Record<RunMethod, string> = { random: '随机推荐', model: '预测模型推荐', llm: '基础 Agent', knowledge: '知识辅助 Agent' }
type NumericKey = { [K in keyof ExperimentConfiguration]-?: ExperimentConfiguration[K] extends number | null ? K : never }[keyof ExperimentConfiguration]
export interface ExperimentSetupPanelProps {
  reference: string; method: RunMethod; config: ExperimentConfiguration; defaults: ExperimentConfiguration
  promptProfile: string; defaultPromptProfile: string; capabilities: ModelCapabilities | null; capabilitiesError: string; customModel: boolean
  disabled: boolean; loading: boolean
  recommendedModel: ModelOverride | null; recommendationProblem: string; onApplyRecommendation: () => void
  onReferenceChange: (value: string) => void; onMethodChange: (value: RunMethod) => void
  onConfigChange: (value: ExperimentConfiguration) => void; onProfileChange: (value: string) => void
}

export function ExperimentSetupPanel(props: ExperimentSetupPanelProps) {
  const { reference, method, config, defaults, promptProfile, defaultPromptProfile, capabilities, capabilitiesError, customModel, disabled, loading,
    recommendedModel, recommendationProblem, onApplyRecommendation, onReferenceChange, onMethodChange, onConfigChange, onProfileChange } = props
  const [section, setSection] = useState<'basic' | 'advanced'>('basic')
  const locked = disabled || loading
  const change = <K extends keyof ExperimentConfiguration>(key: K, value: ExperimentConfiguration[K]) => onConfigChange({ ...config, [key]: value })
  const number = (key: NumericKey, label: string, min = 1, nullable = false, integer = true, max?: number, hint?: string) =>
    <NumberField key={key} label={label} value={config[key]} disabled={locked} min={min} max={max} nullable={nullable} integer={integer} hint={hint}
      onChange={value => change(key, value as never)}/>
  const toggle = (key: 'repair_examples' | 'require_full_batch' | 'save_predictors', label: string) => <label className="gb1-checkbox"><input type="checkbox" checked={config[key]} disabled={locked} onChange={event => change(key, event.target.checked)}/><span>{label}</span></label>
  const usesModel = method === 'llm' || method === 'knowledge'
  const requests = usesModel ? Math.min(config.rounds * config.llm_max_calls_per_round, config.max_model_calls ?? Infinity) : 0
  const assays = Math.min(config.rounds * config.k, config.max_assays === null ? Infinity : Math.floor(config.max_assays / config.k) * config.k)
  return <section className="gb1-control-section" aria-labelledby="gb1-experiment-heading">
    <div className="gb1-section-heading"><h2 id="gb1-experiment-heading">本次实验</h2><Button size="sm" disabled={locked} onClick={() => { onConfigChange(structuredClone(defaults)); onProfileChange(defaultPromptProfile) }}>重置实验参数</Button></div>
    {recommendedModel && <div className="gb1-fields" aria-label="推荐设置">
      <p className="gb1-help"><strong>推荐模型：{recommendedModel.model}</strong><span className="gb1-block">{recommendedModel.provider} · 思考强度 {defaults.llm_reasoning_effort ?? '模型默认'} · 提示版本 {defaultPromptProfile}</span></p>
      <Button size="sm" variant="outline" disabled={locked || Boolean(recommendationProblem)} onClick={onApplyRecommendation}>应用推荐设置</Button>
      <p className="gb1-help" role={recommendationProblem ? 'status' : undefined}>{recommendationProblem || '将推荐模型与默认实验参数填入表单，确认后可保存连接或开始实验。'}</p>
    </div>}
    <SegmentedControl id="gb1-config-section" label="设置范围" value={section} options={[{ value: 'basic', label: '基础' }, { value: 'advanced', label: '高级' }]} onChange={setSection}/>
    {section === 'basic' ? <>
      <label htmlFor="gb1-preset">实验预设</label><select id="gb1-preset" value="" disabled={locked} onChange={event => {
        const preset = event.target.value
        if (!preset) return
        const next = structuredClone(defaults)
        if (preset === 'single') next.rounds = 1
        if (preset === 'feedback') next.rounds = 3
        onConfigChange(next); onProfileChange(defaultPromptProfile)
      }}><option value="">选择并应用预设</option><option value="default">默认实验参数</option><option value="single">单轮观察</option><option value="feedback">三轮反馈比较</option></select>
      <label htmlFor="gb1-method">推荐方式</label><select id="gb1-method" value={method} disabled={locked} onChange={event => onMethodChange(event.target.value as RunMethod)}>
        {Object.entries(METHOD_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
      <div className="gb1-field-pair">{number('rounds', '运行轮数')}{number('seed', '数据随机种子', 0, false, true, 4294967295)}</div>
      <div className="gb1-field-pair"><NumberField label="每轮候选数 C" value={config.proposal_count} disabled={locked} min={1} onChange={value => {
        if (value !== null) onConfigChange({ ...config, proposal_count: value, score_budget: value, k: Math.min(config.k, value) })
      }}/>{number('k', '每轮推荐数 k', 1, false, true, config.proposal_count)}</div>
      <p className="gb1-help">每轮评分上限随候选数同步，推荐数不超过候选数。</p>
      <label htmlFor="gb1-reasoning">模型思考强度</label><select id="gb1-reasoning" value={config.llm_reasoning_effort ?? ''} disabled={locked || !usesModel || !capabilities || customModel} onChange={event => change('llm_reasoning_effort', event.target.value || null)}>
        <option value="">沿用模型默认</option>{capabilities?.reasoningEfforts.map(value => <option key={value} value={value}>{value}</option>)}
      </select>
      <p className="gb1-help">{customModel ? '此控件仅适用于 DSH 已注册模型。独立接口不发送思考强度参数，沿用服务端默认设置。' : capabilitiesError || (capabilities ? `输出额度 ${capabilities.defaultMaxOutputTokens?.toLocaleString() ?? '由模型决定'} · 上下文 ${capabilities.contextWindow?.toLocaleString() ?? '未提供'}` : '选择已注册模型后读取受支持档位。')}</p>
      <details className="gb1-reference-disclosure"><summary>参考序列与突变位置</summary><label htmlFor="gb1-reference">野生型序列</label>
        <textarea id="gb1-reference" rows={3} spellCheck={false} value={reference} disabled={locked} onChange={event => onReferenceChange(event.target.value)} className="gb1-reference"/>
        <p className="gb1-help">56 个氨基酸；可研究位置为 39、40、41、54。</p></details>
    </> : <div className="gb1-advanced">
      <label htmlFor="gb1-profile">科学提示版本</label><select id="gb1-profile" value={promptProfile} disabled={locked} onChange={event => onProfileChange(event.target.value)}>{[...new Set(['v1', 'v2', 'v3', promptProfile])].map(value => <option key={value} value={value}>{value}</option>)}</select>
      <details open><summary>模型请求与输出</summary>
        {number('llm_max_calls_per_round', '每轮最多生成请求')}
        {number('llm_max_output_tokens', '单次输出 token 上限', 128, true)}{number('llm_timeout', '单次请求超时（秒）', 0.001, true, false)}
        {number('llm_temperature', 'Temperature', 0, true, false, 2)}
        <label htmlFor="gb1-format">响应格式</label><select id="gb1-format" aria-describedby="gb1-format-help" value={config.llm_format} disabled={locked} onChange={event => change('llm_format', event.target.value)}><option value="json_schema">JSON Schema</option><option value="json_object">JSON Object</option><option value="text">文本 JSON</option></select>
        <p id="gb1-format-help" className="gb1-help">仅独立 OpenAI 兼容接口生效；DSH 使用提示与本地 JSON 校验。</p>
        <label htmlFor="gb1-oversize">候选数超过上限时</label><select id="gb1-oversize" value={config.oversize_policy} disabled={locked} onChange={event => change('oversize_policy', event.target.value as ExperimentConfiguration['oversize_policy'])}><option value="reject">拒绝整批并反馈</option><option value="first_n">仅检查原始顺序前 C 条</option></select>
        {toggle('repair_examples', '纠正请求提供无标签替补')}{toggle('require_full_batch', '要求候选数达到 C')}
        {number('analysis_max_chars', '公开分析最大字符数')}{number('proposal_text_max_chars', '候选说明最大字符数')}
      </details>
      <details><summary>数据与预测器</summary>
        {number('train_size', '初始训练数')}{number('validation_size', '验证集数量')}{number('test_size', '保留测试集数量')}{number('pool_size', '候选池数量', 1, true)}
        <label>预测模型类别</label>{['ridge', 'random_forest', 'extra_trees'].map(family => <label key={family} className="gb1-checkbox"><input type="checkbox" disabled={locked || (config.model_families.length === 1 && config.model_families.includes(family))} checked={config.model_families.includes(family)} onChange={event => change('model_families', event.target.checked ? [...config.model_families, family] : config.model_families.filter(value => value !== family))}/><span>{family}</span></label>)}
        <NumberListField label="Ridge alpha 列表" value={config.ridge_alphas} integer={false} min={0.000001} disabled={locked} onChange={value => change('ridge_alphas', value)}/>
        {number('tree_estimators', '树的数量')}<NumberListField label="叶节点最少样本数列表" value={config.tree_min_samples_leaf} min={1} disabled={locked} onChange={value => change('tree_min_samples_leaf', value)}/>
        {number('tree_max_depth', '最大树深', 1, true)}{number('n_jobs', '本地并行线程', 1)}
        {number('model_seed', '预测器随机种子', 0, true, true, 4294967295)}{number('design_seed', '候选设计随机种子', 0, true, true, 4294967295)}
        {toggle('save_predictors', '保存每轮预测器')}
      </details>
      <details><summary>上下文与知识规则</summary>
        {number('summary_top', '已知高分摘要数量')}{number('contrast_examples', '对照样例数量', 0)}{number('feedback_rounds', '最近反馈轮数', 0)}
        <label htmlFor="gb1-knowledge-path">项目内公开知识文件</label><Input id="gb1-knowledge-path" value={config.knowledge_path} disabled={locked} onChange={event => change('knowledge_path', event.target.value)}/>
      </details>
      <details><summary>突变范围</summary>
        <label>允许突变位置</label><div className="gb1-checkbox-group">{[39, 40, 41, 54].map(position => <label className="gb1-checkbox" key={position}><input type="checkbox" disabled={locked || (config.allowed_positions.length === 1 && config.allowed_positions.includes(position))} checked={config.allowed_positions.includes(position)} onChange={event => change('allowed_positions', event.target.checked ? [...config.allowed_positions, position].sort((a, b) => a - b) : config.allowed_positions.filter(value => value !== position))}/><span>{position}</span></label>)}</div>
        <label htmlFor="gb1-amino-acids">允许氨基酸</label><Input id="gb1-amino-acids" value={config.allowed_amino_acids} disabled={locked} onChange={event => change('allowed_amino_acids', [...new Set(event.target.value.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, ''))].join(''))}/>
        <div className="gb1-field-pair">{number('min_mutations', '最少替换数', 0, false, true, config.max_mutations)}{number('max_mutations', '最多替换数', config.min_mutations, false, true, 4)}</div>
      </details>
      <details><summary>预算与提前停止</summary>
        {number('max_assays', '最多实测查询数量', 1, true)}{number('max_model_calls', '最多模型请求数量', 1, true)}{number('max_reported_tokens', '已报告 token 阈值', 1, true)}
        {number('target_fitness', '目标 fitness', -Infinity, true, false)}{number('patience', '允许连续未改善轮数', 1, true)}{number('min_improvement', '最低改善量', 0, false, false)}
        <label htmlFor="gb1-early-stop">改善指标</label><select id="gb1-early-stop" value={config.early_stop_metric} disabled={locked} onChange={event => change('early_stop_metric', event.target.value as ExperimentConfiguration['early_stop_metric'])}><option value="best">累计最佳</option><option value="round_mean">本轮推荐均值</option></select>
        <p className="gb1-help">查询预算在整轮开始前检查。token 阈值在请求前检查，单次返回可能超过阈值；用量未知时停止继续请求。</p>
      </details>
    </div>}
    <div className="gb1-budget" aria-live="polite"><strong>本次预算预览</strong><span>最多 {requests.toLocaleString()} 次生成请求（含纠正）</span><span>最多 {assays.toLocaleString()} 条实测查询 · {config.rounds * config.score_budget} 次候选评分</span></div>
  </section>
}
