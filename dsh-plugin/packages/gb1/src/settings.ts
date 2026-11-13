import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { ExperimentConfiguration, ModelOverride } from './contracts.ts'
import { identifier, record } from './model.ts'

export const DEFAULT_CONFIG: ExperimentConfiguration = {
  seed: 11, rounds: 3, k: 10, proposal_count: 40, score_budget: 40,
  train_size: 2000, validation_size: 1000, test_size: 5000, pool_size: null,
  summary_top: 15, contrast_examples: 4, feedback_rounds: 1, top_k_metric: 10, high_fitness_quantile: 0.9,
  ridge_alphas: [0.1, 1, 10], tree_estimators: 100, tree_min_samples_leaf: [1, 3], tree_max_depth: null, n_jobs: 4,
  model_families: ['ridge', 'random_forest', 'extra_trees'], model_seed: null, design_seed: null,
  llm_max_calls_per_round: 2, llm_max_output_tokens: null, llm_timeout: null,
  llm_reasoning_effort: null, llm_temperature: null, llm_format: 'json_schema', knowledge_path: 'configs/knowledge.json',
  oversize_policy: 'reject', repair_examples: true, require_full_batch: false,
  analysis_max_chars: 5000, proposal_text_max_chars: 2000, allowed_positions: [39, 40, 41, 54],
  allowed_amino_acids: 'ACDEFGHIKLMNPQRSTVWY', min_mutations: 0, max_mutations: 4,
  max_assays: null, max_model_calls: null, max_reported_tokens: null, target_fitness: null, patience: null,
  min_improvement: 0, early_stop_metric: 'best', save_predictors: true,
}
export const validSeed = (value: unknown): value is number => Number.isInteger(value) && Number(value) >= 0 && Number(value) <= 0xffffffff
export function validateConfig(value: unknown): ExperimentConfiguration {
  const raw = record(value)
  if (Object.keys(raw).some(key => !Object.hasOwn(DEFAULT_CONFIG, key))) throw new Error('实验配置包含未知字段')
  const config = { ...structuredClone(DEFAULT_CONFIG), ...structuredClone(raw) } as ExperimentConfiguration
  for (const key of ['seed', 'model_seed', 'design_seed'] as const) {
    if (config[key] !== null && !validSeed(config[key])) throw new Error(`${key}必须是非负32位整数`)
  }
  if (config.seed === null) throw new Error('seed不能为空')
  for (const key of ['rounds', 'k', 'proposal_count', 'score_budget', 'train_size', 'validation_size', 'test_size', 'summary_top', 'tree_estimators',
    'llm_max_calls_per_round', 'analysis_max_chars', 'proposal_text_max_chars'] as const) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) throw new Error(`${key}必须是正整数`)
  }
  for (const key of ['contrast_examples', 'feedback_rounds', 'min_mutations', 'max_mutations'] as const) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 0) throw new Error(`${key}必须是非负整数`)
  }
  for (const key of ['pool_size', 'tree_max_depth', 'llm_max_output_tokens', 'max_assays', 'max_model_calls', 'max_reported_tokens', 'patience'] as const) {
    if (config[key] !== null && (!Number.isSafeInteger(config[key]) || config[key]! < 1)) throw new Error(`${key}必须为空或正整数`)
  }
  if (!Number.isSafeInteger(config.n_jobs) || config.n_jobs < 1) throw new Error('n_jobs必须为正整数')
  if (!Number.isSafeInteger(config.top_k_metric) || config.top_k_metric! < 1 || typeof config.high_fitness_quantile !== 'number'
    || !(config.high_fitness_quantile > 0 && config.high_fitness_quantile < 1)) throw new Error('评分指标配置无效')
  if (config.train_size < 4 || config.validation_size < 4 || config.test_size < 4) throw new Error('训练、验证和测试样本数至少为4')
  if (config.llm_max_output_tokens !== null && config.llm_max_output_tokens < 128) throw new Error('输出额度至少为128或为空')
  const requiredRounds = config.max_assays === null ? config.rounds : Math.min(config.rounds, Math.floor(config.max_assays / config.k))
  if (config.pool_size !== null && (config.pool_size < config.k * requiredRounds || config.pool_size < config.score_budget)) throw new Error('候选池不能满足评分或实测预算')
  for (const key of ['llm_timeout', 'llm_temperature', 'target_fitness', 'min_improvement'] as const) {
    if (config[key] !== null && (typeof config[key] !== 'number' || !Number.isFinite(config[key]))) throw new Error(`${key}必须为空或有限数值`)
  }
  if (config.llm_timeout !== null && config.llm_timeout <= 0 || config.llm_temperature !== null && (config.llm_temperature < 0 || config.llm_temperature > 2)
    || typeof config.min_improvement !== 'number' || config.min_improvement < 0 || config.target_fitness !== null && config.target_fitness < 0) throw new Error('超时、温度或改进阈值无效')
  if (config.k > config.proposal_count || config.proposal_count !== config.score_budget) throw new Error('必须满足k <= proposal_count == score_budget')
  if (!Array.isArray(config.ridge_alphas) || !config.ridge_alphas.length || config.ridge_alphas.some(x => typeof x !== 'number' || !Number.isFinite(x) || x < 0)) throw new Error('ridge_alphas必须为非负数数组')
  if (!Array.isArray(config.tree_min_samples_leaf) || !config.tree_min_samples_leaf.length || config.tree_min_samples_leaf.some(x => !Number.isSafeInteger(x) || x < 1)) throw new Error('tree_min_samples_leaf必须为正整数数组')
  if (!Array.isArray(config.model_families) || !config.model_families.length || config.model_families.some(x => !['ridge', 'random_forest', 'extra_trees'].includes(x))
    || new Set(config.model_families).size !== config.model_families.length) throw new Error('评分模型类别无效')
  if (!Array.isArray(config.allowed_positions) || config.allowed_positions.some(x => ![39, 40, 41, 54].includes(x))
    || new Set(config.allowed_positions).size !== config.allowed_positions.length) throw new Error('只支持GB1的39、40、41、54位点')
  if (typeof config.allowed_amino_acids !== 'string' || !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(config.allowed_amino_acids)
    || new Set(config.allowed_amino_acids).size !== config.allowed_amino_acids.length) throw new Error('允许的氨基酸无效')
  if (config.min_mutations > config.max_mutations || config.max_mutations > 4 || config.min_mutations > config.allowed_positions.length) throw new Error('突变数量范围无效')
  for (const key of ['repair_examples', 'require_full_batch', 'save_predictors'] as const) if (typeof config[key] !== 'boolean') throw new Error(`${key}必须为布尔值`)
  if (!['reject', 'first_n'].includes(config.oversize_policy) || !['best', 'round_mean'].includes(config.early_stop_metric)
    || !['json_schema', 'json_object', 'text'].includes(config.llm_format)) throw new Error('输出格式或停止设置无效')
  if (config.llm_reasoning_effort !== null) config.llm_reasoning_effort = identifier(config.llm_reasoning_effort, '思考强度')
  if (typeof config.knowledge_path !== 'string' || !/^configs\/[A-Za-z0-9_/-]+\.json$/.test(config.knowledge_path)
    || config.knowledge_path.split('/').some(part => part === '..' || part.startsWith('.'))) throw new Error('知识文件必须位于项目configs目录内')
  return config
}
export function historicalConfig(seed: number, rounds: number): ExperimentConfiguration {
  return { ...structuredClone(DEFAULT_CONFIG), seed, rounds, llm_max_output_tokens: 8000, llm_timeout: 180 }
}
export function promptProfile(value: unknown): string {
  if (!['v1', 'v2', 'v3'].includes(String(value))) throw new Error('提示词版本无效')
  return String(value)
}
export function modelOverride(value: unknown): ModelOverride {
  const raw = record(value)
  return { provider: identifier(raw.provider, 'Provider'), model: identifier(raw.model, 'Model ID') }
}
export interface WorkbenchDefaults { config: ExperimentConfiguration; promptProfile: string; recommendedModel: ModelOverride | null }
export async function loadDefaults(root: string): Promise<WorkbenchDefaults> {
  try {
    const content = await readFile(join(root, 'configs/workbench-defaults.json'), 'utf8')
    if (Buffer.byteLength(content) > 64 * 1024) throw new Error('默认配置文件过大')
    const raw = record(JSON.parse(content))
    return { config: validateConfig(raw.config ?? raw.defaultConfig), promptProfile: promptProfile(raw.promptProfile ?? raw.defaultPromptProfile ?? 'v3'),
      recommendedModel: raw.recommendedModel == null ? null : modelOverride(raw.recommendedModel) }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new Error('工作台默认配置文件无效')
    return { config: structuredClone(DEFAULT_CONFIG), promptProfile: 'v3', recommendedModel: null }
  }
}
