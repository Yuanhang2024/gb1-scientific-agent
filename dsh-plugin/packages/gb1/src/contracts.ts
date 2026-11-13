/** 公开结果仅包含已选推荐，模型凭据始终保留在 Host。 */
export type ConnectionSource = 'dsh' | 'custom'
export type RunMethod = 'random' | 'model' | 'llm' | 'knowledge'
export type RunStatus = 'queued' | 'running' | 'paused' | 'stopped' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
export interface ExperimentConfiguration {
  seed: number; rounds: number; k: number; proposal_count: number; score_budget: number
  train_size: number; validation_size: number; test_size: number; pool_size: number | null
  summary_top: number; contrast_examples: number; feedback_rounds: number; top_k_metric?: number; high_fitness_quantile?: number
  ridge_alphas: number[]; tree_estimators: number; tree_min_samples_leaf: number[]; tree_max_depth: number | null; n_jobs: number
  model_families: string[]; model_seed: number | null; design_seed: number | null
  llm_max_calls_per_round: number; llm_max_output_tokens: number | null; llm_timeout: number | null
  llm_reasoning_effort: string | null; llm_temperature: number | null; llm_format: string
  knowledge_path: string; oversize_policy: 'reject' | 'first_n'; repair_examples: boolean; require_full_batch: boolean
  analysis_max_chars: number; proposal_text_max_chars: number
  allowed_positions: number[]; allowed_amino_acids: string; min_mutations: number; max_mutations: number
  max_assays: number | null; max_model_calls: number | null; max_reported_tokens: number | null
  target_fitness: number | null; patience: number | null; min_improvement: number; early_stop_metric: 'best' | 'round_mean'
  save_predictors: boolean
}
export type ConversationActor = 'coordinator' | 'model' | 'tool'
export type ConversationKind = 'status' | 'analysis' | 'tool' | 'error' | 'legacy'
export type ConversationMessageStatus = RunStatus
export interface ConversationMessage {
  id: string; order: number; time: string; round: number | null; actor: ConversationActor; kind: ConversationKind
  status: ConversationMessageStatus; title: string; text: string; toolName: string | null; inputSummary: string; outputSummary: string
}
export interface ConversationSnapshot { runId: string; revision: number; status: RunStatus; historyOnly: boolean; truncated: boolean; messages: ConversationMessage[] }
export interface ModelChoice { provider: string; id: string; name: string }
export interface ModelCapabilities {
  provider: string; model: string; reasoningEfforts: string[]; defaultMaxOutputTokens: number | null; contextWindow: number | null
}
export interface ConnectionSettings { source: ConnectionSource; provider: string; model: string; baseUrl: string; hasKey: boolean }
export interface SaveConnectionInput { source: ConnectionSource; provider: string; model: string; baseUrl: string; apiKey?: string; clearKey?: boolean }
export interface ModelIdentity { source: ConnectionSource; provider: string; id: string }
export interface ModelOverride { provider: string; model: string }
export interface WorkbenchInfo {
  ready: boolean; reason: string; reference: string; positions: number[]; connection: ConnectionSettings
  activeRunId: string | null; scoreBudget: number; topK: number; maxRounds: number | null
  defaultConfig: ExperimentConfiguration; defaultPromptProfile: string; recommendedModel: ModelOverride | null
}
export interface StartRunInput {
  operationId: string; reference: string; seed: number; rounds: number; method: RunMethod
  config?: ExperimentConfiguration; promptProfile?: string; modelOverride?: ModelOverride; repeatIndex?: number
}
export interface UsageSummary { inputTokens: number | null; outputTokens: number | null; totalTokens: number | null; missingUsageCount: number }
export interface Recommendation {
  rank: number; variant: string; sequence: string; mutations: string; mutationCount: number; predictedFitness: number; actualFitness: number | null
  hypothesis: string; rationale: string; knowledgeRuleIds: string[]
}
export interface RoundMetrics {
  round: number; status: string; knownCountBefore: number; knownCountAfter: number; validCount: number; scoredCount: number; calls: number
  usage: UsageSummary; meanFitness: number | null; maxFitness: number | null; medianFitness: number | null; cumulativeBest: number | null
}
export interface RoundResult extends RoundMetrics { analysis: string[]; top10: Recommendation[] }
export interface RunSummary {
  id: string; operationId: string; status: RunStatus; stage: string; createdAt: string; updatedAt: string
  seed: number; rounds: number; method: RunMethod; model: ModelIdentity; completedRounds: number; requestCount: number; error: string | null
  formatVersion?: number; config?: ExperimentConfiguration; promptProfile?: string; repeatIndex?: number; batchId?: string | null; stopReason?: string | null
}
export interface RunEvent { time: string; stage: string; message: string }
export interface RunDetail {
  summary: RunSummary; reference: string; initialKnownCount: number; knownCount: number; roundResults: RoundResult[]
  events: RunEvent[]; usage: UsageSummary; roundMetrics?: RoundMetrics[]; totalRounds?: number
}
export interface RunPage { runId: string; offset: number; limit: number; totalRounds: number; roundResults: RoundResult[] }
export interface BatchTarget { provider: string; model: string; reasoningEffort: string | null }
export interface StartBatchInput {
  operationId: string; baseConfig: ExperimentConfiguration; promptProfile: string; methods: RunMethod[]; targets: BatchTarget[]
  seeds: number[]; repeats: number; concurrency: number; perProviderConcurrency: number
}
export type BatchStatus = 'queued' | 'running' | 'paused' | 'completed' | 'cancelled' | 'interrupted'
export interface BatchRun {
  id: string | null; operationId: string; seed: number; method: RunMethod; target: BatchTarget; repeatIndex: number
  status: RunStatus; error: string | null
}
export interface BatchSummary {
  id: string; operationId: string; status: BatchStatus; createdAt: string; updatedAt: string; totalRuns: number; completedRuns: number
  failedRuns: number; activeRuns: number; concurrency: number; perProviderConcurrency: number
}
export interface BatchDetail { summary: BatchSummary; input: StartBatchInput; runs: BatchRun[] }

export interface QuickStartInput { operationId: string; model?: ModelOverride; reasoningEffort?: string | null }
export type QuickStartState = 'preparing' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
export interface QuickStartStep {
  stage: string; status: 'running' | 'completed' | 'failed' | 'cancelled'; message: string; time: string
}
export interface QuickStartStatus {
  id: string; operationId: string; status: QuickStartState; stage: string; message: string
  createdAt: string; updatedAt: string; runId: string | null; errorCode: string | null
  model: ModelOverride | null; reasoningEffort: string | null; requestedRounds: number; maxModelCalls: number
  steps: QuickStartStep[]
}
