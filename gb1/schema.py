# 输入格式与预算检查：先验证配置和模型输出，再交给实验流程使用。
"""Validated experiment configuration and model-generated proposals."""
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

# 20种标准氨基酸的顺序同时决定one-hot编码列，运行中保持固定。
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = (39, 40, 41, 54)
METHODS = ("random", "model", "llm", "knowledge")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentConfig(StrictModel):
    seed: int = Field(default=11, ge=0, le=2**32 - 1, strict=True)
    rounds: int = Field(default=3, ge=1, strict=True)
    k: int = Field(default=10, ge=1, strict=True)
    train_size: int = Field(default=2000, ge=4, strict=True)
    validation_size: int = Field(default=1000, ge=4, strict=True)
    test_size: int = Field(default=5000, ge=4, strict=True)
    pool_size: int | None = Field(default=None, ge=1, strict=True)
    score_budget: int = Field(default=40, ge=1, strict=True)
    proposal_count: int = Field(default=40, ge=1, strict=True)
    summary_top: int = Field(default=15, ge=1, strict=True)
    contrast_examples: int = Field(default=4, ge=0, strict=True)
    feedback_rounds: int = Field(default=1, ge=0, strict=True)
    top_k_metric: int = Field(default=10, ge=1, strict=True)
    high_fitness_quantile: float = Field(default=0.9, gt=0, lt=1)
    ridge_alphas: list[float] = [0.1, 1.0, 10.0]
    tree_estimators: int = Field(default=100, ge=1, strict=True)
    tree_min_samples_leaf: list[int] = [1, 3]
    tree_max_depth: int | None = Field(default=None, ge=1, strict=True)
    n_jobs: int = Field(default=4, ge=1, strict=True)
    model_families: list[Literal["ridge", "random_forest", "extra_trees"]] = ["ridge", "random_forest", "extra_trees"]
    model_seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    design_seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    llm_max_calls_per_round: int = Field(default=2, ge=1, strict=True)
    llm_max_output_tokens: int | None = Field(default=8000, ge=128, strict=True)
    llm_timeout: float | None = Field(default=180, gt=0)
    llm_reasoning_effort: str | None = None
    llm_temperature: float | None = Field(default=None, ge=0, le=2)
    llm_format: Literal["json_schema", "json_object", "text"] = "json_schema"
    knowledge_path: str | None = "configs/knowledge.json"
    oversize_policy: Literal["reject", "first_n"] = "reject"
    repair_examples: bool = True
    require_full_batch: bool = False
    analysis_max_chars: int = Field(default=5000, ge=1, strict=True)
    proposal_text_max_chars: int = Field(default=2000, ge=1, strict=True)
    allowed_positions: list[int] = list(POSITIONS)
    allowed_amino_acids: str = AMINO_ACIDS
    min_mutations: int = Field(default=0, ge=0, le=4, strict=True)
    max_mutations: int = Field(default=4, ge=0, le=4, strict=True)
    max_assays: int | None = Field(default=None, ge=1, strict=True)
    max_model_calls: int | None = Field(default=None, ge=1, strict=True)
    max_reported_tokens: int | None = Field(default=None, ge=1, strict=True)
    target_fitness: float | None = Field(default=None, ge=0)
    patience: int | None = Field(default=None, ge=1, strict=True)
    min_improvement: float = Field(default=0, ge=0)
    early_stop_metric: Literal["best", "round_mean"] = "best"
    save_predictors: bool = True

    @field_validator("ridge_alphas")
    @classmethod
    def valid_alphas(cls, value):
        if not value or any(x < 0 for x in value):
            raise ValueError("ridge_alphas must be a nonempty list of nonnegative values")
        return value

    @field_validator("tree_min_samples_leaf", mode="before")
    @classmethod
    def valid_leaves(cls, value):
        if not isinstance(value, list) or not value or any(type(x) is not int or x < 1 for x in value):
            raise ValueError("tree_min_samples_leaf must contain positive integers")
        return value

    @field_validator("model_families")
    @classmethod
    def valid_families(cls, value):
        if not value or len(value) != len(set(value)):
            raise ValueError("model_families must contain unique supported families")
        return value

    @field_validator("allowed_positions", mode="before")
    @classmethod
    def valid_positions(cls, value):
        if not isinstance(value, list) or any(type(x) is not int or x not in POSITIONS for x in value) or len(value) != len(set(value)):
            raise ValueError("allowed_positions must be a unique subset of the four GB1 positions")
        return value

    @field_validator("allowed_amino_acids")
    @classmethod
    def valid_amino_acids(cls, value):
        if not value or set(value) - set(AMINO_ACIDS) or len(value) != len(set(value)):
            raise ValueError("allowed_amino_acids must contain unique standard amino acids")
        return value

    @model_validator(mode="after")
    def budgets(self):
        if not self.k <= self.proposal_count == self.score_budget:
            raise ValueError("require k <= proposal_count == score_budget for the matched design budget")
        required_rounds = self.rounds if self.max_assays is None else min(self.rounds, self.max_assays // self.k)
        if self.pool_size is not None and (self.pool_size < self.k * required_rounds or self.score_budget > self.pool_size):
            raise ValueError("candidate pool cannot satisfy the scoring and assay budgets")
        if self.min_mutations > self.max_mutations or self.min_mutations > len(self.allowed_positions):
            raise ValueError("mutation bounds cannot be satisfied by allowed_positions")
        return self


def compatible_config(stored, config):
    """Interpret missing fields with their historical compatible defaults."""
    return ExperimentConfig.model_validate(stored).model_dump() == config.model_dump()


def variant_allowed(variant, config):
    if not isinstance(variant, str) or len(variant) != 4 or set(variant) - set(config.allowed_amino_acids):
        return False
    changed = [p for p, old, new in zip(POSITIONS, "VDGV", variant) if old != new]
    return config.min_mutations <= len(changed) <= config.max_mutations and set(changed) <= set(config.allowed_positions)


class Proposal(StrictModel):
    variant: str = Field(pattern="^[ACDEFGHIKLMNPQRSTVWY]{4}$")
    rationale: str = Field(min_length=1, max_length=2000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    knowledge_rule_ids: list[str]


class ProposalBatch(StrictModel):
    analysis: str = Field(min_length=1, max_length=5000)
    proposals: list[Proposal] = Field(min_length=1)


@lru_cache(maxsize=64)
def proposal_models(analysis_max_chars=5000, proposal_text_max_chars=2000):
    if (analysis_max_chars, proposal_text_max_chars) == (5000, 2000):
        return Proposal, ProposalBatch
    item = create_model("ConfiguredProposal", __base__=Proposal,
                        rationale=(str, Field(min_length=1, max_length=proposal_text_max_chars)),
                        hypothesis=(str, Field(min_length=1, max_length=proposal_text_max_chars)))
    batch = create_model("ConfiguredProposalBatch", __base__=ProposalBatch,
                         analysis=(str, Field(min_length=1, max_length=analysis_max_chars)),
                         proposals=(list[item], Field(min_length=1)))
    return item, batch


class KnowledgeRule(StrictModel):
    id: str
    scope: str
    condition: str
    action: str
    source: str
    explanation: str


class KnowledgeBase(StrictModel):
    rules: list[KnowledgeRule] = Field(min_length=1)
