# 提示与上下文：保存版本全文，组织已知例子、反馈和候选纠正信息。
"""Exact prompt snapshots and known-observation context for versioned runs."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .schema import ProposalBatch, proposal_models

PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
LEGACY_VERSION = "gb1-proposals-v1"
# 以下字符串是v1实验的系统提示正文，作为版本记录保存。
LEGACY_SYSTEM_PROMPT = """You design GB1 variants for an IgG-Fc binding selection experiment.
Return only one JSON object matching the supplied schema. Four-site variant order is 39,40,41,54.
Propose your own combinations of exactly four standard amino acids. Avoid duplicate or previously measured variants.
Candidate examples are optional unlabeled suggestions, not a restriction on your designs.
A local membership validator will reject variants absent from the eligible measured candidate pool.
When a correction lists accepted variants and rejections, retain the accepted designs and propose only replacements.
Use known_data as measured observations; candidate fitness is unknown. Predictions are estimates.
Explain a testable hypothesis and a specific rationale for each variant, including uncertainty from epistasis.
Keep each rationale and hypothesis to one concise sentence, preferably at most 30 words each; keep analysis under 200 words.
Do not claim measurements for untested candidates. No file access, browsing, or external tools are provided.
Knowledge rules are hypotheses with limited scope. Cite rule ids only if included in this request.
"""


@dataclass(frozen=True)
class PromptProfile:
    profile: str
    version: str
    system_prompt: str

    def metadata(self):
        return {"profile": self.profile, "version": self.version, "system_prompt": self.system_prompt}


LEGACY_PROFILE = PromptProfile("v1", LEGACY_VERSION, LEGACY_SYSTEM_PROMPT)


def load_prompt_profile(profile="v1"):
    filenames = {"v1": "legacy-v1.txt", "v2": "scientific-v2.txt", "v3": "scientific-v3.txt"}
    if profile not in filenames:
        raise ValueError("prompt profile must be v1, v2 or v3")
    text = (PROMPT_DIRECTORY / filenames[profile]).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("System prompt must not be empty")
    return PromptProfile(profile, f"gb1-proposals-{profile}", text)


# 恢复实验时同时核对版本与提示全文，避免在同一记录里混用不同提示。
def validate_prompt_state(state, profile):
    stored = state.get("prompt_metadata", LEGACY_PROFILE.metadata())
    if stored != profile.metadata():
        raise ValueError("Existing run has a different prompt profile, version or system prompt text; use a new run directory")


# 把本次所需候选数写入JSON Schema；纠正时数量会随缺额变化。
def proposal_schema(requested_count=None, config=None):
    batch = ProposalBatch if config is None else proposal_models(config.analysis_max_chars, config.proposal_text_max_chars)[1]
    schema = deepcopy(batch.model_json_schema())
    if requested_count is not None:
        if type(requested_count) is not int or requested_count < 1:
            raise ValueError("requested_count must be a positive integer")
        schema["properties"]["proposals"].update(minItems=requested_count, maxItems=requested_count)
    return schema


# 从当前已测数据选取高、低和中间表现的例子，供LLM比较不同背景。
def known_examples(known, top_variants, limit=4):
    """Contrasting observations drawn only from this method's known measurements."""
    columns = ["variant", "fitness", "mutations"]
    top_ids = {row["variant"] for row in top_variants}
    remaining = known.loc[~known.variant.isin(top_ids)]
    low = remaining.sort_values(["fitness", "variant"], ascending=[True, True]).head(limit)
    remaining = remaining.loc[~remaining.variant.isin(set(low.variant))]
    median = float(known.fitness.median())
    middle = remaining.assign(distance=(remaining.fitness - median).abs()).sort_values(["distance", "variant"]).head(limit)
    return {"low_fitness_examples": low[columns].to_dict(orient="records"),
            "near_median_examples": middle[columns].to_dict(orient="records"),
            "fitness_median": median,
            "example_scope": "Only current-method known measurements; low and median examples exclude the displayed top examples."}


# 残差定义为实测减预测：负值表示预测偏高，正值表示预测偏低。
# 这些数值描述预测误差，不能单独确定生化失效原因。
def feedback_context(recent_feedback, known_variants):
    """Calculate signed residuals only for already observed feedback."""
    rows = []
    for raw in recent_feedback:
        if raw.get("variant") not in known_variants:
            continue
        actual, predicted = float(raw["actual_fitness"]), float(raw["predicted_fitness"])
        if not np.isfinite([actual, predicted]).all():
            raise ValueError("Known feedback must contain finite actual and predicted values")
        rows.append({**deepcopy(raw), "residual": actual - predicted})
    over = [row for row in rows if row["residual"] < 0]
    under = [row for row in rows if row["residual"] > 0]
    compact = lambda row: {key: row.get(key) for key in ("variant", "predicted_fitness", "actual_fitness", "residual", "hypothesis")}
    summary = {"n_observed_feedback": len(rows), "residual_definition": "actual_fitness - predicted_fitness",
               "largest_overestimation": compact(min(over, key=lambda row: (row["residual"], row["variant"]))) if over else None,
               "largest_underestimation": compact(max(under, key=lambda row: (row["residual"], row["variant"]))) if under else None,
               "interpretation": "Residuals describe prediction error on measured recommendations; fitness alone does not identify expression, growth or technical failure causes."}
    return rows, summary


# 保留已接受候选，仅请求新的替补；可用替补没有真实标签，也不会自动成为模型输出。
def repair_context(base_context, known, eligible, accepted, rejections, config, round_number, attempt_number):
    """Return unlabeled optional replacements without admitting them as proposals."""
    context = deepcopy(base_context)
    remaining = config.proposal_count - len(accepted)
    compact = base_context.get("prompt_version") == "gb1-proposals-v3"
    all_known_ids = set(known.variant)
    known_ids = set(base_context["observed_variants"]) if compact else all_known_ids
    accepted_ids = {row["variant"] for row in accepted}
    rejected_ids = {row["variant"] for row in rejections if isinstance(row.get("variant"), str)}
    forbidden = known_ids | accepted_ids | rejected_ids
    available = sorted(set(eligible) - forbidden - all_known_ids)
    count = min(remaining, len(available))
    seed = config.design_seed if config.design_seed is not None else config.seed
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_number, attempt_number, 300]))
    examples = sorted(rng.choice(available, size=count, replace=False).tolist()) if count and config.repair_examples else []
    context.update(proposal_count=remaining, remaining_count=remaining,
                   accepted_variants=sorted(accepted_ids), observed_variants=sorted(known_ids),
                   forbidden_variants=sorted(forbidden), rejections=deepcopy(rejections),
                   candidate_examples=examples, replacement_examples=examples.copy(),
                   correction="Return only the remaining NEW entries. Accepted entries are already retained; exclude observed, accepted and rejected variants.")
    if compact:
        context.update(stage="repair_only", candidate_examples=[])
    context["response_schema"] = proposal_schema(remaining, config)
    return context


# v3首次设计不提供候选示例；只展示部分已知记录，完整历史仍由本地资格检查使用。
def compact_design_context(context):
    """Expose displayed observations only and reserve candidate examples for repair."""
    context = deepcopy(context)
    feedback_keys = ("variant", "predicted_fitness", "actual_fitness", "residual", "hypothesis", "rationale")
    recent = [{key: row[key] for key in feedback_keys if key in row} for row in context["recent_round_feedback"]]
    visible = {row["variant"] for key in ("top_variants", "low_fitness_examples", "near_median_examples")
               for row in context["known_data"][key]}
    visible.update(row["variant"] for row in recent)
    context.update(prompt_version="gb1-proposals-v3", stage="initial_design", candidate_examples=[],
                   observed_variants=sorted(visible), forbidden_variants=sorted(visible),
                   observed_variants_complete=False,
                   observed_variants_scope="Only IDs in the displayed top, low, median and recent observations; the local validator checks the full known history.",
                   recent_round_feedback=recent)
    return context
