# LLM接口：整理当前可见信息、发送请求，并检查返回的候选和知识引用。
"""Bounded model-visible context and configurable OpenAI JSON generation."""
from copy import deepcopy
import json
import os
from .schema import KnowledgeBase, Proposal, ProposalBatch, POSITIONS, proposal_models, variant_allowed
from .storage import read_json
from .prompts import (LEGACY_VERSION as PROMPT_VERSION, LEGACY_SYSTEM_PROMPT as SYSTEM_PROMPT,
                      compact_design_context, feedback_context, known_examples, load_prompt_profile, proposal_schema)


class MissingCredentials(RuntimeError):
    pass


# known只包含当前方法已知的测量。位点边际统计受组合背景影响，不能直接当单点因果效应。
def summarize(known, top_n):
    top = known.sort_values(["fitness", "variant"], ascending=[False, True]).head(top_n)
    positions = []
    for index, position in enumerate(POSITIONS):
        grouped = known.assign(residue=known.variant.str[index]).groupby("residue").fitness.agg(["count", "mean", "max"])
        positions.append({"position": position, "residue_statistics": grouped.reset_index().to_dict(orient="records")})
    return {"n_known": len(known), "fitness_mean": float(known.fitness.mean()),
            "best_fitness": float(known.fitness.max()),
            "top_variants": top[["variant", "fitness", "mutations"]].to_dict(orient="records"),
            "positions": positions,
            "interpretation": "Marginal residue statistics are observations in sampled backgrounds, not causal substitution effects."}


# 把数据摘要、当前预算和近期反馈组织成LLM可见上下文；候选真实标签不放入其中。
def make_context(known, candidates, config, reference, round_number, knowledge=None, prompt_profile="v1", recent_feedback=()):
    context = {"prompt_version": PROMPT_VERSION, "round": round_number, "reference_sequence": reference,
               "reference_variant": "VDGV", "positions": list(POSITIONS),
               "known_data": summarize(known, config.summary_top),
               "candidate_examples": list(candidates), "proposal_count": config.proposal_count,
               "design_space": "Generate four-site variants using standard amino acids at 39,40,41,54. Membership is checked locally without exposing hidden fitness.",
               "assay_budget": config.k, "selection": "Local predictor scores valid proposals and selects their top k, at most the declared scoring budget.",
               "response_schema": ProposalBatch.model_json_schema()}
    if prompt_profile in ("v2", "v3"):
        context["prompt_version"] = "gb1-proposals-v2"
        context["known_data"].update(known_examples(known, context["known_data"]["top_variants"], config.contrast_examples))
        context["observed_variants"] = sorted(set(known.variant))
        context["forbidden_variants"] = context["observed_variants"].copy()
        context["recent_round_feedback"], context["feedback_summary"] = feedback_context(recent_feedback, set(known.variant))
        context["response_schema"] = proposal_schema(config.proposal_count)
        if prompt_profile == "v3":
            context = compact_design_context(context)
    elif prompt_profile != "v1":
        raise ValueError("prompt profile must be v1, v2 or v3")
    context["response_schema"] = proposal_schema(config.proposal_count if prompt_profile in ("v2", "v3") else None, config)
    constraints = {key: getattr(config, key) for key in
                   ("allowed_positions", "allowed_amino_acids", "min_mutations", "max_mutations")}
    if constraints != {"allowed_positions": list(POSITIONS), "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY", "min_mutations": 0, "max_mutations": 4}:
        context["mutation_constraints"] = constraints
    if knowledge is not None:
        context["knowledge"] = knowledge.model_dump()
    return context


def validate_proposals(text, allowed, config, rule_ids=()):
    _, batch_model = proposal_models(config.analysis_max_chars, config.proposal_text_max_chars)
    batch = batch_model.model_validate_json(text)
    variants = [proposal.variant for proposal in batch.proposals]
    if len(variants) != config.proposal_count:
        raise ValueError(f"Expected exactly {config.proposal_count} proposals")
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate proposed variants")
    if not set(variants) <= set(allowed):
        raise ValueError("Proposed variant outside the allowed unmeasured candidate set")
    for proposal in batch.proposals:
        if not variant_allowed(proposal.variant, config):
            raise ValueError("Proposed variant violates configured mutation constraints")
        if not set(proposal.knowledge_rule_ids) <= set(rule_ids):
            raise ValueError("Unknown knowledge rule citation")
    return batch


# 逐条保留合法候选并记录拒绝原因，纠正请求只补缺额，不丢弃此前的有效提议。
def collect_proposals(text, eligible, config, accepted=(), rule_ids=(), request_count=None):
    """Validate individual suggestions, preserving genuine valid designs during correction."""
    body = json.loads(text)
    if not isinstance(body, dict) or set(body) != {"analysis", "proposals"}:
        raise ValueError("Response must contain analysis and proposals only")
    if not isinstance(body["analysis"], str) or not body["analysis"].strip() or len(body["analysis"]) > config.analysis_max_chars:
        raise ValueError(f"Response analysis must be nonempty and at most {config.analysis_max_chars} characters")
    if not isinstance(body["proposals"], list):
        raise ValueError("Response proposals must be a list")
    expected = config.proposal_count if request_count is None else request_count
    if len(body["proposals"]) > expected and config.oversize_policy == "reject":
        raise ValueError("Response exceeds the requested proposal budget")
    seen = {row["variant"] for row in accepted}
    valid, rejected = [], []
    item_model, batch_model = proposal_models(config.analysis_max_chars, config.proposal_text_max_chars)
    for raw in body["proposals"][:expected]:
        try:
            proposal = item_model.model_validate(raw)
            if not variant_allowed(proposal.variant, config):
                raise ValueError("variant violates configured mutation constraints")
            if proposal.variant in seen:
                raise ValueError("duplicate or already accepted variant")
            if proposal.variant not in eligible:
                raise ValueError("not in the currently eligible candidate pool")
            if not set(proposal.knowledge_rule_ids) <= set(rule_ids):
                raise ValueError("unknown knowledge rule citation")
            seen.add(proposal.variant)
            valid.append(proposal.model_dump())
        except ValueError as exc:
            rejected.append({"variant": raw.get("variant") if isinstance(raw, dict) else None, "reason": str(exc)})
    if valid:
        batch_model.model_validate({"analysis": body["analysis"], "proposals": valid})
    return body["analysis"], valid, rejected


def load_knowledge(path):
    if path is None:
        raise ValueError("Knowledge strategy requires a sourced knowledge JSON file")
    return KnowledgeBase.model_validate(read_json(path))


# 这里仅构造请求参数；实际调用与日志共用它，确保记录和发送内容一致。
def build_request_payload(context, config, model, api_mode, system_prompt):
    """Build SDK keyword arguments without credentials, network access or input mutation."""
    user_text = json.dumps(context, ensure_ascii=False, allow_nan=False)
    schema = deepcopy(context.get("response_schema", ProposalBatch.model_json_schema()))
    payload = {"model": model}
    if api_mode == "responses":
        payload.update(instructions=system_prompt, input=user_text,
                       max_output_tokens=config.llm_max_output_tokens, store=False)
        if config.llm_format != "text":
            format_config = {"type": config.llm_format}
            if config.llm_format == "json_schema":
                format_config.update(name="gb1_proposals", strict=True, schema=schema)
            payload["text"] = {"format": format_config}
        if config.llm_reasoning_effort:
            payload["reasoning"] = {"effort": config.llm_reasoning_effort}
    elif api_mode == "chat":
        payload.update(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
                       max_completion_tokens=config.llm_max_output_tokens)
        if config.llm_format == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "gb1_proposals", "strict": True, "schema": schema}}
        elif config.llm_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if config.llm_reasoning_effort:
            payload["reasoning_effort"] = config.llm_reasoning_effort
    else:
        raise ValueError("api_mode must be responses or chat")
    if config.llm_max_output_tokens is None:
        payload.pop("max_output_tokens", None)
        payload.pop("max_completion_tokens", None)
    if getattr(config, "llm_temperature", None) is not None:
        payload["temperature"] = config.llm_temperature
    return payload


# 兼容两种接口的用量字段；缺总数且输入输出都已报告时才相加。
# 未报告数值按0参与汇总，同时保留缺失标记；reasoning/cache细项不重复累加。
def usage_counts(usage):
    """Normalize top-level token counts; nested reasoning/cache counts are already included."""
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    total = usage.get("total_tokens")
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    return {"input_tokens": incoming or 0, "output_tokens": outgoing or 0, "total_tokens": total or 0,
            "missing_usage_count": int(incoming is None or outgoing is None or total is None)}


# 封装真实模型接口；测试可注入假客户端和显式环境，避免读取凭据或访问网络。
class OpenAIProposer:
    """Reads only project environment variables; no implicit filesystem credentials."""
    def __init__(self, config, client=None, environ=None, prompt_profile="v1"):
        self.prompt = load_prompt_profile(prompt_profile)
        self.prompt_metadata = self.prompt.metadata()
        if environ is None and client is None:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)
        environ = os.environ if environ is None else environ
        self.model = environ.get("OPENAI_MODEL", "").strip()
        self.api_mode = environ.get("OPENAI_API_MODE", "responses").strip()
        self.base_url = environ.get("OPENAI_BASE_URL") or None
        self.config = config
        if self.api_mode not in ("responses", "chat"):
            raise ValueError("OPENAI_API_MODE must be responses or chat")
        if not self.model or (client is None and not environ.get("OPENAI_API_KEY")):
            raise MissingCredentials("Set project OPENAI_API_KEY and OPENAI_MODEL to run LLM methods")
        # 关闭SDK自动重试，由实验控制器记录请求是否完成并管理次数预算。
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=environ["OPENAI_API_KEY"], base_url=self.base_url,
                            timeout=config.llm_timeout, max_retries=0)
        self.client = client

    # 发送请求并保留原始文本、用量和返回模型标识；候选合法性由本地校验函数另行检查。
    def generate(self, context):
        payload = build_request_payload(context, self.config, self.model, self.api_mode, self.prompt.system_prompt)
        if self.api_mode == "responses":
            response = self.client.responses.create(**payload)
            text = response.output_text
            status = getattr(response, "status", None)
            complete = status is None or status == "completed"
            completion_status = {"status": status, "incomplete_details": str(getattr(response, "incomplete_details", None))}
        else:
            response = self.client.chat.completions.create(**payload)
            choice = response.choices[0]
            text = choice.message.content or ""
            complete = choice.finish_reason == "stop" and not getattr(choice.message, "refusal", None)
            completion_status = {"finish_reason": choice.finish_reason, "refusal": getattr(choice.message, "refusal", None)}
        usage = getattr(response, "usage", None)
        return {"text": text, "complete": complete,
                "usage": usage.model_dump(mode="json") if usage is not None else None,
                "response_id": getattr(response, "id", None), "request_id": getattr(response, "_request_id", None),
                "model_requested": self.model, "model_returned": getattr(response, "model", None),
                "api_mode": self.api_mode, "completion_status": completion_status,
                "prompt_metadata": self.prompt.metadata()}

    def close(self):
        self.client.close()
