# 主实验流程：各方法独立提出候选、评分、选择、获得反馈，再进入下一轮。
"""Independent feedback histories, durable round stages and final held-out evaluation."""
from pathlib import Path
from copy import deepcopy
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
from .agent import MissingCredentials, collect_proposals, load_knowledge, make_context, usage_counts
from .data import HiddenOracle, describe_variant, one_hot
from .models import fit_model, regression_metrics
from .schema import METHODS, compatible_config, variant_allowed
from .prompts import load_prompt_profile, repair_context, validate_prompt_state
from .storage import event, read_json, single_writer, write_json


class RoundPaused(BaseException):
    """A durable feedback boundary requested by the run controller."""


class ExperimentStopped(Exception):
    """A configured budget prevents further work."""


def request_stop_reason(method_state, config):
    attempts = [attempt for record in method_state["rounds"] for attempt in record["attempts"]]
    if config.max_model_calls is not None and len(attempts) >= config.max_model_calls:
        return "max_model_calls"
    if config.max_reported_tokens is not None and attempts:
        total = 0
        for attempt in attempts:
            raw = attempt.get("response", {}).get("usage")
            if not isinstance(raw, dict):
                return "reported_token_usage_unknown"
            incoming = raw.get("input_tokens", raw.get("prompt_tokens"))
            outgoing = raw.get("output_tokens", raw.get("completion_tokens"))
            if raw.get("total_tokens") is None and (incoming is None or outgoing is None):
                return "reported_token_usage_unknown"
            total += usage_counts(raw)["total_tokens"]
        if total >= config.max_reported_tokens:
            return "max_reported_tokens"
    return None


def boundary_stop_reason(method_state, config, initial_best):
    evaluated = [record for record in method_state["rounds"] if record["stage"] == "evaluated"]
    assays = sum(len(record["recommendations"]) for record in evaluated)
    if config.max_assays is not None and assays + config.k > config.max_assays:
        return "max_assays"
    if config.target_fitness is not None and max([initial_best] + [record["metrics"]["max_fitness"] for record in evaluated]) >= config.target_fitness:
        return "target_fitness"
    if config.patience is not None:
        best = initial_best if config.early_stop_metric == "best" else None
        stale = 0
        for record in evaluated:
            value = record["metrics"]["max_fitness" if config.early_stop_metric == "best" else "mean_fitness"]
            if best is None or value > best + config.min_improvement:
                best, stale = value, 0
            else:
                stale += 1
        if stale >= config.patience:
            return "patience"
    return None


# 每个方法的known由初始训练加自身已评估反馈组成，其他方法的结果不混入。
def known_for(dataset, method_state):
    observations = []
    for record in method_state["rounds"]:
        if record.get("stage") == "evaluated":
            for row in record["recommendations"]:
                observations.append({**describe_variant(row["variant"], dataset.reference), "fitness": row["actual_fitness"]})
    if not observations:
        return dataset.train.copy()
    result = pd.concat([dataset.train, pd.DataFrame(observations)], ignore_index=True)
    if result.variant.duplicated().any():
        raise ValueError("A method cannot assay or train on the same candidate twice")
    return result


# 为随机/模型对照抽取固定评分预算内的候选；v3的LLM首次设计不使用这份名单。
def candidate_cohort(dataset, known, config, round_number):
    available = sorted(v for v in set(dataset.candidates.variant) - set(known.variant) if variant_allowed(v, config))
    required = config.proposal_count if config.require_full_batch else config.k
    if len(available) < required:
        raise ExperimentStopped("eligible_candidates_exhausted")
    seed = config.design_seed if config.design_seed is not None else config.seed
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_number, 100]))
    count = min(config.score_budget, len(available))
    return sorted(rng.choice(available, size=count, replace=False).tolist())


# 当轮均值、当轮最高与累计最佳分别计算；累计最佳包含此前已知测量。
def _round_metrics(rows, known, threshold):
    actual = np.array([row["actual_fitness"] for row in rows])
    prior_best = float(known.fitness.max())
    prediction_metrics = regression_metrics(actual, [r["predicted_fitness"] for r in rows], len(rows))
    prediction_metrics.pop("top_k")
    prediction_metrics.pop("top_k_hit_rate")
    return {"mean_fitness": float(actual.mean()), "max_fitness": float(actual.max()),
            "cumulative_best": max(prior_best, float(actual.max())), "previous_best": prior_best,
            "improvement_over_previous_best": max(0.0, float(actual.max()) - prior_best),
            "new_high_fitness_count": int((actual >= threshold).sum()),
            "high_fitness_threshold": threshold, "threshold_source": "initial training quantile",
            "assays": len(rows), "duplicate_recommendations": 0,
            "prediction_metrics_on_selected": prediction_metrics}


def _new_state(dataset, selection, config, prompt):
    return {"format_version": 1, "config": config.model_dump(), "dataset_manifest": dataset.manifest,
            "prompt_metadata": prompt.metadata(),
            "selection": selection, "high_fitness_threshold": float(dataset.train.fitness.quantile(config.high_fitness_quantile)),
            "methods": {method: {"status": "pending", "rounds": []} for method in METHODS}}


def _validate_state(state, dataset, selection, config):
    if not compatible_config(state["config"], config) or state["dataset_manifest"] != dataset.manifest or state["selection"] != selection:
        raise ValueError("Existing run uses different data, configuration or selected model; create a new run directory")
    allowed = set(dataset.candidates.variant)
    for method_state in state["methods"].values():
        used = set()
        for index, record in enumerate(method_state["rounds"], 1):
            if record["round"] != index:
                raise ValueError("Round sequence is invalid")
            if record.get("stage") in ("selected", "evaluated"):
                variants = [row["variant"] for row in record["recommendations"]]
                if len(variants) != config.k or len(set(variants)) != config.k or not set(variants) <= allowed or used & set(variants):
                    raise ValueError("Persisted recommendations violate assay budgets or candidate uniqueness")
                scores = record.get("scores", [])
                scored = {row["variant"]: row["predicted_fitness"] for row in scores}
                if (len(scored) != len(scores) or len(scores) != record.get("model_score_count")
                        or len(scores) > config.score_budget or not set(variants) <= set(scored)
                        or any(row["predicted_fitness"] != scored[row["variant"]] for row in record["recommendations"])):
                    raise ValueError("Persisted recommendations require matching completed predictor scores within the budget")
                if record["stage"] == "evaluated":
                    used.update(variants)


# 只发送展示所需的阶段与计数；默认不启用。
def _observe(observer, method, round_number, tool, kind, counts=None, attempt=None, ok=None):
    if observer is None:
        return
    value = {"id": f"{method}-r{round_number}-{tool}" + (f"-a{attempt}" if attempt is not None else ""),
             "kind": kind, "tool": tool, "round": round_number, "counts": dict(counts or {})}
    if ok is not None:
        value["ok"] = ok
    try:
        observer(value)
    except Exception:
        # 观测通道失败只记固定诊断，不能影响候选选择或掩盖核心计算错误。
        logging.getLogger(__name__).warning("GB1 process observer failed")


# 先重放已保存响应，再补充仍需发送的请求；缺额必须来自真实模型回答。
def _generate_proposals(record, method, known, dataset, config, proposer, save, log_path, retry_uncertain, prompt, observer=None, method_state=None):
    knowledge = load_knowledge(config.knowledge_path) if method == "knowledge" else None
    rule_ids = [rule.id for rule in knowledge.rules] if knowledge else []
    eligible = {v for v in set(dataset.candidates.variant) - set(known.variant) if variant_allowed(v, config)}
    accepted = []
    rejections = []
    analyses = []
    base_context = make_context(known, record["cohort"], config, dataset.reference, record["round"], knowledge,
                                prompt.profile, record.get("recent_round_feedback", []))
    if prompt.profile == "v1":
        base_context["recent_round_feedback"] = record.get("recent_round_feedback", [])

    def consume(attempt, observed=False):
        expected = attempt["context"]["proposal_count"]
        active_observer = observer if observed else None
        _observe(active_observer, method, record["round"], "validate", "tool_start",
                 {"requested_count": expected}, attempt=attempt["number"])
        raw_items = None
        if prompt.profile in ("v2", "v3"):
            attempt["expected_count"] = expected
            try:
                body = json.loads(attempt["response"]["text"])
                raw_items = body.get("proposals") if isinstance(body, dict) else None
            except (ValueError, TypeError):
                pass
            attempt["returned_count"] = len(raw_items) if isinstance(raw_items, list) else None
        try:
            if not attempt["response"]["complete"]:
                raise ValueError("Model response was refused or incomplete")
            rejected_ids = {row["variant"] for row in rejections if isinstance(row.get("variant"), str)}
            allowed = eligible - rejected_ids if prompt.profile in ("v2", "v3") else eligible
            analysis, valid, rejected = collect_proposals(attempt["response"]["text"], allowed, config, accepted, rule_ids, expected)
            accepted.extend(valid)
            rejections.extend(rejected)
            analyses.append(analysis)
            status = "succeeded" if not rejected else "partial" if valid else "invalid"
            if prompt.profile in ("v2", "v3") and len(valid) < expected:
                status = "underfilled" if valid else "invalid"
            attempt.update(status=status, valid_count=len(valid), rejected=deepcopy(rejected))
        except ValueError as exc:
            attempt.update(status="invalid", validation_error=str(exc), valid_count=0)
            rejected = [{"variant": None, "reason": str(exc)}]
            if prompt.profile in ("v2", "v3") and isinstance(raw_items, list):
                rejected = [{"variant": row.get("variant"), "reason": str(exc)} for row in raw_items
                            if isinstance(row, dict) and isinstance(row.get("variant"), str)] or rejected
                attempt["rejected"] = deepcopy(rejected)
            rejections.extend(rejected)
        _observe(active_observer, method, record["round"], "validate", "tool_end",
                 {"requested_count": expected, "valid_count": attempt["valid_count"],
                  "rejected_count": len(attempt.get("rejected", []))}, attempt=attempt["number"], ok=attempt["status"] != "invalid")
        save()

    for attempt in record["attempts"]:
        if "response" in attempt:
            consume(attempt)
    if len(accepted) >= config.proposal_count:
        record["analysis"] = analyses
        record["proposal_rejections"] = rejections
        return accepted
    # started表示曾准备发送但结果尚未确认；先检查记录，避免自动重发造成重复调用。
    if any(attempt["status"] == "started" for attempt in record["attempts"]) and not retry_uncertain:
        raise RuntimeError("An API attempt has an unknown outcome; inspect its event record before --retry-uncertain")
    while len(record["attempts"]) < config.llm_max_calls_per_round and len(accepted) < config.proposal_count:
        reason = request_stop_reason(method_state or {"rounds": [record]}, config)
        if reason:
            if len(accepted) >= config.k and not config.require_full_batch:
                record["dispatch_stop_reason"] = reason
                break
            raise ExperimentStopped(reason)
        if proposer is None:
            raise MissingCredentials("LLM method needs a configured project API; offline methods remain resumable")
        context = {**base_context, "proposal_count": config.proposal_count - len(accepted)}
        if record["attempts"]:
            if prompt.profile in ("v2", "v3"):
                context = repair_context(base_context, known, eligible, accepted, rejections, config,
                                         record["round"], len(record["attempts"]) + 1)
            else:
                context.update(accepted_variants=[row["variant"] for row in accepted], rejections=rejections,
                               correction="Propose only replacements for the remaining count. Do not repeat accepted variants.")
                if not config.repair_examples:
                    context["candidate_examples"] = []
        attempt = {"number": len(record["attempts"]) + 1, "status": "started", "context": deepcopy(context),
                   "prompt_metadata": prompt.metadata()}
        record["attempts"].append(attempt)
        save()
        event(log_path, "llm_request", method=method, round=record["round"], attempt=attempt["number"],
              context=attempt["context"], prompt_metadata=prompt.metadata())
        try:
            response = proposer.generate(deepcopy(context))
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            attempt.update(status="failed" if status_code is not None else "started",
                           error={"type": type(exc).__name__, "status_code": status_code})
            save()
            event(log_path, "llm_error", method=method, round=record["round"], error=attempt["error"])
            raise RuntimeError(f"API call failed ({type(exc).__name__}); see the run event log") from None
        attempt.update(status="received", response=response)
        save()
        event(log_path, "llm_response", method=method, round=record["round"], attempt=attempt["number"], response=response)
        consume(attempt, observed=True)
    record["analysis"] = analyses
    record["proposal_rejections"] = rejections
    record["valid_proposal_count"] = len(accepted)
    save()
    required = config.proposal_count if config.require_full_batch else config.k
    if len(accepted) < required:
        raise RuntimeError(f"Only {len(accepted)} valid LLM proposals; fewer than required {required} after the bounded request budget")
    return accepted


# 轮次状态依次为started、selected、evaluated；保存阶段后可从中断处继续。
def run_experiment(dataset, selection, config, output, methods=METHODS, proposer=None, retry_uncertain=False, prompt_profile="v1", observer=None, checkpoint=None):
    if not methods or set(methods) - set(METHODS):
        raise ValueError("Choose random, model, llm and/or knowledge")
    prompt = load_prompt_profile(prompt_profile)
    if proposer is not None and getattr(proposer, "prompt_metadata", prompt.metadata()) != prompt.metadata():
        raise ValueError("Proposer system prompt differs from the selected run profile")
    output = Path(output)
    with single_writer(output):
        path, log_path = output / "state.json", output / "events.jsonl"
        state = read_json(path) if path.exists() else _new_state(dataset, selection, config, prompt)
        _validate_state(state, dataset, selection, config)
        validate_prompt_state(state, prompt)
        save = lambda: write_json(path, state)
        save()
        for method in methods:
            method_state = state["methods"][method]
            if method_state["status"] == "stopped":
                continue
            for round_number in range(1, config.rounds + 1):
                if len(method_state["rounds"]) < round_number:
                    reason = boundary_stop_reason(method_state, config, float(dataset.train.fitness.max()))
                    if reason is None and method in ("llm", "knowledge"):
                        reason = request_stop_reason(method_state, config)
                    known = known_for(dataset, method_state)
                    try:
                        cohort = candidate_cohort(dataset, known, config, round_number) if reason is None else []
                    except ExperimentStopped as exc:
                        reason = str(exc)
                        cohort = []
                    if reason:
                        method_state.update(status="stopped", stop_reason=reason)
                        save()
                        break
                    recent = [row for record in method_state["rounds"][-config.feedback_rounds:] for row in record.get("recommendations", [])] if config.feedback_rounds else []
                    method_state["rounds"].append({"round": round_number, "stage": "started", "attempts": [],
                                                   "recent_round_feedback": recent, "cohort": cohort})
                    save()
                record = method_state["rounds"][round_number - 1]
                if record["stage"] == "evaluated":
                    continue
                known = known_for(dataset, method_state)
                try:
                    if record["stage"] == "started":
                        start = time.perf_counter()
                        _observe(observer, method, round_number, "fit", "tool_start", {"known_count": len(known)})
                        # 每轮用该方法目前的known重新训练，模型类别与超参数保持已选方案。
                        try:
                            model = fit_model(selection["spec"], config, known)
                        except Exception:
                            _observe(observer, method, round_number, "fit", "tool_end", ok=False)
                            raise
                        _observe(observer, method, round_number, "fit", "tool_end", {"known_count": len(known)}, ok=True)
                        record["model_fit_seconds"] = time.perf_counter() - start
                        record["training_count"] = len(known)
                        record["training_variants"] = list(known.variant)
                        if config.save_predictors:
                            joblib.dump(model, output / f"{method}-round-{round_number}.joblib")
                        if method in ("llm", "knowledge"):
                            proposals = _generate_proposals(record, method, known, dataset, config, proposer, save, log_path, retry_uncertain, prompt, observer, method_state)
                        else:
                            proposals = [{"variant": v,
                                          "rationale": "Uniform random design without candidate labels." if method == "random" else "Predictor ranking of uniformly sampled candidates.",
                                          "hypothesis": "Random exploration comparator." if method == "random" else "Predictor ranking identifies promising combinations.",
                                          "knowledge_rule_ids": []} for v in record["cohort"]]
                        record["proposals"] = proposals
                        variants = [row["variant"] for row in proposals]
                        _observe(observer, method, round_number, "predict", "tool_start", {"candidate_count": len(variants)})
                        try:
                            predictions = model.predict(one_hot(variants))
                            if not np.isfinite(predictions).all():
                                raise ValueError("Predictor returned nonfinite scores")
                        except Exception:
                            _observe(observer, method, round_number, "predict", "tool_end", ok=False)
                            raise
                        _observe(observer, method, round_number, "predict", "tool_end", {"scored_count": len(variants)}, ok=True)
                        record["scores"] = [{"variant": v, "predicted_fitness": float(score)} for v, score in zip(variants, predictions)]
                        record["model_score_count"] = len(variants)
                        record["valid_proposal_count"] = len(variants)
                        event(log_path, "fitness_model_scores", method=method, round=round_number, scores=record["scores"])
                        lookup = {row["variant"]: row["predicted_fitness"] for row in record["scores"]}
                        _observe(observer, method, round_number, "select", "tool_start", {"scored_count": len(variants)})
                        # 随机方法的选择不依赖预测值；其余方法按预测分数选Top-k，并用变体ID处理并列。
                        if method == "random":
                            seed = config.design_seed if config.design_seed is not None else config.seed
                            rng = np.random.default_rng(np.random.SeedSequence([seed, round_number, 200]))
                            indices = rng.choice(len(proposals), size=config.k, replace=False)
                            ranked = [proposals[i] for i in indices]
                        else:
                            ranked = sorted(proposals, key=lambda p: (-lookup[p["variant"]], p["variant"]))[:config.k]
                        record["recommendations"] = [{**describe_variant(p["variant"], dataset.reference), **p,
                                                        "predicted_fitness": lookup[p["variant"]], "rank": rank}
                                                       for rank, p in enumerate(ranked, 1)]
                        record["stage"] = "selected"
                        save()
                        event(log_path, "candidates_selected", method=method, round=round_number,
                              recommendations=record["recommendations"], model_score_count=record["model_score_count"])
                        _observe(observer, method, round_number, "select", "tool_end", {"selected_count": len(record["recommendations"])}, ok=True)
                    if record["stage"] == "selected":
                        variants = [row["variant"] for row in record["recommendations"]]
                        # Top-k已写入状态后，才查询这些候选的真实分数，供下一轮反馈。
                        _observe(observer, method, round_number, "oracle", "tool_start", {"selected_count": len(variants)})
                        try:
                            oracle = HiddenOracle(dataset.directory, variants)
                            feedback = {row["variant"]: row["actual_fitness"] for row in oracle.evaluate(variants)}
                        except Exception:
                            _observe(observer, method, round_number, "oracle", "tool_end", ok=False)
                            raise
                        _observe(observer, method, round_number, "oracle", "tool_end", {"feedback_count": len(feedback)}, ok=True)
                        _observe(observer, method, round_number, "feedback", "tool_start", {"feedback_count": len(feedback), "known_count": len(known)})
                        for row in record["recommendations"]:
                            row["actual_fitness"] = feedback[row["variant"]]
                        record["metrics"] = _round_metrics(record["recommendations"], known, state["high_fitness_threshold"])
                        rejected = record.get("proposal_rejections", [])
                        record["metrics"].update(valid_proposal_count=record["valid_proposal_count"],
                                                  requested_proposal_count=config.proposal_count,
                                                  rejected_proposal_count=len(rejected),
                                                  duplicate_proposal_count=sum("duplicate" in row["reason"] for row in rejected),
                                                  scoring_budget_utilization=record["model_score_count"] / config.score_budget)
                        record["stage"] = "evaluated"
                        method_state["status"] = "running"
                        save()
                        event(log_path, "round_evaluated", method=method, round=round_number, metrics=record["metrics"])
                        _observe(observer, method, round_number, "feedback", "tool_end", {"known_count": len(known) + len(feedback), "feedback_count": len(feedback)}, ok=True)
                        if checkpoint is not None:
                            checkpoint(method, round_number)
                except RoundPaused:
                    method_state["status"] = "paused"
                    save()
                    export_results(state, output)
                    return state
                except ExperimentStopped as exc:
                    method_state.update(status="stopped", stop_reason=str(exc))
                    save()
                    break
                except Exception as exc:
                    method_state["status"] = "blocked" if isinstance(exc, MissingCredentials) else "failed"
                    failure = {"round": round_number, "type": type(exc).__name__, "message": str(exc)}
                    method_state.setdefault("failures", []).append(failure)
                    save()
                    event(log_path, "method_failure", method=method, **failure)
                    break
            if len(method_state["rounds"]) == config.rounds and all(r["stage"] == "evaluated" for r in method_state["rounds"]):
                method_state["status"] = "completed"
            save()
        export_results(state, output)
        return state


# 将已保存的状态导出为推荐、指标、用量和失败表；不在导出时补造候选。
def export_results(state, output):
    output = Path(output)
    recommendations, metrics, usages, failures = [], [], [], []
    for method, method_state in state["methods"].items():
        for failure in method_state.get("failures", []):
            failures.append({"method": method, **failure})
        for record in method_state["rounds"]:
            for attempt in record["attempts"]:
                response = attempt.get("response", {})
                usages.append({"method": method, "round": record["round"], "attempt": attempt["number"],
                               "status": attempt["status"], "usage": json.dumps(response.get("usage")),
                               "model_requested": response.get("model_requested"), "model_returned": response.get("model_returned"),
                               "api_mode": response.get("api_mode"), "request_id": response.get("request_id")})
            if record["stage"] != "evaluated":
                continue
            for row in record["recommendations"]:
                recommendations.append({"method": method, "round": record["round"], **row,
                                        "knowledge_rule_ids": ";".join(row["knowledge_rule_ids"])})
            flat_metrics = {key: value for key, value in record["metrics"].items() if not isinstance(value, dict)}
            metrics.append({"method": method, "round": record["round"], **flat_metrics,
                            "training_count": record["training_count"], "model_score_count": record["model_score_count"],
                            "model_fit_seconds": record["model_fit_seconds"], "llm_attempts": len(record["attempts"])})
    for name, rows in (("recommendations", recommendations), ("round_metrics", metrics), ("usage", usages), ("failures", failures)):
        frame = pd.DataFrame(rows)
        frame.insert(0, "seed", state["config"]["seed"])
        frame.to_csv(output / f"{name}.csv", index=False)
    write_json(output / "status.json", {method: record["status"] for method, record in state["methods"].items()})


# 主实验完成所有轮次后，再用独立测试数据评价各方法的最终预测器。
def finalize(dataset, selection, config, output, methods):
    output = Path(output)
    with single_writer(output):
        state = read_json(output / "state.json")
        _validate_state(state, dataset, selection, config)
        for method in methods:
            if state["methods"][method]["status"] != "completed":
                raise ValueError(f"{method} must finish all rounds before held-out test evaluation")
        existing = read_json(output / "final_test_metrics.json") if (output / "final_test_metrics.json").exists() else {}
        # 主实验在finalize阶段使用独立测试标签，前面的模型选择只使用验证集。
        test = pd.read_csv(dataset.directory / "hidden" / "test.csv")
        validation = pd.read_csv(dataset.directory / "validation.csv", usecols=["variant"])
        if set(test.variant) & (set(dataset.train.variant) | set(dataset.candidates.variant) | set(validation.variant)):
            raise ValueError("Held-out test overlaps another data role")
        for method in methods:
            if method in existing:
                continue
            known = known_for(dataset, state["methods"][method])
            model = fit_model(selection["spec"], config, known)
            predictions = model.predict(one_hot(test.variant))
            existing[method] = {**regression_metrics(test.fitness, predictions, config.top_k_metric), "training_count": len(known)}
            if config.save_predictors:
                joblib.dump(model, output / f"{method}-final.joblib")
            pd.DataFrame({"variant": test.variant, "actual_fitness": test.fitness, "predicted_fitness": predictions}).to_csv(output / f"{method}-test-predictions.csv", index=False)
            write_json(output / "final_test_metrics.json", existing)
        return existing
