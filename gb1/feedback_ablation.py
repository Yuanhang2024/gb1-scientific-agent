# 详细反馈消融：在同一已知状态上比较展示或隐藏逐条反馈的影响。
"""Fixed-prefix, single-round detailed-feedback ablation with independent persistence."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from .agent import OpenAIProposer, collect_proposals, make_context, usage_counts
from .data import HiddenOracle, PublicDataset, describe_variant, one_hot
from .experiment import known_for
from .models import fit_model
from .prompts import load_prompt_profile, repair_context
from .schema import ExperimentConfig, KnowledgeBase, compatible_config
from .storage import event, read_json, single_writer, write_json

CONDITIONS = ("full", "withheld")
FEEDBACK_FIELDS = {"recent_round_feedback", "feedback_summary"}
WITHHELD_SUMMARY = {"available": False, "detail": "Detailed prior-round feedback is withheld by design; known_data still includes updated measurements."}


# withheld只隐藏详细反馈字段；两组训练数据与已知统计仍包含已经得到的反馈。
def condition_context(source_context, condition):
    if condition not in CONDITIONS:
        raise ValueError("Condition must be full or withheld")
    result = deepcopy(source_context)
    if condition == "withheld":
        result["recent_round_feedback"] = []
        result["feedback_summary"] = deepcopy(WITHHELD_SUMMARY)
    return result


def _identity(response):
    model, mode = response.get("model_requested"), response.get("api_mode")
    returned = response.get("model_returned")
    if not isinstance(model, str) or not model or not isinstance(mode, str) or not mode:
        raise ValueError("Recorded model ID and API mode are required")
    if returned is not None and not isinstance(returned, str):
        raise ValueError("Returned model ID must be a string or null")
    return {"model_requested": model, "model_returned": returned, "api_mode": mode}


# 取知识增强方法第一轮结束的状态，重建第二轮输入；不加入后续轮次的结果。
def _source_snapshot(source_run, data_directory):
    source = read_json(Path(source_run) / "state.json")
    dataset = PublicDataset(data_directory)
    config = ExperimentConfig.model_validate(source["config"])
    if config.score_budget > 40 or config.proposal_count > 40 or config.k > 10:
        raise ValueError("Ablation permits at most 40 scored proposals and 10 assays per arm")
    if source["dataset_manifest"] != dataset.manifest or not compatible_config(dataset.manifest["config"], config):
        raise ValueError("Source run does not match the frozen prepared dataset")
    selection = source["selection"]
    if not compatible_config(selection["config"], config) or selection["dataset_manifest"] != dataset.manifest:
        raise ValueError("Source predictor selection does not match its data/configuration")
    prompt = deepcopy(source.get("prompt_metadata"))
    if prompt is None or prompt["profile"] != "v3" or prompt != load_prompt_profile("v3").metadata():
        raise ValueError("A source run with the unchanged v3 prompt is required")
    rounds = source["methods"]["knowledge"]["rounds"]
    if len(rounds) < 2 or rounds[0]["round"] != 1 or rounds[0]["stage"] != "evaluated":
        raise ValueError("Source must contain an evaluated knowledge round 1 and the round 2 initial request")
    first = rounds[0]
    if len(first["recommendations"]) != config.k or not rounds[1].get("attempts"):
        raise ValueError("The source prefix or second-round context is incomplete")
    if rounds[1]["round"] != 2:
        raise ValueError("Expected the source round 2 initial request")
    initial_attempt = rounds[1]["attempts"][0]
    context = deepcopy(initial_attempt["context"])
    if initial_attempt.get("prompt_metadata", prompt) != prompt:
        raise ValueError("Source request and system prompt metadata differ")
    # 这里只加入第一轮观测，后续源实验的结果不进入这个固定状态。
    known = known_for(dataset, {"rounds": [first]})
    if len(known) != config.train_size + config.k:
        raise ValueError("Training must contain initial measurements plus exactly round 1 feedback")
    if context.get("knowledge") is None:
        raise ValueError("The knowledge-enhanced source context is required")
    knowledge = KnowledgeBase.model_validate(context["knowledge"])
    expected = make_context(known, [], config, dataset.reference, 2, knowledge,
                            prompt_profile="v3", recent_feedback=first["recommendations"])
    if context != expected:
        raise ValueError("Source round 2 context does not match the round 1-only known snapshot")
    responses = [attempt["response"] for attempt in first["attempts"] if "response" in attempt]
    if not responses:
        raise ValueError("Source round 1 lacks recorded API identity")
    identity = _identity(responses[0])
    if any(_identity(response) != identity for response in responses):
        raise ValueError("Source prefix contains inconsistent model/API identities")
    prefix = {"method": "knowledge", "completed_rounds": [1],
              "recommendations": deepcopy(first["recommendations"]), "training_count": len(known),
              "training_variants": list(known.variant)}
    snapshot = {"seed": config.seed, "config": config.model_dump(), "selection": deepcopy(selection),
                "source_run": str(Path(source_run).resolve()), "data_directory": str(Path(data_directory).resolve()),
                "dataset_manifest": deepcopy(dataset.manifest), "source_prefix": prefix,
                "training_records": known.to_dict(orient="records"),
                "eligible_variants": sorted(set(dataset.candidates.variant) - set(known.variant)),
                "initial_context": context, "prompt_metadata": prompt, "api_identity": identity,
                "model_file": f"models/seed-{config.seed}.joblib", "known_best_fitness": float(known.fitness.max())}
    return snapshot, known


# 每个seed只拟合一次固定预测器，两条件共享它；条件顺序随机且不参考fitness。
def prepare_ablation(source_runs, data_directories, output):
    if len(source_runs) != len(data_directories) or not 1 <= len(source_runs) <= 3:
        raise ValueError("Provide one to three matched source-run/data-directory pairs")
    snapshots, known_tables = {}, {}
    for source, data in zip(source_runs, data_directories):
        snapshot, known = _source_snapshot(source, data)
        key = str(snapshot["seed"])
        if snapshot["seed"] not in (11, 22, 33) or key in snapshots:
            raise ValueError("Source seeds must be unique members of 11,22,33")
        snapshots[key], known_tables[key] = snapshot, known
    reference = next(iter(snapshots.values()))
    for snapshot in snapshots.values():
        if snapshot["prompt_metadata"] != reference["prompt_metadata"] or snapshot["api_identity"] != reference["api_identity"]:
            raise ValueError("Sources must share the same prompt and model/API identity")
        if {k: v for k, v in snapshot["config"].items() if k != "seed"} != {k: v for k, v in reference["config"].items() if k != "seed"}:
            raise ValueError("Sources must share all experiment settings except seed")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Ablation output must be empty; prepared snapshots are immutable")
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    order = []
    for key in sorted(snapshots, key=int):
        snapshot = snapshots[key]
        config = ExperimentConfig.model_validate(snapshot["config"])
        model = fit_model(snapshot["selection"]["spec"], config, known_tables[key])
        joblib.dump(model, output / snapshot["model_file"])
        rng = np.random.default_rng(np.random.SeedSequence([snapshot["seed"], 2500]))
        order.extend({"seed": snapshot["seed"], "condition": str(condition)} for condition in rng.permutation(CONDITIONS))
    manifest = {"format_version": 1, "study": "detailed-feedback-display-ablation", "snapshots": snapshots,
                "order": order, "per_arm_max_requests": 2, "max_requests": 4 * len(snapshots),
                "feedback_fields": sorted(FEEDBACK_FIELDS), "withheld_summary": deepcopy(WITHHELD_SUMMARY),
                "training_feedback_in_both_conditions": True,
                "order_rule": "Per-seed NumPy SeedSequence([seed,2500]) permutation; no fitness inputs"}
    write_json(output / "manifest.json", manifest)
    for item in order:
        snapshot = snapshots[str(item["seed"])]
        state = {"seed": item["seed"], "condition": item["condition"], "stage": "prepared", "attempts": [], "errors": [],
                 "snapshot": deepcopy(snapshot), "initial_context": condition_context(snapshot["initial_context"], item["condition"])}
        write_json(_arm_directory(output, item) / "state.json", state)
    return manifest


def _arm_directory(output, item):
    return Path(output) / f"seed-{item['seed']}" / item["condition"]


def _validate_arm(state, snapshot, condition):
    if state["condition"] != condition:
        raise ValueError("Arm condition metadata differs from its paired slot")
    if state["seed"] != snapshot["seed"] or state["snapshot"] != snapshot:
        raise ValueError("Arm snapshot metadata changed; use a new ablation directory")
    if state["initial_context"] != condition_context(snapshot["initial_context"], state["condition"]):
        raise ValueError("Arm initial context differs from the frozen feedback manipulation")
    if len(state["attempts"]) > 2:
        raise ValueError("Arm exceeds the two-request budget")
    for index, attempt in enumerate(state["attempts"]):
        if attempt["number"] != index + 1 or attempt["prompt_metadata"] != snapshot["prompt_metadata"]:
            raise ValueError("Request numbering or system prompt metadata changed")
        expected = deepcopy(state["initial_context"])
        if index:
            prefix = deepcopy(state)
            prefix["attempts"] = prefix["attempts"][:index]
            accepted, rejected, _ = _validated_proposals(prefix, snapshot)
            expected = repair_context(expected, pd.DataFrame(snapshot["training_records"]), set(snapshot["eligible_variants"]),
                                      accepted, rejected, ExperimentConfig.model_validate(snapshot["config"]), 2, index + 1)
        if attempt["context"] != expected:
            raise ValueError("Saved request context differs from the frozen condition or prescribed repair")


# 从原始回答重建合法候选，保证恢复时使用的提议仍有对应模型响应。
def _validated_proposals(state, snapshot):
    config = ExperimentConfig.model_validate(snapshot["config"])
    eligible = set(snapshot["eligible_variants"])
    rules = {rule["id"] for rule in snapshot["initial_context"]["knowledge"]["rules"]}
    accepted, rejections, analyses = [], [], []
    for attempt in state["attempts"]:
        if "response" not in attempt:
            continue
        expected = attempt["context"]["proposal_count"]
        response = attempt["response"]
        raw = None
        try:
            body = json.loads(response["text"])
            raw = body.get("proposals") if isinstance(body, dict) else None
        except (ValueError, TypeError):
            pass
        validation = {"expected_count": expected, "returned_count": len(raw) if isinstance(raw, list) else None}
        try:
            if _identity(response) != snapshot["api_identity"]:
                raise ValueError("API response model/mode differs from the fixed source")
            if response.get("prompt_metadata", snapshot["prompt_metadata"]) != snapshot["prompt_metadata"]:
                raise ValueError("API response system prompt metadata differs from the fixed source")
            if not response["complete"]:
                raise ValueError("API response was refused or incomplete")
            forbidden = {row["variant"] for row in rejections if isinstance(row.get("variant"), str)}
            analysis, valid, rejected = collect_proposals(response["text"], eligible - forbidden, config, accepted, rules, expected)
            accepted.extend(valid)
            rejections.extend(rejected)
            analyses.append(analysis)
            validation.update(status="succeeded" if len(valid) == expected and not rejected else "underfilled" if valid else "invalid",
                              valid_count=len(valid), rejections=deepcopy(rejected))
        except ValueError as exc:
            rejected = [{"variant": row.get("variant"), "reason": str(exc)} for row in raw
                        if isinstance(row, dict) and isinstance(row.get("variant"), str)] if isinstance(raw, list) else []
            rejected = rejected or [{"variant": None, "reason": str(exc)}]
            rejections.extend(rejected)
            validation.update(status="invalid", valid_count=0, rejections=deepcopy(rejected), error=str(exc))
        attempt["validation"] = validation
    return accepted, rejections, analyses


def _metrics(state, snapshot):
    actual = np.array([row["actual_fitness"] for row in state["recommendations"]], dtype=float)
    return {"mean_fitness": float(actual.mean()), "max_fitness": float(actual.max()), "median_fitness": float(np.median(actual)),
            "known_best_fitness": snapshot["known_best_fitness"], "max_minus_known_best": float(actual.max()) - snapshot["known_best_fitness"],
            "valid_count": len(state["proposals"]), "scored_count": len(state["scores"]), "assay_count": len(actual)}


# 恢复前核对候选来源、冻结模型分数与Top-k；浮点重算允许微差，排序按保存分数核对。
def _validate_selection(state, snapshot, model):
    """Verify stored candidates and saved-score Top-k without obtaining new feedback."""
    config = ExperimentConfig.model_validate(snapshot["config"])
    for attempt in state["attempts"]:
        if "response" in attempt and _identity(attempt["response"]) != snapshot["api_identity"]:
            raise ValueError("Pair API/model identity mismatch")
    accepted, _, _ = _validated_proposals(deepcopy(state), snapshot)
    if state.get("proposals") != accepted or not config.k <= len(accepted) <= config.score_budget:
        raise ValueError("Saved proposals differ from the validated API responses or scoring budget")
    predictions = np.asarray(model.predict(one_hot([row["variant"] for row in accepted])), dtype=float)
    scores = state.get("scores", [])
    if (len(scores) != len(accepted) or any(set(row) != {"variant", "predicted_fitness"} for row in scores)
            or [row["variant"] for row in scores] != [row["variant"] for row in accepted]):
        raise ValueError("Saved score IDs or structure differ from the validated proposals")
    saved = np.asarray([row["predicted_fitness"] for row in scores], dtype=float)
    if not np.isfinite(predictions).all() or not np.isfinite(saved).all() or not np.allclose(saved, predictions, rtol=1e-10, atol=1e-12):
        raise ValueError("Saved scores differ from the frozen predictor")
    lookup = {row["variant"]: row["predicted_fitness"] for row in scores}
    ranked = sorted(accepted, key=lambda row: (-lookup[row["variant"]], row["variant"]))[:config.k]
    reference = snapshot["dataset_manifest"]["audit"]["reference"]
    expected = [{**describe_variant(row["variant"], reference), **row, "rank": rank,
                 "predicted_fitness": lookup[row["variant"]]} for rank, row in enumerate(ranked, 1)]
    recommendations = [{key: value for key, value in row.items() if key != "actual_fitness"}
                       for row in state.get("recommendations", [])]
    if recommendations != expected:
        raise ValueError("Saved recommendations differ from the frozen predictor Top-k")
    if state["stage"] == "evaluated":
        values = [row.get("actual_fitness") for row in state["recommendations"]]
        if any(value is None for value in values) or not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Evaluated recommendations require finite nonnegative fitness")
        if state.get("metrics") != _metrics(state, snapshot):
            raise ValueError("Saved metrics differ from the observed recommendations")


def _arm_tables(state, directory):
    rows = []
    for attempt in state["attempts"]:
        response = attempt.get("response", {})
        usage = response.get("usage") or {}
        rows.append({"attempt": attempt["number"], "status": attempt["status"],
                     "validation_status": attempt.get("validation", {}).get("status"),
                     "expected_count": attempt["context"]["proposal_count"],
                     "returned_count": attempt.get("validation", {}).get("returned_count"),
                     "valid_count": attempt.get("validation", {}).get("valid_count"),
                     "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                     "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                     "total_tokens": usage.get("total_tokens"), "usage": json.dumps(usage),
                     "model_requested": response.get("model_requested"), "model_returned": response.get("model_returned"),
                     "api_mode": response.get("api_mode"), "request_id": response.get("request_id")})
    pd.DataFrame(rows, columns=["attempt", "status", "validation_status", "expected_count", "returned_count", "valid_count",
                               "input_tokens", "output_tokens", "total_tokens", "usage", "model_requested", "model_returned", "api_mode", "request_id"]).to_csv(directory / "usage.csv", index=False)
    pd.DataFrame(state.get("recommendations", []), columns=["rank", "variant", "sequence", "mutations", "mutation_count",
                 "hypothesis", "rationale", "knowledge_rule_ids", "predicted_fitness", "actual_fitness"]).to_csv(directory / "recommendations.csv", index=False)


def _load_manifest(output):
    manifest = read_json(Path(output) / "manifest.json")
    count = len(manifest["snapshots"])
    if (not 1 <= count <= 3 or manifest["max_requests"] != 4 * count
            or manifest["max_requests"] > 12 or manifest["per_arm_max_requests"] != 2):
        raise ValueError("Manifest exceeds the fixed ablation request budget")
    expected = {(snapshot["seed"], condition) for snapshot in manifest["snapshots"].values() for condition in CONDITIONS}
    actual = [(item["seed"], item["condition"]) for item in manifest["order"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Manifest must contain each seed/condition exactly once")
    return manifest


# 每条件最多两次请求；超时后的结果若不明确，就停止自动重发并保留状态。
def run_ablation(output, proposer=None):
    output = Path(output)
    with single_writer(output):
        manifest = _load_manifest(output)
        states = {}
        for item in manifest["order"]:
            directory = _arm_directory(output, item)
            state = read_json(directory / "state.json")
            _validate_arm(state, manifest["snapshots"][str(item["seed"])], item["condition"])
            states[(item["seed"], item["condition"])] = state
        if sum(len(state["attempts"]) for state in states.values()) > manifest["max_requests"]:
            raise ValueError("Global ablation request budget exceeded")
        models = {}
        for item in manifest["order"]:
            state = states[(item["seed"], item["condition"])]
            if state["stage"] in ("selected", "evaluated"):
                snapshot = manifest["snapshots"][str(item["seed"])]
                if item["seed"] not in models:
                    models[item["seed"]] = joblib.load(output / snapshot["model_file"])
                _validate_selection(state, snapshot, models[item["seed"]])
        for item in manifest["order"]:
            state = states[(item["seed"], item["condition"])]
            snapshot = manifest["snapshots"][str(item["seed"])]
            config = ExperimentConfig.model_validate(snapshot["config"])
            directory = _arm_directory(output, item)
            path, events = directory / "state.json", directory / "events.jsonl"
            save = lambda: write_json(path, state)
            if state["stage"] == "evaluated":
                continue
            try:
                if state["stage"] == "failed":
                    continue
                if state["stage"] in ("prepared", "generating", "blocked"):
                    accepted, rejected, analyses = _validated_proposals(state, snapshot)
                    if any(attempt["status"] == "started" for attempt in state["attempts"]):
                        raise RuntimeError("An API request has an unknown outcome; it will not be resent automatically")
                    if any(attempt["status"] == "failed" for attempt in state["attempts"]):
                        raise RuntimeError("A failed API request cannot be retried as a format/eligibility correction")
                    while len(accepted) < config.proposal_count and len(state["attempts"]) < 2:
                        if proposer is None:
                            state["stage"] = "blocked"
                            save()
                            break
                        if (getattr(proposer, "prompt_metadata", None) != snapshot["prompt_metadata"]
                                or getattr(proposer, "model", None) != snapshot["api_identity"]["model_requested"]
                                or getattr(proposer, "api_mode", None) != snapshot["api_identity"]["api_mode"]):
                            raise ValueError("Proposer model/API/system prompt differs from the fixed source")
                        total = sum(len(value["attempts"]) for value in states.values())
                        if total >= manifest["max_requests"]:
                            raise RuntimeError("Global request budget exhausted")
                        context = deepcopy(state["initial_context"])
                        if state["attempts"]:
                            context = repair_context(context, pd.DataFrame(snapshot["training_records"]), set(snapshot["eligible_variants"]),
                                                     accepted, rejected, config, 2, 2)
                        # Manipulation is already applied to the saved initial/repair context before either snapshot or send.
                        attempt = {"number": len(state["attempts"]) + 1, "status": "started", "context": deepcopy(context),
                                   "prompt_metadata": deepcopy(snapshot["prompt_metadata"])}
                        state["attempts"].append(attempt)
                        state["stage"] = "generating"
                        save()
                        event(events, "request_started", seed=item["seed"], condition=item["condition"], attempt=deepcopy(attempt))
                        try:
                            response = proposer.generate(deepcopy(context))
                        except Exception as exc:
                            status_code = getattr(exc, "status_code", None)
                            attempt.update(status="failed" if status_code is not None else "started",
                                           error={"type": type(exc).__name__, "status_code": status_code})
                            save()
                            event(events, "request_error", seed=item["seed"], condition=item["condition"], attempt=attempt["number"], error=attempt["error"])
                            detail = "unknown outcome; automatic resend disabled" if status_code is None else "request failed; no automatic retry"
                            raise RuntimeError(f"API request {detail} ({type(exc).__name__}); inspect the saved request state") from None
                        attempt.update(status="received", response=deepcopy(response))
                        save()
                        event(events, "response_received", seed=item["seed"], condition=item["condition"], attempt=attempt["number"], response=deepcopy(response))
                        if (_identity(response) != snapshot["api_identity"]
                                or response.get("prompt_metadata", snapshot["prompt_metadata"]) != snapshot["prompt_metadata"]):
                            raise ValueError("Response model/API/system prompt differs from the fixed source; no correction requested")
                        accepted, rejected, analyses = _validated_proposals(state, snapshot)
                        save()
                    if state["stage"] == "blocked":
                        _arm_tables(state, directory)
                        continue
                    if len(accepted) < config.k:
                        raise RuntimeError(f"Only {len(accepted)} valid proposals; the arm is incomplete after the two-request budget")
                    if len(accepted) > config.score_budget:
                        raise ValueError("Valid proposals exceed the scoring budget")
                    state.update(proposals=deepcopy(accepted), proposal_rejections=deepcopy(rejected), analysis=analyses)
                    if item["seed"] not in models:
                        models[item["seed"]] = joblib.load(output / snapshot["model_file"])
                    predictions = models[item["seed"]].predict(one_hot([row["variant"] for row in accepted]))
                    if not np.isfinite(predictions).all():
                        raise ValueError("Frozen predictor returned nonfinite values")
                    state["scores"] = [{"variant": row["variant"], "predicted_fitness": float(score)} for row, score in zip(accepted, predictions)]
                    lookup = {row["variant"]: row["predicted_fitness"] for row in state["scores"]}
                    ranked = sorted(accepted, key=lambda row: (-lookup[row["variant"]], row["variant"]))[:config.k]
                    reference = snapshot["dataset_manifest"]["audit"]["reference"]
                    state["recommendations"] = [{**describe_variant(row["variant"], reference), **row, "rank": rank,
                                                 "predicted_fitness": lookup[row["variant"]]} for rank, row in enumerate(ranked, 1)]
                    state["stage"] = "selected"
                    save()
                    event(events, "selected_before_oracle", recommendations=deepcopy(state["recommendations"]), scored_count=len(accepted))
                if state["stage"] == "selected":
                    if item["seed"] not in models:
                        models[item["seed"]] = joblib.load(output / snapshot["model_file"])
                    _validate_selection(state, snapshot, models[item["seed"]])
                    variants = [row["variant"] for row in state["recommendations"]]
                    feedback = HiddenOracle(snapshot["data_directory"], variants).evaluate(variants)
                    lookup = {row["variant"]: row["actual_fitness"] for row in feedback}
                    for row in state["recommendations"]:
                        row["actual_fitness"] = lookup[row["variant"]]
                    state["metrics"] = _metrics(state, snapshot)
                    state["stage"] = "evaluated"
                    save()
                    event(events, "arm_evaluated", metrics=state["metrics"])
            except Exception as exc:
                # Selected candidates can resume their deterministic lookup without regenerating proposals.
                if state["stage"] != "selected":
                    state["stage"] = "failed"
                failure = {"type": type(exc).__name__, "message": str(exc)}
                state["errors"].append(failure)
                save()
                event(events, "arm_error", error=failure)
            _arm_tables(state, directory)
        return summarize_ablation(output, require_complete=False)


def _arm_summary(state):
    inputs = outputs = totals = missing = 0
    for attempt in state["attempts"]:
        counts = usage_counts(attempt.get("response", {}).get("usage"))
        inputs += counts["input_tokens"]
        outputs += counts["output_tokens"]
        totals += counts["total_tokens"]
        missing += counts["missing_usage_count"]
    defaults = {"mean_fitness": None, "max_fitness": None, "median_fitness": None,
                "known_best_fitness": state["snapshot"]["known_best_fitness"], "max_minus_known_best": None,
                "valid_count": len(state["proposals"]) if "proposals" in state else sum(attempt.get("validation", {}).get("valid_count", 0) for attempt in state["attempts"]),
                "scored_count": len(state.get("scores", [])), "assay_count": 0}
    return {"seed": state["seed"], "condition": state["condition"], "stage": state["stage"],
            **defaults, **state.get("metrics", {}), "requests": len(state["attempts"]), "input_tokens": inputs,
            "output_tokens": outputs, "total_tokens": totals, "missing_usage_requests": missing,
            "errors": deepcopy(state["errors"])}


# 检查配对完整性后报告full减withheld的差值；未完成条件明确列出。
def summarize_ablation(output, require_complete=True):
    output = Path(output)
    manifest = _load_manifest(output)
    arms, pairs, incomplete = [], [], []
    for key, snapshot in sorted(manifest["snapshots"].items(), key=lambda pair: int(pair[0])):
        states = {condition: read_json(_arm_directory(output, {"seed": snapshot["seed"], "condition": condition}) / "state.json") for condition in CONDITIONS}
        model = None
        for condition, state in states.items():
            _validate_arm(state, snapshot, condition)
            if state["stage"] in ("selected", "evaluated"):
                if model is None:
                    model = joblib.load(output / snapshot["model_file"])
                _validate_selection(state, snapshot, model)
            arms.append(_arm_summary(state))
            if state["stage"] != "evaluated":
                incomplete.append({"seed": snapshot["seed"], "condition": condition, "stage": state["stage"]})
        full, withheld = states["full"], states["withheld"]
        if {k: v for k, v in full["initial_context"].items() if k not in FEEDBACK_FIELDS} != {k: v for k, v in withheld["initial_context"].items() if k not in FEEDBACK_FIELDS}:
            raise ValueError("A paired context differs beyond the two feedback fields")
        if any(state["stage"] != "evaluated" for state in states.values()):
            continue
        a, b = _arm_summary(full), _arm_summary(withheld)
        row = {"seed": snapshot["seed"]}
        for metric in ("mean_fitness", "max_fitness", "median_fitness", "max_minus_known_best", "valid_count", "scored_count", "requests", "total_tokens"):
            row[f"full_{metric}"] = a[metric]
            row[f"withheld_{metric}"] = b[metric]
            row[f"full_minus_withheld_{metric}"] = a[metric] - b[metric]
        full_ids = {item["variant"] for item in full["recommendations"]}
        withheld_ids = {item["variant"] for item in withheld["recommendations"]}
        row.update(recommendation_overlap=len(full_ids & withheld_ids), recommendation_jaccard=len(full_ids & withheld_ids) / len(full_ids | withheld_ids))
        pairs.append(row)
    result = {"complete": not incomplete, "seeds": sorted(int(key) for key in manifest["snapshots"]),
              "max_requests": manifest["max_requests"], "total_requests": sum(arm["requests"] for arm in arms),
              "training_feedback_in_both_conditions": True, "arms": arms, "paired_results": pairs, "incomplete": incomplete,
              "pair_metadata_checks": {"same_source_prefix": True, "same_training_ids_and_model": True,
                                       "same_system_prompt": True, "only_two_initial_context_fields_differ": True}}
    if result["total_requests"] > min(12, manifest["max_requests"]):
        raise ValueError("Total requests exceed the registered budget")
    write_json(output / "summary.json", result)
    columns = ["seed"] + [f"{prefix}_{metric}" for metric in ("mean_fitness", "max_fitness", "median_fitness", "max_minus_known_best", "valid_count", "scored_count", "requests", "total_tokens")
                           for prefix in ("full", "withheld", "full_minus_withheld")] + ["recommendation_overlap", "recommendation_jaccard"]
    pd.DataFrame(pairs, columns=columns).to_csv(output / "paired_results.csv", index=False)
    pd.DataFrame([{**arm, "errors": json.dumps(arm["errors"])} for arm in arms]).to_csv(output / "condition_results.csv", index=False)
    if require_complete and incomplete:
        raise RuntimeError("Ablation has incomplete paired conditions; see summary.json and arm states")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-prefix detailed-feedback display ablation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-runs", nargs="+", required=True)
    prepare.add_argument("--data-dirs", nargs="+", required=True)
    prepare.add_argument("--output", required=True)
    for command in ("run", "compare", "status"):
        child = commands.add_parser(command)
        child.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_ablation(args.source_runs, args.data_dirs, args.output)
            print(json.dumps({"seeds": list(manifest["snapshots"]), "max_requests": manifest["max_requests"], "order": manifest["order"]}, indent=2))
            return 0
        if args.command in ("status", "compare"):
            summary = summarize_ablation(args.output, require_complete=args.command == "compare")
        else:
            manifest = _load_manifest(args.output)
            snapshot = next(iter(manifest["snapshots"].values()))
            config = ExperimentConfig.model_validate(snapshot["config"])
            # Credentials are loaded only when the user explicitly invokes this independent run command.
            proposer = OpenAIProposer(config, prompt_profile="v3")
            try:
                summary = run_ablation(args.output, proposer)
            finally:
                proposer.close()
        print(json.dumps({"complete": summary["complete"], "total_requests": summary["total_requests"], "incomplete": summary["incomplete"]}, indent=2))
        return 0 if summary["complete"] else 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2
