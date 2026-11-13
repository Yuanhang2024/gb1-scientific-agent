# 提示版本比较：读取已保存结果，先核对实验条件，再计算配对差异。
"""Numerical comparison of frozen prompt runs; reads state files only."""
from collections import Counter
from pathlib import Path
import json
import pandas as pd
from .agent import usage_counts
from .prompts import LEGACY_PROFILE
from .storage import read_json, write_json

METHODS = ("llm", "knowledge")


def _count_generated(attempt):
    if "returned_count" in attempt:
        return attempt["returned_count"]
    try:
        body = json.loads(attempt["response"]["text"])
    except (KeyError, ValueError, TypeError):
        return None
    proposals = body.get("proposals") if isinstance(body, dict) else None
    return len(proposals) if isinstance(proposals, list) else None


def _usage(attempt):
    return usage_counts(attempt.get("response", {}).get("usage"))


# 区分原始生成数、合法候选数、实际评分数和已测推荐数，避免混用这些数量。
def _summarize_records(records):
    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    recommendations = [row for record in records if record.get("stage") == "evaluated"
                       for row in record.get("recommendations", [])]
    actual = [row["actual_fitness"] for row in recommendations]
    raw_counts = [_count_generated(attempt) for attempt in attempts]
    usage = [_usage(attempt) for attempt in attempts]
    return {"evaluated_rounds": sum(record.get("stage") == "evaluated" for record in records),
            "recommendation_count": len(recommendations), "request_count": len(attempts),
            "response_count": sum("response" in attempt for attempt in attempts),
            "request_status_counts": dict(Counter(attempt["status"] for attempt in attempts)),
            "requested_count": sum(attempt["context"]["proposal_count"] for attempt in attempts),
            "raw_generated_count": sum(count for count in raw_counts if count is not None),
            "unknown_raw_count_requests": sum(count is None for count in raw_counts),
            "accepted_valid_count": sum(record.get("valid_proposal_count", 0) for record in records),
            "scored_count": sum(record.get("model_score_count", 0) for record in records),
            "mean_fitness": sum(actual) / len(actual) if actual else None,
            "best_new_fitness": max(actual) if actual else None,
            **{key: sum(item[key] for item in usage) for key in ("input_tokens", "output_tokens", "total_tokens", "missing_usage_count")}}


def _load_runs(directories, expected_profile):
    runs = {}
    metadata = None
    for directory in map(Path, directories):
        state = read_json(directory / "state.json")
        prompt = state.get("prompt_metadata", LEGACY_PROFILE.metadata())
        if prompt["profile"] != expected_profile:
            raise ValueError(f"Expected {expected_profile} run, received {prompt['profile']}")
        if metadata is not None and prompt != metadata:
            raise ValueError("A prompt group contains different versions or system prompt texts")
        metadata = prompt
        seed = state["config"]["seed"]
        if seed in runs:
            raise ValueError("Each prompt group must contain unique seeds")
        runs[seed] = state
    if not runs:
        raise ValueError("Each prompt group must contain at least one run")
    return runs, metadata


# 先确认轮次完整、模型及接口一致，才允许形成同seed的版本配对。
def _completed_method_identity(state, method):
    item = state["methods"][method]
    expected_rounds = state["config"]["rounds"]
    rounds = item["rounds"]
    if (item["status"] != "completed" or len(rounds) != expected_rounds
            or [record["round"] for record in rounds] != list(range(1, expected_rounds + 1))
            or any(record["stage"] != "evaluated" or len(record.get("recommendations", [])) != state["config"]["k"] for record in rounds)):
        raise ValueError(f"{method} requires complete paired rounds for prompt comparison")
    identities = set()
    for record in rounds:
        for attempt in record["attempts"]:
            response = attempt.get("response")
            if response is None:
                continue
            model, api_mode = response.get("model_requested"), response.get("api_mode")
            if not isinstance(model, str) or not model or not isinstance(api_mode, str) or not api_mode:
                raise ValueError("Prompt comparison requires recorded LLM model and API mode")
            identities.add((model, response.get("model_returned"), api_mode))
    if len(identities) != 1:
        raise ValueError("Prompt comparison requires one consistent LLM model and API mode per method")
    return next(iter(identities))


# 配对比较要求相同数据划分、初始预测器和预算；这里读取记录，不重新调用LLM。
def compare_prompt_runs(v1_directories, v2_directories, output, candidate_profile="v2"):
    if candidate_profile not in ("v2", "v3"):
        raise ValueError("candidate_profile must be v2 or v3")
    v1, v1_prompt = _load_runs(v1_directories, "v1")
    candidate, candidate_prompt = _load_runs(v2_directories, candidate_profile)
    if set(v1) != set(candidate):
        raise ValueError(f"v1 and {candidate_profile} must contain identical seed sets")
    checks, method_rows, round_rows, request_rows, difference_rows = [], [], [], [], []
    for seed in sorted(v1):
        first, second = v1[seed], candidate[seed]
        checks.append({"seed": seed, "same_config": first["config"] == second["config"],
                       "same_dataset_manifest": first["dataset_manifest"] == second["dataset_manifest"],
                       "same_selection": first["selection"] == second["selection"]})
        if not all(checks[-1][key] for key in ("same_config", "same_dataset_manifest", "same_selection")):
            raise ValueError(f"Seed {seed} uses different frozen data, selected model or experiment configuration")
        identities = {}
        for method in METHODS:
            baseline_identity = _completed_method_identity(first, method)
            candidate_identity = _completed_method_identity(second, method)
            if baseline_identity != candidate_identity:
                raise ValueError(f"Seed {seed}/{method} uses a different LLM model or API mode")
            identities[method] = {"model_requested": baseline_identity[0], "model_returned": baseline_identity[1], "api_mode": baseline_identity[2]}
        if len({tuple(identity.values()) for identity in identities.values()}) != 1:
            raise ValueError("Compared methods must use the same LLM model and API mode")
        checks[-1].update(same_llm_model_and_api_mode=True, complete_method_round_pairs=True, llm_api_identity=identities)
        for method in METHODS:
            summaries = {}
            for profile, state in (("v1", first), (candidate_profile, second)):
                item = state["methods"][method]
                summary = _summarize_records(item["rounds"])
                summaries[profile] = summary
                method_rows.append({"seed": seed, "profile": profile, "method": method, "method_status": item["status"], **summary})
                for record in item["rounds"]:
                    round_rows.append({"seed": seed, "profile": profile, "method": method, "round": record["round"],
                                       "stage": record["stage"], **_summarize_records([record])})
                    for attempt in record["attempts"]:
                        request_rows.append({"seed": seed, "profile": profile, "method": method, "round": record["round"],
                                             "attempt": attempt["number"], "status": attempt["status"],
                                             "requested_count": attempt["context"]["proposal_count"],
                                             "raw_generated_count": _count_generated(attempt), "valid_count": attempt.get("valid_count"),
                                             **_usage(attempt)})
            difference = {"seed": seed, "method": method,
                          "v1_status": first["methods"][method]["status"], f"{candidate_profile}_status": second["methods"][method]["status"]}
            for metric in ("request_count", "requested_count", "raw_generated_count", "accepted_valid_count", "scored_count",
                           "recommendation_count", "mean_fitness", "best_new_fitness", "total_tokens"):
                a, b = summaries["v1"][metric], summaries[candidate_profile][metric]
                difference[f"v1_{metric}"] = a
                difference[f"{candidate_profile}_{metric}"] = b
                difference[f"delta_{metric}"] = b - a if a is not None and b is not None else None
            difference_rows.append(difference)
    result = {"seeds": sorted(v1), "prompt_metadata": {"v1": v1_prompt, candidate_profile: candidate_prompt},
              "metadata_checks": checks, "methods": method_rows, "rounds": round_rows,
              "requests": request_rows, "per_seed_differences": difference_rows}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prompt_comparison.json", result)
    for name, rows in (("methods", method_rows), ("rounds", round_rows), ("requests", request_rows), ("seed_differences", difference_rows)):
        flat = [{**row, **({"request_status_counts": json.dumps(row["request_status_counts"], sort_keys=True)}
                           if "request_status_counts" in row else {})} for row in rows]
        pd.DataFrame(flat).to_csv(output / f"prompt_comparison_{name}.csv", index=False)
    _plot(round_rows, output)
    return result


# 误差条来自各seed统计量的样本标准差，并在图中说明有效seed数量。
def _plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    titles = ("Round mean measured fitness", "Best measured fitness within each round")
    for axis, title in zip(axes, titles):
        axis.set(title=title, xlabel="Recommendation round", ylabel="S1 fitness (WT=1)")
    seed_count = frame["seed"].nunique() if not frame.empty else 0
    caption = ("1 seed; per-seed values (sample SD unavailable)." if seed_count == 1
               else f"{seed_count} seeds; mean ± sample SD across seeds.")
    figure.supxlabel(f"{caption}\nSource: saved run records, 2026-11-09.", fontsize=9)
    if not frame.empty:
        complete = frame[frame.stage == "evaluated"]
        for axis, metric in zip(axes, ("mean_fitness", "best_new_fitness")):
            for (profile, method), group in complete.groupby(["profile", "method"]):
                summary = group.groupby("round")[metric].agg(["mean", "std", "count"])
                axis.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0),
                              marker="o", capsize=3, label=f"{profile}/{method}")
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend()
    figure.savefig(output / "prompt_comparison.png", dpi=180)
    plt.close(figure)
