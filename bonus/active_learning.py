# 主动选样实验：根据预测均值或分歧选样，获取反馈后仅更新本实验臂。
"""Independent full-pool acquisition and fixed-HD-pool experiments."""
from copy import deepcopy
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from gb1.data import HiddenOracle, describe_variant
from gb1.storage import read_json, write_json, event, single_writer
from .data import BonusDataset, prepare_bonus
from .uncertainty import fit_forest, predict_distribution, evaluate_frozen, interval_bounds, evaluate_predictions


# arm是一个独立实验条件，例如全池greedy或HD1随机；每个条件维护自己的反馈历史。
def arm_definitions(config):
    arms = [{"id": f"global/{strategy}", "hd": None, "strategy": strategy} for strategy in config.global_strategies]
    arms += [{"id": f"hd{group}/{strategy}", "hd": group, "strategy": strategy}
             for group in config.hd_groups for strategy in config.hd_strategies]
    return arms


# 准备公共数据和同一初始模型，并保存独立评价结果；策略和参数在运行前固定。
def prepare_experiment(source, data_directory, output, config, reference, **schema):
    output, data_directory = Path(output), Path(data_directory)
    resolved_output, resolved_data = output.resolve(), data_directory.resolve()
    if resolved_output == resolved_data or resolved_output in resolved_data.parents or resolved_data in resolved_output.parents:
        raise ValueError("Bonus data and result directories must be separate, nonnested locations")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Bonus result destination must be empty; existing experiments are not overwritten")
    manifest = prepare_bonus(source, data_directory, config, reference, **schema)
    dataset = BonusDataset(data_directory)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = fit_forest(dataset.train, config)
    fit_seconds = time.perf_counter() - started
    joblib.dump(model, output / "initial_model.joblib")
    joblib.dump(model, output / "model.joblib")
    calibration, evaluation, predictions = evaluate_frozen(model, dataset, config)
    write_json(output / "initial_calibration.json", calibration)
    write_json(output / "initial_evaluation.json", evaluation)
    predictions.to_csv(output / "initial_test_predictions.csv", index=False)
    model_metadata = {"family": "RandomForestRegressor", "parameters": model.get_params(),
                      "training_count": len(dataset.train), "training_variants": list(dataset.train.variant),
                      "feature_count": 80, "fit_seconds": fit_seconds,
                      "uncertainty": "Sample prediction disagreement across trees, ddof=1", "hyperparameter_selection": "Fixed in advance; no calibration/test tuning"}
    write_json(output / "initial_model_metadata.json", model_metadata)
    state = {"format_version": 1, "study": "bonus-learning", "config": config.model_dump(), "dataset_manifest": manifest,
             "data_directory": str(data_directory.resolve()), "initial_model_metadata": model_metadata,
             "initial_training_rows": dataset.train.to_dict(orient="records"),
             "candidate_variants": list(dataset.candidates.variant),
             "arms": {arm["id"]: {**arm, "status": "pending", "rounds": []} for arm in arm_definitions(config)}}
    write_json(output / "state.json", state)
    return state


# 只追加当前arm已经取得的测量，保证对照之间不共享新标签。
def known_for_arm(dataset, arm):
    observed = [{**describe_variant(row["variant"], dataset.reference), "fitness": row["actual_fitness"]}
                for record in arm["rounds"] if record["stage"] == "evaluated" for row in record["recommendations"]]
    known = pd.concat([dataset.train, pd.DataFrame(observed)], ignore_index=True) if observed else dataset.train.copy()
    if known.variant.duplicated().any():
        raise ValueError("An arm cannot reuse an assayed variant")
    return known


def eligible_for_arm(dataset, known, arm):
    ids = set(dataset.candidates.variant) if arm["hd"] is None else set(dataset.hd_pools[str(arm["hd"])])
    return sorted(ids - set(known.variant))


# 采集函数只接收候选ID、预测均值和分歧：greedy用均值，std_only用分歧，组合用均值+kappa*分歧。
def select_candidates(variants, mean, std, strategy, config, round_number, hd=None):
    """Pure acquisition: calibration quantiles and held-out labels are not accepted inputs."""
    variants = list(variants)
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if len(variants) < config.k or mean.shape != (len(variants),) or std.shape != mean.shape:
        raise ValueError("Candidate counts and prediction arrays cannot satisfy the assay budget")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Acquisition requires finite predictions and nonnegative disagreement")
    if strategy == "random":
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, round_number, 4500, hd or 0]))
        indices = rng.choice(len(variants), config.k, replace=False).tolist()
        acquisition = np.full(len(variants), np.nan)
    else:
        acquisition = mean if strategy == "greedy" else std if strategy == "std_only" else mean + config.kappa * std if strategy == "mean_plus_std" else None
        if acquisition is None:
            raise ValueError("Unknown acquisition strategy")
        indices = sorted(range(len(variants)), key=lambda index: (-acquisition[index], variants[index]))[:config.k]
    return indices, acquisition


def _validate_state(state, dataset, config):
    if state["study"] != "bonus-learning" or state["config"] != config.model_dump() or state["dataset_manifest"] != dataset.manifest:
        raise ValueError("Frozen bonus configuration or dataset changed; create a new experiment")
    if (state["initial_training_rows"] != dataset.train.to_dict(orient="records")
            or state["candidate_variants"] != list(dataset.candidates.variant)):
        raise ValueError("Initial training measurements or candidate IDs changed")
    definitions = {arm["id"]: arm for arm in arm_definitions(config)}
    if set(state["arms"]) != set(definitions):
        raise ValueError("Arm definitions differ from the fixed configuration")
    for name, arm in state["arms"].items():
        if any(arm[key] != value for key, value in definitions[name].items()):
            raise ValueError("Arm identity changed")
        used = set()
        for number, record in enumerate(arm["rounds"], 1):
            if number > config.rounds or record["round"] != number:
                raise ValueError("Round numbering violates the fixed experiment budget")
            if record["stage"] in ("selected", "intervals_ready", "evaluated"):
                variants = [row["variant"] for row in record["recommendations"]]
                allowed = set(dataset.candidates.variant) if arm["hd"] is None else set(dataset.hd_pools[str(arm["hd"])])
                if len(variants) != config.k or len(set(variants)) != config.k or not set(variants) <= allowed or used & set(variants):
                    raise ValueError("Stored selections violate eligibility or uniqueness")
                used.update(variants)


def _round_metrics(dataset, known, arm, record):
    rows = record["recommendations"]
    actual = np.array([row["actual_fitness"] for row in rows])
    mean = np.array([row["predicted_fitness"] for row in rows])
    std = np.array([row["tree_disagreement_std"] for row in rows])
    bounds = {key: np.array([row[key] for row in rows]) for key in ("abs_lower", "abs_upper", "scaled_lower", "scaled_upper")}
    wt = float(dataset.train.loc[dataset.train.variant == "VDGV", "fitness"].iloc[0])
    # 分组实验的此前最佳只在同HD组内计算，避免被其他组的高分掩盖。
    same_group = known if arm["hd"] is None else known[known.mutation_count == arm["hd"]]
    prior = float(same_group.fitness.max()) if len(same_group) else None
    earlier = [row["actual_fitness"] for previous in arm["rounds"] if previous["stage"] == "evaluated" for row in previous["recommendations"]]
    new_best = max(earlier + actual.tolist())
    return {"mean_fitness": float(actual.mean()), "max_fitness": float(actual.max()), "median_fitness": float(np.median(actual)),
            "wt_fitness": wt, "above_wt_count": int((actual > wt).sum()),
            "prior_group_best": prior, "group_cumulative_best": max(prior, float(actual.max())) if prior is not None else float(actual.max()),
            "new_minus_prior_group_best": float(actual.max()) - prior if prior is not None else None,
            "cumulative_new_best": new_best, "training_count": len(known), "scored_count": record["scored_count"],
            "assays": len(rows), "selected_interval_evaluation": evaluate_predictions(actual, mean, std, bounds),
            "selected_coverage_scope": "Empirical coverage of adaptively selected points; no ordinary split-conformal coverage guarantee"}


# 先选定候选，再保存预测区间，最后查询真实反馈；各阶段保存后支持继续运行。
def run_experiment(data_directory, output, config, arms=None):
    output = Path(output)
    dataset = BonusDataset(data_directory)
    with single_writer(output):
        state_path = output / "state.json"
        state = read_json(state_path)
        _validate_state(state, dataset, config)
        chosen = list(state["arms"]) if arms is None else list(arms)
        if set(chosen) - set(state["arms"]):
            raise ValueError("Unknown arm requested")
        initial_model = None
        for name in chosen:
            arm = state["arms"][name]
            directory = output / "arms" / name
            directory.mkdir(parents=True, exist_ok=True)
            log = directory / "events.jsonl"
            save = lambda: write_json(state_path, state)
            try:
                for number in range(1, config.rounds + 1):
                    if len(arm["rounds"]) < number:
                        arm["rounds"].append({"round": number, "stage": "started"})
                        save()
                    record = arm["rounds"][number - 1]
                    if record["stage"] == "evaluated":
                        continue
                    round_directory = directory / f"round-{number}"
                    round_directory.mkdir(parents=True, exist_ok=True)
                    known = known_for_arm(dataset, arm)
                    if record["stage"] == "started":
                        started = time.perf_counter()
                        if number == 1:
                            if initial_model is None:
                                initial_model = joblib.load(output / "initial_model.joblib")
                            model = initial_model
                        else:
                            model = fit_forest(known, config)
                        record["fit_seconds"] = time.perf_counter() - started
                        record["training_count"] = len(known)
                        record["training_variants"] = list(known.variant)
                        eligible = eligible_for_arm(dataset, known, arm)
                        started = time.perf_counter()
                        mean, std = predict_distribution(model, eligible, config.prediction_batch_size)
                        indices, acquisition = select_candidates(eligible, mean, std, arm["strategy"], config, number, arm["hd"])
                        record["score_seconds"] = time.perf_counter() - started
                        record["scored_count"] = len(eligible)
                        if config.save_pool_scores:
                            pd.DataFrame({"variant": eligible, "predicted_fitness": mean, "tree_disagreement_std": std,
                                          "acquisition_score": acquisition}).to_csv(round_directory / "pool_scores.csv", index=False)
                        record["recommendations"] = [{**describe_variant(eligible[index], dataset.reference), "rank": rank,
                                                       "predicted_fitness": float(mean[index]), "tree_disagreement_std": float(std[index]),
                                                       "acquisition_score": None if arm["strategy"] == "random" else float(acquisition[index])}
                                                      for rank, index in enumerate(indices, 1)]
                        joblib.dump(model, round_directory / "model.joblib")
                        record["stage"] = "selected"
                        save()
                        event(log, "selected_without_calibration_or_test_labels", round=number, scored_count=len(eligible),
                              training_variants=list(known.variant), recommendations=deepcopy(record["recommendations"]))
                    # 候选已经固定，随后计算的校准与测试指标只用于评价，不回头改变本轮选择。
                    if record["stage"] == "selected":
                        model = joblib.load(round_directory / "model.joblib")
                        calibration, evaluation, test_predictions = evaluate_frozen(model, dataset, config)
                        write_json(round_directory / "calibration.json", calibration)
                        write_json(round_directory / "heldout_evaluation.json", evaluation)
                        test_predictions.to_csv(round_directory / "test_predictions.csv", index=False)
                        record["calibration"] = calibration
                        record["heldout_evaluation"] = evaluation
                        bounds = interval_bounds([row["predicted_fitness"] for row in record["recommendations"]],
                                                 [row["tree_disagreement_std"] for row in record["recommendations"]], calibration)
                        for index, row in enumerate(record["recommendations"]):
                            row.update({key: float(values[index]) for key, values in bounds.items()})
                        record["stage"] = "intervals_ready"
                        save()
                        event(log, "intervals_saved_before_oracle", round=number, calibration=calibration,
                              recommendations=deepcopy(record["recommendations"]))
                    # 区间已保存后再读选中候选的实测值，才能如实统计它们是否落入区间。
                    if record["stage"] == "intervals_ready":
                        variants = [row["variant"] for row in record["recommendations"]]
                        feedback = HiddenOracle(dataset.directory, variants).evaluate(variants)
                        lookup = {row["variant"]: row["actual_fitness"] for row in feedback}
                        for row in record["recommendations"]:
                            row["actual_fitness"] = lookup[row["variant"]]
                        record["metrics"] = _round_metrics(dataset, known, arm, record)
                        record["stage"] = "evaluated"
                        save()
                        pd.DataFrame(record["recommendations"]).to_csv(round_directory / "recommendations.csv", index=False)
                        event(log, "round_evaluated", round=number, metrics=record["metrics"])
                if "final_evaluation" not in arm:
                    known = known_for_arm(dataset, arm)
                    final_model = fit_forest(known, config)
                    calibration, evaluation, predictions = evaluate_frozen(final_model, dataset, config)
                    joblib.dump(final_model, directory / "final_model.joblib")
                    write_json(directory / "final_calibration.json", calibration)
                    write_json(directory / "final_evaluation.json", evaluation)
                    predictions.to_csv(directory / "final_test_predictions.csv", index=False)
                    arm["final_evaluation"] = evaluation
                    arm["final_training_count"] = len(known)
                    arm["final_calibration"] = calibration
                arm["status"] = "completed"
                save()
            except Exception as exc:
                arm["status"] = "failed"
                arm.setdefault("failures", []).append({"type": type(exc).__name__, "message": str(exc)})
                save()
                event(log, "arm_failed", error=arm["failures"][-1])
        export_seed(state, output)
        return state


def export_seed(state, output):
    rows, selected, coverage = [], [], []
    for name, arm in state["arms"].items():
        for record in arm["rounds"]:
            if record["stage"] != "evaluated":
                continue
            base = {"seed": state["config"]["seed"], "arm": name, "hd": arm["hd"], "strategy": arm["strategy"], "round": record["round"]}
            row = {**base, **{key: value for key, value in record["metrics"].items() if not isinstance(value, dict)},
                   "fit_seconds": record["fit_seconds"], "score_seconds": record["score_seconds"]}
            row.update({f"selected_{key}": value for key, value in record["metrics"]["selected_interval_evaluation"].items()})
            rows.append(row)
            selected.extend({**base, **recommendation} for recommendation in record["recommendations"])
            for group, metrics in {"overall": record["heldout_evaluation"]["overall"], **record["heldout_evaluation"]["by_hd"]}.items():
                coverage.append({**base, "test_hd": group, **metrics})
    base_columns = ["seed", "arm", "hd", "strategy", "round"]
    (pd.DataFrame(rows) if rows else pd.DataFrame(columns=base_columns + ["mean_fitness", "max_fitness", "cumulative_new_best", "scored_count"])).to_csv(Path(output) / "round_metrics.csv", index=False)
    (pd.DataFrame(selected) if selected else pd.DataFrame(columns=base_columns + ["variant", "actual_fitness", "predicted_fitness"])).to_csv(Path(output) / "all_recommendations.csv", index=False)
    (pd.DataFrame(coverage) if coverage else pd.DataFrame(columns=base_columns + ["test_hd", "n", "abs_coverage", "scaled_coverage"])).to_csv(Path(output) / "heldout_coverage.csv", index=False)
    write_json(Path(output) / "status.json", {name: arm["status"] for name, arm in state["arms"].items()})


# 汇总各seed已经保存的结果，同时记录完成状态，避免把缺失实验默默排除。
def summarize(outputs, destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        if not (destination / "summary.json").exists() or read_json(destination / "summary.json").get("study") != "bonus-learning-summary":
            raise ValueError("Summary destination must be new or an existing bonus summary directory")
    states = [read_json(Path(output) / "state.json") for output in outputs]
    seeds = [state["config"]["seed"] for state in states]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Summary requires distinct seed runs")
    reference = {key: value for key, value in states[0]["config"].items() if key != "seed"}
    if any(state["study"] != "bonus-learning" or {key: value for key, value in state["config"].items() if key != "seed"} != reference for state in states):
        raise ValueError("Summary runs have different settings or study identities")
    rounds, recommendations, initial, final, status = [], [], [], [], []
    for output, state in zip(outputs, states):
        baseline = read_json(Path(output) / "initial_evaluation.json")
        for group, metrics in {"overall": baseline["overall"], **baseline["by_hd"]}.items():
            initial.append({"seed": state["config"]["seed"], "test_hd": group, **metrics})
        for name, arm in state["arms"].items():
            status.append({"seed": state["config"]["seed"], "arm": name, "status": arm["status"]})
            if "final_evaluation" in arm:
                for group, metrics in {"overall": arm["final_evaluation"]["overall"], **arm["final_evaluation"]["by_hd"]}.items():
                    final.append({"seed": state["config"]["seed"], "arm": name, "test_hd": group, "training_count": arm["final_training_count"], **metrics})
        for filename, target in (("round_metrics.csv", rounds), ("all_recommendations.csv", recommendations)):
            frame = pd.read_csv(Path(output) / filename)
            target.extend(frame.to_dict(orient="records"))
    destination.mkdir(parents=True, exist_ok=True)
    (pd.DataFrame(rounds) if rounds else pd.DataFrame(columns=["seed", "arm", "hd", "strategy", "round", "mean_fitness"])).to_csv(destination / "round_metrics.csv", index=False)
    (pd.DataFrame(recommendations) if recommendations else pd.DataFrame(columns=["seed", "arm", "round", "variant", "actual_fitness"])).to_csv(destination / "all_recommendations.csv", index=False)
    pd.DataFrame(initial).to_csv(destination / "initial_heldout_coverage.csv", index=False)
    (pd.DataFrame(final) if final else pd.DataFrame(columns=["seed", "arm", "test_hd", "n", "abs_coverage", "scaled_coverage"])).to_csv(destination / "final_heldout_coverage.csv", index=False)
    result = {"study": "bonus-learning-summary", "seeds": seeds, "config": reference, "status": status,
              "complete": all(row["status"] == "completed" for row in status),
              "rounds": len(rounds), "recommendations": len(recommendations),
              "coverage_scope": "Random heldout marginal coverage and empirical selected-point coverage reported separately",
              "uncertainty": "Tree prediction sample disagreement, ddof=1; no Gaussian CI claim"}
    write_json(destination / "summary.json", result)
    _plot_summary(pd.DataFrame(rounds), destination)
    _plot_coverage(pd.DataFrame(final), destination, 1 - reference["alpha"])
    return result


def _plot_summary(frame, destination):
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, grouped in ((axes[0], frame[frame.hd.isna()]), (axes[1], frame[frame.hd.notna()])):
        keys = ["strategy"] if axis is axes[0] else ["hd", "strategy"]
        for key, rows in grouped.groupby(keys):
            stats = rows.groupby("round").mean_fitness.agg(["mean", "std", "count"])
            label = str(key[0] if isinstance(key, tuple) else key) if axis is axes[0] else f"HD{int(key[0])}/{key[1]}"
            axis.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), marker="o", capsize=3, label=label)
        axis.set(xlabel="Acquisition round", ylabel="Measured S1 fitness (WT=1)")
        axis.set_xticks(sorted(int(value) for value in frame["round"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_title("Full-pool acquisition: round mean")
    axes[1].set_title("Fixed HD pools: round mean")
    count = frame.seed.nunique()
    caption = "1 seed; per-seed values (sample SD unavailable)." if count == 1 else f"{count} seeds; error bars show sample SD across seeds."
    figure.supxlabel(f"{caption}\nSource: saved S1 run results.", fontsize=9)
    figure.savefig(destination / "learning_curves.png", dpi=180)
    plt.close(figure)


def _plot_coverage(frame, destination, target):
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overall = frame[frame.test_hd == "overall"]
    names = sorted(overall.arm.unique())
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(names))
    for offset, kind in ((-0.18, "abs"), (0.18, "scaled")):
        coverage = overall.groupby("arm")[f"{kind}_coverage"].agg(["mean", "std"]).reindex(names)
        width = overall.groupby("arm")[f"{kind}_mean_width"].agg(["mean", "std"]).reindex(names)
        axes[0].bar(x + offset, coverage["mean"], yerr=coverage["std"].fillna(0), capsize=2, width=0.35, label=kind)
        axes[1].bar(x + offset, width["mean"], yerr=width["std"].fillna(0), capsize=2, width=0.35, label=kind)
    axes[0].axhline(target, color="black", linestyle="--", label=f"nominal {target:.0%}")
    axes[0].set(title="Final random held-out coverage", ylabel="Empirical coverage", ylim=(0, 1.05))
    axes[1].set(title="Final random held-out mean interval width", ylabel="S1 fitness units")
    for axis in axes:
        axis.set_xticks(x, names, rotation=65, ha="right", fontsize=7)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    count = frame.seed.nunique()
    caption = "1 seed; per-seed values (sample SD unavailable)." if count == 1 else f"{count} seeds; error bars show sample SD across seeds."
    figure.supxlabel(f"{caption}\nSource: saved S1 run results.", fontsize=9)
    figure.savefig(destination / "heldout_coverage.png", dpi=180)
    plt.close(figure)
