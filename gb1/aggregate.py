# 跨种子汇总：比较相同实验设置下的结果，同时保留未完成方法的状态。
"""Combine separately completed random seeds without hiding missing methods."""
from pathlib import Path
import pandas as pd
from .storage import read_json, write_json


# 只合并数据来源与预算一致的独立seed记录，未完成方法的状态也保留。
def aggregate_runs(directories, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, statuses, seeds = [], [], []
    comparison_config = None
    comparison_source = None
    for directory in map(Path, directories):
        state = read_json(directory / "state.json")
        seed = state["config"]["seed"]
        if seed in seeds:
            raise ValueError("Each aggregate input must have a distinct seed")
        seeds.append(seed)
        common = {key: value for key, value in state["config"].items() if key != "seed"}
        if comparison_config is not None and common != comparison_config:
            raise ValueError("Aggregate runs must use the same experiment budgets and settings")
        comparison_config = common
        source = {key: state["dataset_manifest"][key] for key in ("source", "schema", "audit")}
        if comparison_source is not None and source != comparison_source:
            raise ValueError("Aggregate runs must use the same audited source and reference sequence")
        comparison_source = source
        for method, item in state["methods"].items():
            statuses.append({"seed": seed, "method": method, "status": item["status"]})
        frame = pd.read_csv(directory / "round_metrics.csv")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No completed rounds to aggregate")
    combined = pd.concat(frames, ignore_index=True)
    metrics = ["mean_fitness", "max_fitness", "cumulative_best", "new_high_fitness_count", "model_score_count", "llm_attempts"]
    # 每个seed提供一条对应轮次指标；这里跨seed计算均值、样本标准差和有效数量。
    grouped = combined.groupby(["method", "round"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(pair) for pair in grouped.columns]
    grouped.reset_index().to_csv(output / "aggregate_round_metrics.csv", index=False)
    combined.to_csv(output / "all_seed_round_metrics.csv", index=False)
    pd.DataFrame(statuses).to_csv(output / "all_seed_status.csv", index=False)
    write_json(output / "aggregate_manifest.json", {"seeds": seeds, "config": comparison_config,
               "input_directories": [str(Path(path).resolve()) for path in directories],
               "uncertainty": "Sample standard deviation across available seeds; counts reported per method and round."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    table = grouped.reset_index()
    for method, rows in table.groupby("method"):
        for axis, metric, title in zip(axes, ("mean_fitness", "cumulative_best"),
                                       ("Mean measured fitness", "Cumulative best measured fitness")):
            axis.errorbar(rows["round"], rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"].fillna(0),
                          marker="o", capsize=3, label=method)
            axis.set(title=title, xlabel="Recommendation round", ylabel="S1 fitness")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    figure.savefig(output / "aggregate_fitness_progress.png", dpi=180)
    figure.savefig(output / "aggregate_fitness_progress.svg")
    plt.close(figure)
    return {"seeds": seeds, "rows": len(combined), "output": str(output)}
