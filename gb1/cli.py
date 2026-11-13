# 命令分发：解析命令行参数后，调用对应的数据、训练、实验或汇总函数。
"""Command-line entry: prepare, select, run, finalize, plot and status."""
import argparse
import json
from pathlib import Path
from .agent import MissingCredentials, OpenAIProposer
from .data import PublicDataset, prepare
from .experiment import finalize, run_experiment
from .models import load_selection, select_model
from .schema import ExperimentConfig, METHODS, compatible_config
from .storage import read_json
from .prompts import load_prompt_profile, validate_prompt_state


# 绘图只读取已评估轮次的指标；均值曲线与累计最佳曲线回答不同问题。
def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    output = Path(output)
    frame = pd.read_csv(output / "round_metrics.csv")
    if frame.empty:
        raise ValueError("No evaluated rounds are available to plot")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, group in frame.groupby("method", sort=False):
        group = group.sort_values("round")
        axes[0].plot(group["round"], group.mean_fitness, marker="o", label=method)
        axes[1].plot(group["round"], group.cumulative_best, marker="o", label=method)
    for axis, title in zip(axes, ("Mean measured fitness per round", "Cumulative best measured fitness")):
        axis.set(title=title, xlabel="Recommendation round", ylabel="S1 fitness")
        axis.set_xticks(sorted(frame["round"].unique()))
        axis.legend()
        axis.grid(alpha=0.25)
    figure.savefig(output / "fitness_progress.png", dpi=180)
    figure.savefig(output / "fitness_progress.svg")
    plt.close(figure)
    return str(output / "fitness_progress.png")


# 每个子命令对应一个明确步骤；命令行参数先交给配置模型统一检查。
def parser():
    result = argparse.ArgumentParser(description="GB1 virtual directed evolution with separated hidden labels")
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Clean an audited source and create immutable data partitions")
    prep.add_argument("--source", required=True)
    prep.add_argument("--data", required=True)
    prep.add_argument("--config", required=True)
    prep.add_argument("--reference", required=True, help="56-aa domain sequence or a plain sequence file")
    prep.add_argument("--variant-column", required=True)
    prep.add_argument("--fitness-column", required=True)
    prep.add_argument("--sheet", default="0", help="Sheet name, or zero-based integer")
    prep.add_argument("--header", type=int, default=0, help="Zero-based header row")
    prep.add_argument("--seed", type=int, help="Override the declared experiment seed (nonnegative 32-bit integer)")
    for command in ("select", "run", "finalize"):
        child = commands.add_parser(command)
        child.add_argument("--data", required=True)
        child.add_argument("--config", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--seed", type=int, help="Override the declared experiment seed (nonnegative 32-bit integer)")
        if command != "select":
            child.add_argument("--selection", required=True, help="Frozen selection.json path")
            child.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        if command == "run":
            child.add_argument("--prompt-profile", choices=["v1", "v2", "v3"], default="v1")
            child.add_argument("--retry-uncertain", action="store_true", help="Explicitly permit another API attempt after an unknown outcome")
    for command in ("plot", "status"):
        child = commands.add_parser(command)
        child.add_argument("--output", required=True)
    aggregate = commands.add_parser("aggregate", help="Summarize independent seeds with explicit counts and missing statuses")
    aggregate.add_argument("--runs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    comparison = commands.add_parser("compare-prompts", help="Compare frozen prompt states without rerunning experiments")
    comparison.add_argument("--v1-runs", nargs="+", required=True)
    comparison.add_argument("--candidate-runs", "--v2-runs", dest="candidate_runs", nargs="+", required=True)
    comparison.add_argument("--candidate-profile", choices=["v2", "v3"], default="v2")
    comparison.add_argument("--output", required=True)
    return result


# prepare划分数据，select选择预测器，run执行轮次，finalize评价独立测试集。
def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "compare-prompts":
            from .compare_prompts import compare_prompt_runs
            result = compare_prompt_runs(args.v1_runs, args.candidate_runs, args.output, candidate_profile=args.candidate_profile)
            print(json.dumps({"seeds": result["seeds"], "output": args.output}, indent=2))
            return 0
        if args.command == "aggregate":
            from .aggregate import aggregate_runs
            print(json.dumps(aggregate_runs(args.runs, args.output), indent=2))
            return 0
        if args.command == "plot":
            print(plot_results(args.output))
            return 0
        if args.command == "status":
            state = read_json(Path(args.output) / "state.json")
            print(json.dumps({method: item["status"] for method, item in state["methods"].items()}, indent=2))
            return 0
        settings = read_json(args.config)
        if args.seed is not None:
            settings["seed"] = args.seed
        config = ExperimentConfig.model_validate(settings)
        if args.command == "prepare":
            reference = args.reference
            if len(reference) != 56:
                reference = Path(reference).read_text(encoding="utf-8").strip()
            sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
            report = prepare(args.source, args.data, config, reference, args.variant_column, args.fitness_column, sheet, args.header)
            print(json.dumps(report, indent=2))
            return 0
        dataset = PublicDataset(args.data)
        if not compatible_config(dataset.manifest["config"], config):
            raise ValueError("Configuration differs from frozen prepared-data manifest")
        if args.command == "select":
            selected = select_model(dataset, config, args.output)
            print(json.dumps(selected["spec"], indent=2))
            return 0
        selection = load_selection(args.selection, dataset, config)
        if args.command == "finalize":
            print(json.dumps(finalize(dataset, selection, config, args.output, args.methods), indent=2))
            return 0
        state_path = Path(args.output) / "state.json"
        if state_path.exists():
            validate_prompt_state(read_json(state_path), load_prompt_profile(args.prompt_profile))
        # 仅LLM相关方法需要模型客户端；随机与模型推荐可以独立离线运行。
        proposer = None
        if set(args.methods) & {"llm", "knowledge"}:
            try:
                proposer = OpenAIProposer(config, prompt_profile=args.prompt_profile)
            except MissingCredentials as exc:
                print(str(exc))
        try:
            state = run_experiment(dataset, selection, config, args.output, args.methods, proposer, args.retry_uncertain, args.prompt_profile)
        finally:
            if proposer is not None:
                proposer.close()
        statuses = {method: item["status"] for method, item in state["methods"].items()}
        print(json.dumps(statuses, indent=2))
        return 0 if all(statuses[method] == "completed" for method in args.methods) else 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2
