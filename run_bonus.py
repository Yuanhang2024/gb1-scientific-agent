# 主动学习入口：prepare准备数据与初始模型，run执行实验，summarize汇总记录。
"""Independent bonus experiments: python run_bonus.py --help."""
import argparse
import json
from pathlib import Path
from bonus.config import BonusConfig
from bonus.active_learning import prepare_experiment, run_experiment, summarize
from gb1.storage import read_json


# 命令行只负责读取和检查参数；数据、选样与统计逻辑放在bonus包中。
def main(argv=None):
    parser = argparse.ArgumentParser(description="RF disagreement, independent conformal calibration and active learning")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--reference", required=True, help="56-aa reference or plain sequence file")
    prepare.add_argument("--variant-column", default="Variants")
    prepare.add_argument("--fitness-column", default="Fitness")
    prepare.add_argument("--sheet", default="Supplementary Dataset 1")
    prepare.add_argument("--header", type=int, default=0)
    run = commands.add_parser("run")
    run.add_argument("--arms", nargs="+", help="Optional explicit arm IDs, e.g. global/random or hd1/mean_plus_std")
    for child in (prepare, run):
        child.add_argument("--data", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--config", default="configs/bonus.json")
        child.add_argument("--seed", type=int, choices=[44, 55, 66])
    summary = commands.add_parser("summarize")
    summary.add_argument("--outputs", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            result = summarize(args.outputs, args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["complete"] else 2
        values = read_json(args.config)
        if args.seed is not None:
            values["seed"] = args.seed
        config = BonusConfig.model_validate(values)
        if args.command == "prepare":
            reference = args.reference
            if len(reference) != 56:
                reference = Path(reference).read_text(encoding="utf-8").strip()
            sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
            result = prepare_experiment(args.source, args.data, args.output, config, reference,
                                        variant_column=args.variant_column, fitness_column=args.fitness_column,
                                        sheet_name=sheet, header=args.header)
            print(json.dumps({"counts": result["dataset_manifest"]["counts"],
                              "initial_fit_seconds": result["initial_model_metadata"]["fit_seconds"],
                              "arms": list(result["arms"])}, indent=2))
            return 0
        result = run_experiment(args.data, args.output, config, args.arms)
        status = {name: arm["status"] for name, arm in result["arms"].items()}
        print(json.dumps(status, indent=2))
        selected = args.arms or list(result["arms"])
        return 0 if all(status[name] == "completed" for name in selected) else 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
