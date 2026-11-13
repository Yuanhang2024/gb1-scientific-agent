# 反馈必要性对照：复用源v3运行的第1轮上下文与初始预测器，连续三轮不引入任何反馈。
# 与源运行（反馈累积）同种子、同方法(knowledge)、同提示、同预算配对比较。
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "no_feedback"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from gb1.agent import OpenAIProposer, collect_proposals, load_knowledge  # noqa: E402
from gb1.data import one_hot  # noqa: E402
from gb1.prompts import load_prompt_profile, repair_context  # noqa: E402
from gb1.schema import ExperimentConfig  # noqa: E402
from gb1.storage import read_json  # noqa: E402

SEEDS = (11, 22, 33)
ROUNDS = 3


def round1_context(seed):
    path = ROOT / "results" / "prompt-v3" / f"seed-{seed}" / "run" / "events.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if (obj.get("kind") == "llm_request" and obj.get("method") == "knowledge"
                and obj.get("round") == 1 and obj.get("attempt") == 1):
            return obj["context"]
    raise RuntimeError(f"round-1 context not found for seed {seed}")


def run_seed(seed, config, usage_log):
    data_dir = ROOT / "data" / "processed" / f"seed-{seed}"
    run_dir = ROOT / "results" / "prompt-v3" / f"seed-{seed}" / "run"

    base_context = round1_context(seed)
    knowledge = load_knowledge(config.knowledge_path)
    rule_ids = [rule.id for rule in knowledge.rules]
    train = pd.read_csv(data_dir / "train.csv")
    oracle = pd.read_csv(data_dir / "hidden" / "oracle.csv").set_index("variant")["fitness"]
    eligible = set(pd.read_csv(data_dir / "candidates.csv").variant.astype(str)) - set(train.variant.astype(str))

    model = joblib.load(run_dir / "knowledge-round-1.joblib")
    proposer = OpenAIProposer(config, prompt_profile="v3")
    prompt = load_prompt_profile("v3")

    picked, rows = set(), []
    try:
        for round_number in range(1, ROUNDS + 1):
            accepted, rejections = [], []
            attempt_context = {**base_context, "proposal_count": config.proposal_count}
            for attempt_number in range(1, config.llm_max_calls_per_round + 1):
                if attempt_number > 1:
                    attempt_context = repair_context(base_context, train, eligible - picked, accepted,
                                                     rejections, config, round_number, attempt_number)
                response = proposer.generate(dict(attempt_context))
                usage_log.append({"seed": seed, "round": round_number, "attempt": attempt_number,
                                  "model_returned": response.get("model_returned"),
                                  "usage": response.get("usage")})
                rejected_ids = {r["variant"] for r in rejections if isinstance(r.get("variant"), str)}
                allowed = (eligible - picked) - rejected_ids
                expected = config.proposal_count - len(accepted)
                try:
                    if not response["complete"]:
                        raise ValueError("incomplete response")
                    analysis, valid, rejected = collect_proposals(
                        response["text"], allowed, config, accepted, rule_ids, expected)
                    accepted.extend(valid)
                    rejections.extend(rejected)
                except ValueError as exc:
                    rejections.append({"variant": None, "reason": str(exc)})
                if len(accepted) >= config.proposal_count:
                    break

            variants = [row["variant"] for row in accepted]
            predictions = model.predict(one_hot(variants))
            ranked = sorted(zip(variants, (float(p) for p in predictions)),
                            key=lambda t: (-t[1], t[0]))
            top = ranked[: config.k]
            for rank, (variant, predicted) in enumerate(top, start=1):
                rows.append({"seed": seed, "round": round_number, "rank": rank, "variant": variant,
                             "predicted_fitness": predicted, "actual_fitness": float(oracle[variant]),
                             "hypothesis": next(r["hypothesis"] for r in accepted if r["variant"] == variant),
                             "valid_candidates": len(accepted)})
                picked.add(variant)
            print(f"seed{seed} round{round_number}: valid={len(accepted)} "
                  f"mean={np.mean([r['actual_fitness'] for r in rows if r['seed'] == seed and r['round'] == round_number]):.4f}")
    finally:
        proposer.close()
    return rows


def main():
    config = ExperimentConfig.model_validate(read_json(ROOT / "configs" / "experiment.json"))
    usage, all_rows = [], []
    for seed in SEEDS:
        all_rows.extend(run_seed(seed, config, usage))
        pd.DataFrame(all_rows).to_csv(OUT / "recommendations.csv", index=False, encoding="utf-8-sig")

    (OUT / "usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    control = pd.DataFrame(all_rows)

    compare = []
    for seed in SEEDS:
        src = pd.read_csv(ROOT / "results" / "prompt-v3" / f"seed-{seed}" / "run" / "recommendations.csv")
        src = src[src.method == "knowledge"]
        ctl = control[control.seed == seed]
        for label, frame in (("feedback", src), ("no_feedback", ctl)):
            compare.append({"seed": seed, "condition": label,
                            "mean_30": round(float(frame.actual_fitness.mean()), 4),
                            "max_30": round(float(frame.actual_fitness.max()), 4),
                            "round_means": [round(float(g.actual_fitness.mean()), 4)
                                            for _, g in frame.groupby("round")]})
    summary = {"comparison": compare,
               "tokens_total": sum((u.get("usage") or {}).get("total_tokens") or 0 for u in usage),
               "requests": len(usage)}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
