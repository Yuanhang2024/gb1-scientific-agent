# 零上下文记忆检验：不给任何实验数据，直接要求模型排序GB1变体或复述最优变体。
# 共4次真实请求：3个排序批次（每批50个随机变体）+ 1次自由回忆全局最优10个。
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "zero_context"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stats import load_global_fitness  # noqa: E402  同目录统计脚本


def load_client():
    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    from openai import OpenAI
    model = os.environ["OPENAI_MODEL"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("OPENAI_BASE_URL") or None,
                    timeout=180.0, max_retries=0)
    return client, model


RANK_SYSTEM = "You are an expert in protein engineering and deep mutational scanning data."
RANK_TASK = ("Rank the following GB1 domain four-site variants (mutation positions 39, 40, 41 and 54 "
             "relative to wild type VDGV) from highest to lowest measured IgG-Fc binding fitness "
             "(wild type = 1, higher is better). Return all variants exactly once, best first. "
             "No experimental data is provided; use your own knowledge.")
RECALL_TASK = ("List the ten four-letter GB1 variants (mutation positions 39, 40, 41 and 54, wild type "
               "VDGV) with the highest measured IgG-Fc binding fitness in the published deep mutational "
               "scanning dataset, best first.")


def rank_schema(n):
    return {"type": "json_schema", "json_schema": {
        "name": "variant_ranking", "strict": True, "schema": {
            "type": "object",
            "properties": {"ranked_variants": {"type": "array", "items": {"type": "string"},
                                               "minItems": n, "maxItems": n}},
            "required": ["ranked_variants"], "additionalProperties": False}}}


RECALL_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "variant_recall", "strict": True, "schema": {
        "type": "object",
        "properties": {"top_variants": {"type": "array", "items": {"type": "string"},
                                        "minItems": 10, "maxItems": 10}},
        "required": ["top_variants"], "additionalProperties": False}}}


def call(client, model, system, user_text, response_format, usage_log, tag):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_text}],
        response_format=response_format, max_completion_tokens=8000)
    text = response.choices[0].message.content or ""
    usage_log.append({"tag": tag, "model_returned": getattr(response, "model", None),
                      "finish_reason": response.choices[0].finish_reason,
                      "usage": response.usage.model_dump(mode="json") if response.usage else None})
    (OUT / f"{tag}.json").write_text(text, encoding="utf-8")
    return text


def main():
    rng = np.random.default_rng(20261113)
    global_fit = load_global_fitness().reset_index(drop=True)
    gf = global_fit.set_index("variant")["fitness"]

    client, model = load_client()
    usage = []
    batch_reports = []
    for b in range(3):
        sample = global_fit.sample(n=50, random_state=int(rng.integers(1 << 31))).variant.tolist()
        user_text = json.dumps({"task": RANK_TASK, "variants": sample}, ensure_ascii=False)
        text = call(client, model, RANK_SYSTEM, user_text, rank_schema(50), usage, f"rank_batch{b + 1}")
        ranked = json.loads(text)["ranked_variants"]
        valid = [v for v in ranked if v in gf.index]
        if valid:
            actual = np.array([gf[v] for v in valid])
            model_rank = np.arange(len(valid))
            rho = float(spearmanr(model_rank, actual).statistic)
            tau = float(kendalltau(model_rank, actual).statistic)
            true_best10 = set(pd.Series(actual).nlargest(10).index)
            pred_best10 = set(range(10))
            hit10 = len(pred_best10 & true_best10)
            batch_reports.append({"batch": b + 1, "returned": len(ranked), "valid": len(valid),
                                  "spearman_model_rank_vs_actual": round(rho, 4),
                                  "kendall_tau": round(tau, 4),
                                  "top10_in_batch_hits": hit10,
                                  "batch_actual_top": [valid[i] for i in true_best10][:3]})
        print(json.dumps(batch_reports[-1], ensure_ascii=False))

    top100 = set(global_fit.nlargest(100, "fitness").variant)
    top10 = set(global_fit.nlargest(10, "fitness").variant)
    text = call(client, model, RANK_SYSTEM, json.dumps({"task": RECALL_TASK}), RECALL_FORMAT, usage, "free_recall")
    recalled = json.loads(text)["top_variants"]
    recall_report = {"recalled": recalled,
                     "in_global_top10": [v for v in recalled if v in top10],
                     "in_global_top100": [v for v in recalled if v in top100],
                     "recall_fitness": {v: (float(gf[v]) if v in gf.index else None) for v in recalled}}
    print(json.dumps(recall_report, ensure_ascii=False, indent=2))

    (OUT / "usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"model": model, "batches": batch_reports, "free_recall": recall_report}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    total_tokens = sum((u.get("usage") or {}).get("total_tokens") or 0 for u in usage)
    print(f"TOKENS={total_tokens}")


if __name__ == "__main__":
    main()
