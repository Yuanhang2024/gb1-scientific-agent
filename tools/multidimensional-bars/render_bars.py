"""Render audited GB1 quality, reliability and resource bar figures offline."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image
import xml.etree.ElementTree as ET

REPORTS = {
    "phase1": "phase1-audit-20261113-1418",
    "phase2": "phase2-audit-20261113-1425",
    "phase3": "phase3-audit-20261113-final",
}
MODELS = ["gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra", "deepseek-flash", "glm-5.3", "glm-5.3-flash"]
NAMES = {"gpt-6-luna": "GPT-6 Luna", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6.1-terra": "GPT-6.1 Terra",
         "deepseek-flash": "DeepSeek Flash", "glm-5.3": "GLM-5.3", "glm-5.3-flash": "GLM-5.3 Flash"}
COLORS = dict(zip(MODELS, ["#AD9564", "#557E9B", "#6F9587", "#7D8D9B", "#9784A3", "#B08479"]))
EFFORTS = ["off", "low", "medium", "high", "xhigh", "max"]
METRICS = ["mean_fitness", "max_fitness", "completion_percent", "elapsed_seconds_median", "tokens_median", "output_speed_median"]
SCALES = {metric: 1000 if metric == "tokens_median" else 1 for metric in METRICS}
TITLES = ["Mean fitness", "Mean new best fitness", "Completion", "Elapsed time", "Reported tokens", "Effective output speed"]
UNITS = ["Fitness (WT = 1)", "Fitness (WT = 1)", "Trajectories (%)", "Seconds / trajectory", "10³ tokens / trajectory", "Output tokens / generation second"]
VALIDATION = {"checks": {}, "figures": [], "sources": {}}


def require(condition, label):
    if not condition:
        raise ValueError(label)
    VALIDATION["checks"][label] = True


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def close(left, right, label):
    require((left is None and right is None) or (left is not None and right is not None and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)), label)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def speed_metrics(calls, complete):
    outputs = [number((call.get("usage") or {}).get("output_tokens")) for call in calls]
    durations = [number(call.get("generationSeconds")) for call in calls]
    missing_output = sum(value is None or value < 0 for value in outputs)
    missing_time = sum(value is None or value <= 0 for value in durations)
    unknown = sum(bool(call.get("outcomeUnknown")) for call in calls)
    output_sum = sum(outputs) if calls and not missing_output else None
    seconds_sum = sum(durations) if calls and not missing_time else None
    rate = output_sum / seconds_sum if complete and output_sum is not None and seconds_sum is not None and not unknown else None
    return {"output_tokens_sum": output_sum, "generation_seconds_sum": seconds_sum,
            "missing_output_calls": missing_output, "missing_generation_time_calls": missing_time,
            "unknown_outcome_calls": unknown, "output_tokens_per_second": rate}


def usable_median(values):
    return statistics.median(values) if values and all(value is not None for value in values) else None


def collect(study, phase):
    directory = study / "reports" / REPORTS[phase]
    summary, audit = load_json(directory / "summary.json"), load_json(directory / "audit.json")
    require(summary["audit_passed"] and audit["passed"] and summary["audit_errors"] == 0, f"{phase}: source audit passed")
    trajectories = read_csv(directory / "trajectories.csv")
    audit_calls = read_csv(directory / "calls.csv")
    call_index = {(call["trajectory"], int(call["call"])): call for call in audit_calls}
    state = load_json(study / phase / "state.json")
    require(state["phase"] in ("completed", "partial", "interrupted"), f"{phase}: terminal phase")
    conditions = {row["key"]: row for row in state["conditions"]}
    require(set(conditions) == {row["trajectory"] for row in trajectories}, f"{phase}: complete trajectory coverage")
    enriched = []
    for row in trajectories:
        key = row["trajectory"]
        raw = conditions[key]
        complete = row["complete"] == "True"
        require(row["audit_passed"] == "True", f"{key}: audited")
        require(raw["status"] == row["status"] and complete == (raw["status"] == "completed"), f"{key}: terminal status")
        calls = raw.get("calls", [])
        require(len(calls) == int(row["calls"]), f"{key}: all calls included")
        for call in calls:
            exported = call_index[key, call["number"]]
            close(number((call.get("usage") or {}).get("total_tokens")), number(exported["reported_tokens"]), f"{key}/{call['number']}: audited usage")
            close(number(call.get("generationSeconds")), number(exported["generation_seconds"]), f"{key}/{call['number']}: audited generation duration")
        speed = speed_metrics(calls, complete)
        if speed["generation_seconds_sum"] is not None:
            close(speed["generation_seconds_sum"], number(row["model_seconds"]), f"{key}: generation time sum")
        enriched.append({"phase": phase, "trajectory": key, "profile": row["profile"], "model": row["model"],
                         "effort": row["effort"], "seed": int(row["seed"]), "repeat_index": int(row["repeatIndex"]),
                         "status": row["status"], "complete": complete, "calls": len(calls),
                         "elapsed_seconds": number(row["elapsed_seconds"]), "total_tokens": number(row["tokens"]),
                         "mean_fitness": number(row["mean_fitness"]), "max_fitness": number(row["max_fitness"]), **speed})
    rows = []
    for profile in summary["profiles"]:
        subset = [row for row in enriched if row["profile"] == profile["profile"]]
        done = [row for row in subset if row["complete"]]
        planned, completed = profile["planned_trajectories"], profile["completed_trajectories"]
        require((len(subset), len(done)) == (planned, completed), f"{phase}/{profile['profile']}: completion counts")
        elapsed = usable_median([row["elapsed_seconds"] for row in done])
        tokens = usable_median([row["total_tokens"] for row in done])
        speed = usable_median([row["output_tokens_per_second"] for row in done])
        close(elapsed, profile["median_complete_elapsed_seconds"], f"{phase}/{profile['profile']}: elapsed median")
        close(tokens, profile["median_complete_tokens"], f"{phase}/{profile['profile']}: tokens median")
        result = {key: profile[key] for key in ("phase", "profile", "profileKey", "arm", "provider", "model", "effort", "candidatesPerRound", "topK", "rounds", "oversizePolicy")}
        result.update({"planned_n": planned, "completed_n": completed, "completion_percent": 100 * completed / planned,
                       "failed_or_incomplete_n": planned - completed, "elapsed_seconds_median": elapsed, "tokens_median": tokens,
                       "output_speed_median": speed, "cost_n": len(done),
                       "known_total_token_n": sum(row["total_tokens"] is not None for row in done),
                       "known_speed_n": sum(row["output_tokens_per_second"] is not None for row in done),
                       "all_call_reported_tokens_sum": profile["reported_tokens_sum"], "all_call_missing_usage_n": profile["missing_usage_calls"],
                       "quality_condition": "completed trajectories; equal weight per planned data split",
                       "cost_condition": "completed full trajectories; no shared preparation",
                       "speed_definition": "median over complete trajectories of sum(output_tokens)/sum(generationSeconds); all calls; includes reasoning/waiting"})
        for metric in ("mean_fitness", "max_fitness"):
            source = profile[metric]
            values = defaultdict(list)
            for row in done:
                if row[metric] is not None:
                    values[row["seed"]].append(row[metric])
            seeds = {row["seed"] for row in subset}
            expected = statistics.mean(statistics.mean(values[seed]) for seed in seeds) if all(values[seed] for seed in seeds) else None
            close(expected, source["estimate"], f"{phase}/{profile['profile']}: equal-split {metric}")
            for field in ("estimate", "lower", "upper"):
                result[metric if field == "estimate" else f"{metric}_{field}"] = source[field]
            result[f"{metric}_na_reason"] = source["reason"]
            result[f"{metric}_n_per_seed"] = json.dumps(source["usable_per_seed"], sort_keys=True)
            result[f"{metric}_ci_level"] = source["confidence_level"]
            result[f"{metric}_bootstrap_replicates"] = source["replicates"]
        rows.append(result)
    require(len(audit_calls) == sum(row["calls"] for row in enriched), f"{phase}: all audited calls reconciled")
    VALIDATION["sources"][phase] = {"report": str(directory.relative_to(study)), "terminal_state": f"{phase}/state.json", "trajectories": len(enriched), "calls": len(audit_calls)}
    return rows, enriched


def configure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .55, "axes.edgecolor": "#9DA3A9", "text.color": "#30373C",
                         "axes.labelcolor": "#30373C", "xtick.color": "#565D64", "ytick.color": "#565D64",
                         "grid.color": "#D5DADD", "grid.linewidth": .5, "axes.axisbelow": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})


def formatted(value, metric, main=False):
    if value is None:
        return "N/A"
    if metric == "completion_percent":
        return f"{value:.0f}%"
    if metric in ("mean_fitness", "max_fitness"):
        return f"{value:.3f}" if main else f"{value:.2f}"
    if metric == "tokens_median":
        return f"{value / 1000:.1f}"
    return f"{value:.1f}"


def metric_limit(rows, metric):
    values = [row.get(f"{metric}_upper", row.get(metric)) for row in rows]
    maximum = max([value / SCALES[metric] for value in values if value is not None] or [1])
    return 124 if metric == "completion_percent" else maximum * 1.30


def check_layout(fig, axes, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    fonts = set()
    glyph_cache = {}
    missing = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        font_path = font_manager.findfont(text.get_fontproperties())
        fonts.add(text.get_fontproperties().get_name())
        if font_path not in glyph_cache:
            glyph_cache[font_path] = font_manager.get_font(font_path).get_charmap()
        missing.extend(char for char in text.get_text() if not char.isspace() and ord(char) not in glyph_cache[font_path])
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
            outside.append(text.get_text())
    require(not missing, f"{name}: all label glyphs available")
    require(not outside, f"{name}: text within figure bounds {outside}")
    for panel, ax in enumerate(axes):
        for patch in ax.patches:
            baseline = patch.get_y() if name == "confirmation-six-metrics" else patch.get_x()
            require(baseline == 0, f"{name}/{panel}: bars begin at zero")
    VALIDATION["figures"].append({"name": name, "width_mm": fig.get_figwidth() * 25.4, "height_mm": fig.get_figheight() * 25.4,
                                  "fonts": sorted(fonts), "missing_glyphs": missing, "outside_text": outside,
                                  "axes": len(axes), "bars": sum(len(ax.patches) for ax in axes),
                                  "axis_limits": [{"x": list(ax.get_xlim()), "y": list(ax.get_ylim())} for ax in axes]})


def save(fig, axes, output, name):
    check_layout(fig, axes, name)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=600, facecolor="white", metadata={"Creator": "GB1 multidimensional bars"})
    with Image.open(output / f"{name}.png") as image:
        require(all(abs(value - 600) < .1 for value in image.info["dpi"]), f"{name}: PNG resolution 600 dpi")
        VALIDATION["figures"][-1]["png_pixels"] = list(image.size)
        VALIDATION["figures"][-1]["png_dpi"] = list(image.info["dpi"])
    svg = ET.parse(output / f"{name}.svg").getroot()
    require(bool(svg.findall(".//{http://www.w3.org/2000/svg}text")), f"{name}: SVG retains editable text")
    with (output / f"{name}.pdf").open("rb") as handle:
        require(handle.read(5) == b"%PDF-", f"{name}: PDF output header")
    plt.close(fig)


def confirmation(rows, output):
    rows = sorted(rows, key=lambda row: 0 if row["arm"] == "selected" else 1)
    fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 139 / 25.4))
    fig.subplots_adjust(left=.085, right=.985, bottom=.225, top=.89, wspace=.58, hspace=.78)
    fig.suptitle("独立确认：每轨迹查询 30 条变体", x=.08, y=.975, ha="left", fontsize=10, fontweight="normal")
    labels = ["Sol-xhigh", "Luna-high"]
    titles = ["平均 fitness", "每轨迹最高 fitness 均值", "完成率", "轨迹耗时", "已报告 tokens", "有效输出速度"]
    units = ["fitness（野生型 = 1）", "fitness（野生型 = 1）", "完成轨迹（%）", "秒 / 完整轨迹", "千 tokens / 完整轨迹", "输出 tokens / 完整生成秒"]
    for index, (ax, metric, title, unit) in enumerate(zip(axes.flat, METRICS, titles, units)):
        ax.set_title(f"{chr(97 + index)}  {title}", loc="left", pad=8)
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], labels, fontsize=7)
        ax.set_xlim(-.6, 1.6)
        ax.set_ylim(0, metric_limit(rows, metric))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric == "completion_percent":
            ax.set_yticks([0, 50, 100])
        ax.grid(axis="y")
        for x, row in enumerate(rows):
            value = row[metric]
            if value is None:
                ax.text(x, .12, "N/A", transform=ax.get_xaxis_transform(), ha="center")
                continue
            shown = value / SCALES[metric]
            ax.bar(x, shown, width=.53, color=COLORS[row["model"]], linewidth=0)
            top = shown
            if metric in ("mean_fitness", "max_fitness"):
                low, high = row[f"{metric}_lower"], row[f"{metric}_upper"]
                ax.errorbar(x, shown, yerr=[[shown - low], [high - shown]], fmt="none", ecolor="#343B42", capsize=2.5, elinewidth=.8, capthick=.8)
                top = high
            count = f"{row['completed_n']}/{row['planned_n']}" if metric == "completion_percent" else f"n={row['known_speed_n'] if metric == 'output_speed_median' else row['completed_n']}"
            label = count if metric == "completion_percent" else f"{formatted(value, metric, True)}\n{count}"
            ax.annotate(label, (x, top), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.8, linespacing=1.15)
    fig.text(.085, .155, "Sol-xhigh：GPT-6.1 Sol / first_n；Luna-high：GPT-6 Luna / reject", fontsize=7)
    fig.text(.085, .115, "质量：各数据划分等权均值；误差线为 95% 分层 bootstrap 区间（5 划分 × 3 轨迹）。", fontsize=6.6)
    missing_calls = sum(row["all_call_missing_usage_n"] for row in rows)
    missing_complete = sum(row["completed_n"] - row["known_total_token_n"] for row in rows)
    fig.text(.085, .080, f"成本与速度：完整轨迹的中位数；耗时不含共享准备。用量缺失：{missing_calls} 次调用、{missing_complete} 条完整轨迹。", fontsize=6.6)
    fig.text(.085, .045, "速度：每轨迹输出 tokens 总量 / 完整生成总耗时，再取组中位数；包含思考和等待。", fontsize=6.6)
    save(fig, list(axes.flat), output, "confirmation-six-metrics")


def horizontal(rows, output, phase):
    full = phase == "phase1"
    metrics = METRICS if full else ["mean_fitness", "completion_percent", "elapsed_seconds_median", "tokens_median"]
    rows = sorted(rows, key=lambda row: (MODELS.index(row["model"]), EFFORTS.index(row["effort"]), row["candidatesPerRound"], row["oversizePolicy"] == "first_n"))
    n = len(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(14.4 if full else 12.2, 10.7 if full else 7.15), sharey=True)
    fig.subplots_adjust(left=.22 if full else .29, right=.985, bottom=.205 if full else .24, top=.91 if full else .86, wspace=.30)
    labels = []
    for row in rows:
        name = NAMES[row["model"]] if full else ("Sol" if row["model"] == "gpt-6.1-sol" else "Terra")
        config = "" if full else f"  C{row['candidatesPerRound']}/k{row['topK']}/r{row['rounds']} {row['oversizePolicy']}"
        labels.append(f"{name} · {row['effort']}{config}  [{row['completed_n']}/{row['planned_n']}]")
    boundaries = [index - .5 for index in range(1, n) if rows[index]["model"] != rows[index - 1]["model"]]
    for panel, (ax, metric) in enumerate(zip(axes, metrics)):
        index = METRICS.index(metric)
        unit = ["Fitness (WT = 1)", "Fitness (WT = 1)", "Trajectories (%)", "Seconds", "10³ total tokens", "Output tokens/s"][index]
        ax.set_title(f"{chr(97 + panel)}  {TITLES[index]}\n{'Mean + 95% CI' if metric in ('mean_fitness', 'max_fitness') else 'Median' if metric != 'completion_percent' else 'Completed / planned'}", loc="left", fontsize=8.5, pad=12)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_xlim(0, metric_limit(rows, metric))
        ax.set_ylim(n - .45, -.7)
        ax.set_yticks(range(n), labels, fontsize=7.8)
        ax.tick_params(axis="y", length=0, labelleft=panel == 0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="upper"))
        if metric == "completion_percent":
            ax.set_xticks([0, 50, 100])
        ax.grid(axis="x")
        for boundary in boundaries:
            ax.axhline(boundary, color="#B9C0C5", linewidth=.6)
        for y, row in enumerate(rows):
            value = row[metric]
            if value is None:
                reason = row.get(f"{metric}_na_reason")
                ax.text(.04, y, "N/A*" if reason == "missing_data_split" and row["completed_n"] else "N/A", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color="#616970")
                continue
            shown = value / SCALES[metric]
            ax.barh(y, shown, height=.58, color=COLORS[row["model"]], linewidth=0)
            top = shown
            if metric in ("mean_fitness", "max_fitness"):
                low, high = row[f"{metric}_lower"], row[f"{metric}_upper"]
                ax.errorbar(shown, y, xerr=[[shown - low], [high - shown]], fmt="none", ecolor="#343B42", capsize=1.8, elinewidth=.65, capthick=.65)
                top = high
            label = f"{row['completed_n']}/{row['planned_n']}" if metric == "completion_percent" else formatted(value, metric)
            ax.annotate(label, (top, y), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5)
    title = "All 26 model / reasoning combinations" if full else "Candidate-budget configurations at a fixed 30-assay budget"
    fig.suptitle(title, x=.035, y=.975, ha="left", fontsize=12, fontweight="normal")
    if full:
        fig.text(.035, .14, "Phase 1: 3 data splits × 3 independent trajectories per combination. Labels show completed/planned trajectories.", fontsize=8)
        fig.text(.035, .112, "Quality is conditional on completion; means give each split equal weight. Whiskers: audited 95% hierarchical bootstrap CI.", fontsize=8)
        fig.text(.035, .084, "N/A: no completed trajectory. N/A*: at least one planned split has no completed trajectory; quality cannot be estimated.", fontsize=8)
        incomplete = sum(row["failed_or_incomplete_n"] for row in rows)
        missing_calls = sum(row["all_call_missing_usage_n"] for row in rows)
        missing_complete = sum(row["completed_n"] - row["known_total_token_n"] for row in rows)
        fig.text(.035, .056, f"Incomplete trajectories: {incomplete}. Missing usage: {missing_calls} calls ({missing_complete} completed trajectories affected). Costs use completed trajectories only.", fontsize=8)
        fig.text(.035, .028, "Speed = Σ output tokens / Σ full generation seconds per trajectory, then median; includes reasoning/waiting. Shared preparation excluded.", fontsize=7.7)
    else:
        fig.text(.035, .153, "Phase 2: 3 data splits × 2 independent trajectories per configuration. C = candidates/round; k = assays/round; r = rounds.", fontsize=8)
        fig.text(.035, .113, "Sol: GPT-6.1 Sol / xhigh; Terra: GPT-6.1 Terra / max. All configurations query 30 variants. Labels: completed/planned.", fontsize=8)
        fig.text(.035, .073, "Quality: equal-split mean with audited 95% hierarchical bootstrap CI. Cost: median of complete trajectories, excluding shared preparation.", fontsize=7.8)
        fig.text(.035, .033, "reject: reject excess proposals; first_n: validate only the first C proposals in original response order.", fontsize=8)
    save(fig, list(axes), output, "all26-six-metrics" if full else "budget12-four-metrics")


def main():
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--study", type=Path, default=project / "results/default-calibration-20261113")
    parser.add_argument("--output", type=Path, default=project / "results/default-calibration-20261113/bar_figures")
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(output.is_relative_to((project / "results/default-calibration-20261113/bar_figures").resolve()), "output stays within designated directory")
    synthetic = [{"usage": {"output_tokens": 100}, "generationSeconds": 10}, {"usage": {"output_tokens": 600}, "generationSeconds": 30}]
    close(speed_metrics(synthetic, True)["output_tokens_per_second"], 17.5, "speed uses ratio of sums, not mean of call rates")
    require(speed_metrics(synthetic, False)["output_tokens_per_second"] is None, "incomplete trajectory speed is missing")
    require(speed_metrics([{"usage": {}, "generationSeconds": 10}], True)["output_tokens_per_second"] is None, "missing output usage stays missing")
    require(usable_median([10, None]) is None, "missing full-trajectory usage cannot become zero")
    configure_style()
    groups, trajectories = {}, []
    for phase in REPORTS:
        groups[phase], extra = collect(options.study.resolve(), phase)
        trajectories.extend(extra)
        write_csv(output / f"{phase}-bar-metrics.csv", groups[phase])
    require([len(groups[phase]) for phase in REPORTS] == [26, 12, 2], "expected profile counts 26/12/2")
    write_csv(output / "trajectory-speed.csv", trajectories)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with plt.rc_context({"font.family": "Microsoft YaHei"}):
            confirmation(groups["phase3"], output)
        horizontal(groups["phase1"], output, "phase1")
        horizontal(groups["phase2"], output, "phase2")
    require(not caught, f"plot rendering has no warnings {[str(item.message) for item in caught]}")
    VALIDATION["statistics"] = {"quality": "audited equal-split means and 95% hierarchical bootstrap intervals; complete trajectories; missing split remains N/A",
                                "time": "median complete-trajectory elapsed seconds; shared preparation excluded",
                                "tokens": "median reported total tokens of complete trajectories; all generation attempts included",
                                "speed": "sum of reported output_tokens divided by sum of generationSeconds within each complete trajectory, then group median; reasoning/waiting included; not decoding-only throughput",
                                "completion": "completed trajectories / all planned trajectories; failures remain in denominator"}
    (output / "validation.json").write_text(json.dumps(VALIDATION, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "checks": len(VALIDATION["checks"]), "figures": [row["name"] for row in VALIDATION["figures"]], "trajectories": len(trajectories)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
