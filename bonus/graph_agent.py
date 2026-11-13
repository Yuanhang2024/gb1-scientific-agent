# 图谱Agent对照：用相同数据和预测器，比较静态规则与结构加图谱知识。
"""Paired static-JSON versus structure-and-graph knowledge experiment."""
from copy import deepcopy
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gb1.agent import OpenAIProposer, build_request_payload, collect_proposals, load_knowledge, make_context, usage_counts
from gb1.data import HiddenOracle, PublicDataset, describe_variant, one_hot
from gb1.prompts import load_prompt_profile, repair_context
from gb1.schema import ExperimentConfig, KnowledgeBase
from gb1.storage import event, read_json, single_writer, write_json
from .knowledge_graph import build_graph, draw_subgraph, graph_knowledge, save_graph
from .structure import analyze_structure, write_structure_tables

CONDITIONS = ("static_json", "structure_graph")
SEEDS = (44, 55, 66)
API_IDENTITY = {"model_requested": "gpt-6-luna", "model_returned": "gpt-6-luna", "api_mode": "chat"}


def _directory(output, seed, condition):
    return Path(output) / f"seed-{seed}" / condition


# 两条件共享已知数据；structure_graph额外加入结构与图查询知识，评价的是组合增强。
def _context(snapshot, condition):
    config = ExperimentConfig.model_validate(snapshot["config"])
    rules = deepcopy(snapshot["static_knowledge"]["rules"])
    if condition == "structure_graph":
        rules.extend(deepcopy(snapshot["graph_knowledge"]["rules"]))
    elif condition != "static_json":
        raise ValueError("Unknown knowledge condition")
    context = make_context(pd.DataFrame(snapshot["training_records"]), [], config, snapshot["reference"], 1,
                           KnowledgeBase(rules=rules), prompt_profile="v3")
    if condition == "structure_graph":
        context["graph_context"] = deepcopy(snapshot["graph_queries"])
    return context


# 只读取公开训练数据和无标签候选，冻结初始RF、查询结果和输入供两条件共用。
def prepare_graph_agent(data_dirs, model_files, pdb_path, knowledge_path, output):
    """Prepare from public training/candidate data and project-generated initial RF files."""
    if len(data_dirs) != len(model_files) or not 1 <= len(data_dirs) <= 3:
        raise ValueError("Provide one to three paired public-data directories and initial RF model files")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Graph experiment output must be empty")
    knowledge = load_knowledge(knowledge_path)
    snapshots, graphs, models, structure = {}, {}, {}, None
    for data_dir, model_file in zip(data_dirs, model_files):
        dataset = PublicDataset(data_dir)
        seed = dataset.manifest["config"]["seed"]
        if seed not in SEEDS or str(seed) in snapshots:
            raise ValueError("Graph experiment seeds must be unique members of 44,55,66")
        if len(dataset.candidates) < 40 or len(dataset.train) < 4:
            raise ValueError("Need at least four known rows and forty unlabeled candidates")
        config = ExperimentConfig(seed=seed, rounds=1, k=10, proposal_count=40, score_budget=40,
                                  llm_max_calls_per_round=2, llm_max_output_tokens=8000, llm_timeout=180)
        if structure is None:
            structure = analyze_structure(pdb_path, dataset.reference)
        elif structure["metadata"]["experiment_reference"] != dataset.reference:
            raise ValueError("All paired seeds must share the experimental reference")
        model = joblib.load(model_file)
        if not isinstance(model, RandomForestRegressor) or getattr(model, "n_features_in_", None) != 80:
            raise ValueError("Expected a fitted project RandomForestRegressor with 80 one-hot inputs")
        if not np.isfinite(model.predict(one_hot(dataset.train.variant.head(4)))).all():
            raise ValueError("Initial model returned nonfinite predictions")
        graph = build_graph(dataset.train, dataset.reference, structure, f"seed-{seed}-initial-train")
        graph_rules, queries = graph_knowledge(graph, dataset.train)
        all_ids = [rule.id for rule in knowledge.rules + graph_rules.rules]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Static and graph knowledge rule IDs must be unique")
        snapshots[str(seed)] = {"seed": seed, "config": config.model_dump(), "reference": dataset.reference,
                                "dataset_manifest": deepcopy(dataset.manifest), "data_directory": str(Path(data_dir).resolve()),
                                "training_records": dataset.train.to_dict(orient="records"),
                                "eligible_variants": sorted(dataset.candidates.variant),
                                "static_knowledge": knowledge.model_dump(), "graph_knowledge": graph_rules.model_dump(),
                                "graph_queries": queries, "prompt_metadata": load_prompt_profile("v3").metadata(),
                                "api_identity": deepcopy(API_IDENTITY),
                                "model_file": f"seed-{seed}/initial_model.joblib", "source_model_file": str(Path(model_file).resolve()),
                                "known_best_fitness": float(dataset.train.fitness.max())}
        graphs[seed], models[seed] = graph, model
    output.mkdir(parents=True, exist_ok=True)
    write_structure_tables(structure, output / "structure")
    order = []
    for key in sorted(snapshots, key=int):
        seed = int(key)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir()
        joblib.dump(models[seed], output / snapshots[key]["model_file"])
        save_graph(graphs[seed], seed_dir / "knowledge_graph.json")
        write_json(seed_dir / "graph_queries.json", snapshots[key]["graph_queries"])
        draw_subgraph(graphs[seed], seed_dir / "knowledge_subgraph.png",
                      variant=pd.DataFrame(snapshots[key]["training_records"]).sort_values(["fitness", "variant"], ascending=[False, True]).iloc[0].variant)
        conditions = np.random.default_rng(np.random.SeedSequence([seed, 3200])).permutation(CONDITIONS)
        order.extend({"seed": seed, "condition": str(condition)} for condition in conditions)
    manifest = {"format_version": 1, "study": "static-json-versus-structure-graph-knowledge", "snapshots": snapshots,
                "order": order, "per_condition_max_requests": 2, "max_requests": 4 * len(snapshots),
                "comparison_scope": "Combined reference-structure and graph-retrieved knowledge augmentation",
                "order_rule": "Per-seed NumPy SeedSequence([seed,3200]) permutation, without fitness inputs"}
    write_json(output / "manifest.json", manifest)
    for item in order:
        snapshot = snapshots[str(item["seed"])]
        write_json(_directory(output, **item) / "state.json",
                   {"seed": item["seed"], "condition": item["condition"], "stage": "prepared", "attempts": [],
                    "initial_context": _context(snapshot, item["condition"]), "errors": []})
    return manifest


def _manifest(output):
    manifest = read_json(Path(output) / "manifest.json")
    seeds = [snapshot["seed"] for snapshot in manifest["snapshots"].values()]
    expected = {(seed, condition) for seed in seeds for condition in CONDITIONS}
    order = [(row["seed"], row["condition"]) for row in manifest["order"]]
    if (not 1 <= len(seeds) <= 3 or len(seeds) != len(set(seeds)) or not set(seeds) <= set(SEEDS)
            or manifest["max_requests"] != 4 * len(seeds) or manifest["max_requests"] > 12
            or manifest["per_condition_max_requests"] != 2 or len(order) != len(expected) or set(order) != expected):
        raise ValueError("Invalid experiment seed, order or request budget")
    for key, snapshot in manifest["snapshots"].items():
        config = ExperimentConfig.model_validate(snapshot["config"])
        if (str(snapshot["seed"]) != key or config.seed != snapshot["seed"] or config.rounds != 1
                or config.k != 10 or config.score_budget != 40 or config.proposal_count != 40
                or config.llm_max_calls_per_round != 2 or snapshot["prompt_metadata"] != load_prompt_profile("v3").metadata()
                or snapshot["api_identity"] != API_IDENTITY):
            raise ValueError("Prepared snapshot differs from the fixed experimental design")
    return manifest


def _validate_response(response, snapshot):
    for key, value in snapshot["api_identity"].items():
        if response.get(key) != value:
            raise ValueError("Response model/API identity differs from the prepared experiment")
    if response.get("prompt_metadata") != snapshot["prompt_metadata"]:
        raise ValueError("Response prompt differs from the fixed v3 prompt")


# 直接复用核心请求构造函数，让保存的参数与实际SDK调用保持一致。
def request_payload(context, snapshot):
    """Use the same argument builder as the actual SDK call."""
    config = ExperimentConfig.model_validate(snapshot["config"])
    return build_request_payload(context, config, snapshot["api_identity"]["model_requested"],
                                 snapshot["api_identity"]["api_mode"], snapshot["prompt_metadata"]["system_prompt"])


# 重新解析保存的回答并校验规则引用，保证有效候选均能对应实际响应。
def _accepted(state, snapshot):
    config = ExperimentConfig.model_validate(snapshot["config"])
    eligible = set(snapshot["eligible_variants"])
    rule_ids = {row["id"] for row in state["initial_context"]["knowledge"]["rules"]}
    accepted, rejected, analyses = [], [], []
    for attempt in state["attempts"]:
        if "response" not in attempt:
            continue
        response = attempt["response"]
        _validate_response(response, snapshot)
        try:
            if not response.get("complete"):
                raise ValueError("Response was refused or incomplete")
            forbidden = {row["variant"] for row in rejected if isinstance(row.get("variant"), str)}
            analysis, valid, invalid = collect_proposals(response["text"], eligible - forbidden, config, accepted,
                                                        rule_ids, attempt["context"]["proposal_count"])
            accepted.extend(valid)
            rejected.extend(invalid)
            analyses.append(analysis)
            attempt["validation"] = {"valid_count": len(valid), "rejections": deepcopy(invalid)}
        except (ValueError, TypeError) as exc:
            rejected.append({"variant": None, "reason": str(exc)})
            attempt["validation"] = {"valid_count": 0, "rejections": [{"variant": None, "reason": str(exc)}]}
    return accepted, rejected, analyses


def _attempt_context(state, snapshot, accepted, rejected):
    context = deepcopy(state["initial_context"])
    if state["attempts"]:
        context = repair_context(context, pd.DataFrame(snapshot["training_records"]), set(snapshot["eligible_variants"]),
                                 accepted, rejected, ExperimentConfig.model_validate(snapshot["config"]), 1, 2)
    return context


def _validate_state(state, snapshot, condition):
    if state["seed"] != snapshot["seed"] or state["condition"] != condition or state["initial_context"] != _context(snapshot, condition):
        raise ValueError("Saved condition differs from the prepared public snapshot")
    if len(state["attempts"]) > 2:
        raise ValueError("Condition exceeds two requests")
    for index, attempt in enumerate(state["attempts"]):
        prefix = deepcopy(state)
        prefix["attempts"] = prefix["attempts"][:index]
        accepted, rejected, _ = _accepted(prefix, snapshot)
        expected = _attempt_context(prefix, snapshot, accepted, rejected)
        if (attempt["number"] != index + 1 or attempt["context"] != expected
                or attempt["request_payload"] != request_payload(expected, snapshot)):
            raise ValueError("Saved full request differs from its prescribed initial/repair input")


# 核对已保存分数与固定RF是否一致，再按保存分数检查Top-10，允许重算的浮点微差。
def _validate_selection(state, snapshot, model):
    """Check persisted scored candidates and deterministic RF Top-10 before feedback."""
    accepted, _, _ = _accepted(state, snapshot)
    if state.get("proposals") != accepted or not 10 <= len(accepted) <= 40:
        raise ValueError("Saved proposals differ from the validated actual responses")
    predictions = model.predict(one_hot([row["variant"] for row in accepted]))
    if not np.isfinite(predictions).all():
        raise ValueError("Initial RF returned nonfinite predictions")
    saved_scores = state.get("scores", [])
    if (len(saved_scores) != len(accepted)
            or any(set(row) != {"variant", "predicted_fitness"} for row in saved_scores)
            or [row["variant"] for row in saved_scores] != [row["variant"] for row in accepted]):
        raise ValueError("Saved score IDs or structure differ from the validated proposals")
    saved_values = np.asarray([row["predicted_fitness"] for row in saved_scores], dtype=float)
    if not np.isfinite(saved_values).all() or not np.allclose(saved_values, predictions, rtol=1e-10, atol=1e-12):
        raise ValueError("Saved scores differ from the shared initial RF")
    lookup = {row["variant"]: row["predicted_fitness"] for row in saved_scores}
    ranked = sorted(accepted, key=lambda row: (-lookup[row["variant"]], row["variant"]))[:10]
    expected = [{**describe_variant(row["variant"], snapshot["reference"]), **row,
                 "rank": rank, "predicted_fitness": lookup[row["variant"]]}
                for rank, row in enumerate(ranked, 1)]
    actual = [{key: value for key, value in row.items() if key != "actual_fitness"}
              for row in state.get("recommendations", [])]
    if actual != expected:
        raise ValueError("Saved recommendations differ from the shared initial RF Top-10")
    if state["stage"] == "evaluated":
        values = [row.get("actual_fitness") for row in state["recommendations"]]
        if any(value is None for value in values) or not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Evaluated recommendations require finite nonnegative fitness")
        metrics = {"mean_fitness": float(np.mean(values)), "max_fitness": float(np.max(values)),
                   "median_fitness": float(np.median(values)), "known_best_fitness": snapshot["known_best_fitness"]}
        if state.get("metrics") != metrics:
            raise ValueError("Saved metrics differ from the observed recommendations")


def _tables(state, directory):
    for key in ("proposals", "scores", "recommendations"):
        pd.DataFrame(state.get(key, [])).to_csv(directory / f"{key}.csv", index=False)
    rows = []
    for attempt in state["attempts"]:
        response = attempt.get("response", {})
        usage = response.get("usage") or {}
        rows.append({"attempt": attempt["number"], "status": attempt["status"],
                     "valid_count": attempt.get("validation", {}).get("valid_count", 0),
                     "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                     "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                     "total_tokens": usage.get("total_tokens"), "usage": json.dumps(usage),
                     "model_requested": response.get("model_requested"), "model_returned": response.get("model_returned"),
                     "api_mode": response.get("api_mode"), "request_id": response.get("request_id")})
    pd.DataFrame(rows).to_csv(directory / "usage.csv", index=False)


# 先记录请求，再发送；未知请求结果不自动重发。选定Top-10后才查询实验反馈。
def run_graph_agent(output, proposer):
    """Use at most two actual requests per condition, then persist selection before feedback."""
    output = Path(output)
    with single_writer(output):
        manifest = _manifest(output)
        states = {(row["seed"], row["condition"]): read_json(_directory(output, **row) / "state.json") for row in manifest["order"]}
        for (seed, condition), state in states.items():
            snapshot = manifest["snapshots"][str(seed)]
            _validate_state(state, snapshot, condition)
            if (getattr(proposer, "prompt_metadata", None) != snapshot["prompt_metadata"]
                    or getattr(proposer, "model", None) != snapshot["api_identity"]["model_requested"]
                    or getattr(proposer, "api_mode", None) != snapshot["api_identity"]["api_mode"]):
                raise ValueError("Proposer differs from the fixed model/API/v3 prompt")
        if sum(len(state["attempts"]) for state in states.values()) > manifest["max_requests"]:
            raise ValueError("Global request budget exceeded")
        for item in manifest["order"]:
            state = states[(item["seed"], item["condition"])]
            snapshot = manifest["snapshots"][str(item["seed"])]
            directory = _directory(output, **item)
            save = lambda: write_json(directory / "state.json", state)
            if state["stage"] == "evaluated":
                continue
            if state["stage"] == "failed" or any(row["status"] in ("started", "failed") for row in state["attempts"]):
                raise RuntimeError("An earlier request failed or has an unknown outcome; automatic resend is disabled")
            accepted, rejected, analyses = _accepted(state, snapshot)
            config = ExperimentConfig.model_validate(snapshot["config"])
            if state["stage"] != "selected":
                while len(accepted) < config.proposal_count and len(state["attempts"]) < 2:
                    if sum(len(value["attempts"]) for value in states.values()) >= manifest["max_requests"]:
                        raise RuntimeError("Global request budget exhausted")
                    context = _attempt_context(state, snapshot, accepted, rejected)
                    attempt = {"number": len(state["attempts"]) + 1, "status": "started", "context": deepcopy(context),
                               "request_payload": request_payload(context, snapshot)}
                    state["attempts"].append(attempt)
                    state["stage"] = "generating"
                    save()
                    event(directory / "events.jsonl", "request_started", attempt=deepcopy(attempt))
                    try:
                        response = proposer.generate(deepcopy(context))
                    except Exception as exc:
                        attempt["error"] = {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}
                        attempt["status"] = "failed" if attempt["error"]["status_code"] is not None else "started"
                        state["errors"].append(deepcopy(attempt["error"]))
                        save()
                        raise RuntimeError("Request failed or has an unknown outcome; inspect the saved state before further action") from None
                    attempt.update(status="received", response=deepcopy(response))
                    save()
                    event(directory / "events.jsonl", "response_received", attempt=attempt["number"], response=deepcopy(response))
                    accepted, rejected, analyses = _accepted(state, snapshot)
                    save()
                state.update(proposals=accepted, proposal_rejections=rejected, analyses=analyses)
                if len(accepted) < config.k:
                    state["stage"] = "failed"
                    state["errors"].append({"type": "InsufficientProposals", "valid_count": len(accepted)})
                    save()
                    _tables(state, directory)
                    continue
                if len(accepted) > config.score_budget:
                    raise ValueError("Scoring budget exceeded")
                model = joblib.load(output / snapshot["model_file"])
                predicted = model.predict(one_hot([row["variant"] for row in accepted]))
                if not np.isfinite(predicted).all():
                    raise ValueError("Initial RF returned nonfinite predictions")
                lookup = {row["variant"]: float(value) for row, value in zip(accepted, predicted)}
                state["scores"] = [{"variant": variant, "predicted_fitness": value} for variant, value in lookup.items()]
                ranked = sorted(accepted, key=lambda row: (-lookup[row["variant"]], row["variant"]))[:config.k]
                state["recommendations"] = [{**describe_variant(row["variant"], snapshot["reference"]), **row,
                                              "rank": rank, "predicted_fitness": lookup[row["variant"]]}
                                             for rank, row in enumerate(ranked, 1)]
                state["stage"] = "selected"
                save()
                event(directory / "events.jsonl", "selected_before_oracle", recommendations=deepcopy(state["recommendations"]))
            if state["stage"] == "selected":
                _validate_selection(state, snapshot, joblib.load(output / snapshot["model_file"]))
                selected = [row["variant"] for row in state["recommendations"]]
                if (len(selected) != config.k or len(set(selected)) != config.k or not set(selected) <= set(snapshot["eligible_variants"])
                        or not config.k <= len(state["scores"]) <= config.score_budget):
                    raise ValueError("Saved selection violates eligibility or budgets")
                feedback = HiddenOracle(snapshot["data_directory"], selected).evaluate(selected)
                actual = {row["variant"]: row["actual_fitness"] for row in feedback}
                for row in state["recommendations"]:
                    row["actual_fitness"] = actual[row["variant"]]
                values = list(actual.values())
                state["metrics"] = {"mean_fitness": float(np.mean(values)), "max_fitness": float(np.max(values)),
                                    "median_fitness": float(np.median(values)), "known_best_fitness": snapshot["known_best_fitness"]}
                state["stage"] = "evaluated"
                save()
                event(directory / "events.jsonl", "condition_evaluated", metrics=state["metrics"])
            _tables(state, directory)
        return compare_graph_agent(output, require_complete=False)


# 只对完整的同seed条件报告差异，同时保留候选数量、请求数和用量。
def compare_graph_agent(output, require_complete=True):
    output = Path(output)
    manifest = _manifest(output)
    arms, pairs, incomplete, identities = [], [], [], set()
    for key, snapshot in sorted(manifest["snapshots"].items(), key=lambda row: int(row[0])):
        states = {condition: read_json(_directory(output, snapshot["seed"], condition) / "state.json") for condition in CONDITIONS}
        summaries = {}
        for condition, state in states.items():
            _validate_state(state, snapshot, condition)
            if state["stage"] in ("selected", "evaluated"):
                _validate_selection(state, snapshot, joblib.load(output / snapshot["model_file"]))
            totals, missing = 0, 0
            for attempt in state["attempts"]:
                response = attempt.get("response")
                if response:
                    _validate_response(response, snapshot)
                    identities.add((response.get("model_requested"), response.get("model_returned"), response.get("api_mode")))
                counts = usage_counts((response or {}).get("usage"))
                totals += counts["total_tokens"]
                missing += counts["missing_usage_count"]
            row = {"seed": snapshot["seed"], "condition": condition, "stage": state["stage"],
                   "requests": len(state["attempts"]), "total_tokens": totals, "missing_usage_requests": missing,
                   "valid_count": len(state.get("proposals", [])), "scored_count": len(state.get("scores", [])),
                   "assay_count": len(state.get("recommendations", [])), **state.get("metrics", {})}
            if state["stage"] != "evaluated":
                incomplete.append({"seed": snapshot["seed"], "condition": condition, "stage": state["stage"]})
            elif row["assay_count"] != 10 or not 10 <= row["scored_count"] <= 40:
                raise ValueError("Completed condition violates scoring or assay budgets")
            arms.append(row)
            summaries[condition] = row
        if all(state["stage"] == "evaluated" for state in states.values()):
            pair = {"seed": int(key)}
            for metric in ("mean_fitness", "max_fitness", "median_fitness", "valid_count", "requests", "total_tokens"):
                a, b = summaries["static_json"][metric], summaries["structure_graph"][metric]
                pair.update({f"static_json_{metric}": a, f"structure_graph_{metric}": b, f"graph_minus_static_{metric}": b - a})
            a, b = ({row["variant"] for row in states[condition]["recommendations"]} for condition in CONDITIONS)
            pair.update(recommendation_overlap=len(a & b), recommendation_jaccard=len(a & b) / len(a | b))
            pairs.append(pair)
    if len(identities) > 1:
        raise ValueError("Paired conditions have inconsistent returned model/API identities")
    total = sum(row["requests"] for row in arms)
    if total > manifest["max_requests"]:
        raise ValueError("Global request budget exceeded")
    result = {"complete": not incomplete, "total_requests": total, "max_requests": manifest["max_requests"],
              "comparison_scope": manifest["comparison_scope"], "conditions": arms, "paired_results": pairs,
              "incomplete": incomplete, "paired_design": {"same_training": True, "same_initial_rf": True,
                                                          "same_v3_prompt": True, "rounds": 1, "score_budget": 40, "top_k": 10}}
    write_json(output / "summary.json", result)
    pd.DataFrame(arms).to_csv(output / "condition_results.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "paired_results.csv", index=False)
    if require_complete and incomplete:
        raise RuntimeError("Some graph experiment conditions are incomplete; inspect the saved summary")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent GB1 reference structure and knowledge-graph comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    structure = commands.add_parser("structure")
    structure.add_argument("--pdb", required=True)
    structure.add_argument("--reference", required=True)
    structure.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-dirs", nargs="+", required=True)
    prepare.add_argument("--model-files", nargs="+", required=True)
    prepare.add_argument("--pdb", required=True)
    prepare.add_argument("--knowledge", default="configs/knowledge.json")
    prepare.add_argument("--output", required=True)
    for command in ("run", "compare", "status"):
        child = commands.add_parser(command)
        child.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "structure":
            analysis = analyze_structure(args.pdb, args.reference)
            write_structure_tables(analysis, args.output)
            print(json.dumps({"positions": len(analysis["positions"]), "pairs": len(analysis["pairs"])}))
            return 0
        if args.command == "prepare":
            manifest = prepare_graph_agent(args.data_dirs, args.model_files, args.pdb, args.knowledge, args.output)
            print(json.dumps({"seeds": list(manifest["snapshots"]), "max_requests": manifest["max_requests"]}))
            return 0
        if args.command == "run":
            manifest = _manifest(args.output)
            config = ExperimentConfig.model_validate(next(iter(manifest["snapshots"].values()))["config"])
            proposer = OpenAIProposer(config, prompt_profile="v3")
            try:
                summary = run_graph_agent(args.output, proposer)
            finally:
                proposer.close()
        else:
            summary = compare_graph_agent(args.output, require_complete=args.command == "compare")
        print(json.dumps({"complete": summary["complete"], "total_requests": summary["total_requests"], "incomplete": summary["incomplete"]}))
        return 0 if summary["complete"] else 2
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 2
