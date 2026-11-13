"""v3 tests use artificial observations and model responses without external API calls."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import pandas as pd
import pytest
from gb1.agent import make_context
from gb1.cli import main, parser
from gb1.compare_prompts import compare_prompt_runs
from gb1.data import describe_variant
from gb1.experiment import run_experiment
from gb1.prompts import load_prompt_profile, repair_context
from gb1.schema import AMINO_ACIDS
from gb1.storage import read_json, write_json
from test_workflow import setup, SyntheticProposer


class AutonomousProposer:
    """Tests supply a private synthetic eligible set, never a production model fallback."""
    def __init__(self, candidates):
        self.candidates = sorted(candidates)
        self.contexts = []
        self.used = {"llm": set(), "knowledge": set()}

    def generate(self, context):
        self.contexts.append(deepcopy(context))
        method = "knowledge" if "knowledge" in context else "llm"
        ids = [rule["id"] for rule in context.get("knowledge", {}).get("rules", [])][:1]
        variants = [v for v in self.candidates if v not in self.used[method]][:context["proposal_count"]]
        self.used[method].update(variants)
        return {"text": json.dumps({"analysis": "Observations: artificial. Hypotheses: artificial. Feedback: artificial. Design: artificial.",
                                    "proposals": [{"variant": v, "rationale": "Synthetic rationale", "hypothesis": "Synthetic hypothesis", "knowledge_rule_ids": ids} for v in variants]}),
                "complete": True, "usage": {"input_tokens": 10, "output_tokens": 20},
                "model_requested": "synthetic-test", "model_returned": "synthetic-test", "api_mode": "test"}


def visible_ids(context):
    result = {row["variant"] for key in ("top_variants", "low_fitness_examples", "near_median_examples")
              for row in context["known_data"][key]}
    result.update(row["variant"] for row in context["recent_round_feedback"])
    return result


def test_v3_initial_context_is_compact_and_contains_no_candidate_batch(setup):
    variants = ["".join(value) for value in product(AMINO_ACIDS, repeat=4)][:2000]
    known = pd.DataFrame([{**describe_variant(v, setup.reference), "fitness": float(index % 30)} for index, v in enumerate(variants)])
    recent = [{"variant": variants[0], "predicted_fitness": 3, "actual_fitness": 0,
               "hypothesis": "Synthetic hypothesis", "rationale": "Synthetic rationale", "sequence": "must not be copied", "rank": 1, "knowledge_rule_ids": []}]
    examples = list(setup.dataset.candidates.variant[:8])
    previous = make_context(known, examples, setup.config, setup.reference, 2, prompt_profile="v2", recent_feedback=recent)
    snapshot = deepcopy(previous)
    context = make_context(known, examples, setup.config, setup.reference, 2, prompt_profile="v3", recent_feedback=recent)
    assert context["stage"] == "initial_design"
    assert context["candidate_examples"] == [] and "replacement_examples" not in context
    assert context["observed_variants_complete"] is False
    assert set(context["observed_variants"]) == set(context["forbidden_variants"]) == visible_ids(context)
    assert len(context["observed_variants"]) <= 24
    assert "full known history" in context["observed_variants_scope"]
    assert set(context["recent_round_feedback"][0]) == {"variant", "predicted_fitness", "actual_fitness", "residual", "hypothesis", "rationale"}
    assert context["recent_round_feedback"][0]["residual"] == -3
    assert len(json.dumps(context)) < 0.5 * len(json.dumps(previous))
    assert previous == snapshot
    assert previous["candidate_examples"] == examples
    assert previous["observed_variants"] == previous["forbidden_variants"] == sorted(known.variant)
    assert previous["recent_round_feedback"][0]["sequence"] == "must not be copied"
    assert "stage" not in previous


def test_v3_repair_only_examples_have_no_full_known_id_list_or_labels(setup):
    known, eligible = setup.dataset.train, set(setup.dataset.candidates.variant)
    ids = sorted(eligible)
    base = make_context(known, ids, setup.config, setup.reference, 1, prompt_profile="v3")
    accepted = [{"variant": v} for v in ids[:5]]
    rejections = [{"variant": ids[5], "reason": "synthetic rejection"}]
    result = repair_context(base, known, eligible, accepted, rejections, setup.config, 1, 2)
    assert result["stage"] == "repair_only"
    assert result["candidate_examples"] == []
    assert result["proposal_count"] == result["remaining_count"] == len(result["replacement_examples"]) == 3
    assert set(result["observed_variants"]) == set(base["observed_variants"])
    assert set(result["forbidden_variants"]) == set(base["observed_variants"]) | set(ids[:6])
    assert not set(result["replacement_examples"]) & (set(known.variant) | set(ids[:6]))
    assert all(isinstance(v, str) and len(v) == 4 for v in result["replacement_examples"])
    array = result["response_schema"]["properties"]["proposals"]
    assert array["minItems"] == array["maxItems"] == 3


def test_v3_full_known_history_is_still_checked_locally_and_repair_stays_bounded(setup):
    base = make_context(setup.dataset.train, [], setup.config, setup.reference, 1, prompt_profile="v3")
    not_displayed = sorted(set(setup.dataset.train.variant) - set(base["observed_variants"]))[0]
    class KnownCollision(AutonomousProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            if len(self.contexts) == 1:
                body["proposals"][0]["variant"] = not_displayed
            elif context["stage"] == "repair_only":
                body["proposals"] = [{"variant": v, "hypothesis": "Synthetic replacement", "rationale": "Synthetic rationale", "knowledge_rule_ids": []} for v in context["replacement_examples"]]
            response["text"] = json.dumps(body)
            return response
    config = setup.config.model_copy(update={"rounds": 1})
    proposer = KnownCollision(setup.dataset.candidates.variant)
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer, prompt_profile="v3")
    record = state["methods"]["llm"]["rounds"][0]
    assert state["methods"]["llm"]["status"] == "completed"
    assert len(proposer.contexts) == 2
    assert record["attempts"][0]["valid_count"] == 7
    assert proposer.contexts[1]["proposal_count"] == 1
    assert not_displayed in proposer.contexts[1]["forbidden_variants"]
    assert not_displayed not in {row["variant"] for row in record["proposals"]}
    assert record["model_score_count"] == 8 and len(record["recommendations"]) == 3


def test_v3_three_rounds_keep_independent_known_feedback_and_no_test_reads(setup, monkeypatch):
    original = pd.read_csv
    def checked(path, *args, **kwargs):
        assert Path(path).name not in ("test.csv", "validation.csv")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", checked)
    proposer = AutonomousProposer(setup.dataset.candidates.variant)
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm", "knowledge"], proposer=proposer, prompt_profile="v3")
    assert all(state["methods"][m]["status"] == "completed" for m in ("llm", "knowledge"))
    assert len(proposer.contexts) == 6
    assert state["prompt_metadata"] == load_prompt_profile("v3").metadata()
    assert "merely predicting that it is worse" in state["prompt_metadata"]["system_prompt"]
    for context in proposer.contexts:
        assert context["candidate_examples"] == [] and context["stage"] == "initial_design"
        assert set(context["observed_variants"]) == visible_ids(context)
        assert all(set(row) == {"variant", "predicted_fitness", "actual_fitness", "residual", "hypothesis", "rationale"} for row in context["recent_round_feedback"])
    with pytest.raises(ValueError, match="prompt profile"):
        run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer, prompt_profile="v2")


def test_comparison_v3_names_and_complete_pair_metadata_checks(setup):
    v1, v3 = setup.root / "v1", setup.root / "v3"
    run_experiment(setup.dataset, setup.selection, setup.config, v1, methods=["llm", "knowledge"], proposer=SyntheticProposer())
    run_experiment(setup.dataset, setup.selection, setup.config, v3, methods=["llm", "knowledge"], proposer=AutonomousProposer(setup.dataset.candidates.variant), prompt_profile="v3")
    initial = [(path / "state.json").read_text(encoding="utf-8") for path in (v1, v3)]
    output = setup.root / "comparison"
    result = compare_prompt_runs([v1], [v3], output, candidate_profile="v3")
    assert set(result["prompt_metadata"]) == {"v1", "v3"}
    assert {row["profile"] for row in result["methods"]} == {"v1", "v3"}
    for row in result["per_seed_differences"]:
        assert "v3_mean_fitness" in row and "v2_mean_fitness" not in row
        assert row["v1_status"] == row["v3_status"] == "completed"
    assert result["metadata_checks"][0]["same_llm_model_and_api_mode"] is True
    assert result["metadata_checks"][0]["complete_method_round_pairs"] is True
    assert initial == [(path / "state.json").read_text(encoding="utf-8") for path in (v1, v3)]
    assert main(["compare-prompts", "--v1-runs", str(v1), "--candidate-runs", str(v3), "--candidate-profile", "v3", "--output", str(output)]) == 0
    state_path = v3 / "state.json"
    original = read_json(state_path)
    altered = deepcopy(original)
    for record in altered["methods"]["llm"]["rounds"]:
        for attempt in record["attempts"]:
            attempt["response"]["api_mode"] = "different-test-api"
    write_json(state_path, altered)
    with pytest.raises(ValueError, match="different LLM model or API mode"):
        compare_prompt_runs([v1], [v3], setup.root / "bad-api", candidate_profile="v3")
    altered = deepcopy(original)
    altered["methods"]["knowledge"]["rounds"].pop()
    write_json(state_path, altered)
    with pytest.raises(ValueError, match="complete paired rounds"):
        compare_prompt_runs([v1], [v3], setup.root / "bad-rounds", candidate_profile="v3")


def test_v3_cli_run_profile_and_legacy_comparison_alias():
    run = ["run", "--data", "synthetic", "--config", "synthetic", "--output", "synthetic", "--selection", "synthetic", "--prompt-profile", "v3"]
    assert parser().parse_args(run).prompt_profile == "v3"
    compare = ["compare-prompts", "--v1-runs", "synthetic-v1", "--v2-runs", "synthetic-v3", "--candidate-profile", "v3", "--output", "synthetic"]
    assert parser().parse_args(compare).candidate_runs == ["synthetic-v3"]
