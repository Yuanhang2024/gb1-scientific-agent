"""Versioned prompt checks use only synthetic observations and in-process API doubles."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gb1.agent import (PROMPT_VERSION, SYSTEM_PROMPT, OpenAIProposer, make_context)
from gb1.cli import main, parser
from gb1.experiment import run_experiment
from gb1.prompts import (LEGACY_PROFILE, feedback_context, known_examples, load_prompt_profile,
                         proposal_schema, repair_context, validate_prompt_state)
from gb1.schema import ProposalBatch
from gb1.storage import read_json, write_json
from test_workflow import setup, SyntheticProposer


def test_legacy_prompt_and_context_are_preserved_exactly(setup):
    assert load_prompt_profile().metadata() == LEGACY_PROFILE.metadata()
    assert load_prompt_profile().system_prompt == SYSTEM_PROMPT
    context = make_context(setup.dataset.train, ["AAAA", "CCCC"], setup.config, setup.reference, 1)
    known = setup.dataset.train
    positions = []
    for i, position in enumerate((39, 40, 41, 54)):
        grouped = known.assign(residue=known.variant.str[i]).groupby("residue").fitness.agg(["count", "mean", "max"])
        positions.append({"position": position, "residue_statistics": grouped.reset_index().to_dict(orient="records")})
    expected = {
        "prompt_version": "gb1-proposals-v1", "round": 1, "reference_sequence": setup.reference,
        "reference_variant": "VDGV", "positions": [39, 40, 41, 54],
        "known_data": {"n_known": len(known), "fitness_mean": float(known.fitness.mean()),
                       "best_fitness": float(known.fitness.max()),
                       "top_variants": known.sort_values(["fitness", "variant"], ascending=[False, True]).head(setup.config.summary_top)[["variant", "fitness", "mutations"]].to_dict(orient="records"),
                       "positions": positions,
                       "interpretation": "Marginal residue statistics are observations in sampled backgrounds, not causal substitution effects."},
        "candidate_examples": ["AAAA", "CCCC"], "proposal_count": setup.config.proposal_count,
        "design_space": "Generate four-site variants using standard amino acids at 39,40,41,54. Membership is checked locally without exposing hidden fitness.",
        "assay_budget": setup.config.k,
        "selection": "Local predictor scores valid proposals and selects their top k, at most the declared scoring budget.",
        "response_schema": ProposalBatch.model_json_schema(),
    }
    assert context == expected
    assert "prompt_profile" not in setup.config.model_dump()


def test_v2_contrasting_examples_are_known_disjoint_and_bounded(setup):
    context = make_context(setup.dataset.train, [], setup.config, setup.reference, 1, prompt_profile="v2")
    summary = context["known_data"]
    top = {r["variant"] for r in summary["top_variants"]}
    low = {r["variant"] for r in summary["low_fitness_examples"]}
    middle = {r["variant"] for r in summary["near_median_examples"]}
    assert len(top) == setup.config.summary_top and len(low) == 4 and len(middle) == 4
    assert not top & low and not top & middle and not low & middle
    known = setup.dataset.train.set_index("variant").fitness
    for key in ("top_variants", "low_fitness_examples", "near_median_examples"):
        assert all(row["fitness"] == known[row["variant"]] for row in summary[key])
    assert context["observed_variants"] == sorted(known.index)
    assert all(isinstance(v, str) and len(v) == 4 for v in context["observed_variants"])
    assert context["feedback_summary"]["n_observed_feedback"] == 0


def test_feedback_residuals_and_hypotheses_use_only_observed_rows():
    recent = [
        {"variant": "AAAA", "actual_fitness": 1.0, "predicted_fitness": 4.0, "hypothesis": "test high"},
        {"variant": "CCCC", "actual_fitness": 5.0, "predicted_fitness": 2.0, "hypothesis": "test low"},
        {"variant": "DDDD", "actual_fitness": 123456.0, "predicted_fitness": -123456.0, "hypothesis": "hidden sentinel"},
    ]
    before = deepcopy(recent)
    rows, summary = feedback_context(recent, {"AAAA", "CCCC"})
    assert [row["residual"] for row in rows] == [-3, 3]
    assert summary["largest_overestimation"]["variant"] == "AAAA"
    assert summary["largest_underestimation"]["variant"] == "CCCC"
    assert summary["largest_overestimation"]["hypothesis"] == "test high"
    assert "hidden sentinel" not in json.dumps([rows, summary])
    assert recent == before


def test_v2_repair_examples_exclude_known_accepted_rejected_and_match_remaining(setup):
    known = setup.dataset.train
    eligible = set(setup.dataset.candidates.variant)
    ids = sorted(eligible)
    accepted = [{"variant": v} for v in ids[:3]]
    rejected = [{"variant": ids[3], "reason": "synthetic rejection"}, {"variant": None, "reason": "parse failed"}]
    base = make_context(known, ids[:8], setup.config, setup.reference, 1, prompt_profile="v2")
    a = repair_context(base, known, eligible, accepted, rejected, setup.config, 1, 2)
    b = repair_context(base, known, eligible, accepted, rejected, setup.config, 1, 2)
    assert a == b
    assert a["proposal_count"] == a["remaining_count"] == 5
    assert len(a["replacement_examples"]) == 5
    assert a["replacement_examples"] == a["candidate_examples"]
    forbidden = set(known.variant) | set(ids[:4])
    assert set(a["forbidden_variants"]) == forbidden
    assert not set(a["replacement_examples"]) & forbidden
    assert "fitness" not in json.dumps(a["replacement_examples"])
    rejected.append({"variant": ids[4], "reason": "later rejection"})
    assert len(a["rejections"]) == 2


@pytest.mark.parametrize("profile", ["v1", "v2"])
def test_request_context_is_a_snapshot_not_a_mutated_rejection_list(setup, profile):
    class Rejecting(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            body["proposals"][0]["variant"] = "XXXX"
            response["text"] = json.dumps(body)
            return response
    proposer = Rejecting()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                           methods=["llm"], proposer=proposer, prompt_profile=profile)
    events = [json.loads(line) for line in (setup.output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    requests = [row for row in events if row["kind"] == "llm_request"]
    attempts = [attempt for record in state["methods"]["llm"]["rounds"] for attempt in record["attempts"]]
    assert len(attempts) == len(requests) == len(proposer.contexts)
    for attempt, recorded, actual in zip(attempts, requests, proposer.contexts):
        assert attempt["context"] == recorded["context"] == actual
        assert attempt["prompt_metadata"] == recorded["prompt_metadata"] == load_prompt_profile(profile).metadata()
    assert len(attempts[1]["context"]["rejections"]) == 1


def test_v2_underfilled_output_is_recorded_and_never_automatically_completed(setup):
    class Underfilled(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            body["proposals"] = body["proposals"][:4] if not context.get("accepted_variants") else []
            response["text"] = json.dumps(body)
            return response
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                           methods=["llm"], proposer=Underfilled(), prompt_profile="v2")
    for record in state["methods"]["llm"]["rounds"]:
        first, second = record["attempts"]
        assert first["status"] == "underfilled"
        assert first["expected_count"] == 8 and first["returned_count"] == first["valid_count"] == 4
        assert second["expected_count"] == 4 and second["returned_count"] == 0
        assert record["model_score_count"] == 4
        assert len(record["proposals"]) == 4
        assert len(record["recommendations"]) == setup.config.k


def test_v2_overbudget_batch_is_rejected_whole_and_repair_avoids_its_ids(setup):
    ids = sorted(setup.dataset.candidates.variant)
    class Overbudget(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            if not self.contexts or len(self.contexts) == 1:
                body = json.loads(response["text"])
                exemplar = body["proposals"][0]
                body["proposals"] = [{**exemplar, "variant": v} for v in ids[:9]]
                response["text"] = json.dumps(body)
            return response
    config = setup.config.model_copy(update={"rounds": 1})
    proposer = Overbudget()
    state = run_experiment(setup.dataset, setup.selection, config, setup.output,
                           methods=["llm"], proposer=proposer, prompt_profile="v2")
    record = state["methods"]["llm"]["rounds"][0]
    first, second = record["attempts"]
    assert first["returned_count"] == 9 and first["valid_count"] == 0 and first["status"] == "invalid"
    assert set(ids[:9]) <= set(second["context"]["forbidden_variants"])
    assert not set(ids[:9]) & {row["variant"] for row in record["proposals"]}
    assert record["model_score_count"] == 8


def test_v2_profiles_preserve_old_state_and_reject_cross_profile_or_text_changes(setup, monkeypatch):
    run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["random"])
    path = setup.output / "state.json"
    old = read_json(path)
    old.pop("prompt_metadata")
    write_json(path, old)
    before = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="prompt profile"):
        run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["random"], prompt_profile="v2")
    assert path.read_text(encoding="utf-8") == before
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["random"])
    assert resumed == old and "prompt_metadata" not in resumed
    new_output = setup.root / "v2-run"
    new = run_experiment(setup.dataset, setup.selection, setup.config, new_output, methods=["random"], prompt_profile="v2")
    assert new["prompt_metadata"] == load_prompt_profile("v2").metadata()
    with pytest.raises(ValueError, match="prompt profile"):
        run_experiment(setup.dataset, setup.selection, setup.config, new_output, methods=["random"])
    from dataclasses import replace
    import gb1.experiment as experiment
    changed = replace(load_prompt_profile("v2"), system_prompt="Modified synthetic prompt")
    monkeypatch.setattr(experiment, "load_prompt_profile", lambda _: changed)
    with pytest.raises(ValueError, match="system prompt text"):
        run_experiment(setup.dataset, setup.selection, setup.config, new_output, methods=["random"], prompt_profile="v2")


def test_v2_context_only_reads_known_rows_no_hidden_files_and_method_feedback_stays_separate(setup, monkeypatch):
    original = pd.read_csv
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ("test.csv", "validation.csv")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded)
    proposer = SyntheticProposer()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                           methods=["llm", "knowledge"], proposer=proposer, prompt_profile="v2")
    for context in proposer.contexts:
        method = "knowledge" if "knowledge" in context else "llm"
        number = context["round"]
        known_ids = set(setup.dataset.train.variant)
        for record in state["methods"][method]["rounds"][:number - 1]:
            known_ids.update(row["variant"] for row in record["recommendations"])
        assert set(context["observed_variants"]) == known_ids
        assert {row["variant"] for row in context["recent_round_feedback"]} <= known_ids
        assert "oracle.csv" not in json.dumps(context) and "test.csv" not in json.dumps(context)
        for row in context["recent_round_feedback"]:
            assert row["residual"] == row["actual_fitness"] - row["predicted_fitness"]
            assert row["hypothesis"] == "Synthetic test hypothesis"


@pytest.mark.parametrize("mode", ["chat", "responses"])
@pytest.mark.parametrize("profile", ["v1", "v2", "v3"])
def test_api_receives_exact_profile_text_and_context_schema_without_network(setup, mode, profile):
    requests = []
    actual_text = '  {"analysis": "synthetic", "proposals": []}  \n'
    def create(**kwargs):
        requests.append(deepcopy(kwargs))
        if mode == "responses":
            return SimpleNamespace(output_text=actual_text, status="completed", usage=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=actual_text, refusal=None), finish_reason="stop")], usage=None)
    client = SimpleNamespace(responses=SimpleNamespace(create=create), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    adapter = OpenAIProposer(setup.config, client=client, environ={"OPENAI_MODEL": "synthetic", "OPENAI_API_MODE": mode}, prompt_profile=profile)
    context = make_context(setup.dataset.train, [], setup.config, setup.reference, 1, prompt_profile=profile)
    contexts = [context]
    if profile in ("v2", "v3"):
        ids = sorted(setup.dataset.candidates.variant)
        contexts.append(repair_context(context, setup.dataset.train, set(ids), [{"variant": v} for v in ids[:5]], [], setup.config, 1, 2))
    for user_context in contexts:
        response = adapter.generate(user_context)
        request = requests[-1]
        sent_text = request["instructions"] if mode == "responses" else request["messages"][0]["content"]
        sent_schema = request["text"]["format"]["schema"] if mode == "responses" else request["response_format"]["json_schema"]["schema"]
        assert sent_text == load_prompt_profile(profile).system_prompt
        assert response["prompt_metadata"] == load_prompt_profile(profile).metadata()
        assert response["text"] == actual_text
        assert sent_schema == user_context["response_schema"]
        array_schema = sent_schema["properties"]["proposals"]
        if profile in ("v2", "v3"):
            assert array_schema["minItems"] == array_schema["maxItems"] == user_context["proposal_count"]
        else:
            assert array_schema == ProposalBatch.model_json_schema()["properties"]["proposals"]
        assert "uniqueItems" not in array_schema


def test_dynamic_schema_is_fresh_and_counts_are_validated():
    first, second = proposal_schema(40), proposal_schema(3)
    assert first["properties"]["proposals"]["minItems"] == first["properties"]["proposals"]["maxItems"] == 40
    assert second["properties"]["proposals"]["minItems"] == second["properties"]["proposals"]["maxItems"] == 3
    assert "maxItems" not in ProposalBatch.model_json_schema()["properties"]["proposals"]
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            proposal_schema(invalid)


def test_cli_profile_flag_default_and_override():
    common = ["run", "--data", "synthetic", "--config", "synthetic", "--output", "synthetic", "--selection", "synthetic"]
    assert parser().parse_args(common).prompt_profile == "v1"
    assert parser().parse_args(common + ["--prompt-profile", "v2"]).prompt_profile == "v2"
