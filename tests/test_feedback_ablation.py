"""Independent ablation tests use synthetic source runs, labels and API responses."""
from copy import deepcopy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import gb1.feedback_ablation as ablation
from gb1.data import HiddenOracle, describe_variant
from gb1.experiment import run_experiment
from gb1.storage import read_json, write_json
from test_workflow import setup
from test_prompt_v3 import AutonomousProposer


@pytest.fixture
def source(setup):
    setup.source_run = setup.root / "source-run"
    setup.source_state = run_experiment(setup.dataset, setup.selection, setup.config, setup.source_run,
                                       methods=["knowledge"], proposer=AutonomousProposer(setup.dataset.candidates.variant), prompt_profile="v3")
    assert setup.source_state["methods"]["knowledge"]["status"] == "completed"
    setup.ablation_output = setup.root / "ablation"
    return setup


@pytest.fixture
def prepared(source):
    source.manifest = ablation.prepare_ablation([source.source_run], [source.dataset.directory], source.ablation_output)
    return source


class Proposer:
    def __init__(self, manifest):
        self.snapshot = next(iter(manifest["snapshots"].values()))
        self.prompt_metadata = deepcopy(self.snapshot["prompt_metadata"])
        self.model = self.snapshot["api_identity"]["model_requested"]
        self.api_mode = self.snapshot["api_identity"]["api_mode"]
        self.contexts = []

    def generate(self, context):
        self.contexts.append(deepcopy(context))
        choices = [v for v in self.snapshot["eligible_variants"] if v not in context.get("accepted_variants", [])]
        if context.get("recent_round_feedback"):
            choices = list(reversed(choices))
        count = context["proposal_count"]
        rules = [context["knowledge"]["rules"][0]["id"]]
        return {"text": json.dumps({"analysis": "Synthetic ablation response", "proposals": [
            {"variant": v, "rationale": "Synthetic reason", "hypothesis": "Synthetic hypothesis", "knowledge_rule_ids": rules} for v in choices[:count]]}),
                "complete": True, "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                **deepcopy(self.snapshot["api_identity"]), "prompt_metadata": deepcopy(self.prompt_metadata)}


def arm_states(prepared):
    return {condition: read_json(prepared.ablation_output / "seed-11" / condition / "state.json") for condition in ablation.CONDITIONS}


def test_condition_transform_changes_exactly_two_fields_and_is_independent(prepared):
    snapshot = prepared.manifest["snapshots"]["11"]
    original = deepcopy(snapshot["initial_context"])
    full = ablation.condition_context(original, "full")
    withheld = ablation.condition_context(original, "withheld")
    assert full == original
    differences = {key for key in full if full[key] != withheld[key]}
    assert differences == ablation.FEEDBACK_FIELDS
    assert withheld["recent_round_feedback"] == []
    assert withheld["feedback_summary"] == ablation.WITHHELD_SUMMARY
    assert "updated measurements" in withheld["feedback_summary"]["detail"]
    full["known_data"]["n_known"] = 0
    assert original == snapshot["initial_context"]
    assert withheld["known_data"]["n_known"] == prepared.config.train_size + prepared.config.k
    assert "condition" not in withheld


def test_snapshot_excludes_later_outcomes_and_fits_one_shared_model(source, monkeypatch):
    source_path = source.source_run / "state.json"
    changed = read_json(source_path)
    first = changed["methods"]["knowledge"]["rounds"][0]
    for record in changed["methods"]["knowledge"]["rounds"][1:]:
        for row in record["recommendations"]:
            row["actual_fitness"] = 987654321.0
            row["rationale"] = "future-outcome-sentinel"
    write_json(source_path, changed)
    before = source_path.read_text(encoding="utf-8")
    fits = []
    original = ablation.fit_model
    def counted(spec, config, known):
        fits.append(list(known.variant))
        return original(spec, config, known)
    monkeypatch.setattr(ablation, "fit_model", counted)
    manifest = ablation.prepare_ablation([source.source_run], [source.dataset.directory], source.ablation_output)
    snapshot = manifest["snapshots"]["11"]
    expected = set(source.dataset.train.variant) | {row["variant"] for row in first["recommendations"]}
    assert len(fits) == 1 and set(fits[0]) == expected
    assert snapshot["source_prefix"]["training_count"] == 27
    assert set(snapshot["source_prefix"]["training_variants"]) == expected
    assert "987654321" not in json.dumps(snapshot) and "future-outcome-sentinel" not in json.dumps(snapshot)
    assert snapshot["initial_context"]["recent_round_feedback"][0]["predicted_fitness"] == first["recommendations"][0]["predicted_fitness"]
    assert before == source_path.read_text(encoding="utf-8")
    assert manifest["max_requests"] == 4
    with pytest.raises(ValueError, match="immutable"):
        ablation.prepare_ablation([source.source_run], [source.dataset.directory], source.ablation_output)


def test_source_context_must_match_round_one_only(source):
    path = source.source_run / "state.json"
    state = read_json(path)
    state["methods"]["knowledge"]["rounds"][1]["attempts"][0]["context"]["known_data"]["n_known"] += 3
    write_json(path, state)
    with pytest.raises(ValueError, match="round 1-only"):
        ablation.prepare_ablation([source.source_run], [source.dataset.directory], source.ablation_output)
    assert not source.ablation_output.exists()


def test_run_pairs_share_model_and_preserve_exact_request_snapshots(prepared, monkeypatch):
    proposer = Proposer(prepared.manifest)
    source_path = prepared.source_run / "state.json"
    before_source = source_path.read_text(encoding="utf-8")
    loads = []
    original = joblib.load
    def counted(path):
        loads.append(str(path))
        return original(path)
    monkeypatch.setattr(ablation.joblib, "load", counted)
    monkeypatch.setattr(ablation, "fit_model", lambda *args: pytest.fail("No refitting during either ablation arm"))
    summary = ablation.run_ablation(prepared.ablation_output, proposer)
    assert summary["complete"] is True and summary["total_requests"] == 2
    assert loads and len(set(loads)) == 1 and loads[0].endswith("seed-11.joblib")
    assert before_source == source_path.read_text(encoding="utf-8")
    states = arm_states(prepared)
    sent = iter(proposer.contexts)
    event_snapshots = {}
    for item in prepared.manifest["order"]:
        directory = prepared.ablation_output / "seed-11" / item["condition"]
        state = states[item["condition"]]
        assert state["stage"] == "evaluated" and len(state["scores"]) == 8
        assert len(state["recommendations"]) == 3
        assert all(len(row["sequence"]) == 56 and "actual_fitness" in row and "predicted_fitness" in row for row in state["recommendations"])
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        request = next(row for row in events if row["kind"] == "request_started")
        assert request["attempt"]["context"] == state["attempts"][0]["context"] == next(sent)
        event_snapshots[item["condition"]] = (directory / "events.jsonl").read_text(encoding="utf-8")
        assert (directory / "recommendations.csv").exists() and (directory / "usage.csv").exists()
    row = summary["paired_results"][0]
    assert row["full_minus_withheld_mean_fitness"] == row["full_mean_fitness"] - row["withheld_mean_fitness"]
    assert 0 <= row["recommendation_jaccard"] <= 1
    assert all(arm["total_tokens"] == 30 for arm in summary["arms"])
    ablation.run_ablation(prepared.ablation_output, proposer)
    assert len(proposer.contexts) == 2
    for condition, contents in event_snapshots.items():
        assert contents == (prepared.ablation_output / "seed-11" / condition / "events.jsonl").read_text(encoding="utf-8")


def test_prepare_and_run_never_read_final_test_or_validation_labels(source, monkeypatch):
    original = pd.read_csv
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ("test.csv", "validation.csv")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded)
    manifest = ablation.prepare_ablation([source.source_run], [source.dataset.directory], source.ablation_output)
    evaluate = HiddenOracle.evaluate
    def selected_only(oracle, variants):
        states = arm_states(source)
        selected = [state for state in states.values() if state["stage"] == "selected"]
        assert any([row["variant"] for row in state["recommendations"]] == variants for state in selected)
        assert len(variants) == source.config.k
        return evaluate(oracle, variants)
    monkeypatch.setattr(HiddenOracle, "evaluate", selected_only)
    assert ablation.run_ablation(source.ablation_output, Proposer(manifest))["complete"]


def test_correction_keeps_withholding_and_does_not_fill_missing_proposals(prepared):
    class Partial(Proposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            body["proposals"] = body["proposals"][:4] if context["stage"] == "initial_design" else []
            response["text"] = json.dumps(body)
            return response
    proposer = Partial(prepared.manifest)
    summary = ablation.run_ablation(prepared.ablation_output, proposer)
    assert summary["complete"] and summary["total_requests"] == 4
    states = arm_states(prepared)
    for condition, state in states.items():
        first, repair = state["attempts"]
        assert first["validation"]["status"] == "underfilled"
        assert repair["context"]["proposal_count"] == len(repair["context"]["replacement_examples"]) == 4
        assert len(state["proposals"]) == len(state["scores"]) == 4
        assert repair["context"]["recent_round_feedback"] == state["initial_context"]["recent_round_feedback"]
        assert repair["context"]["feedback_summary"] == state["initial_context"]["feedback_summary"]
        assert repair["context"]["response_schema"]["properties"]["proposals"]["maxItems"] == 4
    assert states["withheld"]["attempts"][1]["context"]["recent_round_feedback"] == []
    assert states["full"]["attempts"][1]["context"]["recent_round_feedback"]


def test_invalid_batches_exhaust_budget_and_keep_incomplete_pairs(prepared):
    class Invalid(Proposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            for row in body["proposals"]:
                row["variant"] = "XXXX"
            response["text"] = json.dumps(body)
            return response
    proposer = Invalid(prepared.manifest)
    summary = ablation.run_ablation(prepared.ablation_output, proposer)
    assert not summary["complete"] and summary["total_requests"] == 4
    assert len(summary["incomplete"]) == 2 and summary["paired_results"] == []
    assert all(arm["mean_fitness"] is None and arm["max_fitness"] is None and arm["median_fitness"] is None for arm in summary["arms"])
    assert all(arm["valid_count"] == arm["scored_count"] == arm["assay_count"] == 0 for arm in summary["arms"])
    assert not pd.read_csv(prepared.ablation_output / "paired_results.csv").shape[0]
    with pytest.raises(RuntimeError, match="incomplete paired"):
        ablation.summarize_ablation(prepared.ablation_output)
    ablation.run_ablation(prepared.ablation_output, proposer)
    assert len(proposer.contexts) == 4


def test_unknown_api_outcome_is_not_resent(prepared):
    class Timeout(Proposer):
        def generate(self, context):
            self.contexts.append(deepcopy(context))
            raise TimeoutError("Synthetic timeout; no real request")
    first = Timeout(prepared.manifest)
    summary = ablation.run_ablation(prepared.ablation_output, first)
    assert not summary["complete"] and summary["total_requests"] == 2
    assert all(state["attempts"][0]["status"] == "started" for state in arm_states(prepared).values())
    succeeding = Proposer(prepared.manifest)
    ablation.run_ablation(prepared.ablation_output, succeeding)
    assert succeeding.contexts == []


def test_persisted_response_survives_interruption_without_duplicate_call(prepared, monkeypatch):
    original = ablation.event
    stopped = []
    def interrupt(path, kind, **data):
        if kind == "response_received" and not stopped:
            stopped.append(True)
            raise KeyboardInterrupt("Synthetic process interruption")
        return original(path, kind, **data)
    monkeypatch.setattr(ablation, "event", interrupt)
    first = Proposer(prepared.manifest)
    with pytest.raises(KeyboardInterrupt):
        ablation.run_ablation(prepared.ablation_output, first)
    assert len(first.contexts) == 1
    monkeypatch.setattr(ablation, "event", original)
    succeeding = Proposer(prepared.manifest)
    summary = ablation.run_ablation(prepared.ablation_output, succeeding)
    assert summary["complete"] and summary["total_requests"] == 2
    assert len(succeeding.contexts) == 1


def test_selected_candidates_resume_oracle_without_another_generation(prepared, monkeypatch):
    original = HiddenOracle.evaluate
    monkeypatch.setattr(HiddenOracle, "evaluate", lambda *args: (_ for _ in ()).throw(RuntimeError("Synthetic lookup failure")))
    first = Proposer(prepared.manifest)
    summary = ablation.run_ablation(prepared.ablation_output, first)
    assert not summary["complete"]
    assert all(state["stage"] == "selected" for state in arm_states(prepared).values())
    monkeypatch.setattr(HiddenOracle, "evaluate", original)
    second = Proposer(prepared.manifest)
    assert ablation.run_ablation(prepared.ablation_output, second)["complete"]
    assert second.contexts == []


def test_comparison_rejects_context_and_api_identity_mismatches(prepared):
    ablation.run_ablation(prepared.ablation_output, Proposer(prepared.manifest))
    path = prepared.ablation_output / "seed-11/withheld/state.json"
    original = read_json(path)
    modified = deepcopy(original)
    modified["initial_context"]["known_data"]["n_known"] += 1
    write_json(path, modified)
    with pytest.raises(ValueError, match="frozen feedback manipulation"):
        ablation.summarize_ablation(prepared.ablation_output)
    modified = deepcopy(original)
    modified["attempts"][0]["response"]["model_requested"] = "another-test-model"
    write_json(path, modified)
    with pytest.raises(ValueError, match="identity mismatch"):
        ablation.summarize_ablation(prepared.ablation_output)


def test_proposer_mismatch_fails_before_request_and_cli_is_independent(prepared, monkeypatch):
    proposer = Proposer(prepared.manifest)
    proposer.model = "mismatched-test-model"
    summary = ablation.run_ablation(prepared.ablation_output, proposer)
    assert not summary["complete"] and summary["total_requests"] == 0
    assert proposer.contexts == []
    assert ablation.main(["status", "--output", str(prepared.ablation_output)]) == 2
    with pytest.raises(SystemExit) as exc:
        ablation.main(["--help"])
    assert exc.value.code == 0


def test_three_seed_global_budget_is_twelve_and_orders_are_fixed(source):
    from gb1.data import PublicDataset, prepare
    from gb1.models import select_model
    sources = [source.source_run]
    directories = [source.dataset.directory]
    for seed in (22, 33):
        config = source.config.model_copy(update={"seed": seed})
        directory = source.root / f"data-{seed}"
        prepare(source.source, directory, config, source.reference, "variant", "fitness")
        dataset = PublicDataset(directory)
        selection = select_model(dataset, config, source.root / f"selection-{seed}")
        run = source.root / f"source-{seed}"
        run_experiment(dataset, selection, config, run, methods=["knowledge"],
                       proposer=AutonomousProposer(dataset.candidates.variant), prompt_profile="v3")
        sources.append(run)
        directories.append(directory)
    first = ablation.prepare_ablation(sources, directories, source.ablation_output)
    second = ablation.prepare_ablation(list(reversed(sources)), list(reversed(directories)), source.root / "ablation-copy")
    assert first["order"] == second["order"]
    assert first["max_requests"] == 12 and len(first["order"]) == 6
    class Invalid(Proposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            for row in body["proposals"]:
                row["variant"] = "XXXX"
            response["text"] = json.dumps(body)
            return response
    proposer = Invalid(first)
    summary = ablation.run_ablation(source.ablation_output, proposer)
    assert summary["total_requests"] == 12
    assert len(summary["incomplete"]) == 6 and not summary["complete"]
    assert len(proposer.contexts) == 12
    ablation.run_ablation(source.ablation_output, proposer)
    assert len(proposer.contexts) == 12


def test_independent_cli_full_flow_uses_only_injected_test_client(source, monkeypatch):
    assert ablation.main(["prepare", "--source-runs", str(source.source_run), "--data-dirs", str(source.dataset.directory),
                          "--output", str(source.ablation_output)]) == 0
    manifest = read_json(source.ablation_output / "manifest.json")
    proposer = Proposer(manifest)
    closed = []
    proposer.close = lambda: closed.append(True)
    monkeypatch.setattr(ablation, "OpenAIProposer", lambda *args, **kwargs: proposer)
    assert ablation.main(["run", "--output", str(source.ablation_output)]) == 0
    assert closed == [True]
    assert ablation.main(["compare", "--output", str(source.ablation_output)]) == 0
    assert ablation.main(["status", "--output", str(source.ablation_output)]) == 0
    result = read_json(source.ablation_output / "summary.json")
    assert result["training_feedback_in_both_conditions"] is True
    assert result["arms"][0]["valid_count"] == 8 and result["arms"][1]["scored_count"] == 8


def test_budget_and_condition_metadata_corruption_is_rejected(prepared):
    manifest_path = prepared.ablation_output / "manifest.json"
    manifest = read_json(manifest_path)
    changed = deepcopy(manifest)
    changed["max_requests"] = 13
    write_json(manifest_path, changed)
    with pytest.raises(ValueError, match="request budget"):
        ablation.run_ablation(prepared.ablation_output, Proposer(prepared.manifest))
    write_json(manifest_path, manifest)
    state_path = prepared.ablation_output / "seed-11/full/state.json"
    state = read_json(state_path)
    state["condition"] = "withheld"
    write_json(state_path, state)
    with pytest.raises(ValueError, match="paired slot"):
        ablation.summarize_ablation(prepared.ablation_output)


def test_response_identity_drift_is_not_retried_as_a_format_correction(prepared):
    class Drift(Proposer):
        def generate(self, context):
            response = super().generate(context)
            response["model_returned"] = "different-test-model"
            return response
    proposer = Drift(prepared.manifest)
    result = ablation.run_ablation(prepared.ablation_output, proposer)
    assert not result["complete"] and result["total_requests"] == 2
    assert len(proposer.contexts) == 2
    assert all("no correction requested" in state["errors"][0]["message"] for state in arm_states(prepared).values())


@pytest.mark.parametrize("tamper", ["other_valid_id", "score_value", "score_ids", "proposal_reason", "rank", "sequence", "reason", "recommendation_score"])
def test_saved_selection_preflight_rejects_changes_before_any_api_or_oracle(prepared, monkeypatch, tamper):
    initial = arm_states(prepared)
    with monkeypatch.context() as scope:
        scope.setattr(HiddenOracle, "evaluate", lambda *args: (_ for _ in ()).throw(RuntimeError("Synthetic lookup pause")))
        ablation.run_ablation(prepared.ablation_output, Proposer(prepared.manifest))
    first_condition = prepared.manifest["order"][0]["condition"]
    final_condition = prepared.manifest["order"][-1]["condition"]
    # An earlier prepared arm must not send anything while a later selected arm is invalid.
    write_json(prepared.ablation_output / "seed-11" / first_condition / "state.json", initial[first_condition])
    path = prepared.ablation_output / "seed-11" / final_condition / "state.json"
    state = read_json(path)
    assert state["stage"] == "selected"
    if tamper == "other_valid_id":
        snapshot = prepared.manifest["snapshots"]["11"]
        other = next(v for v in snapshot["eligible_variants"] if v not in {row["variant"] for row in state["proposals"]})
        state["recommendations"][0]["variant"] = other
    elif tamper == "score_value":
        state["scores"][0]["predicted_fitness"] += 1
    elif tamper == "score_ids":
        state["scores"][0]["variant"], state["scores"][1]["variant"] = state["scores"][1]["variant"], state["scores"][0]["variant"]
    elif tamper == "proposal_reason":
        state["proposals"][0]["rationale"] = "Edited synthetic reason"
    elif tamper == "rank":
        state["recommendations"][0]["rank"] = 99
    elif tamper == "sequence":
        state["recommendations"][0]["sequence"] = "C" * 56
    elif tamper == "reason":
        state["recommendations"][0]["rationale"] = "Edited synthetic reason"
    else:
        state["recommendations"][0]["predicted_fitness"] += 1
    write_json(path, state)
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(HiddenOracle, "evaluate", lambda *args: pytest.fail("Preflight must reject before any oracle lookup"))
    proposer = Proposer(prepared.manifest)
    with pytest.raises(ValueError, match="Saved"):
        ablation.run_ablation(prepared.ablation_output, proposer)
    assert proposer.contexts == []
    assert path.read_text(encoding="utf-8") == before


def test_evaluated_summary_verifies_records_without_new_labels_or_rewriting_states(prepared, monkeypatch):
    ablation.run_ablation(prepared.ablation_output, Proposer(prepared.manifest))
    states = {condition: prepared.ablation_output / "seed-11" / condition / "state.json" for condition in ablation.CONDITIONS}
    before = {condition: path.read_text(encoding="utf-8") for condition, path in states.items()}
    monkeypatch.setattr(HiddenOracle, "evaluate", lambda *args: pytest.fail("Comparison must not query labels"))
    assert ablation.summarize_ablation(prepared.ablation_output)["complete"]
    assert before == {condition: path.read_text(encoding="utf-8") for condition, path in states.items()}
    path = states["full"]
    state = read_json(path)
    original = deepcopy(state)
    state["metrics"]["mean_fitness"] += 1
    write_json(path, state)
    with pytest.raises(ValueError, match="metrics"):
        ablation.summarize_ablation(prepared.ablation_output)
    state = deepcopy(original)
    state["scores"][0]["predicted_fitness"] += 1
    write_json(path, state)
    with pytest.raises(ValueError, match="frozen predictor"):
        ablation.summarize_ablation(prepared.ablation_output)


def test_float_roundoff_uses_saved_score_ranking_and_validator_does_not_mutate(prepared):
    ablation.run_ablation(prepared.ablation_output, Proposer(prepared.manifest))
    snapshot = prepared.manifest["snapshots"]["11"]
    state = arm_states(prepared)["full"]
    state["scores"] = [{"variant": row["variant"], "predicted_fitness": 1.0} for row in state["proposals"]]
    order = sorted(row["variant"] for row in state["proposals"])
    lookup = {row["variant"]: row for row in state["proposals"]}
    state["recommendations"] = [{**describe_variant(variant, prepared.reference), **lookup[variant], "rank": rank,
                                 "predicted_fitness": 1.0, "actual_fitness": 0.5} for rank, variant in enumerate(order[:prepared.config.k], 1)]
    state["metrics"] = ablation._metrics(state, snapshot)
    predictions = np.array([1.0 + (order.index(row["variant"]) + 1) * 1e-13 for row in state["proposals"]])
    fresh_ranking = sorted(zip(state["proposals"], predictions), key=lambda pair: (-pair[1], pair[0]["variant"]))[:prepared.config.k]
    assert [row["variant"] for row, _ in fresh_ranking] != order[:prepared.config.k]
    from types import SimpleNamespace
    model = SimpleNamespace(predict=lambda features: predictions)
    before = deepcopy(state)
    ablation._validate_selection(state, snapshot, model)
    assert state == before
    state["scores"][0]["predicted_fitness"] += 0.01
    with pytest.raises(ValueError, match="frozen predictor"):
        ablation._validate_selection(state, snapshot, model)
