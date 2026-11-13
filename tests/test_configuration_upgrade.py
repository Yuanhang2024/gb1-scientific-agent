"""Configurable runs use synthetic measurements and offline model replies only."""
from copy import deepcopy
import json
import pytest
from pydantic import ValidationError
from gb1.agent import build_request_payload, collect_proposals, make_context
from gb1.experiment import RoundPaused, candidate_cohort, run_experiment
from gb1.models import estimator, load_selection, model_specs, select_model
from gb1.schema import ExperimentConfig, variant_allowed
from gb1.storage import read_json, write_json
from test_workflow import setup, SyntheticProposer
from test_plugin_worker import fake_project, offline_host, run_worker, start, worker


@pytest.mark.parametrize("changes", [
    {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"rounds": 0}, {"rounds": 1.1},
    {"k": 41}, {"proposal_count": 30}, {"model_families": []}, {"model_families": ["ridge", "ridge"]},
    {"model_families": ["unsupported"]}, {"model_seed": -1}, {"design_seed": True},
    {"ridge_alphas": [float("nan")]}, {"tree_min_samples_leaf": [1.5]},
    {"allowed_positions": [39, 39]}, {"allowed_positions": [38]}, {"allowed_amino_acids": "AA"},
    {"min_mutations": 3, "max_mutations": 2}, {"allowed_positions": [], "min_mutations": 1},
    {"llm_temperature": 3}, {"target_fitness": float("inf")}, {"patience": 0},
])
def test_invalid_scientific_parameters_are_rejected(changes):
    with pytest.raises(ValidationError):
        ExperimentConfig(**changes)


def test_optional_limits_and_separate_model_seed():
    config = ExperimentConfig(seed=4294967295, rounds=7, model_seed=44, design_seed=55,
                              model_families=["extra_trees", "ridge"], ridge_alphas=[2],
                              tree_min_samples_leaf=[2], llm_max_output_tokens=None, llm_timeout=None)
    specs = model_specs(config)
    assert [spec["family"] for spec in specs] == ["extra_trees", "ridge"]
    assert estimator(specs[0], config).random_state == 44
    for mode in ("chat", "responses"):
        payload = build_request_payload({}, config, "offline", mode, "test")
        assert "max_output_tokens" not in payload and "max_completion_tokens" not in payload
    configured = config.model_copy(update={"llm_temperature": 0.4})
    assert build_request_payload({}, configured, "offline", "chat", "test")["temperature"] == 0.4


def test_oversize_prefix_is_chosen_before_validation_and_scoring(setup):
    variants = list(setup.dataset.candidates.variant[:9])
    body = {"analysis": "synthetic", "proposals": [
        {"variant": v, "hypothesis": "synthetic", "rationale": "synthetic", "knowledge_rule_ids": []} for v in variants]}
    with pytest.raises(ValueError, match="budget"):
        collect_proposals(json.dumps(body), set(variants), setup.config)
    config = setup.config.model_copy(update={"oversize_policy": "first_n"})
    body["proposals"][0]["variant"] = "XXXX"
    _, valid, rejected = collect_proposals(json.dumps(body), set(variants), config)
    assert [row["variant"] for row in valid] == variants[1:8]
    assert len(rejected) == 1 and variants[8] not in {row["variant"] for row in valid}


def test_context_lengths_examples_and_mutation_constraints(setup):
    config = setup.config.model_copy(update={"analysis_max_chars": 6000, "proposal_text_max_chars": 3000,
                                             "contrast_examples": 0, "repair_examples": False})
    context = make_context(setup.dataset.train, [], config, setup.reference, 1, prompt_profile="v3")
    assert not context["known_data"]["low_fitness_examples"]
    assert context["response_schema"]["properties"]["analysis"]["maxLength"] == 6000
    text = json.dumps({"analysis": "x" * 5500, "proposals": [{"variant": setup.dataset.candidates.variant[0],
                      "hypothesis": "x" * 2500, "rationale": "synthetic", "knowledge_rule_ids": []}]})
    assert len(collect_proposals(text, set(setup.dataset.candidates.variant), config)[1]) == 1
    from gb1.prompts import repair_context
    repaired = repair_context(context, setup.dataset.train, set(setup.dataset.candidates.variant), [], [], config, 1, 2)
    assert repaired["replacement_examples"] == []
    restricted = ExperimentConfig(allowed_positions=[39], allowed_amino_acids="ACDGV", min_mutations=1, max_mutations=1)
    assert variant_allowed("ADGV", restricted)
    assert not variant_allowed("VDGA", restricted)
    assert not variant_allowed("VDGV", restricted)
    assert not variant_allowed("WDGV", restricted)


def test_four_rounds_new_seed_custom_k_and_no_saved_predictors(setup):
    config = setup.config.model_copy(update={"seed": 765, "rounds": 4, "k": 5, "proposal_count": 13,
                                             "score_budget": 13, "save_predictors": False,
                                             "model_families": ["ridge"]})
    selection = select_model(setup.dataset, config, setup.root / "new-selection")
    state = run_experiment(setup.dataset, selection, config, setup.output, methods=["random", "model"])
    for method in ("random", "model"):
        assert state["methods"][method]["status"] == "completed"
        assert [r["training_count"] for r in state["methods"][method]["rounds"]] == [24, 29, 34, 39]
        assert all(len(r["recommendations"]) == 5 and r["model_score_count"] == 13 for r in state["methods"][method]["rounds"])
    assert not list(setup.root.rglob("new-selection/*.joblib"))
    assert not list(setup.output.glob("*.joblib"))


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_six_rounds_for_calibration_seeds_without_predictor_files(setup, seed):
    from gb1.data import PublicDataset, prepare
    config = setup.config.model_copy(update={"seed": seed, "rounds": 6, "save_predictors": False})
    directory = setup.root / "six-round-data"
    prepare(setup.source, directory, config, setup.reference, "variant", "fitness")
    dataset = PublicDataset(directory)
    selection = select_model(dataset, config, setup.root / "six-round-selection")
    state = run_experiment(dataset, selection, config, setup.output, methods=["model"])
    rounds = state["methods"]["model"]["rounds"]
    assert state["methods"]["model"]["status"] == "completed"
    assert [record["round"] for record in rounds] == [1, 2, 3, 4, 5, 6]
    assert [record["training_count"] for record in rounds] == [24, 27, 30, 33, 36, 39]
    assert len({row["variant"] for record in rounds for row in record["recommendations"]}) == 18
    assert not list((setup.root / "six-round-selection").glob("*.joblib"))
    assert not list(setup.output.glob("*.joblib"))


@pytest.mark.parametrize("changes, expected_rounds, reason", [
    ({"max_assays": 5}, 1, "max_assays"),
    ({"max_assays": 1}, 0, "max_assays"),
    ({"max_model_calls": 1}, 1, "max_model_calls"),
    ({"max_reported_tokens": 40}, 2, "max_reported_tokens"),
    ({"target_fitness": 0}, 0, "target_fitness"),
    ({"patience": 1, "min_improvement": 100000}, 1, "patience"),
    ({"patience": 1, "min_improvement": 100000, "early_stop_metric": "round_mean"}, 2, "patience"),
])
def test_stop_conditions_preserve_only_completed_rounds(setup, changes, expected_rounds, reason):
    config = setup.config.model_copy(update={"rounds": 4, **changes})
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=SyntheticProposer())
    method = state["methods"]["llm"]
    assert method["status"] == "stopped" and method["stop_reason"] == reason
    assert len(method["rounds"]) == expected_rounds
    assert all(r["stage"] == "evaluated" for r in method["rounds"])


def test_unknown_usage_stops_next_dispatch(setup):
    class MissingUsage(SyntheticProposer):
        def generate(self, context):
            result = super().generate(context)
            result["usage"] = None
            return result
    config = setup.config.model_copy(update={"max_reported_tokens": 1000})
    proposer = MissingUsage()
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer)
    assert len(proposer.contexts) == 1
    assert state["methods"]["llm"]["stop_reason"] == "reported_token_usage_unknown"


def test_full_batch_requirement_checks_available_candidates(setup):
    from types import SimpleNamespace
    from gb1.experiment import ExperimentStopped
    dataset = SimpleNamespace(candidates=setup.dataset.candidates.head(5))
    assert len(candidate_cohort(dataset, setup.dataset.train, setup.config, 1)) == 5
    full = setup.config.model_copy(update={"require_full_batch": True})
    with pytest.raises(ExperimentStopped, match="eligible_candidates_exhausted"):
        candidate_cohort(dataset, setup.dataset.train, full, 1)


def test_legacy_repair_examples_can_be_disabled(setup):
    class Partial(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            if not context.get("accepted_variants"):
                body = json.loads(response["text"])
                body["proposals"] = body["proposals"][:4]
                response["text"] = json.dumps(body)
            return response
    config = setup.config.model_copy(update={"repair_examples": False, "rounds": 1})
    proposer = Partial()
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer)
    assert state["methods"]["llm"]["status"] == "completed"
    assert len(proposer.contexts) == 2 and proposer.contexts[1]["candidate_examples"] == []


def test_reported_total_token_count_is_sufficient_for_dispatch_budget(setup):
    class TotalUsage(SyntheticProposer):
        def generate(self, context):
            result = super().generate(context)
            result["usage"] = {"total_tokens": 30}
            return result
    config = setup.config.model_copy(update={"max_reported_tokens": 40})
    proposer = TotalUsage()
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer)
    assert len(proposer.contexts) == 2
    assert state["methods"]["llm"]["stop_reason"] == "max_reported_tokens"


def test_saved_recommendations_require_completed_scores(setup):
    run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["model"])
    state = read_json(setup.output / "state.json")
    state["methods"]["model"]["rounds"][0]["scores"] = []
    write_json(setup.output / "state.json", state)
    with pytest.raises(ValueError, match="matching completed predictor scores"):
        run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["model"])


def test_require_full_batch_does_not_query_partial_recommendations(setup):
    class Partial(SyntheticProposer):
        def generate(self, context):
            result = super().generate(context)
            body = json.loads(result["text"])
            body["proposals"] = body["proposals"][:2]
            result["text"] = json.dumps(body)
            return result
    config = setup.config.model_copy(update={"require_full_batch": True})
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=Partial())
    assert state["methods"]["llm"]["status"] == "failed"
    assert state["methods"]["llm"]["rounds"][0]["stage"] == "started"
    assert "recommendations" not in state["methods"]["llm"]["rounds"][0]


def test_pause_resume_does_not_repeat_saved_requests_and_feedback(setup):
    config = setup.config.model_copy(update={"feedback_rounds": 2, "rounds": 4})
    proposer = SyntheticProposer()
    def pause(method, number):
        assert number == 1
        raise RoundPaused()
    first = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer, checkpoint=pause)
    saved = deepcopy(first["methods"]["llm"]["rounds"][0])
    assert first["methods"]["llm"]["status"] == "paused" and len(proposer.contexts) == 1
    resumed = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=proposer)
    assert resumed["methods"]["llm"]["status"] == "completed" and len(proposer.contexts) == 4
    assert resumed["methods"]["llm"]["rounds"][0] == saved
    assert len(proposer.contexts[2]["recent_round_feedback"]) == 6
    assert len(proposer.contexts[3]["recent_round_feedback"]) == 6


def test_legacy_configurations_can_resume_without_rewriting_old_fields(setup):
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["random"])
    new_fields = {"contrast_examples", "feedback_rounds", "model_families", "model_seed", "design_seed", "llm_temperature",
                  "oversize_policy", "repair_examples", "require_full_batch", "analysis_max_chars", "proposal_text_max_chars",
                  "allowed_positions", "allowed_amino_acids", "min_mutations", "max_mutations", "max_assays", "max_model_calls",
                  "max_reported_tokens", "target_fitness", "patience", "min_improvement", "early_stop_metric", "save_predictors"}
    selection = deepcopy(setup.selection)
    for key in new_fields:
        state["config"].pop(key)
        selection["config"].pop(key)
    state["selection"] = selection
    write_json(setup.output / "state.json", state)
    legacy_path = setup.root / "legacy-selection.json"
    write_json(legacy_path, selection)
    assert load_selection(legacy_path, setup.dataset, setup.config) == selection
    resumed = run_experiment(setup.dataset, selection, setup.config, setup.output, methods=["random"])
    assert resumed == state and not (new_fields & set(resumed["config"]))


def configured_start(project, **changes):
    values = json.loads((project / "configs/experiment.json").read_text(encoding="utf-8"))
    values.update(seed=765, rounds=4, k=13, proposal_count=17, score_budget=17, model_families=["ridge"],
                  save_predictors=False, llm_max_output_tokens=None, llm_timeout=None, **changes)
    return {**start(method="model"), "seed": values["seed"], "rounds": values["rounds"], "config": values,
            "prompt_profile": "v3", "repeat_index": 0, "features": ["process-events-v1", "round-control-v1"]}


def test_worker_arbitrary_rounds_seed_k_and_model_grid(fake_project):
    project, _ = fake_project
    message = configured_start(project)
    code, values, errors = run_worker(project, message)
    assert code == 0, errors
    summary = values[-1]["summary"]
    assert summary["completed_rounds"] == 4 and summary["calls"] == 0
    assert all(len(record["top10"]) == 13 and record["scored_count"] == 17 for record in summary["rounds"])
    assert [v["round"] for v in values if v["type"] == "checkpoint"] == [1, 2, 3, 4]
    assert not list((project / "results/dsh-plugin/synthetic-run").rglob("*.joblib"))


def test_worker_pause_resume_frozen_configuration_and_session_request_ids(fake_project):
    project, _ = fake_project
    message = configured_start(project)
    message["method"] = "llm"
    host = offline_host(message)
    def pausing(request, project):
        if request["type"] == "checkpoint":
            return {"version": 1, "type": "round_control", "round": request["round"], "action": "pause"}
        return host(request, project)
    code, first, errors = run_worker(project, message, pausing)
    assert code == 0 and first[-1]["status"] == "paused", errors
    assert first[-1]["summary"]["calls"] == 1
    saved = read_json(project / "results/dsh-plugin/synthetic-run/run/state.json")["methods"]["llm"]["rounds"][0]
    code, resumed, errors = run_worker(project, {**message, "resume": True}, host)
    assert code == 0 and resumed[-1]["status"] == "completed", errors
    requests = [row for row in resumed if row["type"] == "llm_request"]
    assert requests[0]["id"] == "request-1" and requests[0]["round"] == 2
    assert resumed[-1]["summary"]["calls"] == 4
    assert all("max_completion_tokens" not in row["payload"] for row in requests)
    assert read_json(project / "results/dsh-plugin/synthetic-run/run/state.json")["methods"]["llm"]["rounds"][0] == saved


def test_worker_stopped_status_and_config_mismatch(fake_project):
    project, _ = fake_project
    message = configured_start(project, max_assays=20)
    code, values, errors = run_worker(project, message)
    assert code == 0 and values[-1]["status"] == "stopped", errors
    assert values[-1]["summary"]["stop_reason"] == "max_assays"
    assert values[-1]["summary"]["completed_rounds"] == 1
    with pytest.raises(worker.ProtocolAbort, match="must match config"):
        worker.validate_start({**message, "seed": 99})


def test_worker_unknown_request_outcome_is_never_resent(fake_project):
    project, _ = fake_project
    message = configured_start(project)
    message["method"] = "llm"
    code, first, _ = run_worker(project, message, lambda request, project: None)
    assert code != 0 and sum(v["type"] == "llm_request" for v in first) == 1
    code, second, _ = run_worker(project, {**message, "resume": True})
    assert code != 0 and second[-1]["code"] == "UNKNOWN_REQUEST_OUTCOME"
    assert not any(v["type"] == "llm_request" for v in second)
