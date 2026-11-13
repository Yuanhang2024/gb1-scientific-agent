"""All measurements and model responses in this module are synthetic test data."""
from itertools import product
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from gb1.agent import OpenAIProposer, MissingCredentials, make_context, validate_proposals
from gb1.cli import main, plot_results
from gb1.data import PublicDataset, HiddenOracle, clean_table, describe_variant, full_sequence, one_hot, prepare, split_table
from gb1.experiment import known_for, run_experiment, finalize
from gb1.models import select_model, load_selection, regression_metrics
from gb1.schema import ExperimentConfig, ProposalBatch
from gb1.storage import read_json, write_json


@pytest.fixture
def setup(tmp_path):
    reference = list("A" * 56)
    for position, aa in zip((39, 40, 41, 54), "VDGV"):
        reference[position - 1] = aa
    reference = "".join(reference)
    # Deliberately artificial sequence and fitness function, never used as experimental results.
    variants = ["".join(x) for x in product("ACDE", repeat=4)]
    source = tmp_path / "synthetic.csv"
    pd.DataFrame({"variant": variants, "fitness": [float(sum("ACDE".index(c) * (i + 1) for i, c in enumerate(v))) for v in variants]}).to_csv(source, index=False)
    knowledge = tmp_path / "test-knowledge.json"
    write_json(knowledge, {"rules": [{"id": "test-only-rule", "scope": "synthetic test", "condition": "always",
                                    "action": "test fixture", "source": "synthetic test", "explanation": "Not a scientific rule"}]})
    config = ExperimentConfig(train_size=24, validation_size=16, pool_size=80, test_size=16,
                              score_budget=8, proposal_count=8, k=3, tree_estimators=5,
                              tree_min_samples_leaf=[1], ridge_alphas=[1.0], n_jobs=1,
                              knowledge_path=str(knowledge))
    data = tmp_path / "prepared"
    prepare(source, data, config, reference, "variant", "fitness")
    dataset = PublicDataset(data)
    selection = select_model(dataset, config, tmp_path / "selection")
    return SimpleNamespace(root=tmp_path, reference=reference, source=source, config=config,
                           dataset=dataset, selection=selection, output=tmp_path / "run")


class SyntheticProposer:
    """In-process API stand-in only for tests, never connected to the production CLI."""
    def __init__(self):
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        ids = [context["knowledge"]["rules"][0]["id"]] if "knowledge" in context else []
        text = json.dumps({"analysis": "Synthetic test response", "proposals": [
            {"variant": v, "rationale": "Synthetic test rationale", "hypothesis": "Synthetic test hypothesis", "knowledge_rule_ids": ids}
            for v in [v for v in context["candidate_examples"] if v not in context.get("accepted_variants", [])][:context["proposal_count"]]]})
        return {"text": text, "complete": True, "usage": {"input_tokens": 10, "output_tokens": 20},
                "model_requested": "synthetic-test", "model_returned": "synthetic-test", "api_mode": "test"}


def test_sequence_and_encoding(setup):
    assert full_sequence("VDGV", setup.reference) == setup.reference
    altered = describe_variant("ADGA", setup.reference)
    assert altered["mutations"] == "V39A:V54A"
    assert len(altered["sequence"]) == 56
    matrix = one_hot(["VDGV", "ADGA"])
    assert matrix.shape == (2, 80)
    assert np.all(matrix.sum(axis=1) == 4)
    for invalid in ("VDG", "VDG*", "AAAAA", "aaaa"):
        with pytest.raises(ValueError):
            one_hot([invalid])


def test_cleaning_missing_zero_duplicates_and_conflict(setup):
    table = pd.DataFrame({"v": ["VDGV", "AAAA", "AAAA", "AAXA", "CCCC", "DDDD"], "f": [0, 2, 2, 3, np.nan, -1]})
    cleaned, audit = clean_table(table, "v", "f", setup.reference)
    assert set(cleaned.variant) == {"VDGV", "AAAA"}
    assert audit["zero_fitness_rows"] == 1
    assert audit["duplicate_rows_removed"] == 1
    table.loc[2, "f"] = 4
    with pytest.raises(ValueError, match="conflicting"):
        clean_table(table, "v", "f", setup.reference)


def test_disjoint_reproducible_splits_and_hidden_candidate_labels(setup):
    directory = setup.dataset.directory
    tables = [pd.read_csv(directory / name) for name in ("train.csv", "validation.csv", "candidates.csv", "hidden/test.csv")]
    roles = [set(table.variant) for table in tables]
    for index, role in enumerate(roles):
        assert all(not role & other for other in roles[index + 1:])
    assert "fitness" not in tables[2]
    clean, _ = clean_table(pd.read_csv(setup.source), "variant", "fitness", setup.reference)
    a, b = split_table(clean, setup.config), split_table(clean, setup.config)
    for role in a:
        pd.testing.assert_frame_equal(a[role], b[role])
    with pytest.raises(ValueError, match="immutable"):
        prepare(setup.source, directory, setup.config, setup.reference, "variant", "fitness")


def test_prepare_reads_synthetic_xlsx_with_openpyxl(setup, monkeypatch):
    source = setup.root / "synthetic.xlsx"
    pd.read_csv(setup.source).to_excel(source, sheet_name="Synthetic", index=False, engine="openpyxl")
    original = pd.read_excel
    seen = []
    def checked(*args, **kwargs):
        seen.append(kwargs.get("engine"))
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_excel", checked)
    report = prepare(source, setup.root / "xlsx-data", setup.config, setup.reference, "variant", "fitness", sheet_name="Synthetic")
    assert seen == ["openpyxl"]
    assert report["valid_rows"] == 256


def test_model_comparison_includes_all_families_and_never_opens_test(setup, monkeypatch):
    original = pd.read_csv
    def guarded(path, *args, **kwargs):
        assert Path(path).name not in ("oracle.csv", "test.csv")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded)
    selection = select_model(setup.dataset, setup.config, setup.root / "selection2")
    assert {row["family"] for row in selection["comparison"]} == {"ridge", "random_forest", "extra_trees"}
    assert "spearman" in selection["comparison"][0]
    assert load_selection(setup.root / "selection2/selection.json", setup.dataset, setup.config)["spec"] == selection["spec"]


def test_four_methods_three_rounds_isolated_histories_and_resume(setup):
    proposer = SyntheticProposer()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, proposer=proposer)
    assert all(method["status"] == "completed" for method in state["methods"].values())
    assert len(proposer.contexts) == 6
    validation = set(pd.read_csv(setup.dataset.directory / "validation.csv").variant)
    for method_state in state["methods"].values():
        selected = []
        for number, record in enumerate(method_state["rounds"], 1):
            assert len(record["recommendations"]) == setup.config.k
            assert record["training_count"] == setup.config.train_size + (number - 1) * setup.config.k
            assert set(record["training_variants"]) == set(setup.dataset.train.variant) | set(selected)
            assert not validation & set(record["training_variants"])
            selected += [row["variant"] for row in record["recommendations"]]
            assert record["model_score_count"] == setup.config.score_budget
            assert all(len(row["sequence"]) == 56 for row in record["recommendations"])
        assert len(set(selected)) == 9
    snapshot = json.loads(json.dumps(state))
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, proposer=proposer)
    assert resumed == snapshot
    assert len(proposer.contexts) == 6
    for context in proposer.contexts:
        assert "fitness" not in context["candidate_examples"]
        assert "oracle" not in json.dumps(context)
        assert "test.csv" not in json.dumps(context)
        assert all(isinstance(variant, str) and len(variant) == 4 for variant in context["candidate_examples"])
    assert not (setup.output / "final_test_metrics.json").exists()
    result = finalize(setup.dataset, setup.selection, setup.config, setup.output, list(state["methods"]))
    assert set(result) == set(state["methods"])
    assert all(item["training_count"] == 33 for item in result.values())
    assert Path(plot_results(setup.output)).stat().st_size > 1000


def test_offline_then_llm_resume_preserves_completed_results(setup):
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output)
    assert state["methods"]["random"]["status"] == "completed"
    assert state["methods"]["model"]["status"] == "completed"
    assert state["methods"]["llm"]["status"] == "blocked"
    assert state["methods"]["knowledge"]["status"] == "blocked"
    old = state["methods"]["model"]["rounds"]
    proposer = SyntheticProposer()
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, proposer=proposer)
    assert resumed["methods"]["model"]["rounds"] == old
    assert len(proposer.contexts) == 6


def test_selected_checkpoint_resumes_without_another_llm_request(setup, monkeypatch):
    proposer = SyntheticProposer()
    original = HiddenOracle.evaluate
    def fail_once(self, variants):
        raise RuntimeError("Synthetic oracle interruption")
    monkeypatch.setattr(HiddenOracle, "evaluate", fail_once)
    first = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer)
    assert first["methods"]["llm"]["rounds"][0]["stage"] == "selected"
    assert len(proposer.contexts) == 1
    monkeypatch.setattr(HiddenOracle, "evaluate", original)
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer)
    assert resumed["methods"]["llm"]["status"] == "completed"
    assert len(proposer.contexts) == 3


def test_invalid_llm_candidates_never_reach_oracle_and_consume_budget(setup, monkeypatch):
    class Invalid(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            for proposal in body["proposals"]:
                proposal["variant"] = "XXXX"
            response["text"] = json.dumps(body)
            return response
    def forbidden(*args):
        pytest.fail("Invalid proposals must not reach the oracle")
    monkeypatch.setattr(HiddenOracle, "evaluate", forbidden)
    proposer = Invalid()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer)
    assert len(proposer.contexts) == setup.config.llm_max_calls_per_round
    assert state["methods"]["llm"]["status"] == "failed"
    assert all(a["status"] == "invalid" for a in state["methods"]["llm"]["rounds"][0]["attempts"])


def test_strict_candidate_validation_and_credential_status(setup):
    with pytest.raises(MissingCredentials):
        OpenAIProposer(setup.config, environ={})
    context = make_context(setup.dataset.train, list(setup.dataset.candidates.variant[:20]), setup.config, setup.reference, 1)
    response = SyntheticProposer().generate(context)
    parsed = validate_proposals(response["text"], context["candidate_examples"], setup.config)
    assert isinstance(parsed, ProposalBatch)
    body = json.loads(response["text"])
    body["proposals"][1] = body["proposals"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_proposals(json.dumps(body), context["candidate_examples"], setup.config)
    body = json.loads(response["text"])
    body["proposals"][0]["knowledge_rule_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="Unknown"):
        validate_proposals(json.dumps(body), context["candidate_examples"], setup.config)


@pytest.mark.parametrize("mode", ["responses", "chat"])
def test_openai_adapter_request_shapes_without_network(setup, mode):
    requests = []
    class Usage:
        def model_dump(self, **kwargs):
            return {"input_tokens": 12, "output_tokens": 4}
    def create(**kwargs):
        requests.append(kwargs)
        if mode == "responses":
            return SimpleNamespace(output_text='{}', status="completed", usage=Usage(), id="test-id", model="synthetic", _request_id="test-request")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}', refusal=None), finish_reason="stop")],
                               usage=Usage(), id="test-id", model="synthetic", _request_id="test-request")
    client = SimpleNamespace(responses=SimpleNamespace(create=create), chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    adapter = OpenAIProposer(setup.config, client=client, environ={"OPENAI_MODEL": "synthetic", "OPENAI_API_MODE": mode})
    response = adapter.generate({"synthetic": True})
    assert response["usage"]["input_tokens"] == 12
    request = requests[0]
    assert "tools" not in request
    if mode == "responses":
        assert request["text"]["format"]["type"] == "json_schema"
        assert request["store"] is False
    else:
        assert request["response_format"]["json_schema"]["strict"] is True


def test_finalization_refuses_incomplete_and_state_config_changes(setup):
    run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["random"])
    with pytest.raises(ValueError, match="finish all rounds"):
        finalize(setup.dataset, setup.selection, setup.config, setup.output, ["llm"])
    changed = setup.config.model_copy(update={"seed": 100})
    with pytest.raises(ValueError, match="different"):
        run_experiment(setup.dataset, setup.selection, changed, setup.output, methods=["random"])


def test_metrics_constant_arrays_are_explicit():
    metrics = regression_metrics([1, 1], [2, 2], 1)
    assert metrics["spearman"] is None and metrics["pearson"] is None
    assert metrics["mse"] == 1


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
