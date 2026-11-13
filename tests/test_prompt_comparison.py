"""Prompt comparison tests use artificial scores and stored in-process model doubles."""
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import pytest
from gb1.cli import main
from gb1.compare_prompts import compare_prompt_runs, _plot, _summarize_records
from gb1.experiment import run_experiment
from gb1.storage import read_json, write_json
from test_workflow import setup, SyntheticProposer


def test_numeric_comparison_preserves_inputs_and_confirms_same_data_model(setup):
    v1_dir, v2_dir = setup.root / "v1", setup.root / "v2"
    first = run_experiment(setup.dataset, setup.selection, setup.config, v1_dir,
                           methods=["llm", "knowledge"], proposer=SyntheticProposer())
    # Exercise compatibility with a real pre-profile state layout.
    first.pop("prompt_metadata")
    write_json(v1_dir / "state.json", first)
    run_experiment(setup.dataset, setup.selection, setup.config, v2_dir,
                   methods=["llm", "knowledge"], proposer=SyntheticProposer(), prompt_profile="v2")
    before = [(directory / "state.json").read_text(encoding="utf-8") for directory in (v1_dir, v2_dir)]
    output = setup.root / "comparison"
    result = compare_prompt_runs([v1_dir], [v2_dir], output)
    assert result["seeds"] == [11]
    check = result["metadata_checks"][0]
    assert check["seed"] == 11
    assert all(check[key] for key in ("same_config", "same_dataset_manifest", "same_selection", "same_llm_model_and_api_mode", "complete_method_round_pairs"))
    assert check["llm_api_identity"]["llm"]["model_requested"] == "synthetic-test"
    assert len(result["requests"]) == 12
    for row in result["methods"]:
        assert row["request_count"] == 3
        assert row["requested_count"] == row["raw_generated_count"] == row["accepted_valid_count"] == row["scored_count"] == 24
        assert row["total_tokens"] == 90
        assert row["missing_usage_count"] == 0
    for row in result["per_seed_differences"]:
        assert row["delta_mean_fitness"] == row["delta_best_new_fitness"] == 0
    assert before == [(directory / "state.json").read_text(encoding="utf-8") for directory in (v1_dir, v2_dir)]
    assert pd.read_csv(output / "prompt_comparison_requests.csv").shape[0] == 12
    assert (output / "prompt_comparison.png").stat().st_size > 1000
    assert main(["compare-prompts", "--v1-runs", str(v1_dir), "--v2-runs", str(v2_dir), "--output", str(output)]) == 0
    second = read_json(v2_dir / "state.json")
    second["selection"]["spec"]["family"] = "different-test-model"
    write_json(v2_dir / "state.json", second)
    with pytest.raises(ValueError, match="different frozen data"):
        compare_prompt_runs([v1_dir], [v2_dir], setup.root / "mismatched")
    assert not (setup.root / "mismatched").exists()


def test_comparison_reports_overbudget_and_underfilled_without_truncating_or_rerunning(setup):
    ids = sorted(setup.dataset.candidates.variant)
    v1_dir, v2_dir = setup.root / "v1", setup.root / "v2"
    class Over(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            body["proposals"] = [{**body["proposals"][0], "variant": v} for v in ids[:9]]
            response["text"] = json.dumps(body)
            return response
    class Under(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            body["proposals"] = body["proposals"][:4] if not context.get("accepted_variants") else []
            response["text"] = json.dumps(body)
            return response
    first = run_experiment(setup.dataset, setup.selection, setup.config, v1_dir, methods=["llm"], proposer=Over())
    second = run_experiment(setup.dataset, setup.selection, setup.config, v2_dir, methods=["llm"], proposer=Under(), prompt_profile="v2")
    baseline = _summarize_records(first["methods"]["llm"]["rounds"])
    candidate = _summarize_records(second["methods"]["llm"]["rounds"])
    assert baseline["raw_generated_count"] == 18 and baseline["accepted_valid_count"] == 0
    assert candidate["accepted_valid_count"] == 12
    assert candidate["request_status_counts"]["underfilled"] == 3
    with pytest.raises(ValueError, match="complete paired rounds"):
        compare_prompt_runs([v1_dir], [v2_dir], setup.root / "comparison")
    assert not (setup.root / "comparison").exists()


@pytest.mark.parametrize("seeds, caption", [
    ((11, 22, 33), "3 seeds; mean ± sample SD across seeds."),
    ((11,), "1 seed; per-seed values (sample SD unavailable)."),
])
def test_comparison_plot_labels_round_maximum_and_seed_uncertainty(tmp_path, monkeypatch, seeds, caption):
    from matplotlib.figure import Figure
    original = Figure.savefig
    captured = []
    def inspect_and_save(figure, *args, **kwargs):
        captured.append({"titles": [axis.get_title() for axis in figure.axes],
                         "ylabels": [axis.get_ylabel() for axis in figure.axes],
                         "texts": [text.get_text() for text in figure.texts]})
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    rows = [{"seed": seed, "profile": "v2", "method": "llm", "round": 1, "stage": "evaluated",
             "mean_fitness": float(index), "best_new_fitness": float(index + 2)}
            for index, seed in enumerate(seeds)]
    _plot(rows, tmp_path)
    assert captured[0]["titles"] == ["Round mean measured fitness", "Best measured fitness within each round"]
    assert captured[0]["ylabels"] == ["S1 fitness (WT=1)", "S1 fitness (WT=1)"]
    assert captured[0]["texts"] == [f"{caption}\nSource: saved run records, 2026-11-09."]
    assert (tmp_path / "prompt_comparison.png").stat().st_size > 1000
