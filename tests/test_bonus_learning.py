"""Bonus learning tests use artificial sequences and scores, never project API credentials."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import importlib.util
import joblib
import numpy as np
import pandas as pd
import pytest
import bonus.active_learning as learning
from bonus.config import BonusConfig
from bonus.data import BonusDataset, split_bonus
from bonus.uncertainty import (predict_distribution, finite_quantile, calibrate, interval_bounds,
                               evaluate_predictions)
from gb1.data import clean_table, one_hot, HiddenOracle
from gb1.storage import read_json, write_json


@pytest.fixture
def synthetic(tmp_path):
    reference = list("A" * 56)
    for position, residue in zip((39, 40, 41, 54), "VDGV"):
        reference[position - 1] = residue
    reference = "".join(reference)
    variants = ["".join(row) for row in product(*[[native, "A", "C", "E", "L"] for native in "VDGV"])]
    fitness = [float(sum((index + 1) * (ord(residue) % 11) for index, residue in enumerate(v)) / 10) for v in variants]
    fitness[variants.index("VDGV")] = 1.0
    source = tmp_path / "artificial.csv"
    pd.DataFrame({"variant": variants, "fitness": fitness}).to_csv(source, index=False)
    table, _ = clean_table(pd.read_csv(source), "variant", "fitness", reference)
    config = BonusConfig(train_size=24, initial_single_count=4, calibration_size=12, test_size=20,
                         reserve_single_count=6, hd_pool_size=6, rounds=2, k=2, rf_estimators=6,
                         n_jobs=1, prediction_batch_size=64, save_pool_scores=True)
    return {"root": tmp_path, "reference": reference, "source": source, "table": table, "config": config,
            "data": tmp_path / "data", "output": tmp_path / "output"}


@pytest.fixture
def prepared(synthetic):
    learning.prepare_experiment(synthetic["source"], synthetic["data"], synthetic["output"], synthetic["config"],
                                synthetic["reference"], variant_column="variant", fitness_column="fitness")
    synthetic["dataset"] = BonusDataset(synthetic["data"])
    return synthetic


def test_split_is_disjoint_exact_and_uses_only_ids_hd_not_fitness(synthetic):
    roles, groups, metadata = split_bonus(synthetic["table"], synthetic["config"])
    modified = synthetic["table"].copy()
    modified["fitness"] = np.arange(len(modified), dtype=float)[::-1] * 999
    repeated, repeated_groups, _ = split_bonus(modified, synthetic["config"])
    for key in roles:
        assert list(roles[key].variant) == list(repeated[key].variant)
    assert groups == repeated_groups
    training = roles["train"]
    assert len(training) == 24
    assert int((training.mutation_count == 0).sum()) == 1
    assert int((training.mutation_count == 1).sum()) == 4
    assert int((training.mutation_count >= 2).sum()) == 19
    sets = [set(rows.variant) for rows in roles.values()]
    assert all(not sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4))
    assert sum(map(len, sets)) == len(synthetic["table"])
    assert metadata["common_heldout_size"] == 32
    assert len(roles["calibration"]) == 12 and len(roles["test"]) == 20
    assert metadata["role_hd_counts"]["pool"]["1"] >= 6
    assert all(len(values) == 6 for values in groups.values())


def test_single_reserve_resampling_has_a_fixed_failure_limit(synthetic):
    config = synthetic["config"].model_copy(update={"test_size": 578, "reserve_single_count": 12, "holdout_max_attempts": 1})
    with pytest.raises(ValueError, match="resampling limit"):
        split_bonus(synthetic["table"], config)


def test_model_interface_and_disagreement_match_individual_tree_predictions(prepared):
    dataset = prepared["dataset"]
    assert "fitness" not in dataset.candidates
    assert dataset.manifest["audit"]["reference"] == prepared["reference"]
    assert dataset.manifest["config"]["seed"] == 44
    assert "1.159" in dataset.manifest["source_calibration_note"]
    assert (prepared["output"] / "model.joblib").exists()
    model = joblib.load(prepared["output"] / "initial_model.joblib")
    variants = list(dataset.candidates.variant[:7])
    encoded = one_hot(variants)
    mean, std = predict_distribution(model, variants, batch_size=3)
    members = np.stack([tree.predict(encoded) for tree in model.estimators_])
    np.testing.assert_allclose(mean, model.predict(encoded))
    np.testing.assert_allclose(std, members.std(axis=0, ddof=1))
    assert model.n_features_in_ == 80
    with pytest.raises(ValueError, match="immutable"):
        learning.prepare_experiment(prepared["source"], prepared["data"], prepared["root"] / "other-output", prepared["config"],
                                    prepared["reference"], variant_column="variant", fitness_column="fitness")


def test_quantiles_two_interval_types_and_finite_sample_rejection(synthetic):
    assert finite_quantile(np.arange(10), 0.1) == (9.0, 10)
    with pytest.raises(ValueError, match="too small"):
        finite_quantile([1, 2], 0.1)
    config = synthetic["config"]
    calibrated = calibrate(np.arange(10), np.zeros(10), np.ones(10), config)
    assert calibrated["q_abs"] == calibrated["q_scaled"] == 9
    assert calibrated["order_index"] == 10
    zero_std = calibrate(np.ones(10), np.zeros(10), np.zeros(10), config)
    assert zero_std["q_scaled"] == 1 / config.std_floor
    bounds = interval_bounds([1.0, 2.0], [0.0, 1.0], zero_std)
    assert (bounds["abs_lower"] >= 0).all() and (bounds["scaled_lower"] >= 0).all()
    empty = evaluate_predictions([], [], [], {name: np.array([]) for name in bounds})
    assert empty["n"] == 0 and all(value is None for key, value in empty.items() if key != "n")
    with pytest.raises(ValueError, match="too small"):
        BonusConfig(calibration_size=2)


def test_all_arms_have_independent_feedback_and_equal_group_budgets(prepared, monkeypatch):
    state = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"])
    assert len(state["arms"]) == 12
    assert all(arm["status"] == "completed" for arm in state["arms"].values())
    initial_ids = set(prepared["dataset"].train.variant)
    for name, arm in state["arms"].items():
        prior = set()
        for number, record in enumerate(arm["rounds"], 1):
            assert set(record["training_variants"]) == initial_ids | prior
            assert record["training_count"] == 24 + (number - 1) * 2
            chosen = {row["variant"] for row in record["recommendations"]}
            assert len(chosen) == 2 and not chosen & prior
            assert all("actual_fitness" in row and "abs_lower" in row and "scaled_upper" in row for row in record["recommendations"])
            expected_count = len(prepared["dataset"].candidates) - len(prior) if arm["hd"] is None else 6 - len(prior)
            assert record["scored_count"] == expected_count
            if arm["hd"] is not None:
                assert all(row["mutation_count"] == arm["hd"] for row in record["recommendations"])
                known = prepared["dataset"].train
                correct_group_best = known[known.mutation_count == arm["hd"]].fitness.max()
                if number == 1:
                    assert record["metrics"]["prior_group_best"] == correct_group_best
            prior.update(chosen)
            assert record["calibration"]["std_ddof"] == 1
            assert "Empirical coverage" in record["metrics"]["selected_coverage_scope"]
        assert arm["final_training_count"] == 28
        assert sum(group["n"] for group in arm["final_evaluation"]["by_hd"].values()) == 20
    before = deepcopy(state)
    resumed = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"])
    assert resumed == before
    from matplotlib.figure import Figure
    original_save = Figure.savefig
    figures = {}
    def capture(figure, path, *args, **kwargs):
        figures[Path(path).name] = {"texts": [text.get_text() for text in figure.texts],
                                  "ticks": [axis.get_xticks().tolist() for axis in figure.axes],
                                  "labels": [axis.get_ylabel() for axis in figure.axes],
                                  "legends": [[text.get_text() for text in axis.get_legend().get_texts()] if axis.get_legend() else [] for axis in figure.axes]}
        return original_save(figure, path, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture)
    result = learning.summarize([prepared["output"]], prepared["root"] / "summary")
    assert "Source: saved S1 run results." in figures["learning_curves.png"]["texts"][0]
    assert "1 seed" in figures["learning_curves.png"]["texts"][0]
    assert figures["learning_curves.png"]["ticks"] == [[1, 2], [1, 2]]
    assert "greedy" in figures["learning_curves.png"]["legends"][0]
    assert all("(" not in label for label in figures["learning_curves.png"]["legends"][0])
    assert figures["heldout_coverage.png"]["labels"][1] == "S1 fitness units"
    assert result["complete"] and result["recommendations"] == 48 and result["rounds"] == 24
    assert (prepared["root"] / "summary/learning_curves.png").stat().st_size > 1000
    assert (prepared["root"] / "summary/heldout_coverage.png").stat().st_size > 1000
    initial = pd.read_csv(prepared["root"] / "summary/initial_heldout_coverage.csv")
    final = pd.read_csv(prepared["root"] / "summary/final_heldout_coverage.csv")
    assert initial.loc[initial.test_hd == "overall", "n"].iloc[0] == 20
    assert (final.loc[final.test_hd == "overall", "n"] == 20).all()


def test_changing_calibration_and_test_labels_cannot_change_acquisition(prepared):
    import shutil
    clone_data, clone_output = prepared["root"] / "clone-data", prepared["root"] / "clone-output"
    shutil.copytree(prepared["data"], clone_data)
    shutil.copytree(prepared["output"], clone_output)
    for name in ("calibration", "test"):
        path = clone_data / f"hidden/{name}.csv"
        table = pd.read_csv(path)
        table["fitness"] = 1000 + np.arange(len(table)) * 17
        table.to_csv(path, index=False)
    arms = ["global/greedy", "global/std_only", "global/mean_plus_std", "hd1/mean_plus_std"]
    first = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], arms)
    second = learning.run_experiment(clone_data, clone_output, prepared["config"], arms)
    for name in arms:
        for a, b in zip(first["arms"][name]["rounds"], second["arms"][name]["rounds"]):
            assert [row["variant"] for row in a["recommendations"]] == [row["variant"] for row in b["recommendations"]]
            assert [row["predicted_fitness"] for row in a["recommendations"]] == [row["predicted_fitness"] for row in b["recommendations"]]
            assert a["calibration"]["q_abs"] != b["calibration"]["q_abs"]


def test_round_intervals_are_saved_before_oracle_and_recover_without_rescoring(prepared, monkeypatch):
    original = HiddenOracle.evaluate
    def interrupted(oracle, variants):
        saved = read_json(prepared["output"] / "state.json")["arms"]["global/greedy"]["rounds"][0]
        assert saved["stage"] == "intervals_ready"
        assert len(saved["recommendations"]) == 2
        assert all("scaled_lower" in row and "actual_fitness" not in row for row in saved["recommendations"])
        raise RuntimeError("Synthetic oracle interruption")
    monkeypatch.setattr(HiddenOracle, "evaluate", interrupted)
    first = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], ["global/greedy"])
    assert first["arms"]["global/greedy"]["status"] == "failed"
    monkeypatch.setattr(HiddenOracle, "evaluate", original)
    resumed = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], ["global/greedy"])
    assert resumed["arms"]["global/greedy"]["status"] == "completed"
    assert len(resumed["arms"]["global/greedy"]["rounds"]) == 2


def test_heldout_reader_is_called_after_selection_is_committed(prepared, monkeypatch):
    original = learning.evaluate_frozen
    def checked(model, dataset, config):
        state = read_json(prepared["output"] / "state.json")
        rounds = state["arms"]["global/random"]["rounds"]
        assert any(record["stage"] == "selected" for record in rounds) or all(record["stage"] == "evaluated" for record in rounds)
        return original(model, dataset, config)
    monkeypatch.setattr(learning, "evaluate_frozen", checked)
    state = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], ["global/random"])
    assert state["arms"]["global/random"]["status"] == "completed"


def test_config_changes_and_duplicate_seed_summaries_are_rejected(prepared):
    changed = prepared["config"].model_copy(update={"kappa": 2.0})
    with pytest.raises(ValueError, match="configuration or dataset changed"):
        learning.run_experiment(prepared["data"], prepared["output"], changed)
    learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], ["global/random"])
    with pytest.raises(ValueError, match="distinct seed"):
        learning.summarize([prepared["output"], prepared["output"]], prepared["root"] / "bad-summary")


def test_cli_artificial_end_to_end_and_no_environment_access(synthetic, monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "run_bonus.py"
    spec = importlib.util.spec_from_file_location("run_bonus_test", module_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config_path = synthetic["root"] / "synthetic-config.json"
    config = synthetic["config"].model_copy(update={"global_strategies": ["random"], "hd_groups": [1], "hd_strategies": ["mean_plus_std"]})
    write_json(config_path, config.model_dump())
    assert cli.main(["prepare", "--source", str(synthetic["source"]), "--data", str(synthetic["data"]), "--output", str(synthetic["output"]),
                     "--config", str(config_path), "--reference", synthetic["reference"], "--variant-column", "variant", "--fitness-column", "fitness"]) == 0
    assert cli.main(["run", "--data", str(synthetic["data"]), "--output", str(synthetic["output"]), "--config", str(config_path)]) == 0
    assert cli.main(["summarize", "--outputs", str(synthetic["output"]), "--output", str(synthetic["root"] / "summary")]) == 0


def test_failed_arms_remain_visible_and_empty_tables_are_readable(prepared, monkeypatch):
    monkeypatch.setattr(learning, "select_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Synthetic acquisition failure")))
    state = learning.run_experiment(prepared["data"], prepared["output"], prepared["config"], ["global/random"])
    assert state["arms"]["global/random"]["status"] == "failed"
    result = learning.summarize([prepared["output"]], prepared["root"] / "failed-summary")
    assert result["complete"] is False and result["rounds"] == result["recommendations"] == 0
    assert any(row["status"] == "failed" for row in result["status"])
    assert pd.read_csv(prepared["root"] / "failed-summary/round_metrics.csv").empty
    assert pd.read_csv(prepared["root"] / "failed-summary/final_heldout_coverage.csv").empty


def test_changed_known_measurements_and_existing_foreign_outputs_are_rejected(prepared):
    training_path = prepared["data"] / "train.csv"
    training = pd.read_csv(training_path)
    training.loc[0, "fitness"] += 1
    training.to_csv(training_path, index=False)
    with pytest.raises(ValueError, match="training measurements"):
        learning.run_experiment(prepared["data"], prepared["output"], prepared["config"])
    foreign = prepared["root"] / "foreign-summary"
    foreign.mkdir()
    write_json(foreign / "summary.json", {"study": "another-project"})
    before = (foreign / "summary.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="bonus summary"):
        learning.summarize([prepared["output"]], foreign)
    assert (foreign / "summary.json").read_text(encoding="utf-8") == before


def test_nonnested_directories_and_invalid_disagreement_config(synthetic):
    with pytest.raises(ValueError, match="nonnested"):
        learning.prepare_experiment(synthetic["source"], synthetic["data"], synthetic["data"] / "output", synthetic["config"], synthetic["reference"])
    with pytest.raises(ValueError):
        BonusConfig(std_ddof=0)
    with pytest.raises(ValueError, match="finite nonnegative"):
        finite_quantile([0, float("nan")], 0.1)
