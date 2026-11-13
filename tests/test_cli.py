"""CLI and aggregation checks run only artificial sequences and scores."""
from pathlib import Path
import pandas as pd
import pytest
from gb1.aggregate import aggregate_runs
from gb1.cli import main
from gb1.data import PublicDataset, prepare
from gb1.experiment import run_experiment
from gb1.models import select_model
from gb1.storage import write_json
from test_workflow import setup


def test_offline_cli_full_workflow_without_loading_project_credentials(setup, monkeypatch):
    import gb1.cli as cli
    def no_client(*args, **kwargs):
        pytest.fail("Offline CLI must not construct an API client")
    monkeypatch.setattr(cli, "OpenAIProposer", no_client)
    config_path = setup.root / "synthetic-config.json"
    write_json(config_path, setup.config.model_dump())
    directory, selected, run = setup.root / "cli-data", setup.root / "cli-selection", setup.root / "cli-run"
    assert main(["prepare", "--source", str(setup.source), "--data", str(directory), "--config", str(config_path),
                 "--reference", setup.reference, "--variant-column", "variant", "--fitness-column", "fitness"]) == 0
    assert main(["select", "--data", str(directory), "--config", str(config_path), "--output", str(selected)]) == 0
    common = ["--data", str(directory), "--config", str(config_path), "--output", str(run),
              "--selection", str(selected / "selection.json"), "--methods", "random", "model"]
    assert main(["run", *common]) == 0
    assert main(["status", "--output", str(run)]) == 0
    assert main(["finalize", *common]) == 0
    assert main(["plot", "--output", str(run)]) == 0
    recommendations = pd.read_csv(run / "recommendations.csv")
    assert len(recommendations) == 18
    assert recommendations.seed.unique().tolist() == [11]
    assert all(recommendations.sequence.str.len() == 56)


def test_aggregate_independent_seed_counts_and_missing_methods(setup):
    directories = []
    for seed in (11, 22, 33):
        config = setup.config.model_copy(update={"seed": seed})
        data_path = setup.root / f"data-{seed}"
        prepare(setup.source, data_path, config, setup.reference, "variant", "fitness")
        dataset = PublicDataset(data_path)
        selection = select_model(dataset, config, setup.root / f"selection-{seed}")
        directory = setup.root / f"run-{seed}"
        run_experiment(dataset, selection, config, directory, methods=["random", "model"])
        directories.append(directory)
    output = setup.root / "aggregate"
    summary = aggregate_runs(directories, output)
    assert summary["seeds"] == [11, 22, 33]
    frame = pd.read_csv(output / "aggregate_round_metrics.csv")
    assert len(frame) == 6
    assert (frame.mean_fitness_count == 3).all()
    assert (output / "aggregate_fitness_progress.png").stat().st_size > 1000
    statuses = pd.read_csv(output / "all_seed_status.csv")
    assert (statuses[statuses.method == "llm"].status == "pending").all()
    with pytest.raises(ValueError, match="distinct"):
        aggregate_runs([directories[0], directories[0]], setup.root / "bad-aggregate")
