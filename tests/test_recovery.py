"""Failure recovery and free-design tests use synthetic data and API responses only."""
import json
from pathlib import Path
import pandas as pd
import pytest
from gb1.agent import collect_proposals, OpenAIProposer
from gb1.experiment import run_experiment
from gb1.storage import read_json, write_json
from test_workflow import setup, SyntheticProposer


def test_llm_can_design_an_eligible_combination_outside_examples(setup):
    all_candidates = list(setup.dataset.candidates.variant)
    class FreeDesigner(SyntheticProposer):
        def generate(self, context):
            context = json.loads(json.dumps(context))
            examples = set(context["candidate_examples"])
            known_top = {row["variant"] for row in context["known_data"]["top_variants"]}
            context["candidate_examples"] = [v for v in all_candidates if v not in examples and v not in known_top]
            return super().generate(context)
    # One round is sufficient to check free design; later collisions are separately validated.
    config = setup.config.model_copy(update={"rounds": 1})
    state = run_experiment(setup.dataset, setup.selection, config, setup.output, methods=["llm"], proposer=FreeDesigner())
    record = state["methods"]["llm"]["rounds"][0]
    assert state["methods"]["llm"]["status"] == "completed"
    assert not set(record["cohort"]) & {row["variant"] for row in record["proposals"]}


def test_partial_valid_output_is_preserved_and_not_filled_by_rules(setup):
    class Partial(SyntheticProposer):
        def generate(self, context):
            response = super().generate(context)
            body = json.loads(response["text"])
            if not context.get("accepted_variants"):
                body["proposals"] = body["proposals"][:4]
            else:
                body["proposals"] = []
            response["text"] = json.dumps(body)
            return response
    proposer = Partial()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer)
    assert state["methods"]["llm"]["status"] == "completed"
    for record in state["methods"]["llm"]["rounds"]:
        assert record["valid_proposal_count"] == 4
        assert record["model_score_count"] == 4
        assert record["metrics"]["scoring_budget_utilization"] == 0.5
        assert len(record["recommendations"]) == 3
        assert len(record["attempts"]) == 2


def test_recent_round_feedback_reaches_only_its_own_method(setup):
    proposer = SyntheticProposer()
    state = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm", "knowledge"], proposer=proposer)
    for context in proposer.contexts:
        method = "knowledge" if "knowledge" in context else "llm"
        if context["round"] == 1:
            assert context["recent_round_feedback"] == []
        else:
            expected = state["methods"][method]["rounds"][context["round"] - 2]["recommendations"]
            assert context["recent_round_feedback"] == expected
            assert all("actual_fitness" in row and "predicted_fitness" in row for row in expected)


def test_unknown_transport_outcome_requires_explicit_retry(setup):
    class Timeout(SyntheticProposer):
        def generate(self, context):
            raise TimeoutError("synthetic timeout")
    first = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=Timeout())
    assert first["methods"]["llm"]["rounds"][0]["attempts"][0]["status"] == "started"
    succeeding = SyntheticProposer()
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=succeeding)
    assert not succeeding.contexts
    assert "unknown outcome" in resumed["methods"]["llm"]["failures"][-1]["message"]
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=succeeding, retry_uncertain=True)
    assert resumed["methods"]["llm"]["status"] == "completed"
    assert len(resumed["methods"]["llm"]["rounds"][0]["attempts"]) == 2


def test_received_response_is_reused_after_process_interruption(setup):
    proposer = SyntheticProposer()
    run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer)
    state = read_json(setup.output / "state.json")
    method = state["methods"]["llm"]
    first = method["rounds"][0]
    method["rounds"] = [first]
    method["status"] = "failed"
    first["stage"] = "started"
    first["attempts"][0]["status"] = "received"
    write_json(setup.output / "state.json", state)
    fresh = SyntheticProposer()
    resumed = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=fresh)
    assert resumed["methods"]["llm"]["status"] == "completed"
    assert len(fresh.contexts) == 2  # Only the remaining two rounds call the API.


def test_oracle_label_changes_cannot_change_preassay_selection(setup):
    first = run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["model"])
    oracle_path = setup.dataset.directory / "hidden/oracle.csv"
    oracle = pd.read_csv(oracle_path)
    oracle["fitness"] = 1000 - oracle.fitness
    oracle.to_csv(oracle_path, index=False)
    second = run_experiment(setup.dataset, setup.selection, setup.config, setup.root / "run2", methods=["model"])
    a = first["methods"]["model"]["rounds"][0]["recommendations"]
    b = second["methods"]["model"]["rounds"][0]["recommendations"]
    assert [row["variant"] for row in a] == [row["variant"] for row in b]
    assert [row["predicted_fitness"] for row in a] == [row["predicted_fitness"] for row in b]
    assert [row["actual_fitness"] for row in a] != [row["actual_fitness"] for row in b]


def test_sdk_loads_only_project_dotenv_without_overriding_environment(setup, monkeypatch):
    calls = []
    import dotenv
    import openai
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "synthetic-test-model")
    monkeypatch.setenv("OPENAI_API_MODE", "chat")
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: object())
    OpenAIProposer(setup.config)
    assert len(calls) == 1
    assert calls[0]["override"] is False
    assert calls[0]["dotenv_path"] == Path(__file__).resolve().parents[1] / ".env"
