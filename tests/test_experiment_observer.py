"""Process observers use artificial data and never become candidate-generation tools."""
from copy import deepcopy
import pytest
import gb1.experiment as experiment
from test_workflow import setup, SyntheticProposer


def test_observer_reports_real_tools_in_order_without_hidden_records(setup):
    events = []
    state = experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                                     methods=["llm"], proposer=SyntheticProposer(), observer=events.append)
    assert state["methods"]["llm"]["status"] == "completed"
    for number in range(1, 4):
        own = [value for value in events if value["round"] == number]
        assert [(value["tool"], value["kind"]) for value in own] == [(tool, kind) for tool in ("fit", "validate", "predict", "select", "oracle", "feedback") for kind in ("tool_start", "tool_end")]
        for begin, end in zip(own[::2], own[1::2]):
            assert begin["id"] == end["id"] and end["ok"] is True and "ok" not in begin
            assert set(begin) == {"id", "kind", "tool", "round", "counts"}
            assert all(type(count) is int and count >= 0 for count in end["counts"].values())
        assert own[-1]["counts"]["known_count"] == setup.config.train_size + number * setup.config.k
    assert len({value["id"] for value in events[::2]}) == 18
    assert not any("variant" in key or "fitness" in key for value in events for key in value["counts"])
    previous = len(events)
    experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                             methods=["llm"], proposer=SyntheticProposer(), observer=events.append)
    assert len(events) == previous


def test_saved_response_revalidation_does_not_claim_new_model_or_validation_execution(setup):
    experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=SyntheticProposer())
    path = setup.output / "state.json"
    saved = experiment.read_json(path)
    method = saved["methods"]["llm"]
    method["rounds"] = method["rounds"][:1]
    method["rounds"][0]["stage"] = "started"
    method["status"] = "failed"
    experiment.write_json(path, saved)
    events = []
    proposer = SyntheticProposer()
    experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output, methods=["llm"], proposer=proposer, observer=events.append)
    assert len(proposer.contexts) == 2
    assert not any(value["round"] == 1 and value["tool"] == "validate" for value in events)


def test_observer_failure_is_safe_and_does_not_change_selected_candidates(setup, caplog):
    baseline = experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                                        methods=["llm"], proposer=SyntheticProposer())
    def broken(_):
        raise RuntimeError("SECRET_DIAGNOSTIC_MUST_NOT_LOG")
    observed = experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.root / "observed",
                                        methods=["llm"], proposer=SyntheticProposer(), observer=broken)
    for a, b in zip(baseline["methods"]["llm"]["rounds"], observed["methods"]["llm"]["rounds"]):
        assert a["recommendations"] == b["recommendations"]
    assert "GB1 process observer failed" in caplog.text and "SECRET_DIAGNOSTIC" not in caplog.text


def test_core_failure_remains_a_failure_and_tool_end_is_not_success(setup, monkeypatch):
    events = []
    monkeypatch.setattr(experiment, "fit_model", lambda *args: (_ for _ in ()).throw(ValueError("Synthetic fit failure")))
    state = experiment.run_experiment(setup.dataset, setup.selection, setup.config, setup.output,
                                     methods=["llm"], proposer=SyntheticProposer(), observer=events.append)
    assert state["methods"]["llm"]["status"] == "failed"
    assert [(event["kind"], event.get("ok")) for event in events] == [("tool_start", None), ("tool_end", False)]
