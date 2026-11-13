"""Worker bridge tests use temporary artificial projects and an offline model broker."""
from copy import deepcopy
from io import BytesIO
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import pandas as pd
import pytest
import gb1.agent as agent
import gb1.prompts as prompts
import gb1.schema as schema
import gb1.storage as storage

PROJECT = Path(__file__).resolve().parents[1]
WORKER_PATH = PROJECT / "dsh-plugin/packages/gb1/python/worker.py"
SPEC = importlib.util.spec_from_file_location("gb1_plugin_worker_test", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def frame(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def start(source="custom", method="llm"):
    return {"version": 1, "type": "start", "reference": worker.REFERENCE, "seed": 11, "rounds": 1,
            "method": method, "model": {"source": source, "provider": "offline-provider", "id": "offline-model"}}


def response(request_id="request-1", mode="chat", **changes):
    body = {"text": "{}", "complete": True, "usage": None, "model_requested": "offline-model",
            "model_returned": "offline-model", "api_mode": mode,
            "prompt_metadata": prompts.load_prompt_profile("v3").metadata()}
    body.update(changes)
    return {"version": 1, "type": "llm_response", "id": request_id, "response": body}


def broker(tmp_path, incoming, source="custom"):
    core = SimpleNamespace(agent=agent, prompts=prompts, storage=storage)
    output = BytesIO()
    channel = worker.Channel(BytesIO(incoming), output)
    config = schema.ExperimentConfig(rounds=1)
    return worker.BrokerProposer(channel, core, config, start(source)["model"], tmp_path), output


@pytest.mark.parametrize("raw, code", [
    (b"", "UNEXPECTED_EOF"),
    (b'{"version":1}', "TRUNCATED_FRAME"),
    (b'not-json\n', "INVALID_JSON"),
    (b'{"version":1,"version":1}\n', "INVALID_JSON"),
    (b'{"version":1,"number":NaN}\n', "INVALID_JSON"),
    (b'{"version":1,"number":1e999}\n', "INVALID_JSON"),
    (b'{"version":true}\n', "INVALID_VERSION"),
    (b'{"version":2}\n', "INVALID_VERSION"),
    (b'{"version":1,"apiKey":"synthetic-forbidden"}\n', "PRIVATE_FIELD"),
    (b'{"version":1,"nested":{"BaseURL":"https://example.invalid"}}\n', "PRIVATE_FIELD"),
    (b'{"version":1,"nested":{"endpoint":"https://example.invalid/chat/completions"}}\n', "PRIVATE_FIELD"),
    (b'{"version":1,"nested":{"Authorization":"synthetic-forbidden"}}\n', "PRIVATE_FIELD"),
])
def test_protocol_rejects_bad_frames_without_printing_contents(raw, code):
    with pytest.raises(worker.ProtocolAbort) as exc:
        worker.Channel(BytesIO(raw), BytesIO()).read()
    assert exc.value.code == code
    assert "synthetic-forbidden" not in exc.value.message


def test_both_frame_size_limits_and_excessive_nesting():
    with pytest.raises(worker.ProtocolAbort, match="16 MiB"):
        worker.Channel(BytesIO(b"x" * (worker.MAX_FRAME_BYTES + 1)), BytesIO()).read()
    with pytest.raises(worker.ProtocolAbort, match="16 MiB"):
        worker.Channel(BytesIO(), BytesIO()).send({"version": 1, "text": "x" * worker.MAX_FRAME_BYTES})
    nested = {}
    for _ in range(66):
        nested = {"value": nested}
    with pytest.raises(worker.ProtocolAbort, match="nesting"):
        worker.Channel(BytesIO(frame({"version": 1, "nested": nested})), BytesIO()).read()


@pytest.mark.parametrize("key, value", [("seed", -1), ("seed", 11.0), ("rounds", True), ("rounds", 0),
                                       ("method", "invalid"), ("reference", "M" * 56), ("extra", "not-allowed")])
def test_start_fields_are_strict(key, value):
    value_frame = start()
    value_frame[key] = value
    with pytest.raises(worker.ProtocolAbort):
        worker.validate_start(value_frame)
    assert worker.validate_start(start()) == start()


@pytest.mark.parametrize("run_id", ["../old-run", "UPPER", "", "con", "x" * 65, "a/b"])
def test_safe_run_id_and_no_overwrite(tmp_path, run_id):
    with pytest.raises(worker.WorkerFailure):
        worker.reserve_run(tmp_path, run_id)
    root = worker.reserve_run(tmp_path, "safe-id")
    (root / "sentinel.txt").write_text("untouched", encoding="utf-8")
    with pytest.raises(worker.WorkerFailure, match="cannot be reused"):
        worker.reserve_run(tmp_path, "safe-id")
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "untouched"


def test_project_env_is_never_an_input_and_main_errors_are_ndjson(tmp_path):
    (tmp_path / ".env").write_text("SYNTHETIC_UNUSED=1", encoding="utf-8")
    with pytest.raises(worker.WorkerFailure, match="credential files"):
        worker._project_file(tmp_path, ".env")
    output = BytesIO()
    result = worker.main(["--project-root", str(tmp_path), "--run-id", "bad/start"],
                         incoming=BytesIO(frame({**start(), "reference": "invalid"})), outgoing=output)
    assert result == 1
    values = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(values) == 1 and values[0]["type"] == "error"
    assert values[0]["code"] == "REFERENCE_MISMATCH"


@pytest.mark.parametrize("source, mode", [("dsh", "dsh"), ("custom", "chat")])
@pytest.mark.parametrize("returned_model", [None, "provider-model-version-20261110"])
def test_broker_uses_shared_chat_payload_and_preserves_public_response(tmp_path, source, mode, returned_model):
    reply = response(mode=mode, model_returned=returned_model, text="Actual synthetic model text", usage={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                     transport_request={"provider": "offline-provider", "model": "offline-model", "mode": mode})
    proposer, outgoing = broker(tmp_path, frame(reply), source)
    context = {"round": 1, "proposal_count": 40, "response_schema": {"type": "object"}}
    result = proposer.generate(context)
    assert result == reply["response"]
    request = json.loads(outgoing.getvalue())
    assert request["id"] == "request-1" and request["round"] == 1
    assert request["payload"] == agent.build_request_payload(context, proposer.config, "offline-model", "chat", proposer.prompt_metadata["system_prompt"])
    assert json.loads(request["payload"]["messages"][1]["content"]) == context
    assert proposer.api_mode == mode
    assert worker.ProtocolAbort.__bases__ == (BaseException,)


@pytest.mark.parametrize("mutation, code", [
    ({"model_returned": 123}, "INVALID_MODEL_REPLY"),
    ({"model_returned": ""}, "INVALID_MODEL_REPLY"),
    ({"model_returned": "  "}, "INVALID_MODEL_REPLY"),
    ({"model_requested": "different-model"}, "MODEL_IDENTITY_MISMATCH"),
    ({"api_mode": "dsh"}, "MODEL_IDENTITY_MISMATCH"),
    ({"prompt_metadata": {}}, "MODEL_IDENTITY_MISMATCH"),
    ({"complete": "yes"}, "INVALID_MODEL_REPLY"),
    ({"usage": {"total_tokens": -1}}, "INVALID_USAGE"),
    ({"usage": {"total_tokens": True}}, "INVALID_USAGE"),
    ({"request_id": 123}, "INVALID_MODEL_REPLY"),
    ({"transport_request": "not-an-object"}, "INVALID_MODEL_REPLY"),
])
def test_broker_rejects_bad_identity_and_field_types(tmp_path, mutation, code):
    reply = response()
    reply["response"].update(mutation)
    proposer, _ = broker(tmp_path, frame(reply))
    with pytest.raises(worker.ProtocolAbort) as exc:
        proposer.generate({"round": 1})
    assert exc.value.code == code


def test_wrong_reply_id_error_status_and_budget(tmp_path):
    proposer, _ = broker(tmp_path, frame(response("another-id")))
    with pytest.raises(worker.ProtocolAbort, match="Reply ID"):
        proposer.generate({"round": 1})
    error = {"version": 1, "type": "llm_error", "id": "request-1", "error": {"code": "TEST_LIMIT", "message": "Synthetic error", "status_code": 429}}
    proposer, _ = broker(tmp_path, frame(error))
    with pytest.raises(worker.HostModelError) as exc:
        proposer.generate({"round": 1})
    assert exc.value.status_code == 429 and proposer.calls == 1
    proposer, _ = broker(tmp_path, frame(response()) + frame(response("request-2")))
    proposer.generate({"round": 1})
    proposer.generate({"round": 1})
    with pytest.raises(worker.ProtocolAbort, match="budget"):
        proposer.generate({"round": 1})
    assert proposer.calls == 2


@pytest.fixture
def fake_project(tmp_path):
    root = tmp_path / "synthetic-project"
    root.mkdir()
    shutil.copytree(PROJECT / "gb1", root / "gb1", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "gb1/__init__.py").write_text('print("Synthetic core diagnostic", flush=True)\n', encoding="utf-8")
    (root / "dotenv.py").write_text('def load_dotenv(*args, **kwargs):\n    raise AssertionError("worker must never load .env")\n', encoding="utf-8")
    (root / "prompts").mkdir()
    for filename in ("legacy-v1.txt", "scientific-v2.txt", "scientific-v3.txt"):
        shutil.copyfile(PROJECT / "prompts" / filename, root / "prompts" / filename)
    (root / "configs").mkdir()
    shutil.copyfile(PROJECT / "configs/knowledge.json", root / "configs/knowledge.json")
    config = json.loads((PROJECT / "configs/experiment.json").read_text(encoding="utf-8"))
    config.update(train_size=24, validation_size=16, test_size=16, pool_size=160, tree_estimators=5, n_jobs=1)
    (root / "configs/experiment.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "data/raw").mkdir(parents=True)
    (root / "data/audit.json").write_text(json.dumps({"synthetic_test": True, "reference_sequence": {"sequence": worker.REFERENCE}}), encoding="utf-8")
    variants = ["".join(row) for row in product("ACDE", repeat=4)]
    table = pd.DataFrame({"Variants": variants, "Fitness": [float(sum((i + 1) * "ACDE".index(c) for i, c in enumerate(v))) for v in variants]})
    table.to_excel(root / "data/raw/elife-16965-supp1-v4.xlsx", sheet_name="Supplementary Dataset 1", index=False, engine="openpyxl")
    return root, table


def run_worker(project, message, handler=None, run_id="synthetic-run"):
    process = subprocess.Popen([sys.executable, "-u", str(WORKER_PATH), "--project-root", str(project), "--run-id", run_id],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", "OPENAI_API_KEY": "SYNTHETIC_MUST_NOT_BE_USED"})
    timer = threading.Timer(35, process.kill)
    timer.start()
    values = []
    try:
        process.stdin.write(frame(message))
        process.stdin.flush()
        for line in iter(process.stdout.readline, b""):
            assert len(line) <= worker.MAX_FRAME_BYTES
            item = json.loads(line)
            values.append(item)
            if item["type"] in ("llm_request", "checkpoint"):
                answer = (handler(item, project) if handler else None) if item["type"] == "llm_request" else {
                    "version": 1, "type": "round_control", "round": item["round"], "action": "continue"}
                if item["type"] == "checkpoint" and handler is not None:
                    answer = handler(item, project)
                if answer is None:
                    process.stdin.close()
                    process.stdin = None
                else:
                    process.stdin.write(frame(answer))
                    process.stdin.flush()
        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        error = process.stderr.read().decode("utf-8")
        code = process.wait(timeout=5)
        return code, values, error
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def offline_host(message, limit=None):
    def handler(request, project):
        if request["type"] == "checkpoint":
            return {"version": 1, "type": "round_control", "round": request["round"], "action": "continue"}
        payload = request["payload"]
        context = json.loads(payload["messages"][1]["content"])
        assert context["prompt_version"] == "gb1-proposals-v3" and context["candidate_examples"] == []
        candidates = pd.read_csv(project / "results/dsh-plugin/synthetic-run/data/candidates.csv")
        assert "fitness" not in candidates
        accepted = set(context.get("accepted_variants", []))
        saved = json.loads((project / "results/dsh-plugin/synthetic-run/run/state.json").read_text(encoding="utf-8"))
        measured = {row["variant"] for record in saved["methods"][message["method"]]["rounds"] if record["stage"] == "evaluated" for row in record["recommendations"]}
        variants = [v for v in candidates.variant if v not in accepted | measured]
        count = min(context["proposal_count"], limit) if limit is not None else context["proposal_count"]
        body = {"analysis": "Observations: synthetic. Hypotheses: synthetic. Feedback: none. Design: synthetic.",
                "proposals": [{"variant": v, "hypothesis": "Synthetic test hypothesis", "rationale": "Synthetic test rationale", "knowledge_rule_ids": []} for v in variants[:count]]}
        return {"version": 1, "type": "llm_response", "id": request["id"], "response": {
            "text": json.dumps(body), "complete": True, "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "model_requested": message["model"]["id"],
            "model_returned": None if message["model"]["source"] == "dsh" else message["model"]["id"] + "-20261110",
            "api_mode": "dsh" if message["model"]["source"] == "dsh" else "chat", "prompt_metadata": request["prompt_metadata"]}}
    return handler


@pytest.mark.parametrize("source, method", [("custom", "llm"), ("dsh", "knowledge")])
def test_single_round_real_core_end_to_end_with_fake_data_and_host(fake_project, source, method):
    project, table = fake_project
    message = start(source, method)
    code, values, errors = run_worker(project, message, offline_host(message))
    assert code == 0, errors
    assert [row["stage"] for row in values if row["type"] == "progress"] == ["prepare", "train", "run"]
    assert not any(row["type"] == "process_event" for row in values)
    assert sum(row["type"] == "llm_request" for row in values) == 1
    result = values[-1]
    assert result["type"] == "result" and result["status"] == "completed"
    summary = result["summary"]
    assert summary["initial_known_count"] == 24 and summary["known_count"] == 34
    assert summary["calls"] == summary["completed_rounds"] == 1
    assert summary["usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "missing_usage_count": 0}
    observed = summary["rounds"][0]
    assert observed["valid_count"] == observed["scored_count"] == 40 and len(observed["top10"]) == 10
    truth = table.set_index("Variants").Fitness
    assert all(row["actual_fitness"] == truth[row["variant"]] for row in observed["top10"])
    assert all(len(row["sequence"]) == 56 for row in observed["top10"])
    encoded = json.dumps(summary)
    assert "training_variants" not in encoded and "eligible_variants" not in encoded
    assert "SYNTHETIC_MUST_NOT_BE_USED" not in encoded and "candidate_examples" not in encoded
    root = project / "results/dsh-plugin/synthetic-run"
    assert len(pd.read_csv(root / "selection/model_comparison.csv")) == 7
    assert not (root / "run/.writer.lock").exists()
    before = (root / "run/state.json").read_text(encoding="utf-8")
    saved_response = json.loads(before)["methods"][method]["rounds"][0]["attempts"][0]["response"]
    assert saved_response["model_returned"] == (None if source == "dsh" else "offline-model-20261110")
    assert saved_response["model_requested"] == "offline-model"
    repeated, repeat_values, _ = run_worker(project, message)
    assert repeated != 0 and repeat_values[-1]["code"] == "RUN_EXISTS"
    assert before == (root / "run/state.json").read_text(encoding="utf-8")
    assert "Synthetic core diagnostic" in errors


def test_underfilled_round_keeps_core_completion_semantics(fake_project):
    project, _ = fake_project
    message = start()
    code, values, errors = run_worker(project, message, offline_host(message, limit=12))
    assert code == 0, errors
    summary = values[-1]["summary"]
    assert values[-1]["status"] == "completed" and summary["calls"] == 2
    assert summary["rounds"][0]["valid_count"] == summary["rounds"][0]["scored_count"] == 24
    assert len(summary["rounds"][0]["top10"]) == 10


def test_worker_feature_negotiation_reports_only_real_paired_safe_tools(fake_project):
    project, _ = fake_project
    message = {**start(), "features": ["process-events-v1"]}
    code, values, errors = run_worker(project, message, offline_host(message))
    assert code == 0, errors
    events = [row["event"] for row in values if row["type"] == "process_event"]
    assert [value["tool"] for value in events[::2]] == ["prepare", "model_selection", "fit", "validate", "predict", "select", "oracle", "feedback"]
    for begin, end in zip(events[::2], events[1::2]):
        assert begin["kind"] == "tool_start" and end["kind"] == "tool_end" and end["ok"] is True
        assert begin["id"] == end["id"] and "ok" not in begin
        assert set(begin["counts"]) <= worker.PROCESS_COUNTS and set(end["counts"]) <= worker.PROCESS_COUNTS
    assert len({event["id"] for event in events[::2]}) == 8
    assert events[-1]["counts"]["known_count"] == 34
    for feature in (["unknown"], ["process-events-v1", "process-events-v1"], "process-events-v1"):
        with pytest.raises(worker.ProtocolAbort):
            worker.validate_start({**start(), "features": feature})


@pytest.mark.parametrize("failure", ["wrong-id", "eof", "model-error"])
def test_failed_protocol_and_model_errors_do_not_retry_and_unlock(fake_project, failure):
    project, _ = fake_project
    message = start()
    def handler(request, project):
        if failure == "eof":
            return None
        if failure == "wrong-id":
            return response("wrong-id")
        return {"version": 1, "type": "llm_error", "id": request["id"], "error": {"code": "TEST_MODEL_ERROR", "message": "Synthetic unavailable model"}}
    code, values, _ = run_worker(project, message, handler)
    assert code != 0
    assert sum(row["type"] == "llm_request" for row in values) == 1
    root = project / "results/dsh-plugin/synthetic-run"
    assert not (root / "run/.writer.lock").exists()
    state = json.loads((root / "run/state.json").read_text(encoding="utf-8"))
    record = state["methods"]["llm"]["rounds"][0]
    assert record["attempts"][0]["status"] == "started"
    if failure == "model-error":
        assert values[-1]["type"] == "result" and values[-1]["status"] == "failed"
        assert values[-1]["summary"]["rounds"][0]["metrics"]["mean_fitness"] is None
        assert values[-1]["summary"]["known_count"] == 24
    else:
        assert values[-1]["type"] == "error"
