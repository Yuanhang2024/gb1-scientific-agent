"""Narrow NDJSON worker; model transports and credentials remain in the Node Host."""
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib
import json
import math
import re
import statistics
import sys

VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024
REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SAFE_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
RESERVED_NAMES = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
PRIVATE_KEYS = {"apikey", "baseurl", "endpoint", "authorization", "proxyauthorization", "xapikey", "password", "credentials", "accesstoken", "secret"}


class WorkerFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


# 协议错误必须穿透core的Exception处理，避免错帧后继续下一轮或重复调用。
class ProtocolAbort(BaseException):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


class HostModelError(Exception):
    def __init__(self, error):
        super().__init__(error["message"])
        self.code = error["code"]
        self.status_code = error.get("status_code")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("Nonfinite JSON number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def _public(value, depth=0):
    if depth > 64:
        raise ProtocolAbort("FRAME_TOO_DEEP", "Protocol nesting exceeds the permitted limit")
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in PRIVATE_KEYS:
                raise ProtocolAbort("PRIVATE_FIELD", "Credential or endpoint fields are not accepted by the worker")
            _public(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _public(item, depth + 1)


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ProtocolAbort("INVALID_FIELDS", "Protocol object has missing or unexpected fields")


def _text(value, limit=200):
    return isinstance(value, str) and 0 < len(value) <= limit and value == value.strip() and not any(ord(char) < 32 for char in value)


class Channel:
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing

    def read(self):
        try:
            line = self.incoming.readline(MAX_FRAME_BYTES + 1)
        except OSError:
            raise ProtocolAbort("INPUT_ERROR", "Worker input could not be read") from None
        if not line:
            raise ProtocolAbort("UNEXPECTED_EOF", "Worker input closed before the matching reply")
        if len(line) > MAX_FRAME_BYTES:
            raise ProtocolAbort("FRAME_TOO_LARGE", "Protocol frame exceeds 16 MiB")
        if not line.endswith(b"\n"):
            raise ProtocolAbort("TRUNCATED_FRAME", "NDJSON frames must end with a newline")
        try:
            value = json.loads(line.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_number, parse_float=_finite_float)
        except (ValueError, UnicodeError, RecursionError):
            raise ProtocolAbort("INVALID_JSON", "Protocol frame is not valid finite UTF-8 JSON") from None
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != VERSION:
            raise ProtocolAbort("INVALID_VERSION", "Protocol requires an object with version 1")
        _public(value)
        return value

    def send(self, value):
        try:
            line = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        except (TypeError, ValueError):
            raise ProtocolAbort("OUTPUT_INVALID", "Worker output could not be serialized") from None
        if len(line) > MAX_FRAME_BYTES:
            raise ProtocolAbort("FRAME_TOO_LARGE", "Worker output frame exceeds 16 MiB")
        try:
            self.outgoing.write(line)
            self.outgoing.flush()
        except OSError:
            raise ProtocolAbort("OUTPUT_CLOSED", "Worker output channel closed") from None


def validate_start(frame):
    _keys(frame, {"version", "type", "reference", "seed", "rounds", "method", "model"},
          {"features", "config", "prompt_profile", "repeat_index", "resume"})
    if type(frame["version"]) is not int or frame["version"] != VERSION:
        raise ProtocolAbort("INVALID_VERSION", "Protocol requires version 1")
    _public(frame)
    features = frame.get("features", [])
    if not isinstance(features, list) or any(value not in ("process-events-v1", "round-control-v1") for value in features) or len(features) != len(set(features)):
        raise ProtocolAbort("INVALID_FEATURES", "Unknown or duplicate worker feature")
    if frame["type"] != "start":
        raise ProtocolAbort("EXPECTED_START", "The first protocol frame must be start")
    if frame["reference"] != REFERENCE:
        raise ProtocolAbort("REFERENCE_MISMATCH", "Reference must match the audited 56-aa assay sequence")
    if type(frame["seed"]) is not int or not 0 <= frame["seed"] <= 2**32 - 1:
        raise ProtocolAbort("INVALID_SEED", "Seed must be a nonnegative 32-bit integer")
    if type(frame["rounds"]) is not int or frame["rounds"] < 1:
        raise ProtocolAbort("INVALID_ROUNDS", "Rounds must be a positive integer")
    if frame["method"] not in ("random", "model", "llm", "knowledge"):
        raise ProtocolAbort("INVALID_METHOD", "Method must be random, model, llm or knowledge")
    if "config" in frame and not isinstance(frame["config"], dict):
        raise ProtocolAbort("INVALID_CONFIG", "Configuration must be an object")
    if frame.get("prompt_profile", "v3") not in ("v1", "v2", "v3"):
        raise ProtocolAbort("INVALID_PROMPT_PROFILE", "Prompt profile must be v1, v2 or v3")
    if "repeat_index" in frame and (type(frame["repeat_index"]) is not int or frame["repeat_index"] < 0):
        raise ProtocolAbort("INVALID_REPEAT_INDEX", "Repeat index must be a nonnegative integer")
    if "resume" in frame and type(frame["resume"]) is not bool:
        raise ProtocolAbort("INVALID_RESUME", "Resume must be a boolean")
    if "config" in frame and any(frame["config"].get(key, frame[key]) != frame[key] for key in ("seed", "rounds")):
        raise ProtocolAbort("CONFIG_MISMATCH", "Top-level seed and rounds must match config")
    _keys(frame["model"], {"source", "provider", "id"})
    if frame["model"]["source"] not in ("dsh", "custom") or not _text(frame["model"]["provider"]) or not _text(frame["model"]["id"]):
        raise ProtocolAbort("INVALID_MODEL", "Model requires a public source, provider and model ID")
    return deepcopy(frame)


def _project_file(project, relative):
    path = (project / relative).resolve()
    if path.name.casefold() == ".env" or path.name.casefold().startswith(".env."):
        raise WorkerFailure("PRIVATE_PROJECT_FILE", "Environment credential files are not worker inputs")
    if not path.is_relative_to(project) or not path.is_file():
        raise WorkerFailure("PROJECT_FILE_MISSING", "A required project file is missing or outside the project")
    return path


def _local_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_object, parse_constant=_invalid_number, parse_float=_finite_float)
    except (OSError, ValueError, UnicodeError):
        raise WorkerFailure("PROJECT_JSON_INVALID", "A required project JSON file could not be read") from None


def project_root(value):
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise WorkerFailure("INVALID_PROJECT_ROOT", "project-root must be an existing absolute directory")
    return path.resolve()


def reserve_run(project, run_id, resume=False):
    if SAFE_RUN_ID.fullmatch(run_id) is None or run_id in RESERVED_NAMES:
        raise WorkerFailure("INVALID_RUN_ID", "run-id must be a safe lowercase identifier of at most 64 characters")
    parent = (project / "results" / "dsh-plugin").resolve()
    if not parent.is_relative_to(project):
        raise WorkerFailure("RESULT_PATH_INVALID", "Result directory must remain inside the project")
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / run_id
    if not root.resolve().is_relative_to(parent):
        raise WorkerFailure("RESULT_PATH_INVALID", "Run directory must remain inside plugin results")
    if resume:
        if not root.is_dir() or not (root / "worker.json").is_file():
            raise WorkerFailure("RESUME_MISSING", "Resume requires an existing worker run")
        return root
    try:
        root.mkdir()
    except FileExistsError:
        raise WorkerFailure("RUN_EXISTS", "The run directory already exists and cannot be reused") from None
    return root


def load_core(project):
    _project_file(project, "gb1/__init__.py")
    sys.path.insert(0, str(project))
    importlib.invalidate_caches()
    package = importlib.import_module("gb1")
    if not Path(package.__file__).resolve().is_relative_to(project):
        raise WorkerFailure("CORE_IMPORT_MISMATCH", "The imported GB1 package is not the requested project")
    data = importlib.import_module("gb1.data")
    models = importlib.import_module("gb1.models")
    agent = importlib.import_module("gb1.agent")
    experiment = importlib.import_module("gb1.experiment")
    prompts = importlib.import_module("gb1.prompts")
    schema = importlib.import_module("gb1.schema")
    storage = importlib.import_module("gb1.storage")
    for module in (data, models, agent, experiment, prompts, schema, storage):
        if not Path(module.__file__).resolve().is_relative_to(project):
            raise WorkerFailure("CORE_IMPORT_MISMATCH", "A GB1 module is outside the requested project")
    return SimpleNamespace(data=data, models=models, agent=agent, experiment=experiment,
                           prompts=prompts, schema=schema, storage=storage)


class BrokerProposer:
    """Only public payloads cross the bridge; this class never creates an SDK client."""
    def __init__(self, channel, core, config, model, root, prompt_profile="v3", previous_rounds=()):
        self.channel, self.core, self.config = channel, core, config
        self.model = model["id"]
        self.api_mode = "dsh" if model["source"] == "dsh" else "chat"
        self.prompt_metadata = core.prompts.load_prompt_profile(prompt_profile).metadata()
        self.root = root
        self.calls, self.errors = 0, []
        self.round_calls = {record["round"]: len(record["attempts"]) for record in previous_rounds}

    def _event(self, kind, **data):
        self.core.storage.event(self.root / "broker_events.jsonl", kind, **deepcopy(data))

    def generate(self, context):
        number = context.get("round")
        if type(number) is not int or not 1 <= number <= self.config.rounds:
            raise ProtocolAbort("INVALID_REQUEST_ROUND", "Core requested an invalid round")
        if self.round_calls.get(number, 0) >= self.config.llm_max_calls_per_round:
            raise ProtocolAbort("REQUEST_BUDGET", "Round request budget would exceed configured calls")
        self.calls += 1
        self.round_calls[number] = self.round_calls.get(number, 0) + 1
        request_id = f"request-{self.calls}"
        # 发给Host的是无凭据chat参数；DSH原生通道的实际转换由Host记录。
        payload = self.core.agent.build_request_payload(context, self.config, self.model, "chat", self.prompt_metadata["system_prompt"])
        request = {"version": VERSION, "type": "llm_request", "id": request_id, "round": number,
                   "payload": payload, "prompt_metadata": deepcopy(self.prompt_metadata)}
        self._event("llm_request", frame=request)
        self.channel.send(request)
        reply = self.channel.read()
        if reply.get("id") != request_id:
            raise ProtocolAbort("REPLY_ID_MISMATCH", "Reply ID does not match the pending model request")
        if reply.get("type") == "llm_error":
            _keys(reply, {"version", "type", "id", "error"})
            error = reply["error"]
            _keys(error, {"code", "message"}, {"status_code"})
            if not _text(error["code"]) or not isinstance(error["message"], str) or not error["message"]:
                raise ProtocolAbort("INVALID_MODEL_ERROR", "Model error requires a code and message")
            status = error.get("status_code")
            if status is not None and (type(status) is not int or not 100 <= status <= 599):
                raise ProtocolAbort("INVALID_MODEL_ERROR", "Model error status_code is invalid")
            self._event("llm_error", frame=reply)
            self.errors.append({"round": number, "type": error["code"], "message": error["message"][:2000]})
            raise HostModelError(error)
        _keys(reply, {"version", "type", "id", "response"})
        if reply["type"] != "llm_response":
            raise ProtocolAbort("EXPECTED_MODEL_REPLY", "Expected llm_response or llm_error")
        response = reply["response"]
        _keys(response, {"text", "complete", "usage", "model_requested", "model_returned", "api_mode", "prompt_metadata"},
              {"request_id", "response_id", "transport_request"})
        if not isinstance(response["text"], str) or type(response["complete"]) is not bool:
            raise ProtocolAbort("INVALID_MODEL_REPLY", "Model text and completion flag have invalid types")
        if (response["model_requested"] != self.model or response["api_mode"] != self.api_mode
                or response["prompt_metadata"] != self.prompt_metadata):
            raise ProtocolAbort("MODEL_IDENTITY_MISMATCH", "Model reply identity or system prompt differs from the selected route")
        # DSH流可能没有服务端返回标识；保留null或真实别名，不用请求值代填。
        returned = response["model_returned"]
        if returned is not None and (not isinstance(returned, str) or not returned.strip()):
            raise ProtocolAbort("INVALID_MODEL_REPLY", "Returned model ID must be a nonempty string or null")
        usage = response["usage"]
        if usage is not None and not isinstance(usage, dict):
            raise ProtocolAbort("INVALID_USAGE", "usage must be a public object or null")
        if usage is not None:
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ProtocolAbort("INVALID_USAGE", "Reported token counts must be nonnegative integers")
        for key in ("request_id", "response_id"):
            if key in response and response[key] is not None and not _text(response[key], 1000):
                raise ProtocolAbort("INVALID_MODEL_REPLY", "Public request/response ID has an invalid type")
        if "transport_request" in response and not isinstance(response["transport_request"], dict):
            raise ProtocolAbort("INVALID_MODEL_REPLY", "transport_request must be a public object")
        self._event("llm_response", frame=reply)
        return deepcopy(response)


def _public_summary(start, run_id, root, config, selection, state, broker, core):
    method = state["methods"][start["method"]]
    counts = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 0}
    rounds, known, completed = [], config.train_size, 0
    for record in method["rounds"]:
        per_round = {key: 0 for key in counts}
        for attempt in record["attempts"]:
            usage = core.agent.usage_counts(attempt.get("response", {}).get("usage"))
            for key in counts:
                per_round[key] += usage[key]
                counts[key] += usage[key]
        measured = record["stage"] == "evaluated"
        top = []
        for row in record.get("recommendations", []):
            top.append({key: row.get(key) for key in ("rank", "variant", "sequence", "mutations", "mutation_count",
                        "predicted_fitness", "actual_fitness", "hypothesis", "rationale", "knowledge_rule_ids")})
        metrics = {key: None for key in ("mean_fitness", "max_fitness", "median_fitness", "cumulative_best")}
        after = known
        if measured:
            completed += 1
            after += len(record["recommendations"])
            for key in ("mean_fitness", "max_fitness", "cumulative_best"):
                metrics[key] = record["metrics"][key]
            metrics["median_fitness"] = float(statistics.median(row["actual_fitness"] for row in record["recommendations"]))
        rounds.append({"round": record["round"], "status": record["stage"], "analysis": record.get("analysis", []),
                       "known_count_before": record.get("training_count", known), "known_count_after": after,
                       "valid_count": record.get("valid_proposal_count", sum(attempt.get("valid_count", 0) for attempt in record["attempts"])),
                       "scored_count": record.get("model_score_count", 0), "calls": len(record["attempts"]),
                       "usage": per_round, "metrics": metrics, "top10": top})
        known = after
    failures = [{"round": row["round"], "type": row["type"], "message": str(row["message"])[:2000]}
                for row in method.get("failures", [])] + deepcopy(broker.errors)
    return {"run_id": run_id, "method": start["method"], "seed": start["seed"], "requested_rounds": start["rounds"],
            "completed_rounds": completed, "model": deepcopy(start["model"]),
            "paths": {"root": str(root), **{key: str(root / key) for key in ("data", "selection", "run")}},
            "selection": deepcopy(selection["spec"]), "initial_known_count": config.train_size, "known_count": known,
            "calls": sum(len(record["attempts"]) for record in method["rounds"]),
            "usage": counts, "rounds": rounds, "failures": failures,
            "stop_reason": method.get("stop_reason"), "config": deepcopy(state["config"]),
            "prompt_profile": start.get("prompt_profile", "v3"), "repeat_index": start.get("repeat_index", 0)}


PROCESS_TOOLS = {"prepare", "model_selection", "fit", "validate", "predict", "select", "oracle", "feedback"}
PROCESS_COUNTS = {"known_count", "candidate_count", "requested_count", "valid_count", "rejected_count",
                  "scored_count", "selected_count", "feedback_count", "validation_count", "configurations"}


def process_observer(channel):
    def observe(value):
        _keys(value, {"id", "kind", "tool", "round", "counts"}, {"ok"})
        if (not _text(value["id"]) or value["kind"] not in ("tool_start", "tool_end") or value["tool"] not in PROCESS_TOOLS
                or type(value["round"]) is not int or value["round"] < 0 or not isinstance(value["counts"], dict)
                or set(value["counts"]) - PROCESS_COUNTS or any(type(count) is not int or count < 0 for count in value["counts"].values())
                or "ok" in value and (value["kind"] != "tool_end" or type(value["ok"]) is not bool)):
            raise ProtocolAbort("INVALID_PROCESS_EVENT", "Worker process event is not a safe stage/count projection")
        channel.send({"version": VERSION, "type": "process_event", "event": deepcopy(value)})
    return observe


def execute(project, run_id, start, channel):
    audit = _local_json(_project_file(project, "data/audit.json"))
    if audit.get("reference_sequence", {}).get("sequence") != start["reference"]:
        raise WorkerFailure("REFERENCE_MISMATCH", "The project audit does not match the requested assay reference")
    values = deepcopy(start.get("config")) if "config" in start else _local_json(_project_file(project, "configs/experiment.json"))
    if not isinstance(values, dict):
        raise WorkerFailure("PROJECT_CONFIG_INVALID", "Experiment configuration must be an object")
    values.update(seed=start["seed"], rounds=start["rounds"])
    core = load_core(project)
    try:
        config = core.schema.ExperimentConfig.model_validate(values)
    except ValueError:
        raise WorkerFailure("INVALID_CONFIG", "Experiment configuration failed scientific parameter validation") from None
    if config.knowledge_path is not None:
        rules = _project_file(project, config.knowledge_path)
        if rules.suffix.lower() != ".json":
            raise WorkerFailure("INVALID_KNOWLEDGE_PATH", "Knowledge must be a project-local public JSON rule file")
        config = config.model_copy(update={"knowledge_path": str(rules)})
    root = reserve_run(project, run_id, start.get("resume", False))
    prompt_profile = start.get("prompt_profile", "v3")
    previous_rounds = []
    if start.get("resume", False):
        original = _local_json(root / "worker.json")["start"]
        for key in ("reference", "seed", "rounds", "method", "model"):
            if original[key] != start[key]:
                raise WorkerFailure("RESUME_MISMATCH", "Resume must retain the original run identity")
        if original.get("prompt_profile", "v3") != prompt_profile or original.get("repeat_index", 0) != start.get("repeat_index", 0):
            raise WorkerFailure("RESUME_MISMATCH", "Resume must retain the original prompt and repeat index")
        saved = _local_json(root / "run/state.json")
        if not core.schema.compatible_config(saved["config"], config):
            raise WorkerFailure("RESUME_MISMATCH", "Resume must retain the original scientific configuration")
        method = saved["methods"][start["method"]]
        previous_rounds = method["rounds"]
        if any(attempt["status"] == "started" for record in previous_rounds for attempt in record["attempts"]):
            raise WorkerFailure("UNKNOWN_REQUEST_OUTCOME", "An earlier model request has an unknown outcome and cannot be resent")
        if method["status"] not in ("paused", "running"):
            raise WorkerFailure("RUN_NOT_PAUSED", "Only a paused run or a saved feedback boundary can be resumed")
    else:
        core.storage.write_json(root / "worker.json", {"version": VERSION, "start": deepcopy(start), "run_id": run_id})
    observer = process_observer(channel) if "process-events-v1" in start.get("features", []) else None
    channel.send({"version": VERSION, "type": "progress", "stage": "prepare"})
    if observer:
        observer({"id": "prepare-data", "kind": "tool_start", "tool": "prepare", "round": 0, "counts": {}})
    try:
        if not start.get("resume", False):
            source = _project_file(project, "data/raw/elife-16965-supp1-v4.xlsx")
            core.data.prepare(source, root / "data", config, start["reference"], "Variants", "Fitness", "Supplementary Dataset 1", 0)
        dataset = core.data.PublicDataset(root / "data")
    except Exception:
        if observer:
            observer({"id": "prepare-data", "kind": "tool_end", "tool": "prepare", "round": 0, "counts": {}, "ok": False})
        raise
    if observer:
        observer({"id": "prepare-data", "kind": "tool_end", "tool": "prepare", "round": 0,
                  "counts": {"known_count": len(dataset.train), "candidate_count": len(dataset.candidates)}, "ok": True})
    channel.send({"version": VERSION, "type": "progress", "stage": "train"})
    if observer:
        observer({"id": "select-model", "kind": "tool_start", "tool": "model_selection", "round": 0,
                  "counts": {"known_count": len(dataset.train), "validation_count": config.validation_size}})
    try:
        selection = (core.models.load_selection(root / "selection/selection.json", dataset, config) if start.get("resume", False)
                     else core.models.select_model(dataset, config, root / "selection"))
    except Exception:
        if observer:
            observer({"id": "select-model", "kind": "tool_end", "tool": "model_selection", "round": 0, "counts": {}, "ok": False})
        raise
    if observer:
        observer({"id": "select-model", "kind": "tool_end", "tool": "model_selection", "round": 0,
                  "counts": {"configurations": len(selection["comparison"])}, "ok": True})
    broker = BrokerProposer(channel, core, config, start["model"], root, prompt_profile, previous_rounds)

    def checkpoint(method, number):
        channel.send({"version": VERSION, "type": "checkpoint", "round": number})
        reply = channel.read()
        _keys(reply, {"version", "type", "round", "action"})
        if reply["type"] != "round_control" or type(reply["round"]) is not int or reply["round"] != number or reply["action"] not in ("continue", "pause"):
            raise ProtocolAbort("INVALID_ROUND_CONTROL", "Round control must match the saved feedback round")
        if reply["action"] == "pause":
            raise core.experiment.RoundPaused()

    channel.send({"version": VERSION, "type": "progress", "stage": "run"})
    state = core.experiment.run_experiment(dataset, selection, config, root / "run", methods=[start["method"]],
                                           proposer=broker, retry_uncertain=False, prompt_profile=prompt_profile, observer=observer,
                                           checkpoint=checkpoint if "round-control-v1" in start.get("features", []) else None)
    summary = _public_summary(start, run_id, root, config, selection, state, broker, core)
    status = state["methods"][start["method"]]["status"]
    if status not in ("completed", "paused", "stopped"):
        status = "failed"
    core.storage.write_json(root / "public-summary.json", summary)
    result = {"version": VERSION, "type": "result", "status": status, "summary": summary}
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > MAX_FRAME_BYTES // 2:
        result = {"version": VERSION, "type": "result", "status": status, "summary_file": True}
    channel.send(result)
    return 0 if status in ("completed", "paused", "stopped") else 1


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise WorkerFailure("INVALID_ARGUMENTS", "Worker requires --project-root and --run-id")


def main(argv=None, incoming=None, outgoing=None):
    channel = Channel(incoming if incoming is not None else sys.stdin.buffer,
                      outgoing if outgoing is not None else sys.stdout.buffer)
    try:
        parser = _Parser(add_help=False)
        parser.add_argument("--project-root", required=True)
        parser.add_argument("--run-id", required=True)
        args = parser.parse_args(argv)
        project = project_root(args.project_root)
        start = validate_start(channel.read())
        # core或第三方库的普通print不得混入stdout协议。
        with redirect_stdout(sys.stderr):
            return execute(project, args.run_id, start, channel)
    except (WorkerFailure, ProtocolAbort) as exc:
        error = {"version": VERSION, "type": "error", "code": exc.code, "message": exc.message}
    except Exception as exc:
        print(f"Worker failed: {type(exc).__name__}", file=sys.stderr)
        error = {"version": VERSION, "type": "error", "code": "WORKER_FAILED", "message": "GB1 worker failed; inspect local run diagnostics"}
    try:
        channel.send(error)
    except ProtocolAbort:
        pass
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
