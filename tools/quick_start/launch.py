"""Run the prepared project with public defaults or a small offline comparison."""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
from uuid import uuid4

from common import DATA_NAME, REFERENCE, inside


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(char) < 32 for char in value):
        raise ValueError("A nonempty public model ID is required.")
    return value.strip()


def connection(root, recommended, *, base_url=None, model=None, api_mode=None, save=False):
    # This explicit application path is the only preparation component that reads an optional project connection.
    from dotenv import dotenv_values, set_key
    saved = dotenv_values(root / ".env") if (root / ".env").is_file() else {}
    current = {name: os.environ.get(name) or saved.get(name) for name in
               ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY", "OPENAI_API_MODE")}
    address = base_url if base_url is not None else current["OPENAI_BASE_URL"]
    identifier = model or current["OPENAI_MODEL"]
    mode = api_mode or current["OPENAI_API_MODE"] or "chat"
    key = current["OPENAI_API_KEY"]
    if not identifier or not key:
        if not sys.stdin.isatty():
            raise ValueError("An independent API connection is missing. Run interactively to enter its model and API key, or use -Offline/-PrepareOnly.")
        if address is None:
            address = input("OpenAI-compatible Base URL (Enter for the SDK public default): ").strip() or None
        if not identifier:
            identifier = input(f"Model ID [{recommended}]: ").strip() or recommended
        if not key:
            key = getpass.getpass("API key (hidden; used only by this process): ").strip()
    identifier = model_text(identifier)
    if not key:
        raise ValueError("An API key is required for the independent model interface.")
    if address:
        parsed = urllib.parse.urlsplit(address)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("The independent Base URL must be an HTTP(S) address without embedded credentials or query parameters.")
    if mode not in ("chat", "responses"):
        raise ValueError("The independent API mode must be chat or responses.")
    print(f"Independent interface model: {identifier}; recommended model: {recommended}.", flush=True)
    values = {"OPENAI_MODEL": identifier, "OPENAI_API_KEY": key, "OPENAI_API_MODE": mode,
              "OPENAI_BASE_URL": address or ""}
    for name, value in values.items():
        os.environ[name] = value
    if not address:
        os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    if save:
        if not sys.stdin.isatty():
            raise ValueError("Saving a project connection requires an explicit interactive confirmation.")
        answer = input("Save this connection, including its API key, into this project's .env? Type yes to save [default: this process only]: ").strip().lower()
        if answer == "yes":
            for name, value in values.items():
                set_key(root / ".env", name, value, quote_mode="always")
            print("The explicitly selected project connection was saved.", flush=True)
    return identifier


def run(root, *, offline=False, base_url=None, model=None, api_mode=None, effort=None, save=False):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    from gb1.schema import ExperimentConfig
    defaults = read(root / "configs/workbench-defaults.json")
    if offline:
        values = read(root / "configs/small-example.json") if (root / "configs/small-example.json").is_file() else {
            "seed": 1234, "rounds": 2, "train_size": 200, "validation_size": 100, "test_size": 100, "pool_size": 200,
            "k": 5, "proposal_count": 20, "score_budget": 20, "tree_estimators": 20, "n_jobs": 1, "save_predictors": False}
        methods, identifier = ["random", "model"], None
    else:
        values = defaults["config"].copy()
        recommended = defaults["recommendedModel"]["model"]
        identifier = connection(root, recommended, base_url=base_url, model=model, api_mode=api_mode, save=save)
        if effort is not None:
            values["llm_reasoning_effort"] = effort or None
        elif identifier != recommended:
            if not sys.stdin.isatty():
                raise ValueError("For a different independent model, explicitly choose --reasoning-effort or use an empty value for its interface default.")
            values["llm_reasoning_effort"] = input("Supported reasoning effort for this independent model (Enter for its interface default): ").strip() or None
        methods = ["knowledge"]
    config = ExperimentConfig.model_validate(values)
    profile = defaults["promptProfile"]
    output = inside(root, f'results/quick-start/cli-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}-{uuid4().hex[:8]}')
    output.mkdir(parents=True, exist_ok=False)
    settings = output / "config.json"
    settings.write_text(json.dumps(config.model_dump(), indent=2), encoding="utf-8")
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    data, selection, experiment = output / "data", output / "selection", output / "run"
    common = ["--data", str(data), "--config", str(settings)]
    commands = [
        ["prepare", *common, "--source", str(root / "data/raw" / DATA_NAME), "--reference", REFERENCE,
         "--variant-column", "Variants", "--fitness-column", "Fitness", "--sheet", "Supplementary Dataset 1"],
        ["select", *common, "--output", str(selection)],
        ["run", *common, "--output", str(experiment), "--selection", str(selection / "selection.json"), "--methods", *methods, "--prompt-profile", profile],
        ["finalize", *common, "--output", str(experiment), "--selection", str(selection / "selection.json"), "--methods", *methods],
        ["plot", "--output", str(experiment)],
    ]
    for arguments in commands:
        print(f"GB1 stage: {arguments[0]}", flush=True)
        completed = subprocess.run([str(python), "-B", str(root / "main.py"), *arguments], cwd=root, check=False)
        if completed.returncode:
            print(json.dumps({"status": "failed", "stage": arguments[0], "exit_code": completed.returncode,
                              "output_relative": str(output.relative_to(root))}), flush=True)
            return completed.returncode
    summary = {"status": "completed", "offline": offline, "methods": methods,
               "actual_model": identifier, "output_relative": str(output.relative_to(root))}
    (output / "launch-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-mode", choices=("chat", "responses"))
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--save-connection", action="store_true")
    options = parser.parse_args(argv)
    try:
        return run(options.project_root, offline=options.offline, base_url=options.base_url, model=options.model,
                   api_mode=options.api_mode, effort=options.reasoning_effort, save=options.save_connection)
    except (ValueError, OSError):
        print("GB1 could not start. Check the public project files or enter the independent model connection interactively; no connection was silently saved.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
