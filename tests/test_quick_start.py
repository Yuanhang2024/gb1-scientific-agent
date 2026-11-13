"""Offline quick-start tests use small synthetic data and owned cancellation processes."""
import ctypes
import http.server
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pytest

PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT / "tools/quick_start"
sys.path.insert(0, str(TOOLS))
import bootstrap
import check_environment
import common
import launch
import validate_data


def fixture_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "requirements.txt").write_text("openpyxl==3.1.5\n", encoding="utf-8")
    for name in ("common", "bootstrap", "check_environment", "validate_data"):
        target = root / f"tools/quick_start/{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TOOLS / f"{name}.py", target)
    return root


def stats():
    return {"variants": 149361, "unique_variants": 149361, "zero_fitness": 29477, "wt_is_one": True}


def test_stdlib_entry_starts_without_site_packages():
    completed = subprocess.run([sys.executable, "-S", str(TOOLS / "bootstrap.py"), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert completed.returncode == 0 and b"--project-root" in completed.stdout


def test_requirements_only_accept_exact_public_pins(tmp_path):
    target = tmp_path / "requirements.txt"
    for text in ("numpy>=2\n", "https://example.invalid/package.whl\n", "--extra-index-url https://example.invalid\n", "numpy==1\nnumpy==2\n"):
        target.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            check_environment.requirements(target)


def test_generated_paths_and_index_never_accept_environment_files_or_url_credentials(tmp_path):
    for relative in ("../outside.json", ".env", ".env.local"):
        with pytest.raises(common.PreparationError):
            common.inside(tmp_path, relative)
    with pytest.raises(common.PreparationError, match="without authentication"):
        bootstrap.Preparer(tmp_path, index_url="https://user:synthetic@example.invalid/simple")


def test_clean_subprocess_environment_does_not_forward_connection_or_proxy_values(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unit-key")
    monkeypatch.setenv("HTTP_PROXY", "http://synthetic-unit-proxy.invalid")
    value = bootstrap.Preparer(tmp_path).environment_variables()
    assert "OPENAI_API_KEY" not in value and "HTTP_PROXY" not in value
    assert value["PYTHON_DOTENV_DISABLED"] == "1" and value["PYTHONNOUSERSITE"] == "1"
    assert Path(value["USERPROFILE"]).is_relative_to(tmp_path)


def test_available_explicit_python_does_not_require_home(tmp_path, monkeypatch):
    preparer = bootstrap.Preparer(tmp_path, python=sys.executable)
    monkeypatch.setattr(bootstrap.Path, "home", lambda: (_ for _ in ()).throw(RuntimeError("No home")))
    monkeypatch.setattr(preparer, "probe", lambda arguments: {"version": [3, 12], "base": sys.executable})
    preparer.locate_python()
    assert preparer.base == Path(sys.executable)


def test_absent_python_has_a_public_error_even_without_home(tmp_path, monkeypatch):
    preparer = bootstrap.Preparer(tmp_path)
    monkeypatch.setattr(preparer, "probe", lambda _: None)
    monkeypatch.setattr(bootstrap.Path, "home", lambda: (_ for _ in ()).throw(RuntimeError("No home")))
    with pytest.raises(common.PreparationError) as caught:
        preparer.locate_python()
    assert caught.value.code == "PYTHON_UNAVAILABLE"


def test_virtual_environment_must_not_resolve_to_shared_python(tmp_path, monkeypatch):
    preparer = bootstrap.Preparer(tmp_path)
    preparer.python.parent.mkdir(parents=True)
    preparer.python.write_bytes(b"synthetic")
    monkeypatch.setattr(preparer, "probe", lambda _: {"prefix": str(tmp_path / "shared"), "base_prefix": str(tmp_path / "shared")})
    with pytest.raises(common.PreparationError) as caught:
        preparer.ensure_environment()
    assert caught.value.code == "INVALID_EXISTING_ENV"


def test_fixed_install_command_requires_the_project_venv(tmp_path, monkeypatch):
    root = fixture_project(tmp_path)
    preparer = bootstrap.Preparer(root)
    first = {"ready": False, "missing_or_mismatched": ["openpyxl"], "failed_imports": []}
    second = {"ready": True, "versions": {"openpyxl": "3.1.5"}}
    states = iter([first, second])
    monkeypatch.setattr(preparer, "environment_check", lambda: next(states))
    monkeypatch.setattr(preparer, "probe", lambda _: {"prefix": str(preparer.environment), "base_prefix": str(tmp_path / "base")})
    calls = []
    monkeypatch.setattr(preparer, "run_command", lambda arguments, **_: (calls.append([str(item) for item in arguments]) or (0, "")))
    preparer.dependencies()
    install = next(arguments for arguments in calls if "install" in arguments)
    assert install[0] == str(preparer.python)
    assert "--require-virtualenv" in install and "--isolated" in install and "--no-cache-dir" in install
    assert "--upgrade" not in install and "https://pypi.org/simple" in install


def test_pip_failure_only_emits_public_diagnostic(tmp_path, monkeypatch, capsys):
    root = fixture_project(tmp_path)
    preparer = bootstrap.Preparer(root)
    monkeypatch.setattr(preparer, "environment_check", lambda: {"ready": False, "missing_or_mismatched": ["openpyxl"], "failed_imports": []})
    monkeypatch.setattr(preparer, "probe", lambda _: {"prefix": str(preparer.environment), "base_prefix": str(tmp_path / "base")})
    monkeypatch.setattr(preparer, "run_command", lambda *_, **__: (1, "Authorization: synthetic-unit-secret"))
    with pytest.raises(common.PreparationError) as caught:
        preparer.dependencies()
    assert caught.value.code == "DEPENDENCY_INSTALL_FAILED"
    preparer.finish("failed", caught.value.code, caught.value.message)
    assert "synthetic-unit-secret" not in capsys.readouterr().out
    assert json.loads(preparer.report_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_existing_valid_data_is_reused_without_download(tmp_path, monkeypatch):
    preparer = bootstrap.Preparer(tmp_path)
    preparer.data.parent.mkdir(parents=True)
    preparer.data.write_bytes(b"existing-synthetic")
    monkeypatch.setattr(preparer, "validate_data", lambda _: stats())
    monkeypatch.setattr(preparer, "download", lambda _: pytest.fail("Cached data must not be downloaded"))
    preparer.data_file()
    assert preparer.counts["data_reused"] and preparer.data.read_bytes() == b"existing-synthetic"
    assert all(step["stage"] != "download" for step in preparer.steps)


def test_invalid_existing_data_is_preserved(tmp_path, monkeypatch):
    preparer = bootstrap.Preparer(tmp_path)
    preparer.data.parent.mkdir(parents=True)
    preparer.data.write_bytes(b"invalid-existing-synthetic")
    monkeypatch.setattr(preparer, "validate_data", lambda _: (_ for _ in ()).throw(common.PreparationError("DATA_VALIDATION_FAILED", "Invalid")))
    with pytest.raises(common.PreparationError) as caught:
        preparer.data_file()
    assert caught.value.code == "INVALID_EXISTING_DATA" and preparer.data.read_bytes() == b"invalid-existing-synthetic"


@pytest.mark.parametrize("cancel", [False, True])
def test_interrupted_download_never_commits_a_partial_file(tmp_path, monkeypatch, cancel):
    preparer = bootstrap.Preparer(tmp_path)
    def interrupted(part):
        part.write_bytes(b"partial-synthetic")
        if cancel:
            raise common.PreparationError("CANCELLED", "Cancelled")
        raise OSError("Synthetic interrupted download")
    monkeypatch.setattr(preparer, "download", interrupted)
    with pytest.raises(common.PreparationError) as caught:
        preparer.data_file()
    assert caught.value.code == ("CANCELLED" if cancel else "DATA_DOWNLOAD_FAILED")
    assert not preparer.data.exists() and not list(preparer.data.parent.glob("*.part"))


def test_offline_missing_data_is_explicit(tmp_path):
    with pytest.raises(common.PreparationError) as caught:
        bootstrap.Preparer(tmp_path, offline=True).data_file()
    assert caught.value.code == "DATA_UNAVAILABLE_OFFLINE"


def test_lock_is_exclusive_and_released(tmp_path):
    first, second = bootstrap.Preparer(tmp_path), bootstrap.Preparer(tmp_path)
    with first.lock():
        with pytest.raises(common.PreparationError) as caught:
            with second.lock():
                pass
        assert caught.value.code == "PREPARATION_BUSY"
    with second.lock():
        pass


def test_pre_cancelled_bootstrap_returns_cancelled_without_creating_environment(tmp_path, capsys):
    root = fixture_project(tmp_path)
    cancel = root / "cancel.flag"
    cancel.write_text("cancel", encoding="utf-8")
    code = bootstrap.main(["--project-root", str(root), "--prepare-only", "--cancel-file", str(cancel)])
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 130 and frames[-1]["status"] == "cancelled" and not (root / ".venv").exists()


def test_part_workbook_is_checked_by_contents_without_extension_assumptions(tmp_path, monkeypatch):
    from openpyxl import Workbook
    monkeypatch.setattr(validate_data, "EXPECTED_ROWS", 3)
    path = tmp_path / "download.part"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = common.SHEET
    sheet.append(["Variants", "Fitness"])
    for row in (("VDGV", 1), ("ADGV", 0), ("VDGA", .2)):
        sheet.append(row)
    workbook.save(path)
    workbook.close()
    value = validate_data.validate(path)
    assert value["variants"] == 3 and value["zero_fitness"] == 1 and value["wt_is_one"]


def test_offline_launcher_never_reads_or_requests_connection(tmp_path, monkeypatch):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs/workbench-defaults.json").write_text(json.dumps({"promptProfile": "v3"}), encoding="utf-8")
    monkeypatch.setattr(launch, "connection", lambda *_, **__: pytest.fail("Offline run must not load a model connection"))
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda arguments, **_: (calls.append(arguments) or subprocess.CompletedProcess(arguments, 0)))
    assert launch.run(root, offline=True) == 0
    run = next(arguments for arguments in calls if "run" in arguments)
    assert run[run.index("--methods") + 1: run.index("--prompt-profile")] == ["random", "model"]


def test_independent_connection_preserves_saved_model_without_saving_or_displaying_key(tmp_path, monkeypatch, capsys):
    from dotenv import dotenv_values
    path = tmp_path / ".env"
    original = "OPENAI_MODEL=synthetic-current\nOPENAI_API_KEY=synthetic-test-only-key\nOPENAI_BASE_URL=https://synthetic.example\n"
    path.write_text(original, encoding="utf-8")
    for name in ("OPENAI_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_MODE"):
        monkeypatch.setenv(name, "")
    assert launch.connection(tmp_path, "synthetic-recommended") == "synthetic-current"
    assert path.read_text(encoding="utf-8") == original
    output = capsys.readouterr().out
    assert "synthetic-test-only-key" not in output and "https://synthetic.example" not in output


def test_new_terminal_key_is_process_only_unless_save_is_explicit(tmp_path, monkeypatch, capsys):
    for name in ("OPENAI_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_MODE"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["https://synthetic.example", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(launch.getpass, "getpass", lambda _: "synthetic-test-only-key")
    assert launch.connection(tmp_path, "synthetic-recommended") == "synthetic-recommended"
    assert not (tmp_path / ".env").exists()
    assert "synthetic-test-only-key" not in capsys.readouterr().out


@pytest.mark.parametrize("case", ["wrong-sheet", "wrong-wt", "duplicate", "negative"])
def test_data_validator_rejects_wrong_measured_table(tmp_path, monkeypatch, case):
    from openpyxl import Workbook
    monkeypatch.setattr(validate_data, "EXPECTED_ROWS", 3)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Supplementary Dataset 2" if case == "wrong-sheet" else common.SHEET
    sheet.append(["Variants", "Fitness"])
    sheet.append(["VDGV", .9 if case == "wrong-wt" else 1])
    sheet.append(["ADGV", 0])
    sheet.append(["ADGV" if case == "duplicate" else "VDGA", -.1 if case == "negative" else .2])
    path = tmp_path / "wrong.xlsx"
    workbook.save(path)
    workbook.close()
    with pytest.raises(ValueError):
        validate_data.validate(path)


def alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        value = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(value))) and value.value == 259
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows owned-process-tree cancellation")
def test_cancel_terminates_an_owned_parent_and_grandchild(tmp_path):
    preparer = bootstrap.Preparer(tmp_path, cancel_file=tmp_path / "cancel.flag")
    child_file = tmp_path / "child-pid.txt"
    code = ("import pathlib,subprocess,sys,time;child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"pathlib.Path({str(child_file)!r}).write_text(str(child.pid));time.sleep(60)")
    def cancel():
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            threading.Event().wait(.05)
        preparer.cancel_file.write_text("cancel", encoding="utf-8")
    thread = threading.Thread(target=cancel, daemon=True)
    thread.start()
    started = time.monotonic()
    with pytest.raises(common.PreparationError) as caught:
        preparer.run_command([sys.executable, "-B", "-c", code])
    thread.join(timeout=1)
    assert caught.value.code == "CANCELLED" and time.monotonic() - started < 8
    assert child_file.exists() and not alive(int(child_file.read_text()))


def test_cancel_stops_an_actual_pip_process(tmp_path):
    started_request, release = threading.Event(), threading.Event()
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            started_request.set()
            release.wait(10)
            self.send_response(404)
            self.end_headers()
        def log_message(self, *_):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    preparer = bootstrap.Preparer(tmp_path, cancel_file=tmp_path / "cancel.flag")
    child_processes = []
    original = preparer.stop_child
    def stop(child):
        child_processes.append(child)
        return original(child)
    preparer.stop_child = stop
    def cancel():
        started_request.wait(5)
        preparer.cancel_file.write_text("cancel", encoding="utf-8")
    cancellation = threading.Thread(target=cancel, daemon=True)
    cancellation.start()
    before = time.monotonic()
    try:
        with pytest.raises(common.PreparationError) as caught:
            preparer.run_command([sys.executable, "-B", "-m", "pip", "--isolated", "--require-virtualenv", "--disable-pip-version-check",
                "download", "--no-deps", "--no-cache-dir", "--trusted-host", "127.0.0.1", "--index-url",
                f"http://127.0.0.1:{server.server_port}/simple", "--dest", str(tmp_path / "download"), "gb1-cancel-test==0"])
        assert caught.value.code == "CANCELLED" and started_request.is_set()
        assert time.monotonic() - before < 8 and child_processes and all(child.poll() is not None for child in child_processes)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        cancellation.join(timeout=1)
