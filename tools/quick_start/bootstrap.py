"""Prepare a project-local Python environment and the official measured GB1 S1 data."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

sys.dont_write_bytecode = True
from common import DATA_NAME, DATA_URL, MAX_DATA_BYTES, REFERENCE, SCHEMA, PreparationError, inside
from check_environment import requirements

CODES = {
    "PROJECT_PATH_INVALID", "PROJECT_LAYOUT_INVALID", "REPORT_PATH_INVALID", "PREPARATION_BUSY",
    "PYTHON_UNAVAILABLE", "INVALID_EXISTING_ENV", "VENV_CREATE_FAILED", "REQUIREMENTS_INVALID",
    "DEPENDENCIES_UNAVAILABLE_OFFLINE", "DEPENDENCY_INSTALL_FAILED", "DEPENDENCY_VALIDATION_FAILED",
    "DATA_UNAVAILABLE_OFFLINE", "DATA_DOWNLOAD_FAILED", "DATA_VALIDATION_FAILED", "INVALID_EXISTING_DATA",
    "DATA_COMMIT_FAILED", "DATA_AUDIT_INVALID", "CANCELLED", "PREPARATION_FAILED", "INDEX_URL_INVALID",
}
SAFE_VARIABLES = ("SystemRoot", "WINDIR", "PATH", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
PROBE = "import json,sys;print(json.dumps({'version':list(sys.version_info[:2]),'base':getattr(sys,'_base_executable',sys.executable),'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"


def now():
    return datetime.now(timezone.utc).isoformat()


class Preparer:
    def __init__(self, root, *, python=None, offline=False, report=None, cancel_file=None, index_url="https://pypi.org/simple"):
        self.root = Path(root).resolve()
        self.explicit_python, self.offline = python, offline
        self.report_path = inside(self.root, report or f".quick-start/reports/{uuid4().hex}.json")
        self.cancel_file = inside(self.root, cancel_file) if cancel_file else None
        self.stage, self.cancelled = "python", threading.Event()
        self.steps, self.counts, self.versions = [], {}, {}
        self.base = None
        self.environment = self.root / ".venv"
        self.python = self.environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.data = inside(self.root, f"data/raw/{DATA_NAME}")
        self.index_url = index_url
        parsed = urllib.parse.urlsplit(index_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PreparationError("INDEX_URL_INVALID", "Use a public HTTPS package index without authentication in its URL.")
        self.child = None

    def environment_variables(self):
        state = inside(self.root, ".quick-start")
        temporary, home = state / "tmp", state / "home"
        temporary.mkdir(parents=True, exist_ok=True)
        home.mkdir(parents=True, exist_ok=True)
        value = {name: os.environ[name] for name in SAFE_VARIABLES if name in os.environ}
        value.update(TEMP=str(temporary), TMP=str(temporary), HOME=str(home), USERPROFILE=str(home),
                     PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1",
                     PYTHONIOENCODING="utf-8", PIP_CONFIG_FILE=os.devnull, MPLCONFIGDIR=str(state / "matplotlib"))
        return value

    def check_cancel(self):
        if self.cancelled.is_set() or self.cancel_file is not None and self.cancel_file.exists():
            raise PreparationError("CANCELLED", "Preparation was cancelled.")

    def emit(self, stage, status, message, **counts):
        self.stage = stage
        self.counts.update(counts)
        frame = {"version": 1, "type": "progress", "stage": stage, "status": status, "message": message, "counts": counts}
        self.steps.append({"stage": stage, "status": status, "counts": counts})
        print(json.dumps(frame, ensure_ascii=False, allow_nan=False), flush=True)

    def stop_child(self, child):
        if child.poll() is not None:
            return
        if os.name == "nt":
            system = os.environ.get("SystemRoot", r"C:\Windows")
            executable = Path(system) / "System32/taskkill.exe"
            if executable.is_file():
                try:
                    subprocess.run([str(executable), "/PID", str(child.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
                                   env=self.environment_variables(), check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if child.poll() is None:
            child.kill()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()

    def run_command(self, arguments, timeout=180, failure="PREPARATION_FAILED"):
        self.check_cancel()
        lines = deque(maxlen=128)
        child = subprocess.Popen([str(argument) for argument in arguments], cwd=self.root, env=self.environment_variables(),
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.child = child
        def consume():
            for raw in iter(child.stdout.readline, b""):
                lines.append(raw[:8192].decode("utf-8", errors="replace"))
        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while child.poll() is None:
                self.check_cancel()
                if time.monotonic() - started > timeout:
                    raise PreparationError(failure, "The preparation step exceeded its time limit.")
                self.cancelled.wait(.2)
            reader.join(timeout=2)
            return child.returncode, "".join(lines)
        finally:
            if child.poll() is None:
                self.stop_child(child)
            child.stdout.close()
            self.child = None

    def probe(self, command):
        try:
            code, text = self.run_command([*command, "-B", "-c", PROBE], timeout=15, failure="PYTHON_UNAVAILABLE")
            value = json.loads(text.strip().splitlines()[-1]) if code == 0 else {}
            return value if value.get("version") == [3, 12] else None
        except PreparationError as error:
            if error.code == "CANCELLED":
                raise
            return None
        except (OSError, ValueError, IndexError):
            return None

    def locate_python(self):
        self.emit("python", "running", "Checking Python 3.12.")
        candidates = []
        if self.python.is_file():
            candidates.append([str(self.python)])
        if self.explicit_python:
            candidates.append([self.explicit_python])
        candidates.append([sys.executable])
        launcher = shutil.which("py")
        if launcher:
            candidates.append([launcher, "-3.12"])
        for name in ("python3.12", "python"):
            executable = shutil.which(name)
            if executable and "windowsapps" not in executable.lower():
                candidates.append([executable])
        for command in candidates:
            info = self.probe(command)
            if info:
                base = Path(info["base"])
                base_info = self.probe([str(base)]) if base.is_file() else None
                if base_info:
                    self.base = base
                    self.emit("python", "completed", "Python 3.12 is available.", python_312=True)
                    return
        try:
            fallback = Path.home() / ".dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe"
        except (RuntimeError, OSError):
            fallback = None
        if fallback is not None and fallback.is_file():
            info = self.probe([str(fallback)])
            if info:
                base = Path(info["base"])
                if base.is_file() and self.probe([str(base)]):
                    self.base = base
                    self.emit("python", "completed", "Python 3.12 is available.", python_312=True)
                    return
        raise PreparationError("PYTHON_UNAVAILABLE", "Python 3.12 is required. Select an existing Python 3.12 executable or the DSH bundled interpreter.")

    def ensure_environment(self):
        self.emit("venv", "running", "Checking the project-local virtual environment.")
        raw = self.root / ".venv"
        if raw.is_symlink() or getattr(raw, "is_junction", lambda: False)() or raw.resolve() != raw:
            raise PreparationError("INVALID_EXISTING_ENV", "The project virtual environment cannot be a directory link.")
        created = not raw.exists()
        ownership = inside(self.root, ".quick-start/venv-creation.json")
        if not self.python.is_file():
            if raw.exists() and not ownership.is_file():
                raise PreparationError("INVALID_EXISTING_ENV", "The existing virtual environment is incomplete and was preserved.")
            ownership.parent.mkdir(parents=True, exist_ok=True)
            ownership.write_text(json.dumps({"schema": SCHEMA, "target": ".venv", "ready": False}), encoding="utf-8")
            code, _ = self.run_command([self.base, "-B", "-m", "venv", str(raw)], timeout=180, failure="VENV_CREATE_FAILED")
            if code != 0:
                raise PreparationError("VENV_CREATE_FAILED", "The project-local virtual environment could not be created.")
        info = self.probe([str(self.python)])
        if not info or Path(info["prefix"]).resolve() != raw or info["prefix"] == info["base_prefix"]:
            raise PreparationError("INVALID_EXISTING_ENV", "The target Python must be isolated inside this project's .venv.")
        self.emit("venv", "completed" if created else "reused", "The project-local virtual environment is available.",
                  environment_created=created, environment_reused=not created)

    def environment_check(self):
        code, text = self.run_command([self.python, "-B", self.root / "tools/quick_start/check_environment.py",
                                       "--requirements", self.root / "requirements.txt"], failure="DEPENDENCY_VALIDATION_FAILED")
        try:
            result = json.loads(text.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {}
        if code != 0 or not result.get("python_312") or Path(result.get("prefix", "")).resolve() != self.environment or result.get("prefix") == result.get("base_prefix"):
            raise PreparationError("DEPENDENCY_VALIDATION_FAILED", "The target virtual environment could not be inspected.")
        return result

    def dependencies(self):
        try:
            wanted = requirements(self.root / "requirements.txt")
        except (OSError, ValueError):
            raise PreparationError("REQUIREMENTS_INVALID", "requirements.txt must contain exact public package pins.") from None
        self.emit("dependencies", "running", "Checking fixed Python dependencies.", required_packages=len(wanted))
        initial = self.environment_check()
        reused = initial["ready"]
        if not reused:
            if self.offline:
                raise PreparationError("DEPENDENCIES_UNAVAILABLE_OFFLINE", "Offline preparation requires every pinned dependency and functional import to be available.")
            self.emit("dependencies", "running", "Installing fixed dependencies only into the project virtual environment.",
                      matched_packages=len(wanted) - len(initial["missing_or_mismatched"]))
            # A second isolation check immediately precedes installation.
            info = self.probe([str(self.python)])
            if not info or Path(info["prefix"]).resolve() != self.environment or info["prefix"] == info["base_prefix"]:
                raise PreparationError("INVALID_EXISTING_ENV", "Dependency installation requires this project's isolated virtual environment.")
            command = [self.python, "-B", "-m", "pip", "--isolated", "--require-virtualenv", "--disable-pip-version-check", "install",
                       "--no-input", "--progress-bar", "off", "--no-cache-dir", "--no-compile", "--only-binary=:all:",
                       "--keyring-provider", "disabled", "--index-url", self.index_url, "-r", self.root / "requirements.txt"]
            if initial["failed_imports"] and not initial["missing_or_mismatched"]:
                command.append("--force-reinstall")
            code, _ = self.run_command(command, timeout=1800, failure="DEPENDENCY_INSTALL_FAILED")
            if code != 0:
                raise PreparationError("DEPENDENCY_INSTALL_FAILED", "Fixed dependency installation failed. Check access to public PyPI or explicitly select a public HTTPS mirror without URL credentials.")
        checked = self.environment_check()
        if not checked["ready"]:
            raise PreparationError("DEPENDENCY_VALIDATION_FAILED", "Pinned versions or functional imports remain unavailable.")
        code, _ = self.run_command([self.python, "-B", "-m", "pip", "--isolated", "--require-virtualenv", "check"], failure="DEPENDENCY_VALIDATION_FAILED")
        if code:
            raise PreparationError("DEPENDENCY_VALIDATION_FAILED", "Installed package dependency constraints are inconsistent.")
        self.versions = checked["versions"]
        self.emit("dependencies", "reused" if reused else "completed", "All fixed dependencies and functional imports are available.",
                  required_packages=len(wanted), matched_packages=len(wanted), imports_ok=True, dependencies_reused=reused)

    def validate_data(self, path):
        code, text = self.run_command([self.python, "-B", self.root / "tools/quick_start/validate_data.py", "--path", path],
                                      timeout=180, failure="DATA_VALIDATION_FAILED")
        try:
            result = json.loads(text.strip().splitlines()[-1])
        except (ValueError, IndexError):
            result = {}
        if code or not result.get("valid"):
            raise PreparationError("DATA_VALIDATION_FAILED", "The workbook does not match the measured GB1 S1 table.")
        return {key: result[key] for key in ("variants", "unique_variants", "zero_fitness", "wt_is_one")}

    def download(self, part):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "GB1-Quick-Start/1.0", "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
        total, started = 0, time.monotonic()
        with opener.open(request, timeout=3) as response, part.open("xb") as handle:
            final = urllib.parse.urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "cdn.elifesciences.org":
                raise PreparationError("DATA_DOWNLOAD_FAILED", "The data response left the official HTTPS source.")
            length = response.headers.get("Content-Length")
            if length is not None and not 1 <= int(length) <= MAX_DATA_BYTES:
                raise PreparationError("DATA_DOWNLOAD_FAILED", "The official data response exceeded the size limit.")
            while True:
                self.check_cancel()
                if time.monotonic() - started > 180:
                    raise PreparationError("DATA_DOWNLOAD_FAILED", "The official data download exceeded its time limit.")
                chunk = response.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DATA_BYTES:
                    raise PreparationError("DATA_DOWNLOAD_FAILED", "The official data response exceeded the size limit.")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        return total

    def data_file(self):
        self.data.parent.mkdir(parents=True, exist_ok=True)
        if self.data.exists():
            self.emit("validate", "running", "Validating the existing official GB1 S1 workbook.")
            try:
                result = self.validate_data(self.data)
            except PreparationError as error:
                if error.code == "CANCELLED":
                    raise
                raise PreparationError("INVALID_EXISTING_DATA", "The existing S1 workbook is invalid and was preserved. Move it aside explicitly before retrying.") from None
            self.emit("validate", "reused", "The existing measured GB1 S1 workbook is valid.", data_reused=True, **result)
        else:
            if self.offline:
                raise PreparationError("DATA_UNAVAILABLE_OFFLINE", "Offline preparation requires a previously validated official S1 workbook.")
            self.emit("download", "running", "Downloading measured GB1 S1 from the official source.", data_reused=False)
            part = inside(self.root, self.data.parent / f".{DATA_NAME}.{uuid4().hex}.part")
            try:
                try:
                    total = self.download(part)
                except PreparationError:
                    raise
                except (OSError, ValueError, urllib.error.URLError):
                    raise PreparationError("DATA_DOWNLOAD_FAILED", "The official HTTPS S1 download failed. Check network access and retry; completed environment setup is reusable.") from None
                self.emit("download", "completed", "The official workbook download completed.", bytes_downloaded=total)
                self.emit("validate", "running", "Validating measured GB1 S1 contents.")
                result = self.validate_data(part)
                self.check_cancel()
                try:
                    os.link(part, self.data)
                except FileExistsError:
                    result = self.validate_data(self.data)
                except OSError:
                    raise PreparationError("DATA_COMMIT_FAILED", "The validated download could not be committed without replacing an existing file.") from None
                self.emit("validate", "completed", "The measured GB1 S1 workbook is valid.", **result)
            finally:
                if part.exists() and part.resolve().is_relative_to(self.data.parent.resolve()):
                    part.unlink()
        audit = inside(self.root, "data/audit.json")
        if audit.exists():
            try:
                current = json.loads(audit.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise PreparationError("DATA_AUDIT_INVALID", "The existing reference audit was preserved but could not be read.") from None
            if current.get("reference_sequence", {}).get("sequence") != REFERENCE:
                raise PreparationError("DATA_AUDIT_INVALID", "The existing reference audit does not match the GB1 assay reference and was preserved.")
        else:
            with audit.open("x", encoding="utf-8") as handle:
                json.dump({"reference_sequence": {"sequence": REFERENCE}, "source_url": DATA_URL, "measured_table": result}, handle, indent=2)

    @contextmanager
    def lock(self):
        path = inside(self.root, ".quick-start/prepare.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise PreparationError("PREPARATION_BUSY", "Another preparation owns this project.") from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def prepare(self):
        if not self.root.is_dir():
            raise PreparationError("PROJECT_LAYOUT_INVALID", "The project directory does not exist.")
        for relative in ("requirements.txt", "tools/quick_start/common.py", "tools/quick_start/check_environment.py", "tools/quick_start/validate_data.py"):
            if not inside(self.root, relative).is_file():
                raise PreparationError("PROJECT_LAYOUT_INVALID", "A required public preparation file is missing.")
        with self.lock():
            self.check_cancel()
            self.locate_python()
            self.ensure_environment()
            self.dependencies()
            self.data_file()
            self.check_cancel()
            self.emit("ready", "completed", "The project environment and measured GB1 data are ready.")

    def finish(self, status, code, message):
        result = {"version": 1, "type": "result", "stage": "ready" if status == "ready" else self.stage,
                  "status": status, "code": code, "message": message, "counts": self.counts}
        report = {"schema": SCHEMA, "created_at": now(), **result, "steps": self.steps, "versions": self.versions,
                  "data_source": DATA_URL, "python": ".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python"}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if self.report_path.exists():
            try:
                previous = json.loads(self.report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
            if previous.get("schema") != SCHEMA:
                raise PreparationError("REPORT_PATH_INVALID", "An unrelated existing report file was preserved.")
        temporary = inside(self.root, self.report_path.with_name(f".{self.report_path.name}.{uuid4().hex}.tmp"))
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.report_path)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--python")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report")
    parser.add_argument("--cancel-file")
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    options = parser.parse_args(argv)
    preparer = None
    try:
        preparer = Preparer(options.project_root, python=options.python, offline=options.offline,
                            report=options.report, cancel_file=options.cancel_file, index_url=options.index_url)
        signal.signal(signal.SIGTERM, lambda *_: preparer.cancelled.set())
        preparer.prepare()
        preparer.finish("ready", "READY", "Preparation completed.")
        return 0
    except KeyboardInterrupt:
        error = PreparationError("CANCELLED", "Preparation was cancelled.")
    except PreparationError as caught:
        error = caught
    except Exception:
        error = PreparationError("PREPARATION_FAILED", "Preparation failed; no ready result was recorded.")
    status = "cancelled" if error.code == "CANCELLED" else "failed"
    if preparer:
        try:
            preparer.finish(status, error.code, error.message)
        except Exception:
            print(json.dumps({"version": 1, "type": "result", "stage": preparer.stage, "status": status,
                              "code": error.code, "message": error.message, "counts": {}}), flush=True)
    else:
        print(json.dumps({"version": 1, "type": "result", "stage": "python", "status": status,
                          "code": error.code, "message": error.message, "counts": {}}), flush=True)
    return 130 if status == "cancelled" else 2


if __name__ == "__main__":
    raise SystemExit(main())
