"""Inspect exact pinned versions and functional imports without exposing process configuration."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

PATTERN = re.compile(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)\Z")
IMPORTS = ("numpy", "pandas", "scipy", "sklearn", "openpyxl", "joblib", "matplotlib", "pydantic", "openai", "dotenv", "docx")


def requirements(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PATTERN.fullmatch(line)
        if match is None:
            raise ValueError("Requirements must contain exact public package pins")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in result:
            raise ValueError("Requirements must contain unique package pins")
        result[normalized] = version
    if not result:
        raise ValueError("Requirements cannot be empty")
    return result


def inspect(path):
    wanted = requirements(path)
    versions, missing = {}, []
    for name, expected in wanted.items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = None
        versions[name] = value
        if value != expected:
            missing.append(name)
    failed = []
    if not missing:
        for name in IMPORTS:
            try:
                importlib.import_module(name)
            except Exception:
                failed.append(name)
    return {"ready": not missing and not failed, "versions": versions, "required": wanted,
            "missing_or_mismatched": missing, "failed_imports": failed, "imports_ok": not missing and not failed,
            "prefix": str(Path(sys.prefix).resolve()), "base_prefix": str(Path(sys.base_prefix).resolve()),
            "python_312": sys.version_info[:2] == (3, 12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True)
    options = parser.parse_args()
    try:
        value = inspect(options.requirements)
    except Exception:
        print(json.dumps({"ready": False, "error": "ENVIRONMENT_CHECK_FAILED"}))
        return 2
    print(json.dumps(value, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
