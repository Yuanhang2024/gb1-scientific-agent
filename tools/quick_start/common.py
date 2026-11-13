"""Public constants and path constraints for the GB1 preparation commands."""
from pathlib import Path

DATA_URL = "https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx"
DATA_NAME = "elife-16965-supp1-v4.xlsx"
SHEET = "Supplementary Dataset 1"
REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
EXPECTED_ROWS = 149361
MAX_DATA_BYTES = 64 * 1024 * 1024
SCHEMA = "gb1-quick-start-report-v1"


class PreparationError(Exception):
    def __init__(self, code, message, **counts):
        super().__init__(message)
        self.code, self.message, self.counts = code, message, counts


def inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root) or any(part.lower() == ".env" or part.lower().startswith(".env.") for part in path.parts):
        raise PreparationError("PROJECT_PATH_INVALID", "Generated paths must remain inside the project and cannot address environment files.")
    return path
