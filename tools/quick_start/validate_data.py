"""Validate the measured GB1 S1 workbook by its table and numeric contents."""
import argparse
import json
import math
from pathlib import Path
import sys
from common import EXPECTED_ROWS, MAX_DATA_BYTES, SHEET

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate(path):
    from openpyxl import load_workbook
    path = Path(path)
    if not path.is_file() or not 1 <= path.stat().st_size <= MAX_DATA_BYTES:
        raise ValueError("Workbook size is outside the permitted range")
    stream = path.open("rb")
    try:
        workbook = load_workbook(stream, read_only=True, data_only=True)
    except Exception:
        stream.close()
        raise
    try:
        if SHEET not in workbook.sheetnames:
            raise ValueError("Required measured-data sheet is missing")
        sheet = workbook[SHEET]
        header = list(next(sheet.iter_rows(min_row=1, max_row=1, values_only=True)))
        if header.count("Variants") != 1 or header.count("Fitness") != 1:
            raise ValueError("Required measured-data columns are missing or duplicated")
        variant_index, fitness_index = header.index("Variants"), header.index("Fitness")
        seen, rows, zeros, wild_type = set(), 0, 0, None
        for row in sheet.iter_rows(min_row=2, max_col=max(variant_index, fitness_index) + 1, values_only=True):
            if all(value is None for value in row):
                continue
            variant, raw = row[variant_index], row[fitness_index]
            if not isinstance(variant, str):
                raise ValueError("Variant must be a four-residue string")
            variant = variant.strip().upper()
            if len(variant) != 4 or not set(variant) <= AMINO_ACIDS or variant in seen or isinstance(raw, bool):
                raise ValueError("Variants must be unique standard four-residue sequences")
            fitness = float(raw)
            if not math.isfinite(fitness) or fitness < 0:
                raise ValueError("Measured fitness must be finite and nonnegative")
            seen.add(variant)
            rows += 1
            zeros += int(fitness == 0)
            if variant == "VDGV":
                wild_type = fitness
            if rows > EXPECTED_ROWS:
                raise ValueError("Unexpected measured-data row count")
        if rows != EXPECTED_ROWS or wild_type != 1.0 or zeros == 0:
            raise ValueError("Measured-data count or wild-type reference is inconsistent")
        return {"valid": True, "variants": rows, "unique_variants": len(seen), "zero_fitness": zeros, "wt_is_one": True}
    finally:
        workbook.close()
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    options = parser.parse_args()
    try:
        result = validate(options.path)
    except Exception:
        print(json.dumps({"valid": False, "code": "DATA_VALIDATION_FAILED"}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
