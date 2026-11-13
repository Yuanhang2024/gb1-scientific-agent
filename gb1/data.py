# 数据准备：把四字母变体还原为完整序列，并分开训练、验证、候选与测试数据。
"""Sequence reconstruction, audited input parsing and disjoint data roles."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .schema import AMINO_ACIDS, POSITIONS, ExperimentConfig
from .storage import read_json, write_json


# 实验位点按1开始编号；访问Python序列时要减1。先核对参考序列，防止位置错配。
def validate_reference(reference):
    if len(reference) != 56 or set(reference) - set(AMINO_ACIDS):
        raise ValueError("Reference must contain exactly 56 standard amino acids")
    if "".join(reference[p - 1] for p in POSITIONS) != "VDGV":
        raise ValueError("Reference positions 39/40/41/54 must map to VDGV")
    return reference


# variant的四个字母依次对应39、40、41、54位；其余位置保留参考氨基酸。
def full_sequence(variant, reference):
    validate_reference(reference)
    if not isinstance(variant, str) or re.fullmatch(f"[{AMINO_ACIDS}]{{4}}", variant) is None:
        raise ValueError(f"Invalid four-site variant: {variant!r}")
    sequence = list(reference)
    for position, residue in zip(POSITIONS, variant):
        sequence[position - 1] = residue
    return "".join(sequence)


# mutation_count是相对参考序列的替换数；例如V39W表示第39位由V换成W。
def describe_variant(variant, reference):
    sequence = full_sequence(variant, reference)
    mutations = [f"{reference[p - 1]}{p}{a}" for p, a in zip(POSITIONS, variant) if reference[p - 1] != a]
    return {"variant": variant, "sequence": sequence, "mutations": ":".join(mutations) or "WT",
            "mutation_count": len(mutations)}


# 每个位点用20个格子表示氨基酸，仅对应格为1；四个位点合计80维。
# 返回矩阵形状为“变体数 × 80”，作为回归模型的输入。
def one_hot(variants):
    variants = list(variants)
    result = np.zeros((len(variants), 80), dtype=np.float64)
    for row, variant in enumerate(variants):
        if not isinstance(variant, str) or len(variant) != 4 or set(variant) - set(AMINO_ACIDS):
            raise ValueError(f"Invalid four-site variant: {variant!r}")
        for position, residue in enumerate(variant):
            result[row, 20 * position + AMINO_ACIDS.index(residue)] = 1.0
    return result


# 保留合法的0分记录，fitness不做对数变换；同一变体出现冲突分数时先停止核查。
def clean_table(table, variant_column, fitness_column, reference):
    validate_reference(reference)
    if variant_column not in table or fitness_column not in table:
        raise ValueError("Configured variant/fitness columns are absent")
    variants = table[variant_column].astype("string").str.strip().str.upper()
    fitness = pd.to_numeric(table[fitness_column], errors="coerce")
    legal = variants.str.fullmatch(f"[{AMINO_ACIDS}]{{4}}", na=False)
    numeric = fitness.notna() & np.isfinite(fitness) & (fitness >= 0)
    cleaned = pd.DataFrame({"variant": variants[legal & numeric], "fitness": fitness[legal & numeric].astype(float)})
    conflicts = cleaned.groupby("variant")["fitness"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Duplicate variants have conflicting fitness; resolve using the source audit")
    duplicate_count = int(cleaned.duplicated("variant").sum())
    cleaned = cleaned.drop_duplicates("variant").sort_values("variant").reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No legal measured variants remain")
    descriptions = pd.DataFrame([describe_variant(v, reference) for v in cleaned.variant])
    cleaned = descriptions.merge(cleaned, on="variant", validate="one_to_one")
    audit = {"input_rows": len(table), "invalid_sequence_rows": int((~legal).sum()),
             "missing_or_invalid_fitness_rows": int((~numeric).sum()),
             "duplicate_rows_removed": duplicate_count, "valid_rows": len(cleaned),
             "zero_fitness_rows": int((cleaned.fitness == 0).sum()), "reference": reference,
             "positions": list(POSITIONS), "fitness_transform": "none", "higher_is_better": True}
    return cleaned, audit


# 按固定seed打乱行号再切分，保证数据角色互斥；划分不按fitness大小挑样本。
def split_table(table, config):
    pool_size = config.pool_size if config.pool_size is not None else len(table) - config.train_size - config.validation_size - config.test_size
    required_rounds = config.rounds if config.max_assays is None else min(config.rounds, config.max_assays // config.k)
    if pool_size < max(config.k * required_rounds, config.score_budget):
        raise ValueError("Measured candidate pool is too small for the experiment budgets")
    counts = [config.train_size, config.validation_size, pool_size, config.test_size]
    if sum(counts) > len(table):
        raise ValueError(f"Requested {sum(counts)} variants but only {len(table)} measured variants are available")
    if table.variant.duplicated().any():
        raise ValueError("Variants must be deduplicated before splitting")
    order = np.random.default_rng(config.seed).permutation(len(table))
    result = {}
    offset = 0
    for role, count in zip(("train", "validation", "pool", "test"), counts):
        result[role] = table.iloc[order[offset:offset + count]].reset_index(drop=True)
        offset += count
    return result


def prepare(source, destination, config, reference, variant_column, fitness_column, sheet_name=0, header=0):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Prepared destination must be empty; existing splits are immutable")
    source = Path(source)
    table = pd.read_excel(source, sheet_name=sheet_name, header=header, engine="openpyxl") if source.suffix.lower() == ".xlsx" else pd.read_csv(source, header=header)
    cleaned, audit = clean_table(table, variant_column, fitness_column, reference)
    splits = split_table(cleaned, config)
    destination.mkdir(parents=True, exist_ok=True)
    hidden = destination / "hidden"
    hidden.mkdir()
    for role in ("train", "validation"):
        splits[role].to_csv(destination / f"{role}.csv", index=False)
    # 候选表只公开序列；真实标签另存，供选定候选后的虚拟评估使用。
    splits["pool"].drop(columns="fitness").to_csv(destination / "candidates.csv", index=False)
    splits["pool"][["variant", "fitness"]].to_csv(hidden / "oracle.csv", index=False)
    splits["test"].to_csv(hidden / "test.csv", index=False)
    write_json(destination / "manifest.json", {"format_version": 1, "config": config.model_dump(),
               "source": str(source.resolve()), "schema": {"variant_column": variant_column,
               "fitness_column": fitness_column, "sheet_name": sheet_name, "header": header},
               "audit": audit, "counts": {role: len(rows) for role, rows in splits.items()}})
    return audit


# 推荐程序读取这个公开视图：初始已测数据，以及不带fitness的候选表。
class PublicDataset:
    """Agent-facing rows contain only initial measurements and unlabeled candidates."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        self.reference = validate_reference(self.manifest["audit"]["reference"])
        self.train = pd.read_csv(self.directory / "train.csv")
        self.candidates = pd.read_csv(self.directory / "candidates.csv")
        if "fitness" in self.candidates:
            raise ValueError("Candidate input must not contain labels")
        if self.train.variant.duplicated().any() or self.candidates.variant.duplicated().any():
            raise ValueError("Public rows must be unique")
        if set(self.train.variant) & set(self.candidates.variant):
            raise ValueError("Training and candidate rows overlap")
        for table in (self.train, self.candidates):
            for row in table.itertuples():
                if full_sequence(row.variant, self.reference) != row.sequence:
                    raise ValueError("Stored sequence does not match the four-site variant")


# 虚拟实验评估器：只查询调用方本轮已选定的变体，返回S1中已有的实测分数。
class HiddenOracle:
    """Only selected, validated candidates can request measured feedback."""
    def __init__(self, directory, allowed):
        self.directory = Path(directory)
        self.allowed = set(allowed)

    def evaluate(self, variants):
        if len(variants) != len(set(variants)) or not set(variants) <= self.allowed:
            raise ValueError("Oracle accepts unique selected candidate variants only")
        table = pd.read_csv(self.directory / "hidden" / "oracle.csv")
        if table.variant.duplicated().any():
            raise ValueError("Oracle variants must be unique")
        lookup = table.set_index("variant").fitness
        if not set(variants) <= set(lookup.index):
            raise ValueError("Selected candidate has no measured oracle value")
        return [{"variant": v, "actual_fitness": float(lookup[v])} for v in variants]
