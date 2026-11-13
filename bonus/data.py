# 扩展数据划分：为单点对照预留候选，再分出互不重叠的校准和测试数据。
"""ID/HD-only allocation with a common held-out calibration/test population."""
from pathlib import Path
import numpy as np
import pandas as pd
from gb1.data import PublicDataset, clean_table
from gb1.storage import read_json, write_json


# HD是相对VDGV不同字母的数量；整个划分只用ID、HD与随机数，不按fitness筛选。
def split_bonus(table, config):
    if table.variant.duplicated().any():
        raise ValueError("Measured variants must be unique")
    table = table.sort_values("variant").reset_index(drop=True)
    hd = table.variant.map(lambda v: sum(a != b for a, b in zip(v, "VDGV")))
    table = table.assign(mutation_count=hd)
    wt = np.flatnonzero(hd.to_numpy() == 0)
    singles = np.flatnonzero(hd.to_numpy() == 1)
    other = np.flatnonzero(hd.to_numpy() >= 2)
    if len(wt) != 1 or len(singles) < config.initial_single_count + config.reserve_single_count:
        raise ValueError("WT or single-mutant coverage cannot satisfy the fixed design")
    other_count = config.train_size - 1 - config.initial_single_count
    if len(other) < other_count:
        raise ValueError("Not enough HD>=2 variants for training")
    # 先保留WT和指定数量的单点训练样本，其余训练样本从HD>=2中抽取。
    train_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4100]))
    training = np.concatenate((wt, train_rng.choice(singles, config.initial_single_count, replace=False),
                               train_rng.choice(other, other_count, replace=False)))
    training = train_rng.permutation(training)
    remaining = np.setdiff1d(np.arange(len(table)), training)
    holdout_size = config.calibration_size + config.test_size
    if len(remaining) < holdout_size + config.rounds * config.k:
        raise ValueError("Not enough rows for the disjoint roles")
    holdout_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4200]))
    # 单点变体数量少，抽取留出集时检查候选仍够用；重抽条件不涉及标签。
    for attempt in range(1, config.holdout_max_attempts + 1):
        heldout = holdout_rng.choice(remaining, holdout_size, replace=False)
        pool = np.setdiff1d(remaining, heldout)
        if int((hd.iloc[pool] == 1).sum()) >= config.reserve_single_count:
            break
    else:
        raise ValueError("Held-out resampling limit exhausted while reserving the single-mutant pool")
    # 先抽共同留出集合，再随机分成校准与测试两部分，二者都不进入训练。
    split_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4300]))
    heldout = split_rng.permutation(heldout)
    roles = {"train": table.iloc[training].reset_index(drop=True),
             "calibration": table.iloc[heldout[:config.calibration_size]].reset_index(drop=True),
             "test": table.iloc[heldout[config.calibration_size:]].reset_index(drop=True),
             "pool": table.iloc[pool].reset_index(drop=True)}
    # 每个HD组抽取相同数量的无标签候选，供同查询预算的分组对照使用。
    groups = {}
    for group in config.hd_groups:
        ids = sorted(roles["pool"].loc[roles["pool"].mutation_count == group, "variant"])
        if len(ids) < config.hd_pool_size:
            raise ValueError(f"HD{group} has fewer than the required fixed candidate count")
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 4400, group]))
        groups[str(group)] = sorted(rng.choice(ids, config.hd_pool_size, replace=False).tolist())
    metadata = {"heldout_sampling_attempts": attempt, "common_heldout_size": holdout_size,
                "heldout_split": "Random partition of one shared ID/HD-selected held-out set; no label-based selection",
                "initial_training": {"WT": 1, "HD1": config.initial_single_count, "HD2plus": other_count},
                "role_hd_counts": {role: {str(group): int((rows.mutation_count == group).sum()) for group in range(5)}
                                   for role, rows in roles.items()},
                "source_hd_counts": {str(group): int((hd == group).sum()) for group in range(5)},
                "group_pool_sizes": {key: len(values) for key, values in groups.items()},
                "group_pool_rule": "Equal-size label-blind subsets; original measurable HD spaces differ in size and composition"}
    return roles, groups, metadata


def prepare_bonus(source, destination, config, reference, variant_column="Variants", fitness_column="Fitness",
                  sheet_name="Supplementary Dataset 1", header=0):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Bonus data destination must be empty; existing splits are immutable")
    source = Path(source)
    raw = pd.read_excel(source, sheet_name=sheet_name, header=header, engine="openpyxl") if source.suffix.lower() == ".xlsx" else pd.read_csv(source, header=header)
    table, audit = clean_table(raw, variant_column, fitness_column, reference)
    roles, groups, metadata = split_bonus(table, config)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "hidden").mkdir()
    roles["train"].to_csv(destination / "train.csv", index=False)
    roles["pool"].drop(columns="fitness").to_csv(destination / "candidates.csv", index=False)
    roles["pool"][["variant", "fitness"]].to_csv(destination / "hidden/oracle.csv", index=False)
    for role in ("calibration", "test"):
        roles[role].to_csv(destination / f"hidden/{role}.csv", index=False)
    write_json(destination / "hd_pools.json", groups)
    manifest = {"format_version": 1, "study": "bonus-learning", "config": config.model_dump(),
                "source": str(source.resolve()), "audit": audit,
                "schema": {"variant_column": variant_column, "fitness_column": fitness_column, "sheet_name": sheet_name, "header": header},
                "counts": {role: len(rows) for role, rows in roles.items()}, "sampling": metadata,
                "source_calibration_note": "S1 single-substitution fitness incorporates Olson et al. 2014 measurements multiplied by 1.159 in Wu et al. 2016; S2 imputation is excluded.",
                "interpretation": "HD groups differ in original library size and calibration provenance; group contrasts are not a causal mutation-count effect"}
    write_json(destination / "manifest.json", manifest)
    return manifest


# 在公开数据视图上附加各HD组的候选ID，加载时再次检查组别和唯一性。
class BonusDataset(PublicDataset):
    def __init__(self, directory):
        super().__init__(directory)
        self.hd_pools = read_json(Path(directory) / "hd_pools.json")
        for group, variants in self.hd_pools.items():
            if len(set(variants)) != len(variants) or not set(variants) <= set(self.candidates.variant):
                raise ValueError("HD pool contains duplicate or noncandidate variants")
            if any(sum(a != b for a, b in zip(variant, "VDGV")) != int(group) for variant in variants):
                raise ValueError("HD pool contains a variant from another mutation-count group")
