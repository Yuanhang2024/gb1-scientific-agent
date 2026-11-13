# 参考结构分析：计算1PGA链A的距离、近邻和溶剂可及表面积。
"""Reference-chain geometry and Shrake-Rupley solvent-accessible surface area."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.PDB.SASA import ShrakeRupley

from gb1.data import validate_reference
from gb1.schema import POSITIONS
from gb1.storage import write_json

PDB_SOURCE = "https://www.rcsb.org/structure/1PGA"
SASA_SOURCE = "https://biopython.org/docs/latest/api/Bio.PDB.SASA.html"
CUTOFFS = (4.0, 4.5, 5.0)


# 所有几何量来自1PGA参考构象；此函数不生成突变体结构或预测fitness。
def analyze_structure(pdb_path, reference, chain_id="A"):
    """Analyze standard residues of isolated 1PGA chain A in its first model."""
    validate_reference(reference)
    structure = PDBParser(QUIET=True).get_structure("1PGA", str(pdb_path))
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError("Requested chain is absent from the first PDB model")
    # 在链的副本上剔除水和其他hetero记录，SASA范围限定为孤立标准蛋白链。
    chain = deepcopy(model[chain_id])
    for residue in list(chain):
        if residue.id[0] != " ":
            chain.detach_child(residue.id)
    residues = list(chain)
    if [residue.id for residue in residues] != [(" ", i, " ") for i in range(1, 57)]:
        raise ValueError("Structure requires standard residues numbered 1 through 56 without insertions")
    try:
        sequence = "".join(protein_letters_3to1[residue.resname] for residue in residues)
    except KeyError as exc:
        raise ValueError("Structure contains a nonstandard polymer residue") from exc
    # 结构MTY与实验MQY仅第2位不同；等长序列的目标位点编号无需偏移。
    differences = [{"position": i, "structure": a, "experiment": b}
                   for i, (a, b) in enumerate(zip(sequence, reference), 1) if a != b]
    if differences != [{"position": 2, "structure": "T", "experiment": "Q"}]:
        raise ValueError("Expected the documented 1PGA MTY / experiment MQY difference at position 2 only")
    if "".join(sequence[p - 1] for p in POSITIONS) != "VDGV":
        raise ValueError("Structure target positions do not map to VDGV")
    coordinates, ca_coordinates = {}, {}
    for position, residue in enumerate(residues, 1):
        if "CA" not in residue:
            raise ValueError(f"Residue {position} has no CA atom")
        heavy = [atom.coord for atom in residue if atom.element.strip().upper() not in ("H", "D")]
        coordinates[position] = np.asarray(heavy, dtype=float)
        ca_coordinates[position] = np.asarray(residue["CA"].coord, dtype=float)
        if not np.isfinite(coordinates[position]).all():
            raise ValueError("Atomic coordinates must be finite")
    areas = {}
    # SASA是溶剂可及表面积，单位Å²；比较两种采样点数，检查数值分辨率的影响。
    for points in (100, 960):
        ShrakeRupley(probe_radius=1.4, n_points=points).compute(chain, level="R")
        areas[points] = {i: float(residue.sasa) for i, residue in enumerate(residues, 1)}
    pairs = []
    # 同时保存Cα距离与最小重原子距离；近邻规则排除序列相邻残基，并比较多个阈值。
    for left, right in combinations(range(1, 57), 2):
        delta = coordinates[left][:, None, :] - coordinates[right][None, :, :]
        minimum = float(np.sqrt(np.sum(delta * delta, axis=-1)).min())
        row = {"position_a": left, "position_b": right,
               "ca_distance_angstrom": float(np.linalg.norm(ca_coordinates[left] - ca_coordinates[right])),
               "min_heavy_distance_angstrom": minimum, "nonadjacent": right - left > 1}
        for cutoff in CUTOFFS:
            row[f"contact_{cutoff:.1f}"] = bool(row["nonadjacent"] and minimum <= cutoff)
        pairs.append(row)
    positions = []
    for position in range(1, 57):
        row = {"position": position, "structure_residue": sequence[position - 1],
               "experiment_residue": reference[position - 1], "target_site": position in POSITIONS,
               "sasa_100_angstrom2": areas[100][position], "sasa_960_angstrom2": areas[960][position],
               "sasa_960_minus_100_angstrom2": areas[960][position] - areas[100][position]}
        for cutoff in CUTOFFS:
            neighbors = sorted(pair["position_b"] if pair["position_a"] == position else pair["position_a"]
                               for pair in pairs if position in (pair["position_a"], pair["position_b"])
                               and pair[f"contact_{cutoff:.1f}"])
            row[f"neighbors_{cutoff:.1f}"] = neighbors
            row[f"neighbor_count_{cutoff:.1f}"] = len(neighbors)
        positions.append(row)
    return {"metadata": {"pdb_id": "1PGA", "source": PDB_SOURCE, "pdb_path": str(Path(pdb_path).resolve()),
                         "model_id": int(model.id), "chain": chain_id, "structure_sequence": sequence,
                         "experiment_reference": reference, "sequence_differences": differences,
                         "target_positions": list(POSITIONS), "target_variant": "VDGV",
                         "distance_units": "angstrom", "sasa_units": "angstrom^2",
                         "contact_definition": "minimum heavy-atom distance <= cutoff and |i-j| > 1",
                         "contact_cutoffs_angstrom": list(CUTOFFS), "sasa_method": "ShrakeRupley",
                         "sasa_points": [100, 960], "probe_radius_angstrom": 1.4,
                         "sasa_source": SASA_SOURCE, "sasa_scope": "isolated chain A standard polymer residues",
                         "scope": "Static 1PGA reference conformation; geometry and absolute SASA describe this reference only."},
            "positions": positions, "pairs": pairs}


# 完整结构表和四个实验位点的子表分别保存，方便检查整体背景与目标位置。
def write_structure_tables(analysis, output):
    """Write the complete distances, residue features and four-site subsets."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Structure output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "structure.json", analysis)
    positions = pd.DataFrame(analysis["positions"])
    pairs = pd.DataFrame(analysis["pairs"])
    positions.to_csv(output / "residue_features.csv", index=False)
    positions.loc[positions.target_site].to_csv(output / "target_features.csv", index=False)
    pairs.to_csv(output / "pair_distances.csv", index=False)
    pairs.loc[pairs.position_a.isin(POSITIONS) & pairs.position_b.isin(POSITIONS)].to_csv(output / "target_distances.csv", index=False)
