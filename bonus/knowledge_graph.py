# 知识图谱：连接位点、突变、性质和已知测量，并为查询结果保留来源。
"""Sourced reference structure and explicitly supplied measurements in a multigraph."""
from copy import deepcopy
import json
from pathlib import Path

import networkx as nx
import numpy as np

from gb1.data import describe_variant, validate_reference
from gb1.schema import AMINO_ACIDS, POSITIONS, KnowledgeBase, KnowledgeRule
from gb1.storage import read_json, write_json

CHEMISTRY_SOURCE = "https://openstax.org/books/biology-2e/pages/3-4-proteins"
MEASUREMENT_SOURCE = "https://elifesciences.org/articles/16965"


# 这些是描述性的侧链类别，具体替换的效果仍取决于完整序列背景。
def amino_acid_properties(amino_acid):
    """Broad side-chain categories used as descriptive design priors."""
    if amino_acid not in AMINO_ACIDS or len(amino_acid) != 1:
        raise ValueError("Expected one standard amino acid")
    categories = {"nonpolar": "GAVLIMPFW", "polar_uncharged": "STCNYQ", "acidic": "DE", "basic": "KRH"}
    result = [name for name, members in categories.items() if amino_acid in members]
    if amino_acid in "FWY":
        result.append("aromatic")
    return result


# 来源使用source_id属性，避免与node-link JSON用于表示边起点的source字段混淆。
def _edge(graph, left, right, relation, source, **attributes):
    graph.add_edge(left, right, key=relation, relation=relation, source_id=source, **attributes)


# 静态结构和化学性质可入图；测量节点只由调用方传入的当前known表建立。
def build_graph(known, reference, structure, known_source="initial-train"):
    """All observed fitness values originate exclusively in the supplied known table."""
    validate_reference(reference)
    if structure["metadata"]["experiment_reference"] != reference:
        raise ValueError("Structure mapping and experimental reference differ")
    graph = nx.MultiDiGraph(format_version=1, reference=reference, observation_scope="explicit known table only",
                            structure_metadata=deepcopy(structure["metadata"]))
    graph.add_node("source:1PGA", entity="Source", label="1PGA chain A", url=structure["metadata"]["source"],
                   metadata=deepcopy(structure["metadata"]))
    graph.add_node("source:chemistry", entity="Source", label="Amino-acid chemistry", url=CHEMISTRY_SOURCE)
    graph.add_node(f"source:{known_source}", entity="Source", label=known_source, url=MEASUREMENT_SOURCE,
                   scope="Only the explicitly supplied known measurement rows")
    for amino_acid in AMINO_ACIDS:
        node = f"aa:{amino_acid}"
        graph.add_node(node, entity="AminoAcid", label=amino_acid)
        for property_name in amino_acid_properties(amino_acid):
            prop = f"property:{property_name}"
            graph.add_node(prop, entity="Property", label=property_name)
            _edge(graph, node, prop, "has_property", "source:chemistry")
    for row in structure["positions"]:
        position = row["position"]
        graph.add_node(f"position:{position}", entity="Position", label=str(position), **deepcopy(row))
        _edge(graph, f"position:{position}", "source:1PGA", "derived_from", "source:1PGA")
    for row in structure["pairs"]:
        left, right = row["position_a"], row["position_b"]
        if row["contact_5.0"] or (left in POSITIONS and right in POSITIONS):
            _edge(graph, f"position:{left}", f"position:{right}", "reference_distance", "source:1PGA", **deepcopy(row))
    for position in POSITIONS:
        for amino_acid in AMINO_ACIDS:
            original = reference[position - 1]
            if amino_acid == original:
                continue
            label = f"{original}{position}{amino_acid}"
            node = f"mutation:{label}"
            graph.add_node(node, entity="Mutation", label=label, position=position, original=original, residue=amino_acid)
            _edge(graph, node, f"position:{position}", "occurs_at", "source:1PGA")
            _edge(graph, node, f"aa:{amino_acid}", "substitutes_to", "source:chemistry")
    add_observations(graph, known, known_source)
    return graph


# 每个Measurement对应完整四位点变体及其来源，不把组合分数归因于单个突变。
def add_observations(graph, known, source_id="initial-train"):
    """Add new explicitly observed variants; conflicting repeated values are rejected."""
    if not {"variant", "fitness"} <= set(known.columns) or known.variant.duplicated().any():
        raise ValueError("Known measurements require unique variant and fitness columns")
    source = f"source:{source_id}"
    if source not in graph:
        graph.add_node(source, entity="Source", label=source_id, url=MEASUREMENT_SOURCE,
                       scope="Explicitly supplied known measurements")
    rows = []
    for raw in known[["variant", "fitness"]].to_dict(orient="records"):
        descriptor = describe_variant(raw["variant"], graph.graph["reference"])
        value = float(raw["fitness"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Known fitness must be finite and nonnegative")
        measurement_id = f"measurement:{source_id}:{raw['variant']}"
        if measurement_id in graph and graph.nodes[measurement_id]["fitness"] != value:
            raise ValueError("Conflicting measurement for this source and variant")
        rows.append((descriptor, value, measurement_id))
    for descriptor, value, measurement_id in rows:
        variant_id = f"variant:{descriptor['variant']}"
        graph.add_node(variant_id, entity="Variant", label=descriptor["variant"], **descriptor)
        for mutation in descriptor["mutations"].split(":"):
            if mutation != "WT":
                _edge(graph, variant_id, f"mutation:{mutation}", "contains", source)
        graph.add_node(measurement_id, entity="Measurement", label=f"{descriptor['variant']} fitness={value:g}",
                       variant=descriptor["variant"], fitness=value, source=source,
                       scope="Observed full four-site background")
        _edge(graph, variant_id, measurement_id, "has_measurement", source)
        _edge(graph, measurement_id, source, "measured_in", source)


def query_position(graph, position):
    if position not in range(1, 57):
        raise ValueError("Position must be between 1 and 56")
    return {"query": {"kind": "position", "position": position},
            "result": deepcopy(dict(graph.nodes[f"position:{position}"])), "source_ids": ["source:1PGA"]}


# 距离关系按较小位点在前存储，查询时先统一两个位点的顺序。
def query_distance(graph, left, right):
    left, right = sorted((left, right))
    attributes = graph.get_edge_data(f"position:{left}", f"position:{right}", "reference_distance")
    if attributes is None:
        raise ValueError("Pair has no stored reference-distance relation")
    return {"query": {"kind": "reference_distance", "positions": [left, right]},
            "result": deepcopy(attributes), "source_ids": ["source:1PGA"]}


# 从图中读取性质和已知测量；没有测量节点时明确返回unknown，不查询隐藏标签。
def query_variant(graph, variant):
    descriptor = describe_variant(variant, graph.graph["reference"])
    substitutions = []
    property_sources = set()
    for position, amino_acid in zip(POSITIONS, variant):
        properties = []
        for _, target, attributes in graph.out_edges(f"aa:{amino_acid}", data=True):
            if attributes["relation"] == "has_property":
                properties.append(graph.nodes[target]["label"])
                property_sources.add(attributes["source_id"])
        substitutions.append({"position": position, "residue": amino_acid,
                              "properties": sorted(set(properties)),
                              "reference_structure": query_position(graph, position)["result"]})
    observations = []
    variant_id = f"variant:{variant}"
    if variant_id in graph:
        for _, target, attributes in graph.out_edges(variant_id, data=True):
            if attributes["relation"] == "has_measurement":
                observations.append({"measurement_id": target, **deepcopy(dict(graph.nodes[target]))})
    return {"query": {"kind": "variant", "variant": variant},
            "result": {**descriptor, "substitutions": substitutions,
                       "fitness_status": "observed" if observations else "unknown", "observed_measurements": observations},
            "source_ids": sorted({"source:1PGA"} | property_sources | {row["source"] for row in observations})}


# 构造Agent上下文时实际查询图谱，把结果连同ID和来源转成可引用知识。
def graph_knowledge(graph, known, observation_limit=3):
    """Execute bounded queries and turn their returned facts into citable rules."""
    rules, queries = [], []
    for position in POSITIONS:
        query = query_position(graph, position)
        queries.append(query)
        result = query["result"]
        facts = {key: result[key] for key in ("position", "structure_residue", "sasa_100_angstrom2", "sasa_960_angstrom2",
                                              "neighbors_4.0", "neighbors_4.5", "neighbors_5.0")}
        rules.append(KnowledgeRule(id=f"graph-position-{position}", scope="1PGA isolated reference chain A",
                                   condition=f"Considering experimental position {position}",
                                   action="Use this reference geometry to formulate a testable structural hypothesis: " + json.dumps(facts),
                                   source=graph.nodes["source:1PGA"]["url"],
                                   explanation=graph.graph["structure_metadata"]["scope"]))
    for left_index, left in enumerate(POSITIONS):
        for right in POSITIONS[left_index + 1:]:
            query = query_distance(graph, left, right)
            queries.append(query)
            facts = {key: value for key, value in query["result"].items() if key not in ("relation", "source_id")}
            rules.append(KnowledgeRule(id=f"graph-distance-{left}-{right}", scope="1PGA isolated reference chain A",
                                       condition=f"Considering joint substitutions at {left} and {right}",
                                       action="Reference pair geometry: " + json.dumps(facts),
                                       source=graph.nodes["source:1PGA"]["url"],
                                       explanation="Distances describe the reference conformation and can guide hypotheses about interactions."))
    for variant in known.sort_values(["fitness", "variant"], ascending=[False, True]).head(observation_limit).variant:
        query = query_variant(graph, variant)
        queries.append(query)
        measurements = query["result"]["observed_measurements"]
        if not measurements:
            raise ValueError("Known observation was not found in the graph")
        rules.append(KnowledgeRule(id=f"graph-observation-{variant}", scope="Known full four-site GB1 background",
                                   condition=f"Using observed variant {variant} as a design anchor",
                                   action="Known graph measurement: " + json.dumps(measurements),
                                   source="; ".join(graph.nodes[row["source"]]["url"] for row in measurements),
                                   explanation="This measurement belongs to the stated complete variant background; component substitutions remain conditional hypotheses."))
    return KnowledgeBase(rules=rules), queries


# 保存与加载使用同一组node-link字段名，保留节点、关系及来源属性。
def save_graph(graph, path):
    write_json(path, nx.node_link_data(graph, edges="edges"))


def load_graph(path):
    graph = nx.node_link_graph(read_json(path), directed=True, multigraph=True, edges="edges")
    if not isinstance(graph, nx.MultiDiGraph) or graph.graph.get("format_version") != 1:
        raise ValueError("Expected a version 1 GB1 MultiDiGraph")
    return graph


# 图示只挑一条突变关系和完整变体的测量路径，便于阅读；完整图数据保持不变。
def draw_subgraph(graph, path, variant=None):
    """Render one complete measurement and one mutation path in fixed visual layers."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def targets(node, relation):
        return sorted((target, key) for _, target, key, data in graph.out_edges(node, keys=True, data=True)
                      if data["relation"] == relation)

    if variant is None:
        variants = sorted(node for node, data in graph.nodes(data=True) if data["entity"] == "Variant")
        variant_node = next((node for node in variants if targets(node, "contains")), variants[0] if variants else None)
    else:
        variant_node = f"variant:{variant}"
    if variant_node not in graph or not targets(variant_node, "has_measurement"):
        raise ValueError("The displayed variant must have a known complete-variant measurement")
    edges, layout = [], {variant_node: (0.0, 2.4)}
    measurement, key = targets(variant_node, "has_measurement")[0]
    edges.append((variant_node, measurement, key))
    layout[measurement] = (4.2, 2.4)
    source, key = targets(measurement, "measured_in")[0]
    edges.append((measurement, source, key))
    layout[source] = (8.4, 2.4)
    mutations = sorted(targets(variant_node, "contains"), key=lambda item: (graph.nodes[item[0]]["position"], item[0]))
    if mutations:
        mutation, key = mutations[0]
        edges.append((variant_node, mutation, key))
        layout[mutation] = (0.0, 0.0)
        position, key = targets(mutation, "occurs_at")[0]
        edges.append((mutation, position, key))
        layout[position] = (4.2, 0.0)
        structure_source, key = targets(position, "derived_from")[0]
        edges.append((position, structure_source, key))
        layout[structure_source] = (8.4, 0.0)
        amino_acid, key = targets(mutation, "substitutes_to")[0]
        edges.append((mutation, amino_acid, key))
        layout[amino_acid] = (0.0, -2.4)
        properties = targets(amino_acid, "has_property")
        for index, (property_node, key) in enumerate(properties):
            edges.append((amino_acid, property_node, key))
            layout[property_node] = (4.2, -2.4 + (len(properties) - 1) * 0.75 - index * 1.5)
    view = graph.edge_subgraph(edges)
    colors = {"Position": "#d6eaf8", "Variant": "#d5f5e3", "Mutation": "#fcf3cf",
              "Measurement": "#fadbd8", "Source": "#e8daef", "AminoAcid": "#fdebd0", "Property": "#e5e7e9"}
    figure, axes = plt.subplots(figsize=(15, 9))
    width, height = 2.75, 0.9
    for left, right, data in view.edges(data=True):
        start, end = np.array(layout[left]), np.array(layout[right])
        delta = end - start
        scale = min((width / 2 + 0.08) / abs(delta[0]) if delta[0] else float("inf"),
                    (height / 2 + 0.08) / abs(delta[1]) if delta[1] else float("inf"))
        axes.add_patch(FancyArrowPatch(start + scale * delta, end - scale * delta,
                                      arrowstyle="-|>", mutation_scale=16, color="#566573", linewidth=1.3))
        midpoint = (start + end) / 2
        vertical = abs(delta[0]) < 0.01
        axes.text(midpoint[0] + (0.12 if vertical else 0), midpoint[1] + (0 if vertical else 0.16),
                  data["relation"], fontsize=9, ha="left" if vertical else "center", va="center",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.96, "pad": 2})
    for node, data in view.nodes(data=True):
        x, y = layout[node]
        axes.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                      boxstyle="round,pad=0.04,rounding_size=0.08", linewidth=1.2,
                                      edgecolor="#566573", facecolor=colors[data["entity"]]))
        if data["entity"] == "Measurement":
            label = f"{data['variant']} | fitness = {data['fitness']:.6g}"
        else:
            label = data["label"].replace("_", " ")
        label = textwrap.fill(label, width=25)
        axes.text(x, y, f"{data['entity']}\n{label}", fontsize=10, ha="center", va="center")
    axes.set_title("GB1: one measured variant and one mutation path", fontsize=16, pad=20)
    axes.set_xlim(-1.8, 10.2)
    axes.set_ylim(-4.0, 3.3)
    axes.axis("off")
    measurement_scope = ("The fitness measurement belongs to the complete four-site variant, not to the displayed single substitution."
                         if mutations else "The fitness measurement belongs to the complete four-site reference variant.")
    caption = (measurement_scope + "\n"
               "This is a selected graph view. The complete graph is stored separately in knowledge_graph.json.")
    figure.text(0.5, 0.045, caption, fontsize=11, ha="center", va="center", linespacing=1.6)
    figure.subplots_adjust(left=0.035, right=0.985, top=0.88, bottom=0.14)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return view
