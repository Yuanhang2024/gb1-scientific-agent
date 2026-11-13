"""Synthetic offline tests for reference geometry, knowledge visibility and paired agents."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from Bio.Data.IUPACData import protein_letters_1to3
from sklearn.ensemble import RandomForestRegressor

from bonus import graph_agent as ga
from bonus.knowledge_graph import (add_observations, build_graph, draw_subgraph, graph_knowledge, load_graph,
                                   query_distance, query_variant, save_graph)
from bonus.structure import analyze_structure, write_structure_tables
from gb1.agent import OpenAIProposer, collect_proposals
from gb1.data import describe_variant, one_hot
from gb1.prompts import load_prompt_profile
from gb1.schema import ExperimentConfig
from gb1.storage import read_json, write_json

REFERENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic_structure(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reference-structure")
    pdb = directory / "reference.pdb"
    sequence = "MT" + REFERENCE[2:]
    lines, serial = [], 1
    for position, amino_acid in enumerate(sequence, 1):
        for name, element, offset in (("CA", "C", 0.0), ("C", "C", 1.0), ("H", "H", 2.9)):
            x = (position - 1) * 3 + offset
            lines.append(f"ATOM  {serial:5d} {name:^4s} {protein_letters_1to3[amino_acid].upper():3s} A{position:4d}    "
                         f"{x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          {element:>2s}\n")
            serial += 1
    pdb.write_text("".join(lines) + "TER\nEND\n", encoding="ascii")
    return pdb, analyze_structure(pdb, REFERENCE)


def known_table(variants=("VDGV", "ADGV", "VDGA")):
    return pd.DataFrame([{**describe_variant(variant, REFERENCE), "fitness": index + 0.5}
                         for index, variant in enumerate(variants)])


def test_structure_mapping_distances_neighbors_and_sasa(synthetic_structure, tmp_path):
    _, analysis = synthetic_structure
    assert len(analysis["positions"]) == 56
    assert len(analysis["pairs"]) == 56 * 55 // 2
    assert analysis["metadata"]["sequence_differences"] == [{"position": 2, "structure": "T", "experiment": "Q"}]
    pair = next(row for row in analysis["pairs"] if row["position_a"] == 39 and row["position_b"] == 41)
    assert pair["ca_distance_angstrom"] == pytest.approx(6)
    assert pair["min_heavy_distance_angstrom"] == pytest.approx(5)
    assert pair["contact_5.0"] and not pair["contact_4.5"] and not pair["contact_4.0"]
    adjacent = next(row for row in analysis["pairs"] if row["position_a"] == 39 and row["position_b"] == 40)
    assert not adjacent["nonadjacent"] and not adjacent["contact_5.0"]
    for row in analysis["positions"]:
        assert row["sasa_100_angstrom2"] >= 0 and row["sasa_960_angstrom2"] >= 0
        assert row["neighbor_count_4.0"] <= row["neighbor_count_4.5"] <= row["neighbor_count_5.0"]
    write_structure_tables(analysis, tmp_path / "tables")
    assert len(pd.read_csv(tmp_path / "tables" / "target_features.csv")) == 4
    assert len(pd.read_csv(tmp_path / "tables" / "target_distances.csv")) == 6
    with pytest.raises(ValueError, match="empty"):
        write_structure_tables(analysis, tmp_path / "tables")


def test_structure_rejects_undocumented_reference_difference(synthetic_structure):
    pdb, _ = synthetic_structure
    with pytest.raises(ValueError, match="documented"):
        analyze_structure(pdb, "MA" + REFERENCE[2:])


def test_graph_unknown_visibility_observation_update_and_roundtrip(synthetic_structure, tmp_path):
    _, structure = synthetic_structure
    known = known_table()
    graph = build_graph(known, REFERENCE, structure)
    assert isinstance(graph, nx.MultiDiGraph)
    assert len([n for n, d in graph.nodes(data=True) if d["entity"] == "Measurement"]) == len(known)
    assert not any(d["relation"] == "improves" for _, _, d in graph.edges(data=True))
    assert all(d["entity"] == "Measurement" for _, d in graph.nodes(data=True) if "fitness" in d)
    unknown = query_variant(graph, "FDGV")
    assert unknown["result"]["observed_measurements"] == []
    assert unknown["result"]["fitness_status"] == "unknown"
    assert '"fitness":' not in json.dumps(unknown)
    assert "variant:FDGV" not in graph
    observed = query_variant(graph, "ADGV")["result"]
    assert observed["fitness_status"] == "observed"
    assert observed["observed_measurements"][0]["fitness"] == 1.5
    add_observations(graph, pd.DataFrame([{"variant": "FDGV", "fitness": 4.25}]), "round-1")
    assert query_variant(graph, "FDGV")["result"]["observed_measurements"][0]["fitness"] == 4.25
    with pytest.raises(ValueError, match="Conflicting"):
        add_observations(graph, pd.DataFrame([{"variant": "FDGV", "fitness": 9}]), "round-1")
    save_graph(graph, tmp_path / "graph.json")
    restored = load_graph(tmp_path / "graph.json")
    assert nx.node_link_data(graph, edges="edges") == nx.node_link_data(restored, edges="edges")
    assert dict(graph.edges) == dict(restored.edges)
    assert all("source_id" in attributes and "source" not in attributes
               for _, _, attributes in restored.edges(data=True))
    assert query_distance(restored, 41, 39)["result"]["min_heavy_distance_angstrom"] == 5
    assert query_variant(restored, "FDGV")["result"]["fitness_status"] == "observed"


def test_variant_properties_are_queried_from_serialized_graph(synthetic_structure, tmp_path, monkeypatch):
    _, structure = synthetic_structure
    graph = build_graph(known_table(), REFERENCE, structure)
    graph.remove_edge("aa:F", "property:aromatic", "has_property")
    graph.add_node("property:custom", entity="Property", label="custom-property")
    graph.add_node("source:custom", entity="Source", label="Synthetic property source")
    graph.add_edge("aa:F", "property:custom", key="has_property", relation="has_property", source_id="source:custom")
    save_graph(graph, tmp_path / "graph.json")
    restored = load_graph(tmp_path / "graph.json")
    def classification_must_not_run(*args):
        raise AssertionError("Variant queries must read graph relationships")
    monkeypatch.setattr("bonus.knowledge_graph.amino_acid_properties", classification_must_not_run)
    query = query_variant(restored, "FDGV")
    assert query["result"]["substitutions"][0]["properties"] == ["custom-property", "nonpolar"]
    assert "source:custom" in query["source_ids"]
    assert query["result"]["fitness_status"] == "unknown"
    assert query["result"]["observed_measurements"] == []


def test_subgraph_displays_properties_and_known_measurement(synthetic_structure, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    _, structure = synthetic_structure
    graph = build_graph(known_table(("VDGV", "ADGV", "FAGW")), REFERENCE, structure)
    original = nx.node_link_data(graph, edges="edges")
    captured = []
    save_figure = Figure.savefig
    def record_figure(figure, *args, **kwargs):
        captured.append(figure)
        return save_figure(figure, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", record_figure)
    view = draw_subgraph(graph, tmp_path / "subgraph.png", variant="FAGW")
    assert nx.node_link_data(graph, edges="edges") == original
    entities = [attributes["entity"] for _, attributes in view.nodes(data=True)]
    assert {"Position", "Variant", "Mutation", "AminoAcid", "Property", "Measurement", "Source"} <= set(entities)
    assert entities.count("Mutation") == entities.count("Position") == entities.count("Measurement") == 1
    relations = {attributes["relation"] for _, _, attributes in view.edges(data=True)}
    assert relations == {"contains", "has_measurement", "measured_in", "occurs_at", "substitutes_to", "has_property", "derived_from"}
    assert {"property:nonpolar", "property:aromatic"} <= set(view)
    assert len(view) <= 10
    assert (tmp_path / "subgraph.png").stat().st_size > 1000
    assert len(captured) == 1
    figure = captured[0]
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    labels = figure.axes[0].texts
    boxes = [label.get_window_extent(renderer) for label in labels]
    assert not any(left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1:])
    captions = " ".join(text.get_text() for text in figure.texts)
    assert "complete four-site variant" in captions and "single substitution" in captions
    assert "knowledge_graph.json" in captions


def test_graph_knowledge_queries_are_sourced_and_citable(synthetic_structure):
    _, structure = synthetic_structure
    known = known_table()
    graph = build_graph(known, REFERENCE, structure)
    knowledge, queries = graph_knowledge(graph, known)
    assert len(knowledge.rules) == len(queries) == 13
    assert all(rule.source.startswith("https://") for rule in knowledge.rules)
    context = {rule.id for rule in knowledge.rules}
    text = json.dumps({"analysis": "Test known graph facts", "proposals": [{"variant": "FDGV", "hypothesis": "Check this background",
                        "rationale": "Reference distance motivates a test", "knowledge_rule_ids": ["graph-distance-39-41"]}]})
    config = ExperimentConfig(k=1, proposal_count=1, score_budget=1)
    assert len(collect_proposals(text, {"FDGV"}, config, rule_ids=context)[1]) == 1
    assert len(collect_proposals(text.replace("graph-distance-39-41", "invented-id"), {"FDGV"}, config, rule_ids=context)[2]) == 1


@pytest.fixture
def prepared(tmp_path, synthetic_structure, monkeypatch):
    pdb, structure = synthetic_structure
    monkeypatch.setattr(ga, "analyze_structure", lambda *a, **kw: deepcopy(structure))
    variants = ["".join(chars) for chars in product("ACDV", repeat=4)][:116]
    table = pd.DataFrame([{**describe_variant(variant, REFERENCE), "fitness": float(index % 9) / 3}
                          for index, variant in enumerate(variants)])
    data = tmp_path / "data"
    data.mkdir()
    train, candidates = table.iloc[:16], table.iloc[16:]
    train.to_csv(data / "train.csv", index=False)
    candidates.drop(columns="fitness").to_csv(data / "candidates.csv", index=False)
    (data / "hidden").mkdir()
    candidates[["variant", "fitness"]].to_csv(data / "hidden" / "oracle.csv", index=False)
    write_json(data / "manifest.json", {"config": {"seed": 44}, "audit": {"reference": REFERENCE}})
    model = RandomForestRegressor(n_estimators=5, random_state=44).fit(one_hot(train.variant), train.fitness)
    model_file = tmp_path / "initial_model.joblib"
    joblib.dump(model, model_file)
    output = tmp_path / "graph-study"
    read_csv = pd.read_csv
    def public_only(path, *args, **kwargs):
        assert "hidden" not in Path(path).parts, "prepare accessed hidden labels"
        return read_csv(path, *args, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(pd, "read_csv", public_only)
        manifest = ga.prepare_graph_agent([data], [model_file], pdb, PROJECT / "configs" / "knowledge.json", output)
    return output, manifest, candidates


class FakeProposer:
    model = "gpt-6-luna"
    api_mode = "chat"
    prompt_metadata = load_prompt_profile("v3").metadata()

    def __init__(self, variants, limit=None, invalid_citation=False, timeout=False):
        self.variants = list(variants)
        self.contexts = []
        self.limit = limit
        self.invalid_citation = invalid_citation
        self.timeout = timeout

    def generate(self, context):
        self.contexts.append(deepcopy(context))
        if self.timeout:
            raise TimeoutError("Synthetic timeout")
        eligible = [v for v in self.variants if v not in context.get("forbidden_variants", [])]
        count = context["proposal_count"]
        if self.limit is not None:
            count = min(count, self.limit)
        rules = context["knowledge"]["rules"]
        proposals = [{"variant": v, "hypothesis": "Test the proposed full background", "rationale": "Use a sourced prior",
                      "knowledge_rule_ids": [rules[-1]["id"]]} for v in eligible[:count]]
        if self.invalid_citation and context["stage"] == "initial_design":
            proposals[0]["knowledge_rule_ids"] = ["nonexistent"]
        return {"text": json.dumps({"analysis": "Synthetic design", "proposals": proposals}), "complete": True,
                "model_requested": self.model, "model_returned": self.model, "api_mode": self.api_mode,
                "prompt_metadata": deepcopy(self.prompt_metadata), "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def test_prepare_pair_inputs_and_offline_status(prepared):
    output, manifest, _ = prepared
    assert manifest["max_requests"] == 4
    snapshot = manifest["snapshots"]["44"]
    a = read_json(output / "seed-44" / "static_json" / "state.json")["initial_context"]
    b = read_json(output / "seed-44" / "structure_graph" / "state.json")["initial_context"]
    assert {k: v for k, v in a.items() if k != "knowledge"} == {k: v for k, v in b.items() if k not in ("knowledge", "graph_context")}
    assert b["knowledge"]["rules"][:len(a["knowledge"]["rules"])] == a["knowledge"]["rules"]
    assert b["graph_context"] == snapshot["graph_queries"]
    assert a["candidate_examples"] == b["candidate_examples"] == []
    assert (output / "seed-44" / "knowledge_subgraph.png").stat().st_size > 1000
    status = ga.compare_graph_agent(output, require_complete=False)
    assert status["total_requests"] == 0 and not status["complete"]


def test_paired_agent_actual_proposals_selection_and_oracle_order(prepared, monkeypatch):
    output, manifest, candidates = prepared
    proposer = FakeProposer(candidates.variant)
    oracle = ga.HiddenOracle
    calls = []
    class GuardedOracle(oracle):
        def evaluate(self, variants):
            selected_states = [read_json(output / "seed-44" / c / "state.json") for c in ga.CONDITIONS]
            assert any(state["stage"] == "selected" and [row["variant"] for row in state["recommendations"]] == variants
                       for state in selected_states)
            calls.append(list(variants))
            return super().evaluate(variants)
    monkeypatch.setattr(ga, "HiddenOracle", GuardedOracle)
    summary = ga.run_graph_agent(output, proposer)
    assert summary["complete"] and summary["total_requests"] == 2
    assert len(calls) == 2 and all(len(call) == 10 for call in calls)
    for condition in ga.CONDITIONS:
        state = read_json(output / "seed-44" / condition / "state.json")
        assert len(state["proposals"]) == len(state["scores"]) == 40
        assert len(state["recommendations"]) == 10
        for attempt in state["attempts"]:
            payload = attempt["request_payload"]
            assert json.loads(payload["messages"][1]["content"]) == attempt["context"]
            assert payload["messages"][0]["content"] == proposer.prompt_metadata["system_prompt"]
            raw = json.loads(attempt["response"]["text"])
            assert state["proposals"] == raw["proposals"]
        expected = sorted(state["scores"], key=lambda row: (-row["predicted_fitness"], row["variant"]))[:10]
        assert [row["variant"] for row in state["recommendations"]] == [row["variant"] for row in expected]
    assert ga.run_graph_agent(output, proposer)["total_requests"] == 2
    assert len(proposer.contexts) == 2


def test_repair_keeps_valid_candidates_and_checks_rule_ids(prepared):
    output, _, candidates = prepared
    proposer = FakeProposer(candidates.variant, invalid_citation=True)
    summary = ga.run_graph_agent(output, proposer)
    assert summary["complete"] and summary["total_requests"] == 4
    for condition in ga.CONDITIONS:
        state = read_json(output / "seed-44" / condition / "state.json")
        assert state["attempts"][0]["validation"]["valid_count"] == 39
        repair = state["attempts"][1]["context"]
        assert repair["stage"] == "repair_only" and repair["proposal_count"] == 1
        assert len(repair["accepted_variants"]) == 39
        assert len(state["proposals"]) == 40
        if condition == "structure_graph":
            assert repair["graph_context"] == state["initial_context"]["graph_context"]


def test_underfilled_response_is_preserved_without_synthetic_completion(prepared):
    output, _, candidates = prepared
    proposer = FakeProposer(candidates.variant, limit=12)
    summary = ga.run_graph_agent(output, proposer)
    assert summary["complete"] and summary["total_requests"] == 4
    assert all(row["valid_count"] == 24 and row["scored_count"] == 24 for row in summary["conditions"])


def test_unknown_request_outcome_is_never_resent(prepared):
    output, _, candidates = prepared
    proposer = FakeProposer(candidates.variant, timeout=True)
    with pytest.raises(RuntimeError, match="unknown outcome"):
        ga.run_graph_agent(output, proposer)
    with pytest.raises(RuntimeError, match="automatic resend"):
        ga.run_graph_agent(output, proposer)
    assert len(proposer.contexts) == 1
    assert ga.compare_graph_agent(output, require_complete=False)["total_requests"] == 1


def test_saved_context_tampering_stops_before_request(prepared):
    output, _, candidates = prepared
    path = output / "seed-44" / "structure_graph" / "state.json"
    state = read_json(path)
    state["initial_context"]["graph_context"][0]["result"]["sasa_960_angstrom2"] = -1
    write_json(path, state)
    proposer = FakeProposer(candidates.variant)
    with pytest.raises(ValueError, match="snapshot"):
        ga.run_graph_agent(output, proposer)
    assert proposer.contexts == []


def test_request_snapshot_matches_unchanged_openai_sdk_arguments(prepared):
    output, manifest, _ = prepared
    snapshot = manifest["snapshots"]["44"]
    context = read_json(output / "seed-44" / "structure_graph" / "state.json")["initial_context"]
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}", refusal=None), finish_reason="stop")],
                               usage=None, model="gpt-6-luna", id="synthetic")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    proposer = OpenAIProposer(ExperimentConfig.model_validate(snapshot["config"]), client=client,
                             environ={"OPENAI_MODEL": "gpt-6-luna", "OPENAI_API_MODE": "chat"}, prompt_profile="v3")
    proposer.generate(context)
    assert captured == [ga.request_payload(context, snapshot)]


def test_saved_scores_and_metrics_are_verified(prepared):
    output, _, candidates = prepared
    ga.run_graph_agent(output, FakeProposer(candidates.variant))
    path = output / "seed-44" / "structure_graph" / "state.json"
    state = read_json(path)
    state["scores"][0]["predicted_fitness"] += 1
    write_json(path, state)
    with pytest.raises(ValueError, match="shared initial RF"):
        ga.compare_graph_agent(output)


def test_prediction_roundoff_preserves_saved_score_ranking(prepared):
    output, manifest, candidates = prepared
    ga.run_graph_agent(output, FakeProposer(candidates.variant))
    state = read_json(output / "seed-44" / "structure_graph" / "state.json")
    snapshot = manifest["snapshots"]["44"]
    state["scores"] = [{"variant": row["variant"], "predicted_fitness": 1.0} for row in state["proposals"]]
    ranked = sorted(state["proposals"], key=lambda row: row["variant"])[:10]
    state["recommendations"] = [{**describe_variant(row["variant"], REFERENCE), **row, "rank": rank,
                                 "predicted_fitness": 1.0, "actual_fitness": 0.5} for rank, row in enumerate(ranked, 1)]
    state["metrics"] = {"mean_fitness": 0.5, "max_fitness": 0.5, "median_fitness": 0.5,
                        "known_best_fitness": snapshot["known_best_fitness"]}
    recomputed = 1.0 + np.arange(len(state["scores"])) * 1e-13
    reordered = sorted(zip(state["scores"], recomputed), key=lambda pair: (-pair[1], pair[0]["variant"]))[:10]
    assert [row["variant"] for row, _ in reordered] != [row["variant"] for row in ranked]
    model = SimpleNamespace(predict=lambda features: recomputed)
    ga._validate_selection(state, snapshot, model)
    state["scores"][0]["predicted_fitness"] += 0.01
    with pytest.raises(ValueError, match="shared initial RF"):
        ga._validate_selection(state, snapshot, model)


def test_score_ids_are_checked_exactly(prepared):
    output, manifest, candidates = prepared
    ga.run_graph_agent(output, FakeProposer(candidates.variant))
    state = read_json(output / "seed-44" / "structure_graph" / "state.json")
    state["scores"][0]["variant"], state["scores"][1]["variant"] = state["scores"][1]["variant"], state["scores"][0]["variant"]
    model = joblib.load(output / manifest["snapshots"]["44"]["model_file"])
    with pytest.raises(ValueError, match="IDs or structure"):
        ga._validate_selection(state, manifest["snapshots"]["44"], model)


@pytest.mark.parametrize("returned_model", ["different-model", None])
def test_returned_model_identity_is_required_before_scoring(prepared, monkeypatch, returned_model):
    output, manifest, candidates = prepared
    assert manifest["snapshots"]["44"]["api_identity"]["model_returned"] == "gpt-6-luna"
    class UnexpectedModel(FakeProposer):
        def generate(self, context):
            response = super().generate(context)
            response["model_returned"] = returned_model
            return response
    def no_oracle(*args):
        raise AssertionError("A response from an unexpected model must not obtain feedback")
    monkeypatch.setattr(ga.HiddenOracle, "evaluate", no_oracle)
    proposer = UnexpectedModel(candidates.variant)
    with pytest.raises(ValueError, match="model/API identity"):
        ga.run_graph_agent(output, proposer)
    with pytest.raises(ValueError, match="model/API identity"):
        ga.run_graph_agent(output, proposer)
    assert len(proposer.contexts) == 1


def test_manifest_request_budget_cannot_expand(prepared):
    output, manifest, _ = prepared
    manifest["max_requests"] = 100
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="budget"):
        ga.compare_graph_agent(output, require_complete=False)
