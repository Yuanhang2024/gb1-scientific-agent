"""Pure request/usage checks use independent expected arguments and injected SDK doubles."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from gb1.agent import OpenAIProposer, build_request_payload, usage_counts
from gb1.compare_prompts import _usage
from gb1.feedback_ablation import _arm_summary, _arm_tables
import bonus.graph_agent as graph_agent
from test_bonus_graph import prepared as graph_prepared, synthetic_structure, FakeProposer

CONTEXT = {"note": "中文", "response_schema": {"type": "object", "properties": {"variant": {"type": "string"}},
           "required": ["variant"], "additionalProperties": False}}
USER_TEXT = '{"note": "中文", "response_schema": {"type": "object", "properties": {"variant": {"type": "string"}}, "required": ["variant"], "additionalProperties": false}}'
SCHEMA = {"type": "object", "properties": {"variant": {"type": "string"}}, "required": ["variant"], "additionalProperties": False}


@pytest.mark.parametrize("mode", ["chat", "responses"])
@pytest.mark.parametrize("format_name", ["json_schema", "json_object", "text"])
@pytest.mark.parametrize("effort", [None, "xhigh"])
def test_literal_request_contract_for_builder_sdk_and_record(mode, format_name, effort):
    config = SimpleNamespace(llm_format=format_name, llm_max_output_tokens=321, llm_reasoning_effort=effort)
    if mode == "chat":
        expected = {"model": "test-model", "messages": [{"role": "system", "content": "Scientist 中文"}, {"role": "user", "content": USER_TEXT}],
                    "max_completion_tokens": 321}
        if format_name == "json_schema":
            expected["response_format"] = {"type": "json_schema", "json_schema": {"name": "gb1_proposals", "strict": True, "schema": deepcopy(SCHEMA)}}
        elif format_name == "json_object":
            expected["response_format"] = {"type": "json_object"}
        if effort:
            expected["reasoning_effort"] = "xhigh"
    else:
        expected = {"model": "test-model", "instructions": "Scientist 中文", "input": USER_TEXT, "max_output_tokens": 321, "store": False}
        if format_name == "json_schema":
            expected["text"] = {"format": {"type": "json_schema", "name": "gb1_proposals", "strict": True, "schema": deepcopy(SCHEMA)}}
        elif format_name == "json_object":
            expected["text"] = {"format": {"type": "json_object"}}
        if effort:
            expected["reasoning"] = {"effort": "xhigh"}
    context = deepcopy(CONTEXT)
    original = deepcopy(context)
    assert build_request_payload(context, config, "test-model", mode, "Scientist 中文") == expected
    assert context == original
    captured = []
    def create(**kwargs):
        captured.append(deepcopy(kwargs))
        return SimpleNamespace(output_text="Synthetic response", status="completed", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="Synthetic response", refusal=None), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), responses=SimpleNamespace(create=create), close=lambda: None)
    proposer = OpenAIProposer(config, client=client, environ={"OPENAI_MODEL": "test-model", "OPENAI_API_MODE": mode})
    proposer.prompt = SimpleNamespace(system_prompt="Scientist 中文", metadata=lambda: {"profile": "synthetic"})
    assert proposer.generate(context)["text"] == "Synthetic response"
    assert captured == [expected]
    snapshot = {"config": {"llm_format": format_name, "llm_max_output_tokens": 321, "llm_reasoning_effort": effort},
                "api_identity": {"model_requested": "test-model", "api_mode": mode},
                "prompt_metadata": {"system_prompt": "Scientist 中文"}}
    assert graph_agent.request_payload(context, snapshot) == expected
    assert context == original


def test_payload_owns_schema_and_rejects_nonjson_input():
    config = SimpleNamespace(llm_format="json_schema", llm_max_output_tokens=321, llm_reasoning_effort=None)
    context = deepcopy(CONTEXT)
    payload = build_request_payload(context, config, "test-model", "chat", "Scientist")
    payload["response_format"]["json_schema"]["schema"]["required"].append("new-field")
    assert context == CONTEXT
    with pytest.raises(ValueError):
        build_request_payload({"bad": float("nan")}, config, "test-model", "chat", "Scientist")
    with pytest.raises(ValueError, match="api_mode"):
        build_request_payload(context, config, "test-model", "unknown", "Scientist")


USAGES = [
    (None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}),
    ({}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}),
    ({"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}, {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16, "missing_usage_count": 0}),
    ({"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}, {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16, "missing_usage_count": 0}),
    ({"prompt_tokens": 12, "completion_tokens": 4}, {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16, "missing_usage_count": 0}),
    ({"input_tokens": 12, "output_tokens": 4, "total_tokens": None}, {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16, "missing_usage_count": 0}),
    ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 0}),
    ({"total_tokens": 8}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 8, "missing_usage_count": 1}),
    ({"input_tokens": 12}, {"input_tokens": 12, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}),
    ({"input_tokens": None, "output_tokens": 4, "total_tokens": 4}, {"input_tokens": 0, "output_tokens": 4, "total_tokens": 4, "missing_usage_count": 1}),
    ({"input_tokens": 12, "output_tokens": 4, "total_tokens": 99, "input_tokens_details": {"cached_tokens": 10}, "output_tokens_details": {"reasoning_tokens": 3}},
     {"input_tokens": 12, "output_tokens": 4, "total_tokens": 99, "missing_usage_count": 0}),
    ("unavailable", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "missing_usage_count": 1}),
]


@pytest.mark.parametrize("usage, expected", USAGES)
def test_usage_rules_and_existing_aggregation_fields(usage, expected):
    original = deepcopy(usage)
    assert usage_counts(usage) == expected
    attempt = {"response": {"usage": usage}}
    assert _usage(attempt) == expected
    state = {"seed": 11, "condition": "full", "stage": "prepared", "attempts": [attempt],
             "snapshot": {"known_best_fitness": 1.0}, "errors": []}
    summary = _arm_summary(state)
    assert summary["input_tokens"] == expected["input_tokens"]
    assert summary["output_tokens"] == expected["output_tokens"]
    assert summary["total_tokens"] == expected["total_tokens"]
    assert summary["missing_usage_requests"] == expected["missing_usage_count"]
    assert usage == original


@pytest.mark.parametrize("usage, total, missing", [
    ({"prompt_tokens": 12, "completion_tokens": 4}, 16, 0),
    ({"total_tokens": 8}, 8, 1),
    ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, 0),
])
def test_graph_summary_uses_unified_rule_but_raw_csv_is_unchanged(graph_prepared, usage, total, missing):
    output, _, candidates = graph_prepared
    class UsageProposer(FakeProposer):
        def generate(self, context):
            response = super().generate(context)
            response["usage"] = deepcopy(usage)
            return response
    summary = graph_agent.run_graph_agent(output, UsageProposer(candidates.variant))
    assert summary["complete"]
    for row in summary["conditions"]:
        assert row["total_tokens"] == total and row["missing_usage_requests"] == missing
        raw = pd.read_csv(output / "seed-44" / row["condition"] / "usage.csv")
        assert json.loads(raw.loc[0, "usage"]) == usage
        if "total_tokens" not in usage:
            assert pd.isna(raw.loc[0, "total_tokens"])
        if "prompt_tokens" not in usage:
            assert pd.isna(raw.loc[0, "input_tokens"])


def test_ablation_raw_csv_preserves_unreported_counts(tmp_path):
    state = {"attempts": [{"number": 1, "status": "received", "context": {"proposal_count": 40},
                           "response": {"usage": {"prompt_tokens": 12, "completion_tokens": 4}}}]}
    _arm_tables(state, tmp_path)
    row = pd.read_csv(tmp_path / "usage.csv").iloc[0]
    assert row.input_tokens == 12 and row.output_tokens == 4 and pd.isna(row.total_tokens)
    assert json.loads(row.usage) == {"prompt_tokens": 12, "completion_tokens": 4}


def test_sdk_creation_keeps_zero_retries_without_loading_project_environment(monkeypatch):
    import openai
    import dotenv
    captured = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.append(kwargs) or SimpleNamespace())
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **kwargs: pytest.fail("Explicit test environment must not load project .env"))
    config = SimpleNamespace(llm_timeout=17)
    OpenAIProposer(config, environ={"OPENAI_MODEL": "test-model", "OPENAI_API_KEY": "synthetic-key", "OPENAI_API_MODE": "chat", "OPENAI_BASE_URL": "https://example.invalid/v1"})
    assert captured == [{"api_key": "synthetic-key", "base_url": "https://example.invalid/v1", "timeout": 17, "max_retries": 0}]
