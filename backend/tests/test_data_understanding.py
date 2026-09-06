"""Tests for the Data Understanding Agent.

Covers the deterministic tool functions in isolation (pure pandas, no
network), the LLM-free quick_stats() heuristic, both API endpoints, and the
agentic Claude tool-use loop against a mocked client — no real
ANTHROPIC_API_KEY is required for any of these.
"""

import io
import os
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.agents.data_understanding import (
    DataUnderstandingAgent,
    check_duplicates,
    get_correlations,
    inspect_column,
    list_columns,
    quick_stats,
    sample_rows,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 45, 22, 40, 25, None],
            "income": [50000, 60000, 70000, 80000, 50000, 55000],
            "city": ["NY", "LA", "NY", "SF", "NY", "LA"],
            "target": [0, 1, 1, 0, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# list_columns
# ---------------------------------------------------------------------------


def test_list_columns_reports_missing_and_unique(sample_df):
    overview = list_columns(sample_df)
    by_name = {c["name"]: c for c in overview}
    assert by_name["age"]["missing_count"] == 1
    assert by_name["age"]["n_unique"] == 4
    assert by_name["city"]["n_unique"] == 3
    assert by_name["city"]["missing_count"] == 0


# ---------------------------------------------------------------------------
# inspect_column
# ---------------------------------------------------------------------------


def test_inspect_column_numeric_stats(sample_df):
    result = inspect_column(sample_df, "income")
    assert result["is_numeric"] is True
    assert result["count"] == 6
    assert result["mean"] == pytest.approx(60833.33, rel=1e-3)
    assert result["min"] == 50000
    assert result["max"] == 80000
    assert "skew" in result
    assert "std" in result


def test_inspect_column_categorical_top_values(sample_df):
    result = inspect_column(sample_df, "city")
    assert result["is_numeric"] is False
    assert result["cardinality"] == 3
    top = {v["value"]: v["count"] for v in result["top_values"]}
    assert top["NY"] == 3
    assert top["LA"] == 2
    assert top["SF"] == 1


def test_inspect_column_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        inspect_column(sample_df, "does_not_exist")


# ---------------------------------------------------------------------------
# sample_rows
# ---------------------------------------------------------------------------


def test_sample_rows_returns_requested_count(sample_df):
    rows = sample_rows(sample_df, n=3)
    assert len(rows) == 3
    assert set(rows[0].keys()) == set(sample_df.columns)


def test_sample_rows_caps_at_dataframe_length(sample_df):
    rows = sample_rows(sample_df, n=100)
    assert len(rows) == len(sample_df)


# ---------------------------------------------------------------------------
# check_duplicates
# ---------------------------------------------------------------------------


def test_check_duplicates_counts_exact_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    result = check_duplicates(df)
    assert result["n_duplicate_rows"] == 1
    assert result["percent"] == pytest.approx(25.0)
    assert len(result["example_rows"]) == 1


def test_check_duplicates_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = check_duplicates(df)
    assert result["n_duplicate_rows"] == 0
    assert result["example_rows"] == []


# ---------------------------------------------------------------------------
# get_correlations
# ---------------------------------------------------------------------------


def test_get_correlations_finds_strongly_correlated_pair():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 1, 9, 2, 7],
        }
    )
    pairs = get_correlations(df, threshold=0.85)
    assert len(pairs) == 1
    assert {pairs[0]["column_a"], pairs[0]["column_b"]} == {"a", "b"}
    assert pairs[0]["correlation"] == pytest.approx(1.0)


def test_get_correlations_respects_threshold():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    assert get_correlations(df, threshold=0.999)
    assert get_correlations(df, threshold=1.5) == []


def test_get_correlations_ignores_non_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "label": ["x", "y", "z"]})
    assert get_correlations(df) == []


# ---------------------------------------------------------------------------
# quick_stats (deterministic report)
# ---------------------------------------------------------------------------


def test_quick_stats_produces_full_report(sample_df):
    report = quick_stats(sample_df, "ds-1")
    assert report.dataset_id == "ds-1"
    assert report.n_rows == 6
    assert report.n_columns == 4
    assert report.duplicates is not None
    assert report.correlations is not None
    assert report.target_candidate == "target"
    assert report.problem_type == "classification"
    assert report.narrative
    assert report.key_findings


def test_quick_stats_flags_duplicates_in_key_findings():
    df = pd.DataFrame({"a": [1, 1, 2], "target": [0, 0, 1]})
    report = quick_stats(df, "ds-2")
    assert any("duplicate" in f.lower() for f in report.key_findings)


# ---------------------------------------------------------------------------
# Agentic loop (mocked Anthropic client — no real API key needed)
# ---------------------------------------------------------------------------


def _tool_use_block(name, tool_input, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeAnthropicClient:
    """Stand-in for anthropic.Anthropic: returns a scripted sequence of
    responses regardless of arguments, so no real API key is needed."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        # Snapshot the messages list — the real API sends a point-in-time
        # copy, but the agent's `messages` list is mutated in place after
        # each call, so a bare reference would silently show later state.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def test_agent_run_executes_tools_and_builds_report_from_tool_results(sample_df):
    responses = [
        _response("tool_use", [_tool_use_block("check_duplicates", {}, "call_1")]),
        _response(
            "tool_use",
            [_tool_use_block("get_correlations", {"threshold": 0.85}, "call_2")],
        ),
        _response(
            "tool_use",
            [
                _tool_use_block(
                    "submit_report",
                    {
                        "target_candidate": "target",
                        "problem_type": "classification",
                        "reasoning": "Target looks binary and categorical.",
                        "narrative": (
                            "A small dataset with a binary target column and no "
                            "major issues."
                        ),
                        "key_findings": [
                            "No duplicate rows found.",
                            "No highly correlated columns.",
                        ],
                    },
                    "call_3",
                )
            ],
        ),
    ]
    client = FakeAnthropicClient(responses)
    agent = DataUnderstandingAgent(client=client)

    report = agent.run(sample_df, "ds-42")

    assert len(client.calls) == 3
    assert report.dataset_id == "ds-42"
    assert report.target_candidate == "target"
    assert report.problem_type == "classification"
    assert report.reasoning == "Target looks binary and categorical."
    # Factual fields must come from the actual tool results, not from
    # Claude's free text.
    assert report.duplicates is not None
    assert report.duplicates.n_duplicate_rows == check_duplicates(sample_df)["n_duplicate_rows"]
    expected_correlations = get_correlations(sample_df, threshold=0.85)
    assert [c.model_dump() for c in report.correlations] == expected_correlations


def test_agent_run_forces_submission_after_tool_call_cap(sample_df):
    responses = [
        _response("tool_use", [_tool_use_block("check_duplicates", {}, f"call_{i}")])
        for i in range(8)
    ]
    responses.append(
        _response(
            "tool_use",
            [
                _tool_use_block(
                    "submit_report",
                    {
                        "target_candidate": None,
                        "problem_type": "unclear",
                        "reasoning": "Ran out of budget.",
                        "narrative": "Investigation capped at the tool-call budget.",
                        "key_findings": [],
                    },
                    "call_final",
                )
            ],
        )
    )
    client = FakeAnthropicClient(responses)
    agent = DataUnderstandingAgent(client=client)

    report = agent.run(sample_df, "ds-cap")

    assert len(client.calls) == 9
    assert report.problem_type == "unclear"


def test_agent_run_marks_failed_tool_call_as_error(sample_df):
    responses = [
        _response(
            "tool_use",
            [_tool_use_block("inspect_column", {"column_name": "does_not_exist"}, "call_1")],
        ),
        _response(
            "tool_use",
            [
                _tool_use_block(
                    "submit_report",
                    {
                        "target_candidate": "target",
                        "problem_type": "classification",
                        "reasoning": "Recovered after a bad column lookup.",
                        "narrative": (
                            "One tool call failed but the agent adapted and "
                            "submitted a report."
                        ),
                        "key_findings": [],
                    },
                    "call_2",
                )
            ],
        ),
    ]
    client = FakeAnthropicClient(responses)
    agent = DataUnderstandingAgent(client=client)

    report = agent.run(sample_df, "ds-err")

    second_call_messages = client.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert tool_result_message["content"][0]["is_error"] is True
    assert report.reasoning == "Recovered after a bad column lookup."


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY for manual sanity-checking",
)
def test_agent_run_against_live_api(sample_df):
    import anthropic

    agent = DataUnderstandingAgent(client=anthropic.Anthropic())
    report = agent.run(sample_df, "ds-live")
    assert report.problem_type in {"classification", "regression", "unclear"}


def test_agent_default_client_comes_from_configured_provider(monkeypatch):
    """No client passed -> the agent must go through get_client(), i.e.
    AGENTDS_LLM_PROVIDER actually routes construction, not just a hardcoded
    anthropic.Anthropic()."""
    sentinel = object()
    monkeypatch.setattr(
        "app.agents.data_understanding.get_client", lambda: sentinel
    )
    agent = DataUnderstandingAgent()
    assert agent.client is sentinel


def test_agent_explicit_client_bypasses_provider_selection(monkeypatch):
    """An explicitly-passed client (as every mocked-loop test above does)
    must never trigger get_client() / provider selection at all."""

    def boom():
        raise AssertionError("get_client() should not be called when client= is given")

    monkeypatch.setattr("app.agents.data_understanding.get_client", boom)
    sentinel = object()
    agent = DataUnderstandingAgent(client=sentinel)
    assert agent.client is sentinel


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client_and_dataset_id(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    test_client = TestClient(app)

    csv_bytes = b"age,income,target\n25,50000,0\n30,60000,1\n25,50000,0\n"
    resp = test_client.post(
        "/datasets/upload",
        files={"file": ("people.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    dataset_id = resp.json()["dataset_id"]
    return test_client, dataset_id


def test_quick_stats_endpoint_returns_report(client_and_dataset_id):
    test_client, dataset_id = client_and_dataset_id
    resp = test_client.post(f"/datasets/{dataset_id}/quick-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dataset_id
    assert body["n_rows"] == 3
    assert body["n_columns"] == 3


def test_quick_stats_endpoint_unknown_dataset_404(client_and_dataset_id):
    test_client, _ = client_and_dataset_id
    resp = test_client.post("/datasets/does-not-exist/quick-stats")
    assert resp.status_code == 404


def test_analyze_endpoint_503_when_ollama_unreachable_default_provider(
    client_and_dataset_id, monkeypatch
):
    """AGENTDS_LLM_PROVIDER defaults to "ollama" for /analyze. If Ollama
    isn't reachable, this must 503 with a clear message naming the host —
    not a raw connection-error traceback, and not silently succeed just
    because ANTHROPIC_API_KEY happens to be unset or set."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    test_client, dataset_id = client_and_dataset_id
    resp = test_client.post(f"/datasets/{dataset_id}/analyze")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_analyze_endpoint_503_when_anthropic_provider_missing_key(
    client_and_dataset_id, monkeypatch
):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    test_client, dataset_id = client_and_dataset_id
    resp = test_client.post(f"/datasets/{dataset_id}/analyze")
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_analyze_endpoint_caches_report(client_and_dataset_id, monkeypatch):
    from app.storage.dataset_store import get_report

    test_client, dataset_id = client_and_dataset_id
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    canned = _response(
        "tool_use",
        [
            _tool_use_block(
                "submit_report",
                {
                    "target_candidate": "target",
                    "problem_type": "classification",
                    "reasoning": "binary target",
                    "narrative": "small clean dataset",
                    "key_findings": [],
                },
                "call_1",
            )
        ],
    )
    monkeypatch.setattr(
        "app.routers.analyze.DataUnderstandingAgent",
        lambda: DataUnderstandingAgent(client=FakeAnthropicClient([canned])),
    )

    assert get_report(dataset_id) is None
    resp = test_client.post(f"/datasets/{dataset_id}/analyze")
    assert resp.status_code == 200
    cached = get_report(dataset_id)
    assert cached is not None
    assert cached["dataset_id"] == dataset_id
    assert cached["problem_type"] == "classification"
