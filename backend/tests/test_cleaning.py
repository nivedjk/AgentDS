"""Tests for the Cleaning Agent (Module 2)."""

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.cleaning import (
    MAX_ACTION_CALLS,
    MAX_TOTAL_TOOL_CALLS,
    CleaningAgent,
    CleaningReport,
    CleaningStep,
    SubmitCleaningReportInput,
    apply_drop_column,
    apply_encode_column,
    apply_flag_outliers,
    apply_impute,
    apply_remove_duplicates,
    apply_scale_column,
)


def test_submit_cleaning_report_input_requires_list_summary():
    ok = SubmitCleaningReportInput(summary=["imputed age with median"], remaining_issues=[])
    assert ok.summary == ["imputed age with median"]
    with pytest.raises(ValidationError):
        SubmitCleaningReportInput(summary="a string, not a list", remaining_issues=[])


def test_cleaning_report_builds_from_steps():
    step = CleaningStep(
        order=1,
        tool="impute_column",
        reason="age had 12% missing",
        params={"column": "age", "strategy": "median"},
        result={"missing_before": 6, "missing_after": 0},
    )
    report = CleaningReport(
        dataset_id="ds-1",
        target_candidate="target",
        problem_type="classification",
        initial_shape=[100, 5],
        final_shape=[98, 5],
        initial_missing_cells=6,
        final_missing_cells=0,
        steps=[step],
        summary=["imputed age"],
        remaining_issues=[],
    )
    assert report.cleaned_dataset_id is None
    assert report.steps[0].tool == "impute_column"


@pytest.fixture
def df_missing():
    return pd.DataFrame(
        {
            "age": [25.0, None, 22.0, 40.0, None, 30.0],
            "city": ["NY", "LA", None, "SF", "NY", "LA"],
            "target": [0, 1, 1, 0, 0, 1],
        }
    )


def test_apply_impute_median_fills_all_missing(df_missing):
    new_df, result = apply_impute(df_missing, "age", "median")
    assert result["missing_before"] == 2
    assert result["missing_after"] == 0
    assert result["fill_value"] == 27.5
    assert new_df["age"].isna().sum() == 0


def test_apply_impute_mode_on_categorical(df_missing):
    new_df, result = apply_impute(df_missing, "city", "mode")
    assert result["missing_after"] == 0
    assert new_df["city"].isna().sum() == 0
    assert result["fill_value"] == "LA"  # first mode of ["NY","LA","SF","NY","LA"]


def test_apply_impute_constant_requires_value(df_missing):
    with pytest.raises(ValueError, match="constant_value"):
        apply_impute(df_missing, "city", "constant")


def test_apply_impute_mean_rejects_non_numeric(df_missing):
    with pytest.raises(ValueError, match="not numeric"):
        apply_impute(df_missing, "city", "mean")


def test_apply_impute_drop_rows_removes_na_rows(df_missing):
    new_df, result = apply_impute(df_missing, "age", "drop_rows")
    assert len(new_df) == 4
    assert result["missing_before"] == 2
    assert result["missing_after"] == 0


def test_apply_impute_drop_rows_guarded_when_over_half_lost():
    df = pd.DataFrame({"a": [1.0, None, None, None, 5.0]})
    with pytest.raises(ValueError, match="remove 3 of 5 rows"):
        apply_impute(df, "a", "drop_rows")


def test_apply_impute_unknown_column(df_missing):
    with pytest.raises(ValueError, match="not found"):
        apply_impute(df_missing, "nope", "median")


def test_apply_drop_column_removes_and_lists_remaining(df_missing):
    new_df, result = apply_drop_column(df_missing, "city")
    assert "city" not in new_df.columns
    assert result["dropped"] is True
    assert result["columns_remaining"] == ["age", "target"]


def test_apply_drop_column_unknown(df_missing):
    with pytest.raises(ValueError, match="not found"):
        apply_drop_column(df_missing, "nope")


def test_apply_remove_duplicates_counts_full_row_dupes():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    new_df, result = apply_remove_duplicates(df)
    assert result["rows_before"] == 4
    assert result["rows_after"] == 3
    assert result["rows_removed"] == 1


def test_apply_remove_duplicates_guarded_when_over_half_lost():
    df = pd.DataFrame({"a": [1, 1, 1, 1, 2]})
    with pytest.raises(ValueError, match="remove 3 of 5 rows"):
        apply_remove_duplicates(df)


@pytest.fixture
def df_outlier():
    # 1000 is a clear high outlier; the rest are 10-19
    return pd.DataFrame({"v": [10, 11, 12, 13, 14, 15, 16, 17, 18, 1000]})


def test_flag_outliers_iqr_flag_adds_boolean_column(df_outlier):
    new_df, result = apply_flag_outliers(df_outlier, "v", "iqr", "flag")
    assert result["n_affected"] == 1
    assert result["new_column"] == "v_outlier"
    assert new_df["v_outlier"].tolist() == [False] * 9 + [True]
    assert result["bounds"]["upper"] < 1000


def test_flag_outliers_iqr_cap_winsorizes(df_outlier):
    new_df, result = apply_flag_outliers(df_outlier, "v", "iqr", "cap")
    assert new_df["v"].max() == result["bounds"]["upper"]
    assert len(new_df) == 10


def test_flag_outliers_zscore_remove_drops_rows():
    df = pd.DataFrame({"v": list(range(30)) + [10_000]})
    new_df, result = apply_flag_outliers(df, "v", "zscore", "remove")
    assert result["n_affected"] == 1
    assert result["rows_after"] == 30
    assert len(new_df) == 30
    assert 10_000 not in new_df["v"].tolist()


def test_flag_outliers_zero_spread_is_a_noop():
    df = pd.DataFrame({"v": [5, 5, 5, 5]})
    new_df, result = apply_flag_outliers(df, "v", "zscore", "flag")
    assert result["n_affected"] == 0
    assert new_df["v_outlier"].tolist() == [False] * 4


def test_flag_outliers_rejects_non_numeric():
    df = pd.DataFrame({"c": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="not numeric"):
        apply_flag_outliers(df, "c", "iqr", "flag")


def test_encode_one_hot_creates_prefixed_int_columns():
    df = pd.DataFrame({"city": ["NY", "LA", "NY", "SF"], "n": [1, 2, 3, 4]})
    new_df, result = apply_encode_column(df, "city", "one_hot")
    assert "city" not in new_df.columns
    assert set(result["new_columns"]) == {"city_LA", "city_NY", "city_SF"}
    assert result["original_dropped"] is True
    assert new_df["city_NY"].tolist() == [1, 0, 1, 0]


def test_encode_label_is_deterministic_by_sorted_value():
    df = pd.DataFrame({"size": ["m", "s", "l", "m", None]})
    new_df, result = apply_encode_column(df, "size", "label")
    assert result["mapping"] == {"l": 0, "m": 1, "s": 2}
    assert new_df["size"].tolist()[:4] == [1, 2, 0, 1]
    assert pd.isna(new_df["size"].tolist()[4])


def test_encode_rejects_numeric_column():
    df = pd.DataFrame({"n": [1, 2, 3]})
    with pytest.raises(ValueError, match="numeric"):
        apply_encode_column(df, "n", "one_hot")


def test_scale_standard_centers_and_unit_scales():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    new_df, result = apply_scale_column(df, "x", "standard")
    assert result["before"]["mean"] == 25.0
    assert new_df["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert new_df["x"].std() == pytest.approx(1.0, rel=1e-9)
    assert result["after"]["std"] == pytest.approx(1.0, rel=1e-9)


def test_scale_minmax_maps_to_unit_interval():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    new_df, result = apply_scale_column(df, "x", "minmax")
    assert new_df["x"].min() == 0.0
    assert new_df["x"].max() == 1.0
    assert result["before"] == {"min": 10.0, "max": 40.0}


def test_scale_robust_uses_median_and_iqr():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    new_df, result = apply_scale_column(df, "x", "robust")
    # median 3, IQR = 4 - 2 = 2  ->  (x-3)/2
    assert new_df["x"].tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_scale_rejects_zero_variance():
    df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
    with pytest.raises(ValueError, match="zero"):
        apply_scale_column(df, "x", "standard")


def test_scale_rejects_non_numeric():
    df = pd.DataFrame({"c": ["a", "b"]})
    with pytest.raises(ValueError, match="not numeric"):
        apply_scale_column(df, "c", "minmax")


@pytest.fixture
def prepared_agent(df_missing):
    agent = CleaningAgent(client=object())  # client unused by _execute_tool
    agent._prepare(
        df_missing,
        "ds-1",
        {"target_candidate": "target", "problem_type": "classification"},
    )
    return agent


def test_execute_tool_impute_mutates_df_and_logs_step(prepared_agent):
    result = prepared_agent._execute_tool(
        "impute_column",
        {"column": "age", "strategy": "median", "reason": "age is 33% missing"},
    )
    assert result["missing_after"] == 0
    assert prepared_agent.df["age"].isna().sum() == 0
    assert len(prepared_agent.steps) == 1
    step = prepared_agent.steps[0]
    assert step.order == 1
    assert step.tool == "impute_column"
    assert step.reason == "age is 33% missing"
    assert step.params == {"column": "age", "strategy": "median"}


def test_execute_tool_inspect_column_does_not_log_a_step(prepared_agent):
    out = prepared_agent._execute_tool("inspect_column", {"column_name": "age"})
    assert out["name"] == "age"
    assert prepared_agent.steps == []


def test_execute_tool_blocks_dropping_the_target(prepared_agent):
    with pytest.raises(ValueError, match="target column"):
        prepared_agent._execute_tool(
            "drop_column", {"column": "target", "reason": "unwanted"}
        )
    assert "target" in prepared_agent.df.columns
    assert prepared_agent.steps == []


def test_execute_tool_references_current_state_after_a_drop(prepared_agent):
    prepared_agent._execute_tool("drop_column", {"column": "city", "reason": "too sparse"})
    with pytest.raises(ValueError, match="not found"):
        prepared_agent._execute_tool(
            "impute_column", {"column": "city", "strategy": "mode", "reason": "retry"}
        )
    assert len(prepared_agent.steps) == 1


def test_execute_tool_steps_are_ordered(prepared_agent):
    prepared_agent._execute_tool("impute_column", {"column": "age", "strategy": "median", "reason": "a"})
    prepared_agent._execute_tool("impute_column", {"column": "city", "strategy": "mode", "reason": "b"})
    assert [s.order for s in prepared_agent.steps] == [1, 2]
    assert [s.tool for s in prepared_agent.steps] == ["impute_column", "impute_column"]


# ---------------------------------------------------------------------------
# Run loop test helpers and tests
# ---------------------------------------------------------------------------


def _tool_use_block(name, tool_input, block_id):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


CTX = {"target_candidate": "target", "problem_type": "classification"}


def _submit_block(block_id, summary=None, remaining=None):
    return _tool_use_block(
        "submit_cleaning_report",
        {"summary": summary or ["done"], "remaining_issues": remaining or []},
        block_id,
    )


def test_run_executes_ordered_steps_and_builds_report(df_missing):
    responses = [
        _response("tool_use", [_tool_use_block(
            "impute_column", {"column": "age", "strategy": "median", "reason": "33% missing"}, "c1"
        )]),
        _response("tool_use", [_tool_use_block(
            "encode_column", {"column": "city", "method": "one_hot", "reason": "categorical"}, "c2"
        )]),
        _response("tool_use", [_submit_block("c3", summary=["imputed age", "one-hot city"])]),
    ]
    client = FakeAnthropicClient(responses)
    report = CleaningAgent(client=client).run(df_missing, "ds-1", CTX)

    assert len(client.calls) == 3
    assert report.dataset_id == "ds-1"
    assert report.cleaned_dataset_id is None
    assert report.target_candidate == "target"
    assert report.problem_type == "classification"
    assert [s.tool for s in report.steps] == ["impute_column", "encode_column"]
    assert [s.order for s in report.steps] == [1, 2]
    assert report.initial_shape == [6, 3]
    assert report.initial_missing_cells == 3
    assert report.final_missing_cells == 0
    assert report.summary == ["imputed age", "one-hot city"]


def test_run_feeds_tool_errors_back_and_recovers(df_missing):
    responses = [
        _response("tool_use", [_tool_use_block(
            "drop_column", {"column": "target", "reason": "unwanted"}, "c1"
        )]),
        _response("tool_use", [_submit_block("c2", summary=["left target alone"])]),
    ]
    client = FakeAnthropicClient(responses)
    report = CleaningAgent(client=client).run(df_missing, "ds-1", CTX)

    second_call_msgs = client.calls[1]["messages"]
    tool_result_msg = second_call_msgs[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["is_error"] is True
    assert report.steps == []
    assert report.summary == ["left target alone"]


def test_run_forces_submission_after_action_cap(df_missing):
    # 25 successful impute calls (idempotent after the first), then a forced submit
    responses = [
        _response("tool_use", [_tool_use_block(
            "impute_column", {"column": "age", "strategy": "median", "reason": f"r{i}"}, f"c{i}"
        )])
        for i in range(MAX_ACTION_CALLS)
    ]
    responses.append(_response("tool_use", [_submit_block("cfinal", summary=["capped"])]))
    client = FakeAnthropicClient(responses)
    report = CleaningAgent(client=client).run(df_missing, "ds-1", CTX)

    assert len(client.calls) == MAX_ACTION_CALLS + 1
    forcing_msg = client.calls[-1]["messages"][-1]
    assert forcing_msg["role"] == "user"
    assert "submit_cleaning_report" in forcing_msg["content"]
    assert report.summary == ["capped"]


def test_run_total_call_backstop_trips_on_inspect_only_loop(df_missing):
    responses = [
        _response("tool_use", [_tool_use_block("inspect_column", {"column_name": "age"}, f"c{i}")])
        for i in range(MAX_TOTAL_TOOL_CALLS)
    ]
    responses.append(_response("tool_use", [_submit_block("cfinal", summary=["inspected a lot"])]))
    client = FakeAnthropicClient(responses)
    report = CleaningAgent(client=client).run(df_missing, "ds-1", CTX)

    assert len(client.calls) == MAX_TOTAL_TOOL_CALLS + 1
    assert report.steps == []  # inspect_column never logs a step
    assert report.summary == ["inspected a lot"]


def test_run_synthesizes_summary_when_never_submitted(df_missing):
    responses = [
        _response("tool_use", [_tool_use_block(
            "impute_column", {"column": "age", "strategy": "median", "reason": f"r{i}"}, f"c{i}"
        )])
        for i in range(MAX_ACTION_CALLS)
    ]
    # after the forcing prompt the model still refuses to submit:
    responses.append(_response("end_turn", [SimpleNamespace(type="text", text="no")]))
    client = FakeAnthropicClient(responses)
    report = CleaningAgent(client=client).run(df_missing, "ds-1", CTX)

    assert report.summary == [
        "Cleaning halted at the tool-call budget without a final summary."
    ]
    assert report.remaining_issues == []
    assert len(report.steps) >= 1


def test_agent_default_client_comes_from_configured_provider(monkeypatch):
    """No client passed -> the agent must go through get_client(), i.e.
    AGENTDS_LLM_PROVIDER actually routes construction, not just a hardcoded
    anthropic.Anthropic()."""
    sentinel = object()
    monkeypatch.setattr("app.agents.cleaning.get_client", lambda: sentinel)
    agent = CleaningAgent()
    assert agent.client is sentinel


def test_agent_explicit_client_bypasses_provider_selection(monkeypatch):
    """An explicitly-passed client (as every mocked-loop test above does)
    must never trigger get_client() / provider selection at all."""

    def boom():
        raise AssertionError("get_client() should not be called when client= is given")

    monkeypatch.setattr("app.agents.cleaning.get_client", boom)
    sentinel = object()
    agent = CleaningAgent(client=sentinel)
    assert agent.client is sentinel


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    tc = TestClient(app)
    csv_bytes = b"age,income,target\n25,50000,0\n30,60000,1\n25,50000,0\n40,,1\n"
    resp = tc.post(
        "/datasets/upload",
        files={"file": ("people.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    return tc, resp.json()["dataset_id"]


def test_clean_endpoint_503_when_ollama_unreachable_default_provider(clean_client, monkeypatch):
    """AGENTDS_LLM_PROVIDER defaults to "ollama" for /clean. If Ollama isn't
    reachable, this must 503 naming the host — not a raw connection-error
    traceback, and not silently succeed because ANTHROPIC_API_KEY happens
    to be set or unset."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    tc, dataset_id = clean_client
    resp = tc.post(f"/datasets/{dataset_id}/clean")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_clean_endpoint_503_when_anthropic_provider_missing_key(clean_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tc, dataset_id = clean_client
    resp = tc.post(f"/datasets/{dataset_id}/clean")
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_clean_endpoint_unknown_dataset_404(clean_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tc, _ = clean_client
    resp = tc.post("/datasets/does-not-exist/clean")
    assert resp.status_code == 404


def test_clean_endpoint_runs_module1_when_no_cached_report(clean_client, monkeypatch):
    from app.storage.dataset_store import get_cleaning_report, get_dataset_path, get_report

    tc, dataset_id = clean_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_du_run(self, df, ds_id):
        from app.agents.data_understanding import DataUnderstandingReport, ColumnOverview

        return DataUnderstandingReport(
            dataset_id=ds_id,
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in __import__(
                "app.agents.data_understanding", fromlist=["list_columns"]
            ).list_columns(df)],
            problem_type="classification",
            target_candidate="target",
            reasoning="stub",
            narrative="stub",
            key_findings=[],
        )

    def fake_cleaning_run(self, df, ds_id, context_report):
        from app.agents.cleaning import CleaningReport

        self.df = df.copy()
        return CleaningReport(
            dataset_id=ds_id,
            target_candidate=context_report.get("target_candidate"),
            problem_type=context_report.get("problem_type", "unclear"),
            initial_shape=[len(df), df.shape[1]],
            final_shape=[len(df), df.shape[1]],
            initial_missing_cells=int(df.isna().sum().sum()),
            final_missing_cells=int(df.isna().sum().sum()),
            steps=[],
            summary=["stub run"],
            remaining_issues=[],
        )

    monkeypatch.setattr(
        "app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run
    )
    monkeypatch.setattr("app.agents.cleaning.CleaningAgent.run", fake_cleaning_run)

    assert get_report(dataset_id) is None
    resp = tc.post(f"/datasets/{dataset_id}/clean")
    assert resp.status_code == 200
    body = resp.json()

    assert get_report(dataset_id) is not None          # Module 1 ran and was cached
    assert body["cleaned_dataset_id"]                   # a new id was assigned
    assert body["cleaned_dataset_id"] != dataset_id
    assert get_dataset_path(body["cleaned_dataset_id"]).is_file()  # cleaned CSV exists on disk
    assert get_cleaning_report(dataset_id) is not None  # cleaning report cached


def test_clean_endpoint_leaves_original_dataset_untouched(clean_client, monkeypatch):
    from app.storage.dataset_store import get_cleaning_report, get_dataset_path, get_report

    tc, dataset_id = clean_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Read the original uploaded CSV bytes BEFORE calling /clean
    original_bytes = get_dataset_path(dataset_id).read_bytes()

    def fake_du_run(self, df, ds_id):
        from app.agents.data_understanding import DataUnderstandingReport, ColumnOverview

        return DataUnderstandingReport(
            dataset_id=ds_id,
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in __import__(
                "app.agents.data_understanding", fromlist=["list_columns"]
            ).list_columns(df)],
            problem_type="classification",
            target_candidate="target",
            reasoning="stub",
            narrative="stub",
            key_findings=[],
        )

    def fake_cleaning_run(self, df, ds_id, context_report):
        from app.agents.cleaning import CleaningReport

        self.df = df.copy()
        return CleaningReport(
            dataset_id=ds_id,
            target_candidate=context_report.get("target_candidate"),
            problem_type=context_report.get("problem_type", "unclear"),
            initial_shape=[len(df), df.shape[1]],
            final_shape=[len(df), df.shape[1]],
            initial_missing_cells=int(df.isna().sum().sum()),
            final_missing_cells=int(df.isna().sum().sum()),
            steps=[],
            summary=["stub run"],
            remaining_issues=[],
        )

    monkeypatch.setattr(
        "app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run
    )
    monkeypatch.setattr("app.agents.cleaning.CleaningAgent.run", fake_cleaning_run)

    # Call /clean
    resp = tc.post(f"/datasets/{dataset_id}/clean")
    assert resp.status_code == 200
    body = resp.json()

    # Assert the original file's bytes are byte-identical
    assert get_dataset_path(dataset_id).read_bytes() == original_bytes
    # Assert a new dataset was created
    assert body["cleaned_dataset_id"] != dataset_id
