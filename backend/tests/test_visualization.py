"""Tests for the Visualization Agent (Module 3)."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.visualization import (
    ChartSpec,
    SelectedChart,
    VisualizationNarration,
    VisualizationReport,
    MAX_TARGET_FEATURE_CHARTS,
    _candidate_digest,
    _resolve_source_df,
    build_categorical_charts,
    build_chart_candidates,
    build_correlation_chart,
    build_numeric_charts,
    build_target_relationship_charts,
    build_temporal_charts,
    run_visualization,
)
from app.storage.dataset_store import save_cleaning_report, save_dataset


def test_chartspec_defaults_insight_to_empty_string():
    spec = ChartSpec(
        chart_type="histogram",
        title="Distribution of age",
        columns=["age"],
        plotly={"data": [{"type": "bar", "x": [1], "y": [2]}], "layout": {}},
    )
    assert spec.insight == ""
    assert spec.plotly["data"][0]["type"] == "bar"


def test_visualization_narration_requires_selected_list():
    ok = VisualizationNarration(
        selected=[SelectedChart(index=0, insight="right-skewed")], narrative="n"
    )
    assert ok.selected[0].index == 0
    with pytest.raises(ValidationError):
        VisualizationNarration(selected="nope", narrative="n")


def test_visualization_report_builds_and_counts():
    charts = [
        ChartSpec(chart_type="bar", title="t", columns=["c"], plotly={"data": [], "layout": {}})
    ]
    report = VisualizationReport(
        dataset_id="ds-1",
        source="original",
        target_column="target",
        problem_type="classification",
        n_charts=len(charts),
        charts=charts,
        skipped=["notes: nothing"],
        narrative="overview",
    )
    assert report.n_charts == 1
    assert report.source == "original"
    assert report.charts[0].chart_type == "bar"


# Task 3 tests

CLS = {"target_candidate": "y", "problem_type": "classification"}


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))


def _numeric_df():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "const": [7, 7, 7, 7, 7],
            "empty": [None, None, None, None, None],
            "y": [0, 1, 0, 1, 0],
        }
    )


def test_build_numeric_charts_emits_histogram_and_box_for_age():
    charts, skipped = build_numeric_charts(_numeric_df(), CLS)
    by_type = {c.chart_type for c in charts}
    assert by_type == {"histogram", "box"}
    assert all(c.columns == ["age"] for c in charts)  # y excluded, const/empty skipped

    hist = next(c for c in charts if c.chart_type == "histogram")
    assert hist.plotly["data"][0]["type"] == "bar"
    assert sum(hist.plotly["data"][0]["y"]) == 5           # every row binned
    assert len(hist.plotly["data"][0]["x"]) == len(hist.plotly["data"][0]["y"])

    box = next(c for c in charts if c.chart_type == "box")
    assert box.plotly["data"][0]["median"] == [40.0]
    assert box.plotly["data"][0]["q1"] == [30.0]
    assert box.plotly["data"][0]["q3"] == [50.0]


def test_build_numeric_charts_records_skip_reasons():
    _, skipped = build_numeric_charts(_numeric_df(), CLS)
    joined = " ".join(skipped)
    assert "const" in joined and "constant" in joined
    assert "empty" in joined and "null" in joined


def test_build_numeric_charts_plotly_is_json_serializable():
    charts, _ = build_numeric_charts(_numeric_df(), CLS)
    for c in charts:
        json.dumps(c.plotly)  # must not raise


def test_build_categorical_charts_bar_counts_are_ground_truth():
    df = pd.DataFrame({"city": ["NY", "NY", "LA", "SF"], "y": [0, 1, 0, 1]})
    charts, skipped = build_categorical_charts(df, CLS)
    assert len(charts) == 1
    bar = charts[0]
    assert bar.chart_type == "bar" and bar.columns == ["city"]
    assert bar.plotly["data"][0]["x"] == ["NY", "LA", "SF"]
    assert bar.plotly["data"][0]["y"] == [2, 1, 1]


def test_build_categorical_charts_skips_high_cardinality():
    df = pd.DataFrame({"code": [f"v{i}" for i in range(25)], "y": list(range(25))})
    charts, skipped = build_categorical_charts(df, {"target_candidate": "y", "problem_type": "regression"})
    assert charts == []
    assert "code" in skipped[0] and "exceeds" in skipped[0]


def test_resolve_source_df_returns_original_when_no_cleaning_report():
    ds_id = save_dataset("d.csv", b"a,b\n1,2\n3,4\n")
    df, source = _resolve_source_df(ds_id)
    assert source == "original"
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_resolve_source_df_prefers_cleaned_when_present():
    orig = save_dataset("d.csv", b"a,b\n1,2\n")
    cleaned = save_dataset("d_cleaned.csv", b"a,b,c\n1,2,3\n9,9,9\n")
    save_cleaning_report(orig, {"cleaned_dataset_id": cleaned, "steps": []})
    df, source = _resolve_source_df(orig)
    assert source == "cleaned"
    assert list(df.columns) == ["a", "b", "c"] and len(df) == 2


def test_resolve_source_df_falls_back_when_cleaned_file_missing():
    orig = save_dataset("d.csv", b"a,b\n1,2\n")
    save_cleaning_report(orig, {"cleaned_dataset_id": "gone-id", "steps": []})
    df, source = _resolve_source_df(orig)
    assert source == "original"


# Task 4 tests

NOCLS = {"target_candidate": None, "problem_type": "unclear"}


def test_build_temporal_charts_line_over_parsed_dates():
    df = pd.DataFrame(
        {
            "ts": ["2021-01-15", "2021-02-15", "2021-03-15", "2021-04-15", "2021-05-15", "2021-06-15"],
            "value": [10, 12, 11, 15, 14, 18],
        }
    )
    charts, skipped = build_temporal_charts(df, NOCLS)
    assert len(charts) == 1
    line = charts[0]
    assert line.chart_type == "line"
    assert line.columns[0] == "ts"
    xs = line.plotly["data"][0]["x"]
    assert xs == sorted(xs)                       # ascending
    assert sum(line.plotly["data"][0]["y"]) == 6  # row-count trace covers every row


def test_build_temporal_charts_notes_unparseable_date_named_column():
    df = pd.DataFrame({"order_date": ["nope", "not a date", "xxx"], "n": [1, 2, 3]})
    charts, skipped = build_temporal_charts(df, NOCLS)
    assert charts == []
    joined = " ".join(skipped)
    assert "order_date" in joined and "parse" in joined
    assert "no parseable datetime column found" in joined


def test_build_temporal_charts_detects_real_datetime_dtype():
    df = pd.DataFrame(
        {"when": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"]), "v": [1, 2, 3, 4]}
    )
    charts, _ = build_temporal_charts(df, NOCLS)
    assert len(charts) == 1 and charts[0].columns[0] == "when"


def test_build_correlation_chart_values_are_ground_truth():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [1, 0, 1, 0]})
    charts, skipped = build_correlation_chart(df)
    assert len(charts) == 1
    hm = charts[0]
    assert hm.chart_type == "heatmap"
    z = hm.plotly["data"][0]["z"]
    cols = hm.plotly["data"][0]["x"]
    ia, ib = cols.index("a"), cols.index("b")
    assert z[ia][ib] == pytest.approx(1.0)     # a,b perfectly correlated
    assert z[ia][ia] == pytest.approx(1.0)     # diagonal
    assert hm.plotly["data"][0]["zmin"] == -1


def test_build_correlation_chart_drops_zero_variance_and_notes_it():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "k": [5, 5, 5, 5]})
    charts, skipped = build_correlation_chart(df)
    assert len(charts) == 1
    assert "k" not in charts[0].plotly["data"][0]["x"]
    assert any("k" in s and "zero variance" in s for s in skipped)


def test_build_correlation_chart_skips_with_one_numeric_column():
    df = pd.DataFrame({"a": [1, 2, 3], "label": ["x", "y", "z"]})
    charts, skipped = build_correlation_chart(df)
    assert charts == []
    assert "fewer than 2" in skipped[0]


# Task 5 tests

REG = {"target_candidate": "price", "problem_type": "regression"}


def _mixed_df():
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat": ["x", "x", "y", "y", "x", "y"],
            "y": [0, 0, 1, 1, 0, 1],
        }
    )


def test_target_relationship_classification_emits_expected_chart_types():
    charts, skipped = build_target_relationship_charts(_mixed_df(), CLS)
    types = {c.chart_type for c in charts}
    assert "bar" in types           # class balance of y
    assert "box" in types           # f1 by y
    assert "grouped_bar" in types   # cat vs y

    balance = next(c for c in charts if c.chart_type == "bar" and c.columns == ["y"])
    assert sorted(balance.plotly["data"][0]["y"]) == [3, 3]

    box = next(c for c in charts if c.chart_type == "box")
    assert box.columns == ["f1", "y"]
    assert len(box.plotly["data"]) == 2   # one trace per class

    grouped = next(c for c in charts if c.chart_type == "grouped_bar")
    assert grouped.columns == ["cat", "y"]
    assert len(grouped.plotly["data"]) == 2


def test_target_relationship_regression_uses_histogram_and_quartile_bins():
    df = pd.DataFrame(
        {
            "size": [10, 20, 30, 40, 50, 60, 70, 80],
            "grade": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    charts, skipped = build_target_relationship_charts(df, REG)
    assert any(c.chart_type == "histogram" and c.columns == ["price"] for c in charts)
    assert any(c.chart_type == "box" and c.columns == ["size", "price"] for c in charts)
    assert any(c.chart_type == "grouped_bar" and c.columns == ["grade", "price"] for c in charts)


def test_target_relationship_no_target_returns_skip_note():
    charts, skipped = build_target_relationship_charts(_mixed_df(), NOCLS)
    assert charts == []
    assert "no target column identified" in skipped[0]


def test_target_relationship_target_absent_from_df():
    charts, skipped = build_target_relationship_charts(
        _mixed_df(), {"target_candidate": "ghost", "problem_type": "classification"}
    )
    assert charts == []
    assert "ghost" in skipped[0] and "not present" in skipped[0]


def test_target_relationship_respects_feature_budget():
    cols = {f"n{i}": list(range(20)) for i in range(MAX_TARGET_FEATURE_CHARTS + 3)}
    cols["y"] = [0, 1] * 10
    charts, skipped = build_target_relationship_charts(
        pd.DataFrame(cols), {"target_candidate": "y", "problem_type": "classification"}
    )
    assert any("budget" in s for s in skipped)


def test_build_chart_candidates_concatenates_all_builders():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "city": ["NY", "LA", "NY", "SF", "LA", "NY"],
            "ts": pd.to_datetime(
                ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01", "2021-06-01"]
            ),
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    charts, skipped = build_chart_candidates(df, CLS)
    assert charts and all(isinstance(c, ChartSpec) for c in charts)
    assert all(c.insight == "" for c in charts)          # not narrated yet
    present = {c.chart_type for c in charts}
    assert {"histogram", "bar", "line"} <= present
    for c in charts:
        json.dumps(c.plotly)                              # all JSON-serializable
    assert isinstance(skipped, list)


# Task 6 tests


def _fake_narrate_client(selected, narrative="the story"):
    """Fake anthropic client whose single response is a submit_visualization tool call."""
    payload = {
        "selected": [{"index": i, "insight": f"insight-{i}"} for i in selected],
        "narrative": narrative,
    }
    block = SimpleNamespace(type="tool_use", name="submit_visualization", input=payload, id="c1")
    resp = SimpleNamespace(stop_reason="tool_use", content=[block])

    class _C:
        def __init__(self):
            self.calls = []
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kw):
            self.calls.append({**kw, "messages": list(kw["messages"])})
            return resp

    return _C()


def _viz_df():
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "city": ["NY", "LA", "NY", "SF", "LA", "NY"],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )


def test_candidate_digest_is_metadata_only():
    charts, _ = build_chart_candidates(_viz_df(), CLS)
    digest = _candidate_digest(charts)
    assert len(digest) == len(charts)
    for d in digest:
        assert set(d) == {"index", "chart_type", "title", "columns", "summary"}
        blob = json.dumps(d)
        assert '"z"' not in blob and '"data"' not in blob


def test_run_visualization_applies_llm_order_and_insights():
    charts, _ = build_chart_candidates(_viz_df(), CLS)
    client = _fake_narrate_client(selected=[2, 0])
    report = run_visualization(_viz_df(), "ds-1", CLS, client=client, source="cleaned")

    assert report.source == "cleaned"
    assert report.n_charts == 2
    assert report.charts[0].plotly == charts[2].plotly     # order follows the LLM
    assert report.charts[1].plotly == charts[0].plotly
    assert report.charts[0].insight == "insight-2"
    assert report.narrative == "the story"
    assert len(client.calls) == 1


def test_run_visualization_drops_out_of_range_indices():
    client = _fake_narrate_client(selected=[0, 999])
    report = run_visualization(_viz_df(), "ds-1", CLS, client=client)
    assert report.n_charts == 1
    assert report.charts[0].insight == "insight-0"


def test_run_visualization_falls_back_to_candidates_when_selection_empty():
    client = _fake_narrate_client(selected=[], narrative="")
    report = run_visualization(_viz_df(), "ds-1", CLS, client=client)
    candidates, _ = build_chart_candidates(_viz_df(), CLS)
    assert report.n_charts == min(len(candidates), 12)
    assert report.narrative  # deterministic fallback, non-empty


def test_run_visualization_skips_llm_when_no_candidates():
    df = pd.DataFrame({"empty": [None, None, None]})
    called = {"n": 0}

    class _Boom:
        def __init__(self):
            self.messages = SimpleNamespace(create=self._c)

        def _c(self, **kw):
            called["n"] += 1
            raise AssertionError("LLM must not be called when there are no candidates")

    report = run_visualization(
        df, "ds-1", {"target_candidate": None, "problem_type": "unclear"}, client=_Boom()
    )
    assert report.n_charts == 0
    assert report.charts == []
    assert report.skipped  # reasons recorded
    assert called["n"] == 0


def test_run_visualization_chart_data_is_never_the_llms():
    """The LLM adds only insight/order; every plotly dict matches the builder output."""
    df = _viz_df()
    candidates, _ = build_chart_candidates(df, CLS)
    client = _fake_narrate_client(selected=list(range(len(candidates))))
    report = run_visualization(df, "ds-1", CLS, client=client)
    for chart in report.charts:
        assert any(chart.plotly == c.plotly for c in candidates)


# Task 6 fixes: pandas 3.0.5 dtype handling and robustness


def test_build_target_relationship_charts_regression_with_string_target_returns_bar():
    """Regression with string/category target should emit bar chart, not histogram."""
    df = pd.DataFrame({
        "size": [10, 20, 30, 40, 50, 60],
        "grade": ["a", "a", "b", "b", "c", "c"],
    })
    # Regression on a string column (not numeric)
    reg_report = {"target_candidate": "grade", "problem_type": "regression"}
    charts, skipped = build_target_relationship_charts(df, reg_report)

    # Should have at least one chart for univariate target
    assert len(charts) >= 1

    # The univariate target chart should be bar (not histogram)
    univariate = [c for c in charts if "grade" in c.columns and len(c.columns) == 1]
    assert len(univariate) >= 1
    assert univariate[0].chart_type == "bar"


def test_run_visualization_caps_at_max_selected():
    """run_visualization should cap final charts at MAX_SELECTED."""
    from app.agents.visualization import MAX_SELECTED

    # Build a wide dataframe with many columns to generate many candidates
    n_rows = 50
    n_cols = 20
    df = pd.DataFrame({
        f"col_{i}": [i + j*0.1 for j in range(n_rows)]
        for i in range(n_cols)
    })
    df["y"] = [0, 1] * (n_rows // 2)

    # Create a fake client that selects ALL candidates
    num_candidates = len(build_chart_candidates(df, CLS)[0])
    selected = list(range(num_candidates))
    client = _fake_narrate_client(selected=selected)

    report = run_visualization(df, "ds-1", CLS, client=client)

    # Final report should have at most MAX_SELECTED charts
    assert report.n_charts <= MAX_SELECTED
    assert len(report.charts) <= MAX_SELECTED


def test_build_temporal_charts_mixed_offset_strings_handled_with_utc():
    """build_temporal_charts should handle mixed-offset strings by converting to UTC."""
    df = pd.DataFrame({
        "mixed_ts": [
            "2021-01-01T00:00:00+00:00",
            "2021-02-01T00:00:00+05:30",  # Different offset
            "2021-03-01T00:00:00+00:00",
        ],
        "value": [10, 20, 30],
    })

    charts, skipped = build_temporal_charts(df, NOCLS)

    # With utc=True, mixed-offset strings should parse successfully (converted to UTC)
    assert len(charts) == 1
    assert charts[0].chart_type == "line"
    assert "mixed_ts" in charts[0].columns


# Task 7: Endpoint tests


@pytest.fixture
def viz_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    tc = TestClient(app)
    csv = b"age,city,target\n25,NY,0\n30,LA,1\n25,NY,0\n40,SF,1\n35,LA,0\n"
    resp = tc.post("/datasets/upload", files={"file": ("p.csv", io.BytesIO(csv), "text/csv")})
    return tc, resp.json()["dataset_id"]


def _stub_report(dataset_id, source="original"):
    return VisualizationReport(
        dataset_id=dataset_id,
        source=source,
        target_column="target",
        problem_type="classification",
        n_charts=0,
        charts=[],
        skipped=[],
        narrative="stub",
    )


def test_get_visualize_404_before_run(viz_client):
    tc, ds = viz_client
    assert tc.get(f"/datasets/{ds}/visualize").status_code == 404


def test_get_visualize_returns_cached_after_run(viz_client):
    from app.storage.dataset_store import save_artifact

    tc, ds = viz_client
    save_artifact(ds, "visualization", _stub_report(ds, source="cleaned").model_dump())

    resp = tc.get(f"/datasets/{ds}/visualize")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == ds
    assert body["source"] == "cleaned"
    assert body["narrative"] == "stub"


def test_visualize_503_when_ollama_unreachable_default_provider(viz_client, monkeypatch):
    """AGENTDS_LLM_PROVIDER defaults to "ollama". If it isn't reachable, this
    must 503 with a clear message naming the host — not silently succeed
    just because ANTHROPIC_API_KEY happens to be set or unset."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    tc, ds = viz_client
    resp = tc.post(f"/datasets/{ds}/visualize")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_visualize_requires_api_key(viz_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tc, ds = viz_client
    assert tc.post(f"/datasets/{ds}/visualize").status_code == 503


def test_visualize_unknown_dataset_404(viz_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tc, _ = viz_client
    assert tc.post("/datasets/does-not-exist/visualize").status_code == 404


def test_visualize_runs_and_caches_report(viz_client, monkeypatch):
    from app.storage.dataset_store import get_artifact, save_report

    tc, ds = viz_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    save_report(ds, {"dataset_id": ds, "target_candidate": "target", "problem_type": "classification"})

    captured = {}

    def fake_run(df, dataset_id, understanding, client=None, source="original"):
        captured["cols"] = list(df.columns)
        captured["source"] = source
        return _stub_report(dataset_id, source)

    monkeypatch.setattr("app.routers.visualize.run_visualization", fake_run)

    resp = tc.post(f"/datasets/{ds}/visualize")
    assert resp.status_code == 200
    assert resp.json()["narrative"] == "stub"
    assert get_artifact(ds, "visualization") is not None
    assert captured["source"] == "original"
    assert "age" in captured["cols"]


def test_visualize_operates_on_cleaned_dataset_when_present(viz_client, monkeypatch):
    from app.storage.dataset_store import save_cleaning_report, save_dataset, save_report

    tc, ds = viz_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    save_report(ds, {"dataset_id": ds, "target_candidate": "target", "problem_type": "classification"})
    cleaned = save_dataset("c.csv", b"age,city,target,extra\n25,NY,0,9\n30,LA,1,8\n")
    save_cleaning_report(ds, {"cleaned_dataset_id": cleaned, "steps": []})

    seen = {}

    def fake_run(df, dataset_id, understanding, client=None, source="original"):
        seen["cols"] = list(df.columns)
        return _stub_report(dataset_id, source)

    monkeypatch.setattr("app.routers.visualize.run_visualization", fake_run)
    resp = tc.post(f"/datasets/{ds}/visualize")
    assert resp.status_code == 200
    assert resp.json()["source"] == "cleaned"
    assert "extra" in seen["cols"]


def test_visualize_regenerates_understanding_when_absent(viz_client, monkeypatch):
    from app.storage.dataset_store import get_artifact

    tc, ds = viz_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_du_run(self, df, dataset_id):
        from app.agents.data_understanding import ColumnOverview, DataUnderstandingReport, list_columns

        return DataUnderstandingReport(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in list_columns(df)],
            problem_type="classification",
            target_candidate="target",
            reasoning="stub",
            narrative="stub",
            key_findings=[],
        )

    monkeypatch.setattr("app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run)
    monkeypatch.setattr(
        "app.routers.visualize.run_visualization",
        lambda df, dataset_id, understanding, client=None, source="original": _stub_report(dataset_id, source),
    )

    assert get_artifact(ds, "understanding") is None
    resp = tc.post(f"/datasets/{ds}/visualize")
    assert resp.status_code == 200
    assert get_artifact(ds, "understanding") is not None
