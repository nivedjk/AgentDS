"""Tests for the Report Agent (Module 7)."""

import pytest
from pydantic import ValidationError

from app.agents.report import (
    PROSE_SECTION_KEYS,
    SECTION_ORDER,
    UPSTREAM_KINDS,
    FinalReport,
    ReportNarration,
    ReportSection,
)


def test_section_order_is_ten_sections_bookended_by_report_prose():
    keys = [k for k, _, _ in SECTION_ORDER]
    assert keys[0] == "executive_summary"
    assert keys[-1] == "conclusion"
    assert keys == [
        "executive_summary", "dataset_overview", "understanding", "cleaning",
        "visualization", "recommendation", "training", "explainability",
        "whatif", "conclusion",
    ]
    assert UPSTREAM_KINDS == [
        "understanding", "cleaning", "visualization", "recommendation",
        "training", "explainability", "whatif",
    ]
    assert PROSE_SECTION_KEYS == keys[1:-1]


def test_report_section_status_is_constrained():
    ok = ReportSection(
        key="cleaning", title="Cleaning", body_markdown="x",
        source_module="cleaning", status="not_run",
    )
    assert ok.status == "not_run"
    with pytest.raises(ValidationError):
        ReportSection(
            key="cleaning", title="Cleaning", body_markdown="x",
            source_module="cleaning", status="skipped",
        )


def test_report_narration_defaults_section_intros_to_empty_dict():
    n = ReportNarration(executive_summary="s", conclusion="c")
    assert n.section_intros == {}


from app.agents.report import (
    render_cleaning_section,
    render_dataset_overview,
    render_explainability_section,
    render_recommendation_section,
    render_training_section,
    render_understanding_section,
    render_visualization_section,
    render_whatif_section,
)

FAKE_UNDERSTANDING = {
    "n_rows": 100, "n_columns": 4, "target_candidate": "target",
    "problem_type": "classification",
    "columns": [{"name": "age", "dtype": "int64", "missing_pct": 0.0, "n_unique": 40}],
    "duplicates": {"n_duplicate_rows": 2, "percent": 2.0, "example_rows": []},
    "correlations": [{"column_a": "age", "column_b": "income", "correlation": 0.91}],
    "key_findings": ["2 duplicate rows (2.0%)."],
    "narrative": "Small clean classification dataset.",
}
FAKE_CLEANING = {
    "cleaned_dataset_id": "ds-1-clean", "initial_shape": [100, 4], "final_shape": [98, 4],
    "initial_missing_cells": 6, "final_missing_cells": 0,
    "steps": [{"order": 1, "tool": "impute_column",
               "params": {"column": "age", "strategy": "median"},
               "reason": "12% missing", "result": {"missing_before": 6, "missing_after": 0}}],
    "summary": ["imputed age"], "remaining_issues": [],
}
FAKE_TRAINING = {
    "best_model": "xgboost", "primary_metric": "f1_macro", "best_score": 0.87,
    "n_rows": 98, "train_rows": 78, "test_rows": 20, "cv_splits": 5,
    "leaderboard": [
        {"rank": 1, "name": "xgboost", "cv_mean": 0.85, "cv_std": 0.03,
         "test_score": 0.87, "failed": False},
        {"rank": 2, "name": "ridge", "cv_mean": None, "cv_std": None,
         "test_score": None, "failed": True},
    ],
    "candidates": [{"name": "ridge", "error": "ValueError: bad"}],
    "warnings": ["stratified split fell back to non-stratified"],
    "narrative": "",
}


def test_render_understanding_none_is_not_run():
    s = render_understanding_section(None)
    assert s.status == "not_run"
    assert s.key == "understanding"
    assert "has not been run" in s.body_markdown


def test_render_understanding_pulls_facts_verbatim():
    s = render_understanding_section(FAKE_UNDERSTANDING)
    assert s.status == "ok"
    assert "0.91" in s.body_markdown              # correlation, verbatim
    assert "age" in s.body_markdown and "int64" in s.body_markdown
    assert "> Small clean classification dataset." in s.body_markdown
    assert "2 duplicate rows" in s.body_markdown


def test_render_dataset_overview_uses_cleaned_id_when_present():
    s = render_dataset_overview(
        FAKE_UNDERSTANDING, FAKE_CLEANING,
        dataset_id="ds-1", source_dataset_id="ds-1-clean",
    )
    assert s.status == "ok"
    assert "ds-1-clean" in s.body_markdown
    assert "| Rows | 100 |" in s.body_markdown


def test_render_dataset_overview_without_cleaning_says_not_run():
    s = render_dataset_overview(
        FAKE_UNDERSTANDING, None, dataset_id="ds-1", source_dataset_id="ds-1",
    )
    assert "cleaning not run" in s.body_markdown


def test_render_cleaning_section_lists_steps_and_shapes():
    s = render_cleaning_section(FAKE_CLEANING)
    assert s.status == "ok"
    assert "[100, 4]" in s.body_markdown and "[98, 4]" in s.body_markdown
    assert "impute_column" in s.body_markdown
    assert "12% missing" in s.body_markdown


def test_render_visualization_none_is_not_run():
    assert render_visualization_section(None).status == "not_run"


def test_render_visualization_table_excludes_plotly():
    art = {"n_charts": 1, "source": "cleaned",
           "charts": [{"title": "Age hist", "chart_type": "histogram",
                       "columns": ["age"], "insight": "right-skewed",
                       "plotly": {"data": [{"x": [1, 2, 3]}]}}],
           "skipped": [], "narrative": "One useful chart."}
    s = render_visualization_section(art)
    assert "Age hist" in s.body_markdown and "right-skewed" in s.body_markdown
    # No plotly payload leaks into the markdown (the word "dataset" legitimately
    # appears in the summary line, so check for the actual payload content).
    assert "plotly" not in s.body_markdown
    assert "[1, 2, 3]" not in s.body_markdown


def test_render_recommendation_section_verbatim_metric_and_models():
    art = {"primary_metric": "roc_auc", "used_cleaned_dataset": True,
           "candidates": [{"name": "xgboost", "library": "xgboost", "rationale": "nonlinear"}],
           "preprocessing_recommendations": ["scale numerics"], "reasoning": "binary target"}
    s = render_recommendation_section(art)
    assert "roc_auc" in s.body_markdown
    assert "xgboost" in s.body_markdown and "nonlinear" in s.body_markdown
    assert "> binary target" in s.body_markdown


def test_render_training_section_leaderboard_and_errors():
    s = render_training_section(FAKE_TRAINING)
    assert s.status == "ok"
    assert "xgboost" in s.body_markdown and "0.87" in s.body_markdown
    assert "ridge: ValueError: bad" in s.body_markdown
    assert "non-stratified" in s.body_markdown


def test_render_explainability_tolerates_missing_keys():
    s = render_explainability_section({"feature_importance": [{"feature": "age", "mean_abs_shap": 0.4}]})
    assert s.status == "ok"
    assert "age" in s.body_markdown and "0.4" in s.body_markdown
    assert render_explainability_section(None).status == "not_run"


def test_render_whatif_none_and_empty_are_not_run():
    assert render_whatif_section(None).status == "not_run"
    assert render_whatif_section({"experiments": []}).status == "not_run"


def test_render_whatif_renders_each_experiment():
    art = {"experiments": [
        {"question": "What if income +10%?", "summary": "prediction flips to 1",
         "deltas": [{"label": "P(class=1)", "change": "+0.22"}]},
    ]}
    s = render_whatif_section(art)
    assert "What if income +10%?" in s.body_markdown
    assert "+0.22" in s.body_markdown


def test_final_report_round_trips():
    fr = FinalReport(
        dataset_id="ds-1",
        generated_at="2026-09-01T12:00:00+00:00",
        source_dataset_id="ds-1",
        modules_present=["understanding"],
        modules_missing=["cleaning", "visualization", "recommendation",
                         "training", "explainability", "whatif"],
        sections=[ReportSection(key="executive_summary", title="Executive Summary",
                                body_markdown="...", source_module="report", status="ok")],
        executive_summary="...",
        markdown="# AgentDS Final Report",
    )
    assert FinalReport(**fr.model_dump()) == fr


from app.agents.report import (
    ReportNarration,
    ReportSection,
    assemble_markdown,
    build_report_digest,
)


def test_digest_collapses_arrays_to_counts_and_names():
    artifacts = {
        "understanding": FAKE_UNDERSTANDING,
        "cleaning": FAKE_CLEANING,
        "visualization": {"n_charts": 3, "source": "cleaned",
                          "charts": [{"chart_type": "bar"}, {"chart_type": "bar"},
                                     {"chart_type": "histogram"}]},
        "recommendation": {"primary_metric": "roc_auc",
                           "candidates": [{"name": "ridge"}, {"name": "xgboost"}]},
        "training": FAKE_TRAINING,
        "explainability": None,
        "whatif": None,
    }
    d = build_report_digest(artifacts)
    assert d["modules_present"] == ["understanding", "cleaning", "visualization",
                                    "recommendation", "training"]
    assert d["modules_missing"] == ["explainability", "whatif"]
    assert d["understanding"]["n_high_corr_pairs"] == 1
    assert d["cleaning"]["n_steps"] == 1
    assert d["cleaning"]["missing_cells_removed"] == 6
    assert d["visualization"]["chart_types"] == ["bar", "histogram"]
    assert d["recommendation"]["candidate_names"] == ["ridge", "xgboost"]
    assert d["training"]["n_failed"] == 1
    # no heavy arrays leaked
    assert "charts" not in d["visualization"]
    assert "columns" not in d["understanding"]
    assert "leaderboard" not in d["training"]


def test_digest_omits_absent_modules_cleanly():
    d = build_report_digest({k: None for k in
        ["understanding", "cleaning", "visualization", "recommendation",
         "training", "explainability", "whatif"]})
    assert d["modules_present"] == []
    assert set(d["modules_missing"]) == set(d["modules_missing"])
    assert "understanding" not in d


def test_assemble_markdown_orders_sections_and_weaves_prose():
    sections = [
        ReportSection(key="executive_summary", title="Executive Summary",
                      body_markdown="EXEC.", source_module="report", status="ok"),
        ReportSection(key="cleaning", title="Cleaning", body_markdown="STEPS TABLE",
                      source_module="cleaning", status="ok"),
        ReportSection(key="whatif", title="What-If Analysis", body_markdown="",
                      source_module="whatif", status="not_run"),
        ReportSection(key="conclusion", title="Conclusion & Limitations",
                      body_markdown="CONCL.", source_module="report", status="ok"),
    ]
    prose = ReportNarration(
        executive_summary="EXEC.",
        section_intros={"cleaning": "Cleaning tidied the frame.", "whatif": "n/a"},
        conclusion="CONCL.",
    )
    md = assemble_markdown(sections, prose)
    assert md.index("## Executive Summary") < md.index("## Cleaning") < md.index("## What-If Analysis") < md.index("## Conclusion")
    assert "Cleaning tidied the frame." in md
    assert "STEPS TABLE" in md
    assert "_This module has not been run for this dataset._" in md
    assert "EXEC." in md and "CONCL." in md


def test_default_narration_is_schema_valid_and_factual():
    d = build_report_digest({
        "understanding": FAKE_UNDERSTANDING, "cleaning": None, "visualization": None,
        "recommendation": None, "training": None, "explainability": None, "whatif": None,
    })
    from app.agents.report import _default_narration
    n = _default_narration(d)
    assert isinstance(n, ReportNarration)
    assert "100" in n.executive_summary  # n_rows, verbatim from the digest


import json
from types import SimpleNamespace

from app.agents.report import run_report
from app.storage.dataset_store import save_artifact


def _tool_use_block(name, tool_input, block_id="c1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _response(content, stop_reason="tool_use"):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


PROSE = {
    "executive_summary": "This dataset was analysed end to end.",
    "section_intros": {"cleaning": "Cleaning removed missing values.",
                       "training": "Four models were compared."},
    "conclusion": "Results are preliminary.",
}


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))


def test_run_report_with_only_understanding_marks_rest_not_run():
    save_artifact("ds-1", "understanding", FAKE_UNDERSTANDING)
    client = FakeAnthropicClient([_response([_tool_use_block("submit_report_prose", PROSE)])])
    fr = run_report("ds-1", client=client)

    assert len(client.calls) == 1
    assert fr.modules_present == ["understanding"]
    assert set(fr.modules_missing) == {"cleaning", "visualization", "recommendation",
                                       "training", "explainability", "whatif"}
    by_key = {s.key: s for s in fr.sections}
    assert [s.key for s in fr.sections] == [k for k, _, _ in
        __import__("app.agents.report", fromlist=["SECTION_ORDER"]).SECTION_ORDER]
    assert by_key["cleaning"].status == "not_run"
    assert by_key["training"].status == "not_run"
    assert by_key["executive_summary"].body_markdown == PROSE["executive_summary"]
    assert by_key["conclusion"].body_markdown == PROSE["conclusion"]
    assert fr.executive_summary == PROSE["executive_summary"]
    assert "# AgentDS Final Report" in fr.markdown
    assert "_Generated " in fr.markdown


def test_run_report_splices_llm_intros_and_verbatim_facts():
    save_artifact("ds-2", "understanding", FAKE_UNDERSTANDING)
    save_artifact("ds-2", "cleaning", FAKE_CLEANING)
    save_artifact("ds-2", "training", FAKE_TRAINING)
    client = FakeAnthropicClient([_response([_tool_use_block("submit_report_prose", PROSE)])])
    fr = run_report("ds-2", client=client)

    assert fr.source_dataset_id == "ds-1-clean"          # cleaned id from the cleaning sidecar
    assert "Cleaning removed missing values." in fr.markdown   # LLM intro
    assert "impute_column" in fr.markdown                      # verbatim cleaning fact
    assert "0.87" in fr.markdown                               # verbatim training metric
    digest_sent = json.loads(client.calls[0]["messages"][0]["content"])
    assert digest_sent["training"]["best_model"] == "xgboost"
    assert "leaderboard" not in digest_sent["training"]        # digest stayed compact


def test_run_report_falls_back_to_template_prose_on_llm_failure():
    save_artifact("ds-3", "understanding", FAKE_UNDERSTANDING)
    client = FakeAnthropicClient([_response([SimpleNamespace(type="text", text="no")],
                                            stop_reason="end_turn"),
                                 _response([SimpleNamespace(type="text", text="no")],
                                            stop_reason="end_turn")])
    fr = run_report("ds-3", client=client)
    assert "100" in fr.executive_summary            # template prose, from the digest
    assert fr.sections[0].key == "executive_summary"


def test_run_report_is_deterministic_given_fixed_sidecars_and_prose():
    for ds in ("ds-a", "ds-b"):
        save_artifact(ds, "understanding", FAKE_UNDERSTANDING)
        save_artifact(ds, "cleaning", FAKE_CLEANING)
    r1 = run_report("ds-a", client=FakeAnthropicClient(
        [_response([_tool_use_block("submit_report_prose", PROSE)])]))
    r2 = run_report("ds-a", client=FakeAnthropicClient(
        [_response([_tool_use_block("submit_report_prose", PROSE)])]))
    assert [s.model_dump() for s in r1.sections] == [s.model_dump() for s in r2.sections]
    # markdown identical once the single generated-timestamp line is dropped
    strip = lambda m: "\n".join(l for l in m.splitlines() if not l.startswith("_Generated "))
    assert strip(r1.markdown) == strip(r2.markdown)


def test_run_report_without_client_uses_template_prose():
    save_artifact("ds-4", "understanding", FAKE_UNDERSTANDING)
    fr = run_report("ds-4", client=None)
    assert fr.executive_summary  # non-empty template prose, no exception


import io

from fastapi.testclient import TestClient


@pytest.fixture
def report_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    tc = TestClient(app)
    csv = b"age,income,target\n25,50000,0\n30,60000,1\n41,72000,1\n22,,0\n"
    ds = tc.post("/datasets/upload",
                 files={"file": ("people.csv", io.BytesIO(csv), "text/csv")}).json()["dataset_id"]
    return tc, ds


def test_post_report_503_when_ollama_unreachable_default_provider(report_client, monkeypatch):
    """AGENTDS_LLM_PROVIDER defaults to "ollama". If it isn't reachable, this
    must 503 with a clear message naming the host — not silently succeed
    just because ANTHROPIC_API_KEY happens to be set or unset."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    tc, ds = report_client
    resp = tc.post(f"/datasets/{ds}/report")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_post_report_requires_api_key(report_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tc, ds = report_client
    assert tc.post(f"/datasets/{ds}/report").status_code == 503


def test_post_report_unknown_dataset_404(report_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tc, _ = report_client
    assert tc.post("/datasets/nope/report").status_code == 404


def test_post_report_passes_a_real_client_to_run_report(report_client, monkeypatch):
    """Regression: post_report used to call run_report(dataset_id) with no
    client, so the narration always fell back to deterministic template
    prose. It must now hand run_report a live client so the summary is
    genuinely LLM-written."""
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    tc, ds = report_client

    fake = FakeAnthropicClient([_response([_tool_use_block("submit_report_prose", PROSE)])])
    monkeypatch.setattr("app.routers.report.get_client", lambda: fake)

    def fake_du_run(self, df, ds_id):
        from app.agents.data_understanding import ColumnOverview, DataUnderstandingReport, list_columns
        return DataUnderstandingReport(
            dataset_id=ds_id, n_rows=len(df), n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in list_columns(df)],
            problem_type="classification", target_candidate="target",
            reasoning="stub", narrative="stub narrative", key_findings=[],
        )

    monkeypatch.setattr("app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run)

    resp = tc.post(f"/datasets/{ds}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["executive_summary"] == PROSE["executive_summary"]
    assert len(fake.calls) == 1  # the narration call actually happened


def test_get_report_404_before_build(report_client):
    tc, ds = report_client
    assert tc.get(f"/datasets/{ds}/report").status_code == 404


def test_post_then_get_report_round_trips_with_only_module_1(report_client, monkeypatch):
    tc, ds = report_client
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def fake_du_run(self, df, ds_id):
        from app.agents.data_understanding import ColumnOverview, DataUnderstandingReport, list_columns
        return DataUnderstandingReport(
            dataset_id=ds_id, n_rows=len(df), n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in list_columns(df)],
            problem_type="classification", target_candidate="target",
            reasoning="stub", narrative="stub narrative", key_findings=[],
        )

    def fake_run_report(dataset_id, client=None):
        from app.agents.report import run_report as real
        # exercise the real assembler, but with a deterministic fake narrate client
        return real(dataset_id, client=FakeAnthropicClient(
            [_response([_tool_use_block("submit_report_prose", PROSE)])]))

    monkeypatch.setattr("app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run)
    monkeypatch.setattr("app.routers.report.run_report", fake_run_report)

    posted = tc.post(f"/datasets/{ds}/report")
    assert posted.status_code == 200
    body = posted.json()
    assert body["modules_present"] == ["understanding"]
    assert body["dataset_id"] == ds
    assert "# AgentDS Final Report" in body["markdown"]

    got = tc.get(f"/datasets/{ds}/report")           # no key needed
    assert got.status_code == 200
    assert got.json()["markdown"] == body["markdown"]


def test_get_report_works_without_api_key(report_client, monkeypatch):
    tc, ds = report_client
    from app.storage.dataset_store import save_artifact as _sa
    # hand-place a minimal cached final report
    from app.agents.report import FinalReport
    fr = FinalReport(dataset_id=ds, generated_at="2026-09-01T00:00:00+00:00",
                     source_dataset_id=ds, modules_present=["understanding"],
                     modules_missing=[], sections=[], executive_summary="e",
                     markdown="# AgentDS Final Report")
    _sa(ds, "final", fr.model_dump())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert tc.get(f"/datasets/{ds}/report").status_code == 200
