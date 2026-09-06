"""Tests for the Model Recommendation Agent (Module 4)."""

import pytest
from pydantic import ValidationError

from app.agents.recommendation import (
    CLASSIFICATION_METRICS,
    DEFAULT_METRIC,
    MODEL_CATALOG,
    REGRESSION_METRICS,
    CandidateChoice,
    ModelCandidate,
    RecommendationLLMInput,
    RecommendationReport,
    catalog_names,
    get_catalog_entry,
)

_EXPECTED = {
    "classification": {
        "logistic_regression",
        "decision_tree_classifier",
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
        "xgboost_classifier",
        "lightgbm_classifier",
    },
    "regression": {
        "ridge",
        "decision_tree_regressor",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    },
}


@pytest.mark.parametrize("ptype", ["classification", "regression"])
def test_catalog_has_the_six_spec_models(ptype):
    assert {e["name"] for e in MODEL_CATALOG[ptype]} == _EXPECTED[ptype]
    assert len(MODEL_CATALOG[ptype]) == 6


@pytest.mark.parametrize("ptype", ["classification", "regression"])
def test_every_entry_has_library_estimator_hyperparameters(ptype):
    for e in MODEL_CATALOG[ptype]:
        assert e["library"] in {"sklearn", "xgboost", "lightgbm"}
        assert isinstance(e["estimator"], str) and "." in e["estimator"]
        assert isinstance(e["hyperparameters"], dict)
        assert isinstance(e["blurb"], str) and e["blurb"]


def test_random_state_is_42_everywhere():
    for ptype in ("classification", "regression"):
        for e in MODEL_CATALOG[ptype]:
            assert e["hyperparameters"]["random_state"] == 42


def test_library_matches_name_family():
    assert get_catalog_entry("classification", "xgboost_classifier")["library"] == "xgboost"
    assert get_catalog_entry("classification", "lightgbm_classifier")["library"] == "lightgbm"
    assert get_catalog_entry("regression", "ridge")["library"] == "sklearn"


def test_catalog_names_and_get_entry():
    assert catalog_names("classification") == [e["name"] for e in MODEL_CATALOG["classification"]]
    assert len(catalog_names("regression")) == 6
    assert get_catalog_entry("regression", "ridge")["hyperparameters"] == {"alpha": 1.0, "random_state": 42}
    with pytest.raises(KeyError):
        get_catalog_entry("classification", "not_a_model")


def test_metric_constants():
    assert DEFAULT_METRIC["classification"] in CLASSIFICATION_METRICS
    assert DEFAULT_METRIC["regression"] in REGRESSION_METRICS
    assert DEFAULT_METRIC == {"classification": "f1_macro", "regression": "r2"}


def test_model_candidate_requires_rationale():
    ok = ModelCandidate(name="ridge", library="sklearn", rationale="linear baseline", hyperparameters={})
    assert ok.rationale
    with pytest.raises(ValidationError):
        ModelCandidate(name="ridge", library="sklearn", hyperparameters={})


def test_llm_input_enforces_candidate_count_bounds():
    good = RecommendationLLMInput(
        candidates=[CandidateChoice(name=n, rationale="r") for n in ("a", "b", "c")],
        preprocessing_recommendations=["scale"],
        primary_metric="roc_auc",
        reasoning="because",
    )
    assert len(good.candidates) == 3
    with pytest.raises(ValidationError):
        RecommendationLLMInput(
            candidates=[CandidateChoice(name="a", rationale="r")],
            preprocessing_recommendations=["scale"], primary_metric="roc_auc", reasoning="x",
        )
    with pytest.raises(ValidationError):
        RecommendationLLMInput(
            candidates=[CandidateChoice(name=n, rationale="r") for n in "abcde"],
            preprocessing_recommendations=["scale"], primary_metric="roc_auc", reasoning="x",
        )


def test_recommendation_report_identity_fields_default_to_none():
    r = RecommendationReport(
        problem_type="classification",
        modeling_profile={},
        candidates=[ModelCandidate(name="ridge", library="sklearn", rationale="r", hyperparameters={})],
        preprocessing_recommendations=["scale"],
        primary_metric="f1_macro",
        reasoning="x",
    )
    assert r.dataset_id is None
    assert r.source_dataset_id is None
    assert r.used_cleaned_dataset is False


# ---------------------------------------------------------------------------
# Task 3: build_modeling_profile + _resolve_source_df
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.agents.recommendation import (  # noqa: E402
    _looks_like_datetime,
    _resolve_source_df,
    _size_bucket,
    build_modeling_profile,
)
from app.storage.dataset_store import save_cleaning_report, save_dataset  # noqa: E402


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))


@pytest.fixture
def clf_frame():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50_000, 12_000, n),
        "city": rng.choice(["NY", "LA", "SF"], n),
        "zipcode": [f"z{i:04d}" for i in range(n)],          # 120 uniques -> high cardinality
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "target": rng.choice([0, 0, 0, 1], n),               # imbalanced binary
    })
    df.loc[:11, "income"] = np.nan                            # 12 missing cells
    return df


CLF_REPORT = {"target_candidate": "target", "problem_type": "classification"}


def test_size_bucket_cutoffs():
    assert _size_bucket(999) == "small"
    assert _size_bucket(1_000) == "medium"
    assert _size_bucket(99_999) == "medium"
    assert _size_bucket(100_000) == "large"


def test_looks_like_datetime_by_name_and_parse():
    assert _looks_like_datetime("signup_date", pd.Series(["2021-01-01", "2021-02-01"]))
    assert not _looks_like_datetime("city", pd.Series(["NY", "LA", "SF"]))
    assert not _looks_like_datetime("age", pd.Series([1, 2, 3]))
    assert _looks_like_datetime("ts", pd.to_datetime(pd.Series(["2021-01-01"])))  # dtype path


def test_profile_shapes_and_feature_split(clf_frame):
    p = build_modeling_profile(clf_frame, CLF_REPORT)
    assert p["n_rows"] == 120
    assert p["target_column"] == "target"
    assert p["n_features"] == 5                               # 6 cols - target
    assert p["n_numeric_features"] + p["n_categorical_features"] == 5
    assert p["n_numeric_features"] == 2                       # age, income
    assert p["problem_type"] == "classification"


def test_profile_class_balance_and_classes(clf_frame):
    p = build_modeling_profile(clf_frame, CLF_REPORT)
    assert p["n_classes"] == 2
    assert set(p["class_balance"]) == {"0", "1"}
    assert abs(sum(p["class_balance"].values()) - 1.0) < 1e-9
    assert p["minority_class_fraction"] == min(p["class_balance"].values())
    assert p["target_kind"] == "numeric"                     # 0/1 ints


def test_profile_missing_pct_and_high_cardinality(clf_frame):
    p = build_modeling_profile(clf_frame, CLF_REPORT)
    assert p["overall_missing_pct"] == pytest.approx(12 / (120 * 6) * 100, abs=0.01)
    assert p["n_high_cardinality_categoricals"] == 1          # zipcode only
    assert p["size_bucket"] == "small"
    assert p["has_datetime_column"] is True


def test_profile_regression_target_has_no_class_fields():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "price": [10.5, 20.1, 33.7, 41.2]})
    p = build_modeling_profile(df, {"target_candidate": "price", "problem_type": "regression"})
    assert p["n_classes"] is None
    assert p["class_balance"] is None
    assert p["minority_class_fraction"] is None
    assert p["target_kind"] == "numeric"
    assert p["n_features"] == 1


def test_profile_when_target_absent_counts_all_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    p = build_modeling_profile(df, {"target_candidate": "missing_col", "problem_type": "classification"})
    assert p["target_column"] is None
    assert p["n_features"] == 2
    assert p["n_classes"] is None


def test_resolve_prefers_cleaned_frame(clf_frame):
    orig_id = save_dataset("orig.csv", clf_frame.to_csv(index=False).encode())
    cleaned = clf_frame.iloc[:50]
    cleaned_id = save_dataset("clean.csv", cleaned.to_csv(index=False).encode())
    save_cleaning_report(orig_id, {"cleaned_dataset_id": cleaned_id})

    df, src_id, used = _resolve_source_df(orig_id)
    assert used is True
    assert src_id == cleaned_id
    assert len(df) == 50


def test_resolve_falls_back_to_original_without_cleaning_report(clf_frame):
    orig_id = save_dataset("orig.csv", clf_frame.to_csv(index=False).encode())
    df, src_id, used = _resolve_source_df(orig_id)
    assert (src_id, used) == (orig_id, False)
    assert len(df) == 120


def test_resolve_falls_back_when_cleaned_file_missing(clf_frame):
    orig_id = save_dataset("orig.csv", clf_frame.to_csv(index=False).encode())
    save_cleaning_report(orig_id, {"cleaned_dataset_id": "ghost-id"})
    df, src_id, used = _resolve_source_df(orig_id)
    assert (src_id, used) == (orig_id, False)


def test_resolve_raises_when_nothing_on_disk():
    with pytest.raises(FileNotFoundError):
        _resolve_source_df("no-such-dataset")


# ---------------------------------------------------------------------------
# Task 4: recommend_models - the single reasoning call
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from app.agents.recommendation import get_catalog_entry, recommend_models  # noqa: E402

_CLF_PROFILE = {"problem_type": "classification", "n_rows": 800, "size_bucket": "small",
                "n_features": 6, "minority_class_fraction": 0.2}
_REG_PROFILE = {"problem_type": "regression", "n_rows": 5000, "size_bucket": "medium", "n_features": 9}


def _tool_use_block(payload, block_id="b1"):
    return SimpleNamespace(type="tool_use", name="submit_recommendation", input=payload, id=block_id)


def _resp(content):
    return SimpleNamespace(stop_reason="tool_use", content=content)


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def _payload(names, metric="roc_auc", prep=None, reasoning="because"):
    return {
        "candidates": [{"name": n, "rationale": f"rationale for {n}"} for n in names],
        "preprocessing_recommendations": prep or ["scale numeric", "one-hot city"],
        "primary_metric": metric,
        "reasoning": reasoning,
    }


def test_recommend_copies_library_and_hyperparameters_from_catalog():
    names = ["random_forest_classifier", "xgboost_classifier", "logistic_regression"]
    client = FakeAnthropicClient([_resp([_tool_use_block(_payload(names))])])
    report = recommend_models(_CLF_PROFILE, "classification", client=client)

    assert len(client.calls) == 1
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_recommendation"}
    assert [c.name for c in report.candidates] == names
    assert report.candidates[0].library == "sklearn"
    assert report.candidates[1].library == "xgboost"
    assert report.candidates[0].hyperparameters == get_catalog_entry(
        "classification", "random_forest_classifier")["hyperparameters"]
    assert report.candidates[0].hyperparameters["random_state"] == 42
    assert report.candidates[0].rationale == "rationale for random_forest_classifier"
    assert report.primary_metric == "roc_auc"
    assert report.problem_type == "classification"
    assert report.reasoning == "because"
    assert report.modeling_profile == _CLF_PROFILE
    assert report.dataset_id is None


def test_recommend_drops_unknown_names_and_backfills_to_three():
    names = ["catboost_clf", "random_forest_classifier", "not_real"]     # only 1 valid
    client = FakeAnthropicClient([_resp([_tool_use_block(_payload(names))])])
    report = recommend_models(_CLF_PROFILE, "classification", client=client)

    got = [c.name for c in report.candidates]
    assert len(got) >= 3
    assert set(got) <= set(catalog_names("classification"))
    assert "random_forest_classifier" in got
    assert "catboost_clf" not in got


def test_recommend_truncates_to_four_valid_candidates():
    names = ["random_forest_classifier", "xgboost_classifier",
             "lightgbm_classifier", "logistic_regression"]
    client = FakeAnthropicClient([_resp([_tool_use_block(_payload(names))])])
    report = recommend_models(_CLF_PROFILE, "classification", client=client)
    assert len(report.candidates) == 4


def test_recommend_rejects_invalid_metric_per_problem_type():
    client = FakeAnthropicClient([_resp([_tool_use_block(
        _payload(["ridge", "xgboost_regressor", "lightgbm_regressor"], metric="roc_auc"))])])  # clf metric on reg
    report = recommend_models(_REG_PROFILE, "regression", client=client)
    assert report.primary_metric == "r2"
    assert [c.library for c in report.candidates] == ["sklearn", "xgboost", "lightgbm"]


def test_recommend_falls_back_to_default_when_llm_returns_no_tool_block():
    empty = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="no")])
    client = FakeAnthropicClient([empty, empty])
    report = recommend_models(_CLF_PROFILE, "classification", client=client)

    assert len(report.candidates) == 3
    assert [c.name for c in report.candidates] == catalog_names("classification")[:3]
    assert report.primary_metric == "f1_macro"
    assert "fallback" in report.reasoning.lower()


def test_recommend_never_exposes_numeric_metric_fields():
    client = FakeAnthropicClient([_resp([_tool_use_block(
        _payload(["ridge", "random_forest_regressor", "lightgbm_regressor"], metric="mae"))])])
    report = recommend_models(_REG_PROFILE, "regression", client=client)
    assert isinstance(report.primary_metric, str)
    assert not hasattr(report, "metrics")
    assert report.model_dump().get("primary_metric") == "mae"


# ---------------------------------------------------------------------------
# Task 5: POST /datasets/{id}/recommend endpoint
# ---------------------------------------------------------------------------
import io  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.storage.dataset_store import get_artifact, get_report, save_report  # noqa: E402


@pytest.fixture
def rec_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    tc = TestClient(app)
    rows = "\n".join(f"{20 + i},{1000 * i},{'NY' if i % 2 else 'LA'},{i % 2}" for i in range(40))
    csv_bytes = ("age,income,city,target\n" + rows + "\n").encode()
    resp = tc.post("/datasets/upload", files={"file": ("p.csv", io.BytesIO(csv_bytes), "text/csv")})
    return tc, resp.json()["dataset_id"]


_CLF_REPORT = {"target_candidate": "target", "problem_type": "classification",
               "dataset_id": "x", "n_rows": 40, "n_columns": 4}


def _stub_recommend(monkeypatch, capture=None):
    from app.agents.recommendation import ModelCandidate, RecommendationReport

    def fake(profile, problem_type, client=None):
        if capture is not None:
            capture["profile"] = profile
        return RecommendationReport(
            problem_type=problem_type,
            modeling_profile=profile,
            candidates=[
                ModelCandidate(name=n, library="sklearn", rationale="r", hyperparameters={})
                for n in ("logistic_regression", "random_forest_classifier",
                          "hist_gradient_boosting_classifier")
            ],
            preprocessing_recommendations=["scale"],
            primary_metric="roc_auc",
            reasoning="stub",
        )

    monkeypatch.setattr("app.routers.recommend.recommend_models", fake)


def test_get_recommend_404_before_run(rec_client):
    tc, dsid = rec_client
    assert tc.get(f"/datasets/{dsid}/recommend").status_code == 404


def test_get_recommend_returns_cached_after_run(rec_client):
    from app.agents.recommendation import ModelCandidate, RecommendationReport
    from app.storage.dataset_store import save_artifact

    tc, dsid = rec_client
    stub = RecommendationReport(
        dataset_id=dsid,
        problem_type="classification",
        modeling_profile={},
        candidates=[
            ModelCandidate(name="ridge", library="sklearn", rationale="r", hyperparameters={})
        ],
        preprocessing_recommendations=["scale"],
        primary_metric="roc_auc",
        reasoning="stub",
    )
    save_artifact(dsid, "recommendation", stub.model_dump())

    resp = tc.get(f"/datasets/{dsid}/recommend")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dsid
    assert body["primary_metric"] == "roc_auc"
    assert body["reasoning"] == "stub"


def test_recommend_endpoint_503_when_ollama_unreachable_default_provider(rec_client, monkeypatch):
    """AGENTDS_LLM_PROVIDER defaults to "ollama". If it isn't reachable, this
    must 503 with a clear message naming the host — not silently succeed
    just because ANTHROPIC_API_KEY happens to be set or unset."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    tc, dsid = rec_client
    resp = tc.post(f"/datasets/{dsid}/recommend")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_recommend_endpoint_requires_api_key(rec_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tc, dsid = rec_client
    assert tc.post(f"/datasets/{dsid}/recommend").status_code == 503


def test_recommend_endpoint_unknown_dataset_404(rec_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, _ = rec_client
    assert tc.post("/datasets/does-not-exist/recommend").status_code == 404


def test_recommend_endpoint_422_when_problem_type_unclear(rec_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dsid = rec_client
    save_report(dsid, {"target_candidate": "target", "problem_type": "unclear"})
    assert tc.post(f"/datasets/{dsid}/recommend").status_code == 422


def test_recommend_endpoint_happy_path_caches_artifact(rec_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dsid = rec_client
    save_report(dsid, _CLF_REPORT)
    _stub_recommend(monkeypatch)

    resp = tc.post(f"/datasets/{dsid}/recommend")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dsid
    assert body["problem_type"] == "classification"
    assert len(body["candidates"]) == 3
    assert body["used_cleaned_dataset"] is False
    assert body["source_dataset_id"] == dsid

    cached = get_artifact(dsid, "recommendation")
    assert cached is not None
    assert cached["primary_metric"] == "roc_auc"


def test_recommend_endpoint_profiles_cleaned_frame_when_present(rec_client, monkeypatch):
    from app.storage.dataset_store import save_cleaning_report, save_dataset

    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dsid = rec_client
    save_report(dsid, _CLF_REPORT)
    cleaned_csv = ("age,income,city,target\n" + "\n".join(
        f"{20 + i},{100 * i},{'NY' if i % 2 else 'LA'},{i % 2}" for i in range(10)) + "\n").encode()
    cleaned_id = save_dataset("c.csv", cleaned_csv)
    save_cleaning_report(dsid, {"cleaned_dataset_id": cleaned_id})

    capture: dict = {}
    _stub_recommend(monkeypatch, capture)
    resp = tc.post(f"/datasets/{dsid}/recommend")

    assert resp.status_code == 200
    body = resp.json()
    assert body["source_dataset_id"] == cleaned_id
    assert body["used_cleaned_dataset"] is True
    assert capture["profile"]["n_rows"] == 10          # the CLEANED frame was profiled


def test_recommend_endpoint_runs_module1_when_no_cached_report(rec_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dsid = rec_client

    def fake_du_run(self, df, ds_id):
        from app.agents.data_understanding import ColumnOverview, DataUnderstandingReport, list_columns

        return DataUnderstandingReport(
            dataset_id=ds_id, n_rows=len(df), n_columns=df.shape[1],
            columns=[ColumnOverview(**c) for c in list_columns(df)],
            problem_type="classification", target_candidate="target",
            reasoning="stub", narrative="stub", key_findings=[],
        )

    monkeypatch.setattr("app.agents.data_understanding.DataUnderstandingAgent.run", fake_du_run)
    _stub_recommend(monkeypatch)

    assert get_report(dsid) is None
    resp = tc.post(f"/datasets/{dsid}/recommend")
    assert resp.status_code == 200
    assert get_report(dsid) is not None
