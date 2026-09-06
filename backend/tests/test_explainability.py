"""Tests for the Explainability Agent (Module 6).

Explainer-branch selection is the part to be most rigorous about: each of
tree/linear/kernel gets its own real (not mocked) SHAP computation, plus a
kernel-branch test that proves the background/explain-row caps actually work
(not just present in code) via a shape assertion and a wall-clock bound.
"""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.agents.explainability import (
    MAX_BACKGROUND_ROWS,
    ExplainabilityError,
    _compute_shap_values,
    run_explainability,
    select_explainer,
)
from app.agents.training import RANDOM_STATE, TEST_SIZE, build_preprocessor
from app.storage.dataset_store import (
    get_artifact,
    model_dir,
    save_artifact,
    save_cleaning_report,
    save_dataset,
)


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "u"))


# ---------------------------------------------------------------------------
# select_explainer - one real test per branch, no mocking
# ---------------------------------------------------------------------------
def test_select_explainer_tree_branch():
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )
    model = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)

    explainer, mode = select_explainer(model, X, "classification")

    assert mode == "tree"
    shap_values = np.asarray(explainer.shap_values(X[:3]))
    assert shap_values.shape[0] == 3
    assert shap_values.shape[1] == 6


def test_select_explainer_tree_branch_regression():
    X, y = make_regression(n_samples=60, n_features=5, noise=5.0, random_state=42)
    reg = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    explainer, mode = select_explainer(reg, X, "regression")

    assert mode == "tree"
    shap_values = np.asarray(explainer.shap_values(X[:3]))
    assert shap_values.shape == (3, 5)


def test_select_explainer_linear_branch():
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)

    explainer, mode = select_explainer(model, X, "classification")

    assert mode == "linear"
    shap_values = np.asarray(explainer.shap_values(X[:3]))
    assert shap_values.shape[0] == 3
    assert shap_values.shape[1] == 6


def test_select_explainer_linear_branch_regression():
    X, y = make_regression(n_samples=60, n_features=5, noise=5.0, random_state=42)
    model = Ridge(alpha=1.0, random_state=42).fit(X, y)

    explainer, mode = select_explainer(model, X, "regression")

    assert mode == "linear"
    shap_values = np.asarray(explainer.shap_values(X[:3]))
    assert shap_values.shape == (3, 5)


def test_select_explainer_kernel_branch_caps_background_and_is_fast():
    # 200 background rows fed in - well over the 50-row cap - so this test
    # actually proves the cap works rather than merely being fast because the
    # input happened to be small.
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )
    model = HistGradientBoostingClassifier(random_state=42).fit(X, y)

    started = time.perf_counter()
    explainer, mode = select_explainer(model, X, "classification")

    assert mode == "kernel"
    background_size = explainer.data.data.shape[0]
    assert background_size <= MAX_BACKGROUND_ROWS
    assert background_size < len(X)  # proves summarization actually happened

    shap_values = np.asarray(explainer.shap_values(X[:5]))
    elapsed = time.perf_counter() - started

    assert shap_values.shape[0] == 5
    assert shap_values.shape[1] == 6
    # Time-box sanity check (brief section "Verification before done" #5):
    # this must be a couple of seconds, not tens of seconds.
    assert elapsed < 10.0, f"kernel branch took {elapsed:.3f}s - caps may not be working"
    print(f"\nKERNEL BRANCH WALL-CLOCK: {elapsed:.3f}s (background capped to {background_size})")


def test_select_explainer_kernel_branch_regression():
    X, y = make_regression(n_samples=60, n_features=5, noise=5.0, random_state=42)
    model = HistGradientBoostingRegressor(random_state=42).fit(X, y)

    explainer, mode = select_explainer(model, X, "regression")

    assert mode == "kernel"
    shap_values = np.asarray(explainer.shap_values(X[:3]))
    assert shap_values.shape == (3, 5)


def test_select_explainer_unexplainable_object_raises_with_name():
    class Bogus:
        """Neither predict nor predict_proba - should never reach shap."""

    with pytest.raises(ExplainabilityError, match="Bogus"):
        select_explainer(Bogus(), np.zeros((5, 3)), "classification")


def test_compute_shap_values_wraps_real_shap_failure():
    # A genuine shap.LinearExplainer failure (feature-count mismatch), proving
    # the wrapper converts a raw shap/numpy exception into ExplainabilityError
    # rather than letting it surface as an unhandled 500.
    X, y = make_classification(n_samples=60, n_features=6, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    explainer, mode = select_explainer(model, X, "classification")

    with pytest.raises(ExplainabilityError, match="SHAP failed to explain"):
        _compute_shap_values(explainer, mode, np.zeros((3, 4)), "LogisticRegression")


# ---------------------------------------------------------------------------
# run_explainability - orchestration
# ---------------------------------------------------------------------------
def _persist_trained_pipeline(
    dataset_id, df, target, feature_columns, problem_type, estimator, name, stratified=True
):
    """Fit exactly the split/pipeline shape run_training produces, and persist
    a training-report sidecar + joblib model so run_explainability can load it."""
    pre, _roles = build_preprocessor(df, target, feature_columns)
    X, y = df[feature_columns], df[target]
    if problem_type == "classification" and stratified:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    pipe = Pipeline([("pre", pre), ("model", estimator)])
    pipe.fit(X_train, y_train)

    path = model_dir(dataset_id) / f"{name}.joblib"
    joblib.dump(pipe, path)

    training_report = {
        "dataset_id": dataset_id,
        "problem_type": problem_type,
        "target": target,
        "feature_columns": feature_columns,
        "stratified": stratified,
        "best_model": name,
        "candidates": [{"name": name, "model_path": str(path)}],
    }
    save_artifact(dataset_id, "training", training_report)
    return training_report, (X_train, X_test, y_train, y_test)


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3, n_redundant=0, random_state=42
    )
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    d["target"] = y
    return d


@pytest.fixture
def reg_df():
    X, y = make_regression(n_samples=80, n_features=5, noise=5.0, random_state=42)
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    d["target"] = y
    return d


def test_run_explainability_tree_branch_happy_path(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-tree",
        clf_df,
        "target",
        feature_columns,
        "classification",
        RandomForestClassifier(n_estimators=20, random_state=42),
        "random_forest_classifier",
    )

    report = run_explainability(clf_df, "ds-tree", client=None)

    assert report.explainer_type == "tree"
    assert report.model_name == "random_forest_classifier"
    assert report.problem_type == "classification"
    assert report.target == "target"
    assert report.n_rows_explained == 16  # 80 rows * test_size=0.2
    assert len(report.feature_importance) == 5
    vals = [f.mean_abs_shap for f in report.feature_importance]
    assert vals == sorted(vals, reverse=True)  # sorted descending
    assert all(f.mean_abs_shap >= 0 for f in report.feature_importance)
    assert 1 <= len(report.sample_explanations) <= 5
    for sample in report.sample_explanations:
        assert 1 <= len(sample.top_reasons) <= 3
        shap_mags = [abs(r.shap_value) for r in sample.top_reasons]
        assert shap_mags == sorted(shap_mags, reverse=True)
    assert report.narrative == ""
    assert report.warnings == []


def test_run_explainability_linear_branch_happy_path(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-linear",
        clf_df,
        "target",
        feature_columns,
        "classification",
        LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression",
    )

    report = run_explainability(clf_df, "ds-linear", client=None)

    assert report.explainer_type == "linear"
    assert len(report.feature_importance) == 5


def test_run_explainability_regression_tree_branch(reg_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-reg-tree",
        reg_df,
        "target",
        feature_columns,
        "regression",
        RandomForestRegressor(n_estimators=20, random_state=42),
        "random_forest_regressor",
    )

    report = run_explainability(reg_df, "ds-reg-tree", client=None)

    assert report.explainer_type == "tree"
    assert report.problem_type == "regression"
    for sample in report.sample_explanations:
        assert isinstance(sample.predicted_value, float)


def test_run_explainability_kernel_branch_end_to_end():
    X, y = make_classification(
        n_samples=300, n_features=5, n_informative=3, n_redundant=0, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-kernel",
        df,
        "target",
        feature_columns,
        "classification",
        HistGradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting_classifier",
    )

    started = time.perf_counter()
    report = run_explainability(df, "ds-kernel", client=None)
    elapsed = time.perf_counter() - started

    assert report.explainer_type == "kernel"
    assert report.n_rows_explained == 50  # 300*0.2=60 test rows, capped to 50
    assert any("KernelExplainer" in w for w in report.warnings)
    assert any("capped" in w.lower() for w in report.warnings)
    assert elapsed < 15.0, f"kernel end-to-end took {elapsed:.3f}s"
    print(f"\nKERNEL END-TO-END WALL-CLOCK: {elapsed:.3f}s")


def test_run_explainability_multiclass_uses_predicted_class_shap():
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0, n_classes=3, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-mc",
        df,
        "target",
        feature_columns,
        "classification",
        RandomForestClassifier(n_estimators=20, random_state=42),
        "random_forest_classifier",
    )

    report = run_explainability(df, "ds-mc", client=None)

    assert report.explainer_type == "tree"
    predicted_labels = {int(float(s.predicted_value)) for s in report.sample_explanations}
    assert predicted_labels <= {0, 1, 2}


def test_run_explainability_no_training_report_raises():
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(ExplainabilityError, match="run /train first"):
        run_explainability(df, "ds-missing", client=None)


def test_run_explainability_null_best_model_raises():
    save_artifact("ds-nobest", "training", {"best_model": None, "candidates": []})
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(ExplainabilityError, match="all candidates failed"):
        run_explainability(df, "ds-nobest", client=None)


def test_run_explainability_missing_candidate_entry_raises():
    save_artifact(
        "ds-nocand",
        "training",
        {"best_model": "ghost_model", "candidates": [{"name": "other", "model_path": "x"}]},
    )
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(ExplainabilityError, match="ghost_model"):
        run_explainability(df, "ds-nocand", client=None)


def test_run_explainability_missing_target_column_raises(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-badtarget",
        clf_df,
        "target",
        feature_columns,
        "classification",
        LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression",
    )
    broken_df = clf_df.drop(columns=["target"])

    with pytest.raises(ExplainabilityError, match="target"):
        run_explainability(broken_df, "ds-badtarget", client=None)


def test_run_explainability_missing_feature_column_raises(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-badfeat",
        clf_df,
        "target",
        feature_columns,
        "classification",
        LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression",
    )
    broken_df = clf_df.drop(columns=["f0"])

    with pytest.raises(ExplainabilityError, match="f0"):
        run_explainability(broken_df, "ds-badfeat", client=None)


# ---------------------------------------------------------------------------
# narrate() guard
# ---------------------------------------------------------------------------
def _tool_use_block(name, tool_input, block_id="c1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeAnthropicClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class BoomClient:
    def __init__(self):
        self.messages = SimpleNamespace(create=self._boom)

    def _boom(self, **kwargs):
        raise RuntimeError("no network")


def test_run_explainability_narrates_with_client(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-narr",
        clf_df,
        "target",
        feature_columns,
        "classification",
        RandomForestClassifier(n_estimators=20, random_state=42),
        "random_forest_classifier",
    )
    client = FakeAnthropicClient(
        [
            _response(
                "tool_use",
                [_tool_use_block("submit_shap_narrative", {"narrative": "f0 drives predictions."})],
            )
        ]
    )

    report = run_explainability(clf_df, "ds-narr", client=client)

    assert report.narrative == "f0 drives predictions."
    assert len(client.calls) == 1


def test_run_explainability_narration_failure_is_non_fatal(clf_df):
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        "ds-narr-fail",
        clf_df,
        "target",
        feature_columns,
        "classification",
        RandomForestClassifier(n_estimators=20, random_state=42),
        "random_forest_classifier",
    )

    report = run_explainability(clf_df, "ds-narr-fail", client=BoomClient())

    assert report.narrative == ""
    assert len(report.feature_importance) == 5


# ---------------------------------------------------------------------------
# POST /datasets/{id}/explain endpoint
# ---------------------------------------------------------------------------
@pytest.fixture
def explain_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "u"))
    from app.main import app

    tc = TestClient(app)
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3, n_redundant=0, random_state=42
    )
    lines = ["f0,f1,f2,f3,f4,target"]
    for row, label in zip(X, y):
        lines.append(",".join(f"{v:.4f}" for v in row) + f",{label}")
    csv_bytes = ("\n".join(lines) + "\n").encode()
    resp = tc.post(
        "/datasets/upload", files={"file": ("synth.csv", io.BytesIO(csv_bytes), "text/csv")}
    )
    return tc, resp.json()["dataset_id"]


def _fake_anthropic_ctor():
    return FakeAnthropicClient(
        [
            _response(
                "tool_use",
                [_tool_use_block("submit_shap_narrative", {"narrative": "top feature drives it."})],
            )
        ]
    )


def test_get_explain_404_before_run(explain_client):
    tc, dataset_id = explain_client
    assert tc.get(f"/datasets/{dataset_id}/explain").status_code == 404


def test_get_explain_returns_cached_after_run(explain_client):
    from app.agents.explainability import ExplainabilityReport
    from app.storage.dataset_store import save_artifact

    tc, dataset_id = explain_client
    stub = ExplainabilityReport(
        dataset_id=dataset_id,
        model_name="random_forest_classifier",
        problem_type="classification",
        target="target",
        explainer_type="tree",
        n_rows_explained=16,
        feature_importance=[],
        sample_explanations=[],
        narrative="stub narrative",
    )
    save_artifact(dataset_id, "explainability", stub.model_dump())

    resp = tc.get(f"/datasets/{dataset_id}/explain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == dataset_id
    assert body["explainer_type"] == "tree"
    assert body["narrative"] == "stub narrative"


def test_explain_endpoint_503_when_ollama_unreachable_default_provider(explain_client, monkeypatch):
    """AGENTDS_LLM_PROVIDER defaults to "ollama". If it isn't reachable, this
    must 503 with a clear message naming the host — not silently succeed
    just because ANTHROPIC_API_KEY happens to be set or unset."""
    import httpx

    monkeypatch.delenv("AGENTDS_LLM_PROVIDER", raising=False)

    def fake_get(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    tc, dataset_id = explain_client
    resp = tc.post(f"/datasets/{dataset_id}/explain")
    assert resp.status_code == 503
    assert "localhost:11434" in resp.json()["detail"]


def test_explain_endpoint_requires_api_key(explain_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tc, dataset_id = explain_client
    resp = tc.post(f"/datasets/{dataset_id}/explain")
    assert resp.status_code == 503


def test_explain_endpoint_unknown_dataset_404(explain_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, _ = explain_client
    resp = tc.post("/datasets/does-not-exist/explain")
    assert resp.status_code == 404


def test_explain_endpoint_422_when_no_training_report(explain_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dataset_id = explain_client
    resp = tc.post(f"/datasets/{dataset_id}/explain")
    assert resp.status_code == 422
    assert "run /train first" in resp.json()["detail"]


def test_explain_endpoint_happy_path_persists_artifact(explain_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dataset_id = explain_client
    monkeypatch.setattr(
        "app.routers.explain.get_client", lambda: _fake_anthropic_ctor()
    )

    from app.storage.dataset_store import get_dataset_path

    df = pd.read_csv(get_dataset_path(dataset_id))
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        dataset_id,
        df,
        "target",
        feature_columns,
        "classification",
        RandomForestClassifier(n_estimators=20, random_state=42),
        "random_forest_classifier",
    )

    resp = tc.post(f"/datasets/{dataset_id}/explain")

    assert resp.status_code == 200
    body = resp.json()
    assert body["explainer_type"] == "tree"
    assert body["model_name"] == "random_forest_classifier"
    assert body["narrative"] == "top feature drives it."

    cached = get_artifact(dataset_id, "explainability")
    assert cached is not None
    assert cached["explainer_type"] == "tree"


def test_explain_endpoint_uses_cleaned_dataset_when_present(explain_client, monkeypatch):
    monkeypatch.setenv("AGENTDS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    tc, dataset_id = explain_client
    monkeypatch.setattr(
        "app.routers.explain.get_client", lambda: _fake_anthropic_ctor()
    )

    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3, n_redundant=0, random_state=7
    )
    lines = ["f0,f1,f2,f3,f4,target"]
    for row, label in zip(X, y):
        lines.append(",".join(f"{v:.4f}" for v in row) + f",{label}")
    csv_bytes = ("\n".join(lines) + "\n").encode()
    cleaned_id = save_dataset("cleaned.csv", csv_bytes)
    save_cleaning_report(dataset_id, {"cleaned_dataset_id": cleaned_id})

    from app.storage.dataset_store import get_dataset_path

    cleaned_df = pd.read_csv(get_dataset_path(cleaned_id))
    feature_columns = [f"f{i}" for i in range(5)]
    _persist_trained_pipeline(
        dataset_id,
        cleaned_df,
        "target",
        feature_columns,
        "classification",
        LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression",
    )

    resp = tc.post(f"/datasets/{dataset_id}/explain")

    assert resp.status_code == 200
    assert resp.json()["explainer_type"] == "linear"
