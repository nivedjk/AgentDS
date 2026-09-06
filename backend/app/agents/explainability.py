"""Explainability Agent (Module 6).

Loads the best fitted pipeline from Module 5's training report, reproduces the
exact train/test split ``run_training`` used (no raw arrays are persisted, so
the split is re-derived deterministically from the recorded split parameters),
and computes SHAP feature-importance / per-prediction explanations. Every
number in the final report comes from a real SHAP computation; a single
``narrate()`` call at the end only writes prose over the already-computed
facts. No agent loop, no tool-use loop.

Explainer selection is explicit: a fixed set of tree-model class names goes to
``shap.TreeExplainer``, a fixed set of linear-model class names goes to
``shap.LinearExplainer``, and everything else (deliberately including
``HistGradientBoosting*``) goes to a capped ``shap.KernelExplainer`` so a
big/exotic model can never hang the endpoint.
"""

from __future__ import annotations

from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from app.agents._llm import narrate
from app.agents.training import RANDOM_STATE, TEST_SIZE
from app.storage.dataset_store import get_artifact

# ---------------------------------------------------------------------------
# Fixed knobs
# ---------------------------------------------------------------------------
MAX_EXPLAIN_ROWS = 50
MAX_BACKGROUND_ROWS = 50
MAX_SAMPLE_EXPLANATIONS_SMALL = 10  # if n_explain <= this, sample every row
MAX_SAMPLE_EXPLANATIONS = 5  # else sample only the first this many
SHAP_ROUND = 6

# Explicit, exhaustive class-name sets for explainer selection - no guessing,
# no generic "try tree first" fallback chain.
TREE_MODEL_NAMES: set[str] = {
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "XGBClassifier",
    "XGBRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",  # not in our catalog; included for completeness
}
LINEAR_MODEL_NAMES: set[str] = {"LogisticRegression", "LinearRegression", "Ridge"}


class ExplainabilityError(Exception):
    """A SHAP explanation could not be produced.

    Raised for a missing training report, a null ``best_model``, a split that
    can't be reproduced against the current dataframe, or a SHAP failure.
    Never an ``HTTPException`` - the router maps it to a response.
    """


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------
class FeatureImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class Reason(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class SamplePrediction(BaseModel):
    row_index: int
    predicted_value: float | str
    top_reasons: list[Reason]


class ExplainabilityReport(BaseModel):
    dataset_id: str
    model_name: str
    problem_type: Literal["classification", "regression"]
    target: str
    explainer_type: Literal["tree", "linear", "kernel"]
    n_rows_explained: int
    feature_importance: list[FeatureImportance]
    sample_explanations: list[SamplePrediction]
    narrative: str = ""
    warnings: list[str] = []


class _ExplainabilityNarration(BaseModel):
    narrative: str


_NARRATION_SYSTEM = (
    "You write a two-to-three sentence plain-English summary of a SHAP "
    "feature-importance report. Name the top 2-3 features and, briefly, how "
    "they push predictions. No markdown, no hype, no numbers you were not "
    "given verbatim."
)
_NARRATION_TOOL_DESCRIPTION = "Return the SHAP explanation narrative as a single short paragraph."


# ---------------------------------------------------------------------------
# Explainer selection - the part to be most rigorous about
# ---------------------------------------------------------------------------
def select_explainer(
    estimator: Any, background: np.ndarray, problem_type: str
) -> tuple[Any, Literal["tree", "linear", "kernel"]]:
    """Pick the right SHAP explainer for ``estimator`` by exact class name.

    - Tree-family name -> ``shap.TreeExplainer(estimator)``.
    - Linear-family name -> ``shap.LinearExplainer(estimator, background)``.
    - Anything else (e.g. ``HistGradientBoosting*``, deliberately excluded
      from the tree set even though shap can sometimes explain them as trees)
      -> a capped ``shap.KernelExplainer``: the background is summarized to
      at most ``MAX_BACKGROUND_ROWS`` points via ``shap.sample`` (chosen over
      ``shap.kmeans`` specifically because it takes an explicit
      ``random_state`` and is therefore trivially deterministic), and the
      predict function is ``predict_proba`` for classification when
      available, else ``predict``.
    """
    name = type(estimator).__name__

    if name in TREE_MODEL_NAMES:
        return shap.TreeExplainer(estimator), "tree"

    if name in LINEAR_MODEL_NAMES:
        return shap.LinearExplainer(estimator, background), "linear"

    if problem_type == "classification" and hasattr(estimator, "predict_proba"):
        predict_fn = estimator.predict_proba
    elif hasattr(estimator, "predict"):
        predict_fn = estimator.predict
    else:
        raise ExplainabilityError(
            f"Cannot build any SHAP explainer for {name!r}: no predict/predict_proba"
        )

    background_summary = shap.sample(
        background, min(len(background), MAX_BACKGROUND_ROWS), random_state=RANDOM_STATE
    )
    return shap.KernelExplainer(predict_fn, background_summary), "kernel"


def _compute_shap_values(explainer: Any, mode: str, X: np.ndarray, model_name: str) -> np.ndarray:
    """Run the actual SHAP computation, converting any shap/numpy failure into
    a clean ``ExplainabilityError`` rather than an unhandled 500.

    Normalizes the result to a numpy array: some shap explainer/version
    combinations return a list of per-class arrays instead of a single
    ``(n_rows, n_features, n_classes)`` array - this collapses that case to
    the same 3-D shape so downstream code has one shape to handle.
    """
    try:
        values = explainer.shap_values(X)
    except Exception as exc:  # noqa: BLE001 - never let a raw shap exception surface
        raise ExplainabilityError(f"SHAP failed to explain {model_name!r} ({mode}): {exc}") from exc

    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    return np.asarray(values)


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=float)


def _json_scalar(value: Any) -> float | str:
    if isinstance(value, str):
        return value
    return float(value)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_explainability(
    df: pd.DataFrame, dataset_id: str, client: Any | None = None
) -> ExplainabilityReport:
    """Load the best trained pipeline, reproduce its test split, and compute
    SHAP-based global + per-prediction explanations.

    ``df`` is the already-resolved working dataframe (cleaned dataset if one
    exists, else the original) - callers replicate the same cleaned-vs-
    original fallback ``app/routers/train.py`` uses before calling this.
    """
    training_report = get_artifact(dataset_id, "training")
    if training_report is None:
        raise ExplainabilityError(
            f"no training report for dataset {dataset_id!r}; run /train first"
        )

    best_name = training_report.get("best_model")
    if not best_name:
        raise ExplainabilityError(
            "no best_model in training report; all candidates failed"
        )

    candidate = next(
        (c for c in training_report.get("candidates", []) if c.get("name") == best_name),
        None,
    )
    if candidate is None or not candidate.get("model_path"):
        raise ExplainabilityError(
            f"no model_path recorded for best_model {best_name!r} in the training report"
        )

    try:
        pipeline = joblib.load(candidate["model_path"])
    except (FileNotFoundError, OSError) as exc:
        raise ExplainabilityError(
            f"could not load persisted model for {best_name!r}: {exc}"
        ) from exc

    target = training_report.get("target")
    feature_columns = training_report.get("feature_columns") or []
    problem_type = training_report.get("problem_type")
    stratified = training_report.get("stratified", True)

    if problem_type not in ("classification", "regression"):
        raise ExplainabilityError(
            f"training report problem_type is {problem_type!r}; cannot explain"
        )
    if not target or target not in df.columns:
        raise ExplainabilityError(
            f"target {target!r} from the training report is not a column in the dataset"
        )
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ExplainabilityError(
            f"feature columns from the training report are missing from the dataset: {missing}"
        )

    df = df.dropna(subset=[target])
    X, y = df[feature_columns], df[target]

    if problem_type == "classification" and stratified:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    if len(X_test) == 0:
        raise ExplainabilityError("reproduced test split has zero rows; cannot explain")

    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    X_train_t = _dense(pre.transform(X_train))

    n_explain = min(len(X_test), MAX_EXPLAIN_ROWS)
    X_test_sub = X_test.iloc[:n_explain]
    X_test_t = _dense(pre.transform(X_test_sub))

    feature_names = list(pre.get_feature_names_out())

    warn_msgs: list[str] = []
    if len(X_test) > MAX_EXPLAIN_ROWS:
        warn_msgs.append(
            f"Explained rows capped to {n_explain} of {len(X_test)} held-out test rows."
        )

    try:
        explainer, mode = select_explainer(model, X_train_t, problem_type)
    except ExplainabilityError:
        raise

    if mode == "kernel":
        warn_msgs.append(
            f"{type(model).__name__} is not a tree or linear model; used a capped "
            f"KernelExplainer with a background summarized to at most {MAX_BACKGROUND_ROWS} points."
        )

    shap_values = _compute_shap_values(explainer, mode, X_test_t, type(model).__name__)
    y_pred = pipeline.predict(X_test_sub)

    per_class = shap_values.ndim == 3  # (n_rows, n_features, n_classes)
    if per_class:
        classes = list(getattr(model, "classes_", []))
        global_abs = np.abs(shap_values).mean(axis=(0, 2))
        row_shap = np.zeros((n_explain, shap_values.shape[1]))
        for i in range(n_explain):
            try:
                class_idx = classes.index(y_pred[i])
            except ValueError:
                class_idx = 0
            row_shap[i] = shap_values[i, :, class_idx]
    else:
        global_abs = np.abs(shap_values).mean(axis=0)
        row_shap = shap_values

    order = np.argsort(-global_abs)
    feature_importance = [
        FeatureImportance(
            feature=feature_names[i], mean_abs_shap=round(float(global_abs[i]), SHAP_ROUND)
        )
        for i in order
    ]

    n_samples = n_explain if n_explain <= MAX_SAMPLE_EXPLANATIONS_SMALL else MAX_SAMPLE_EXPLANATIONS
    sample_explanations: list[SamplePrediction] = []
    for i in range(n_samples):
        row_vals = row_shap[i]
        reason_order = np.argsort(-np.abs(row_vals))[:3]
        reasons = [
            Reason(
                feature=feature_names[j],
                shap_value=round(float(row_vals[j]), SHAP_ROUND),
                feature_value=float(X_test_t[i, j]),
            )
            for j in reason_order
        ]
        sample_explanations.append(
            SamplePrediction(
                row_index=i,
                predicted_value=_json_scalar(y_pred[i]),
                top_reasons=reasons,
            )
        )

    narrative = ""
    if client is not None:
        try:
            result = narrate(
                client,
                system=_NARRATION_SYSTEM,
                user=_narration_summary(best_name, feature_importance, sample_explanations),
                schema=_ExplainabilityNarration,
                tool_name="submit_shap_narrative",
                tool_description=_NARRATION_TOOL_DESCRIPTION,
                default=_ExplainabilityNarration(narrative=""),
            )
            narrative = result.narrative
        except Exception:  # noqa: BLE001 - narration is never fatal
            narrative = ""

    return ExplainabilityReport(
        dataset_id=dataset_id,
        model_name=best_name,
        problem_type=problem_type,
        target=target,
        explainer_type=mode,
        n_rows_explained=n_explain,
        feature_importance=feature_importance,
        sample_explanations=sample_explanations,
        narrative=narrative,
        warnings=warn_msgs,
    )


def _narration_summary(
    model_name: str,
    feature_importance: list[FeatureImportance],
    sample_explanations: list[SamplePrediction],
) -> str:
    lines = [f"Best model: {model_name}.", "Global feature importance (mean |SHAP value|):"]
    for fi in feature_importance[:10]:
        lines.append(f"  {fi.feature}: {fi.mean_abs_shap}")
    lines.append("Example predictions and their top reasons:")
    for sample in sample_explanations[:2]:
        reasons = ", ".join(
            f"{r.feature}={r.shap_value}" for r in sample.top_reasons
        )
        lines.append(f"  row {sample.row_index} predicted {sample.predicted_value}: {reasons}")
    return "\n".join(lines)
