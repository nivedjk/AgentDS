"""Model Recommendation Agent (Module 4).

Deterministically profiles a (cleaned) tabular dataset, then makes ONE LLM
reasoning call (provider selected by AGENTDS_LLM_PROVIDER, same as every
other module) to pick 3-4 candidate models from a FIXED catalog, a prose
rationale for each, the preprocessing the data needs, and a primary evaluation
metric. No training happens here and no metric numbers are produced: the LLM
only chooses catalog names and writes prose. ``library`` and the fixed default
hyperparameters are copied from the catalog verbatim - they are the contract
Module 5 consumes.

``MODEL_CATALOG`` is plain dict metadata on purpose: this module never imports
scikit-learn / xgboost / lightgbm.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from app.agents._llm import narrate
from app.storage.dataset_store import get_cleaning_report, get_dataset_path

# ---------------------------------------------------------------------------
# Fixed model catalog - the contract Module 5 (Training & AutoML) consumes.
# Each entry: {name, library, estimator (dotted import path), hyperparameters, blurb}.
# ---------------------------------------------------------------------------
MODEL_CATALOG: dict[str, list[dict[str, Any]]] = {
    "classification": [
        {
            "name": "logistic_regression",
            "library": "sklearn",
            "estimator": "sklearn.linear_model.LogisticRegression",
            "hyperparameters": {"max_iter": 1000, "C": 1.0, "random_state": 42},
            "blurb": "Linear baseline; fast, calibrated, needs scaled numeric features.",
        },
        {
            "name": "decision_tree_classifier",
            "library": "sklearn",
            "estimator": "sklearn.tree.DecisionTreeClassifier",
            "hyperparameters": {"random_state": 42},
            "blurb": "Single CART tree; interpretable, handles mixed types, prone to overfitting.",
        },
        {
            "name": "random_forest_classifier",
            "library": "sklearn",
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "hyperparameters": {"n_estimators": 200, "n_jobs": 1, "random_state": 42},
            "blurb": "Bagged trees; strong default, handles mixed feature types, little tuning.",
        },
        {
            "name": "hist_gradient_boosting_classifier",
            "library": "sklearn",
            "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
            "hyperparameters": {"random_state": 42},
            "blurb": "Histogram gradient boosting; native missing-value support, fast on wide data.",
        },
        {
            "name": "xgboost_classifier",
            "library": "xgboost",
            "estimator": "xgboost.XGBClassifier",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "random_state": 42,
                "tree_method": "hist",
                "verbosity": 0,
                "eval_metric": "logloss",
            },
            "blurb": "Gradient-boosted trees; top-tier accuracy on tabular data, more tuning surface.",
        },
        {
            "name": "lightgbm_classifier",
            "library": "lightgbm",
            "estimator": "lightgbm.LGBMClassifier",
            "hyperparameters": {"n_estimators": 200, "n_jobs": 1, "random_state": 42, "verbose": -1},
            "blurb": "Leaf-wise gradient boosting; very fast, great on large / high-cardinality data.",
        },
    ],
    "regression": [
        {
            "name": "ridge",
            "library": "sklearn",
            "estimator": "sklearn.linear_model.Ridge",
            "hyperparameters": {"alpha": 1.0, "random_state": 42},
            "blurb": "L2-regularized linear baseline; needs scaled numeric features.",
        },
        {
            "name": "decision_tree_regressor",
            "library": "sklearn",
            "estimator": "sklearn.tree.DecisionTreeRegressor",
            "hyperparameters": {"random_state": 42},
            "blurb": "Single CART tree; interpretable, handles mixed types, prone to overfitting.",
        },
        {
            "name": "random_forest_regressor",
            "library": "sklearn",
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "hyperparameters": {"n_estimators": 200, "n_jobs": 1, "random_state": 42},
            "blurb": "Bagged trees; robust default, handles mixed feature types.",
        },
        {
            "name": "hist_gradient_boosting_regressor",
            "library": "sklearn",
            "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
            "hyperparameters": {"random_state": 42},
            "blurb": "Histogram gradient boosting; native missing-value support, fast.",
        },
        {
            "name": "xgboost_regressor",
            "library": "xgboost",
            "estimator": "xgboost.XGBRegressor",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": 1,
                "random_state": 42,
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "verbosity": 0,
            },
            "blurb": "Gradient-boosted trees; top-tier tabular accuracy.",
        },
        {
            "name": "lightgbm_regressor",
            "library": "lightgbm",
            "estimator": "lightgbm.LGBMRegressor",
            "hyperparameters": {"n_estimators": 200, "n_jobs": 1, "random_state": 42, "verbose": -1},
            "blurb": "Leaf-wise gradient boosting; very fast on large data.",
        },
    ],
}

CLASSIFICATION_METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc")
REGRESSION_METRICS = ("r2", "mae", "rmse")
DEFAULT_METRIC = {"classification": "f1_macro", "regression": "r2"}


def catalog_names(problem_type: str) -> list[str]:
    return [e["name"] for e in MODEL_CATALOG[problem_type]]


def get_catalog_entry(problem_type: str, name: str) -> dict:
    for e in MODEL_CATALOG[problem_type]:
        if e["name"] == name:
            return e
    raise KeyError(f"{name!r} is not in the {problem_type} catalog")


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------
class ModelCandidate(BaseModel):
    name: str
    library: Literal["sklearn", "xgboost", "lightgbm"]
    rationale: str
    hyperparameters: dict[str, Any]


class CandidateChoice(BaseModel):
    """One LLM-chosen catalog name + why. The LLM never supplies hyperparameters."""

    name: str
    rationale: str = Field(min_length=1)


class RecommendationLLMInput(BaseModel):
    """The forced-tool payload shape for the single reasoning call."""

    candidates: list[CandidateChoice] = Field(min_length=3, max_length=4)
    preprocessing_recommendations: list[str] = Field(min_length=1)
    primary_metric: str
    reasoning: str = Field(min_length=1)


class RecommendationReport(BaseModel):
    dataset_id: str | None = None
    source_dataset_id: str | None = None
    used_cleaned_dataset: bool = False
    problem_type: Literal["classification", "regression"]
    modeling_profile: dict[str, Any]
    candidates: list[ModelCandidate]
    preprocessing_recommendations: list[str]
    primary_metric: str
    reasoning: str


# ---------------------------------------------------------------------------
# Deterministic modeling profile + source-frame resolver
# ---------------------------------------------------------------------------
HIGH_CARDINALITY_THRESHOLD = 50
_DATETIME_NAME_RE = re.compile(r"(date|time|timestamp|datetime)", re.I)


def _size_bucket(n_rows: int) -> Literal["small", "medium", "large"]:
    if n_rows < 1_000:
        return "small"
    if n_rows < 100_000:
        return "medium"
    return "large"


def _looks_like_datetime(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not _DATETIME_NAME_RE.search(str(name)):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def build_modeling_profile(df: pd.DataFrame, understanding_report: dict) -> dict:
    """Pure: derive a JSON-safe modeling profile from the frame + Module 1 report."""
    problem_type = understanding_report.get("problem_type", "unclear")
    target = understanding_report.get("target_candidate")
    has_target = target in df.columns

    feature_cols = [c for c in df.columns if not (has_target and c == target)]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_pct = (
        round(float(df.isna().sum().sum()) / total_cells * 100, 2) if total_cells else 0.0
    )

    n_classes = class_balance = minority = None
    if problem_type == "classification" and has_target:
        vc = df[target].value_counts(normalize=True, dropna=True)
        n_classes = int(df[target].nunique(dropna=True))
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}
        minority = min(class_balance.values()) if class_balance else None

    target_kind = None
    if has_target:
        target_kind = "numeric" if pd.api.types.is_numeric_dtype(df[target]) else "categorical"

    return {
        "problem_type": problem_type,
        "target_column": target if has_target else None,
        "target_dtype": str(df[target].dtype) if has_target else None,
        "target_kind": target_kind,
        "n_rows": int(n_rows),
        "n_features": len(feature_cols),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
        "n_classes": n_classes,
        "class_balance": class_balance,
        "minority_class_fraction": minority,
        "overall_missing_pct": missing_pct,
        "n_high_cardinality_categoricals": sum(
            1
            for c in categorical_features
            if df[c].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
            and not _looks_like_datetime(c, df[c])
        ),
        "size_bucket": _size_bucket(int(n_rows)),
        "has_datetime_column": any(_looks_like_datetime(c, df[c]) for c in df.columns),
    }


def _resolve_source_df(dataset_id: str) -> tuple[pd.DataFrame, str, bool]:
    """Return ``(df, source_dataset_id, used_cleaned)``.

    Prefer the Module 2 cleaned frame when its id is recorded and its file is on
    disk; otherwise fall back to the original. Propagates ``FileNotFoundError``
    when the original is also absent.
    """
    cleaning = get_cleaning_report(dataset_id)
    cleaned_id = (cleaning or {}).get("cleaned_dataset_id")
    if cleaned_id:
        try:
            path = get_dataset_path(cleaned_id)
            return pd.read_csv(path), cleaned_id, True
        except FileNotFoundError:
            pass
    return pd.read_csv(get_dataset_path(dataset_id)), dataset_id, False


# ---------------------------------------------------------------------------
# The single reasoning call
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You recommend candidate models for a tabular machine-learning problem. "
    "You only choose model names from the catalog you are given, and you only "
    "write prose. You never invent or change hyperparameters, never produce "
    "metric numbers, and never claim any model has been trained or evaluated."
)


def _user_prompt(profile: dict, problem_type: str) -> str:
    entries = MODEL_CATALOG[problem_type]
    catalog_lines = "\n".join(f"- {e['name']}: {e['blurb']}" for e in entries)
    metrics = CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
    return (
        f"Problem type: {problem_type}\n\n"
        f"Modeling profile (computed deterministically - treat as ground truth):\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f"Candidate model catalog (choose 3-4 by exact name):\n{catalog_lines}\n\n"
        f"Allowed primary metrics: {', '.join(metrics)}\n\n"
        "Call submit_recommendation with 3-4 catalog names, a specific rationale "
        "for each given this profile, the preprocessing the data needs, one "
        "primary metric from the allowed list, and your overall reasoning."
    )


def recommend_models(
    profile: dict,
    problem_type: Literal["classification", "regression"],
    client: Any | None = None,
) -> RecommendationReport:
    """ONE forced-tool reasoning call, then re-derive every factual field.

    The LLM only picks catalog names and writes prose. ``library`` and
    ``hyperparameters`` are copied from the catalog verbatim; ``primary_metric``
    is validated against the allowed vocabulary for ``problem_type``.
    """
    names = catalog_names(problem_type)
    default = RecommendationLLMInput(
        candidates=[
            CandidateChoice(name=n, rationale="Strong default baseline for this profile.")
            for n in names[:3]
        ],
        preprocessing_recommendations=[
            "Impute any residual missing values.",
            "Scale numeric features for the linear model.",
            "Encode categorical features (one-hot for low cardinality, ordinal/target for high).",
        ],
        primary_metric=DEFAULT_METRIC[problem_type],
        reasoning="Deterministic fallback - the reasoning call did not return a usable response.",
    )

    llm = narrate(
        client,
        system=_SYSTEM,
        user=_user_prompt(profile, problem_type),
        schema=RecommendationLLMInput,
        tool_name="submit_recommendation",
        tool_description=(
            "Return 3-4 candidate model names from the catalog, a rationale for "
            "each, the preprocessing the data needs, a primary metric, and overall reasoning."
        ),
        default=default,
    )

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for c in llm.candidates:
        if c.name in names and c.name not in seen:
            chosen.append((c.name, c.rationale))
            seen.add(c.name)
    for n in names:
        if len(chosen) >= 3:
            break
        if n not in seen:
            chosen.append(
                (n, "Included as a strong default baseline for this problem type and dataset size.")
            )
            seen.add(n)
    chosen = chosen[:4]

    allowed_metrics = (
        CLASSIFICATION_METRICS if problem_type == "classification" else REGRESSION_METRICS
    )
    metric = (
        llm.primary_metric
        if llm.primary_metric in allowed_metrics
        else DEFAULT_METRIC[problem_type]
    )

    candidates = [
        ModelCandidate(
            name=name,
            library=get_catalog_entry(problem_type, name)["library"],
            rationale=rationale,
            hyperparameters=dict(get_catalog_entry(problem_type, name)["hyperparameters"]),
        )
        for name, rationale in chosen
    ]

    return RecommendationReport(
        problem_type=problem_type,
        modeling_profile=profile,
        candidates=candidates,
        preprocessing_recommendations=list(llm.preprocessing_recommendations),
        primary_metric=metric,
        reasoning=llm.reasoning,
    )
