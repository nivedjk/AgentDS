"""Training & AutoML (Module 5) - fully deterministic training, with one
optional, best-effort LLM narration call at the end.

Everything but that narration is pure/mechanical: given a cleaned frame, a
target and a list of catalog model names, build a preprocessing + estimator
pipeline, run a fixed 5-fold cross-validation, score a held-out test split,
and rank candidates by a single primary metric. The narration call (provider
selected by AGENTDS_LLM_PROVIDER, same as every other module) only ever
writes prose over the already-computed leaderboard — it's skipped entirely
if no client is passed, and its failure never fails the run (see
``run_training``'s ``if client is not None`` guard below).

Determinism contract: ``random_state=42`` everywhere it is accepted, ``n_jobs=1``
throughout, and every metric float is ``round(v, METRIC_ROUND)`` so re-running
on the same inputs yields byte-identical reports.
"""

from __future__ import annotations

import time
import warnings
from importlib import import_module
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.agents._llm import narrate
from app.agents.recommendation import MODEL_CATALOG
from app.storage.dataset_store import model_dir

# ---------------------------------------------------------------------------
# Fixed knobs - the deterministic contract.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
MAX_ONEHOT_CARDINALITY = 20
METRIC_ROUND = 6

HIGHER_IS_BETTER: set[str] = {
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "r2",
}
LOWER_IS_BETTER: set[str] = {"mae", "rmse"}

# Default AutoML shortlist per problem type (all catalog names).
DEFAULT_CANDIDATES: dict[str, list[str]] = {
    "classification": [
        "logistic_regression",
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
        "xgboost_classifier",
        "lightgbm_classifier",
    ],
    "regression": [
        "ridge",
        "random_forest_regressor",
        "hist_gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    ],
}


def _all_catalog_entries() -> list[dict[str, Any]]:
    return [entry for entries in MODEL_CATALOG.values() for entry in entries]


def estimator_from_catalog(name: str) -> Any:
    """Return a fresh, unfitted sklearn-compatible estimator for a catalog name.

    Searches both the ``classification`` and ``regression`` catalog lists for the
    entry whose ``name`` matches. Unknown name -> ``ValueError``. Optional
    boosters (xgboost / lightgbm) are imported lazily and a missing wheel is
    re-raised as a friendly ``ImportError``. Hyperparameters are copied from the
    catalog verbatim - the catalog is the contract.
    """
    entry = next((e for e in _all_catalog_entries() if e["name"] == name), None)
    if entry is None:
        known = sorted(e["name"] for e in _all_catalog_entries())
        raise ValueError(f"Unknown model {name!r}; known: {known}")

    if entry["library"] in {"xgboost", "lightgbm"}:
        try:
            import_module(entry["library"])
        except ImportError as exc:  # pragma: no cover - exercised only w/o wheel
            raise ImportError(
                f"Candidate {name!r} needs the optional {entry['library']!r} package"
            ) from exc

    module, _, cls = entry["estimator"].rpartition(".")
    return getattr(import_module(module), cls)(**entry["hyperparameters"])


# ---------------------------------------------------------------------------
# Task 3: column-role classification + preprocessing
# ---------------------------------------------------------------------------
class ColumnRoles(BaseModel):
    """Which feature columns feed the model, and which were dropped and why.

    Every list preserves the DataFrame's column order. The target column never
    appears in any list.
    """

    numeric: list[str] = []
    categorical: list[str] = []
    dropped_high_cardinality: list[str] = []
    dropped_datetime: list[str] = []


def _looks_like_datetime(series: pd.Series) -> bool:
    """True for real datetime dtype, or an object column that is >=95% parseable."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        with warnings.catch_warnings():
            # Mixed/absent date formats here are expected - we only want the hit rate.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(non_null, errors="coerce")
        return bool(parsed.notna().mean() >= 0.95)
    return False


def classify_columns(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str] | None = None,
    *,
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY,
) -> ColumnRoles:
    """Bucket each feature column into numeric / categorical / dropped.

    ``feature_columns`` defaults to every column except ``target``; ``target`` is
    always excluded even if passed in explicitly. Deterministic and pure.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target]
    else:
        feature_columns = [c for c in feature_columns if c != target]

    numeric: list[str] = []
    categorical: list[str] = []
    dropped_high_cardinality: list[str] = []
    dropped_datetime: list[str] = []

    for col in feature_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
        elif _looks_like_datetime(series):
            dropped_datetime.append(col)
        elif series.nunique(dropna=True) <= max_onehot_cardinality:
            categorical.append(col)
        else:
            dropped_high_cardinality.append(col)

    return ColumnRoles(
        numeric=numeric,
        categorical=categorical,
        dropped_high_cardinality=dropped_high_cardinality,
        dropped_datetime=dropped_datetime,
    )


def build_preprocessor(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str] | None = None,
    *,
    max_onehot_cardinality: int = MAX_ONEHOT_CARDINALITY,
) -> tuple[ColumnTransformer, ColumnRoles]:
    """Build a deterministic ``ColumnTransformer`` plus the ``ColumnRoles`` used.

    Numeric: median-impute then standard-scale. Categorical: most-frequent-impute
    then one-hot (``handle_unknown="ignore"``). A transformer whose column list is
    empty is omitted. No randomness anywhere.
    """
    roles = classify_columns(
        df, target, feature_columns, max_onehot_cardinality=max_onehot_cardinality
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if roles.numeric:
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("num", num_pipe, roles.numeric))
    if roles.categorical:
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, roles.categorical))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, roles


# ---------------------------------------------------------------------------
# Task 4: fixed-fold cross-validation + held-out test metrics
# ---------------------------------------------------------------------------
def _make_cv(problem_type: str, n_splits: int, random_state: int):
    """Deterministic K-fold splitter: stratified for classification, plain KFold otherwise."""
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _round(value: float) -> float:
    return round(float(value), METRIC_ROUND)


def evaluate_candidate(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Any,
    y_test: Any,
    problem_type: Literal["classification", "regression"],
    *,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Cross-validate ``pipeline`` on the train split, then score it on the held-out test split.

    Positional argument order mirrors ``sklearn.model_selection.train_test_split``'s
    return: ``(X_train, X_test, y_train, y_test)``.

    Returns ``{"cv": {metric: {"mean", "std"}}, "test": {metric: value}}`` with every
    float ``round(v, METRIC_ROUND)``.

    Classification metrics: ``accuracy``, ``precision_macro``, ``recall_macro``,
    ``f1_macro``; plus ``roc_auc`` only when ``y_train`` has exactly two classes
    and the estimator exposes ``predict_proba`` / ``decision_function``.
    Regression metrics: ``r2``, ``mae``, ``rmse``.

    Side effect: ``pipeline`` is fitted **in place** on ``(X_train, y_train)`` so the
    caller can persist that exact fitted object. Fully deterministic - the only
    seed used is ``random_state`` and ``n_jobs=1`` throughout.
    """
    if problem_type == "classification":
        cv_result, test_result = _evaluate_classification(
            pipeline, X_train, y_train, X_test, y_test, cv_splits, random_state
        )
    else:
        cv_result, test_result = _evaluate_regression(
            pipeline, X_train, y_train, X_test, y_test, cv_splits, random_state
        )
    return {"cv": cv_result, "test": test_result}


def _summarise_cv(cv_results: dict, keys, sign: dict | None = None) -> dict:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        arr = np.asarray(cv_results[f"test_{key}"], dtype=float)
        if sign is not None:
            arr = arr * sign[key]
        out[key] = {"mean": _round(np.mean(arr)), "std": _round(np.std(arr))}
    return out


def _evaluate_classification(
    pipeline, X_train, y_train, X_test, y_test, cv_splits, random_state
):
    n_classes = len(np.unique(np.asarray(y_train)))
    has_scores = hasattr(pipeline, "predict_proba") or hasattr(pipeline, "decision_function")
    use_roc_auc = n_classes == 2 and has_scores

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    if use_roc_auc:
        scoring["roc_auc"] = "roc_auc"

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=_make_cv("classification", cv_splits, random_state),
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    cv_out = _summarise_cv(cv_results, scoring.keys())

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_out = {
        "accuracy": _round(accuracy_score(y_test, y_pred)),
        "precision_macro": _round(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": _round(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": _round(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if use_roc_auc:
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)
        test_out["roc_auc"] = _round(roc_auc_score(y_test, y_score))

    return cv_out, test_out


def _evaluate_regression(
    pipeline, X_train, y_train, X_test, y_test, cv_splits, random_state
):
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    sign = {"r2": 1.0, "mae": -1.0, "rmse": -1.0}

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=_make_cv("regression", cv_splits, random_state),
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    cv_out = _summarise_cv(cv_results, scoring.keys(), sign=sign)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_out = {
        "r2": _round(r2_score(y_test, y_pred)),
        "mae": _round(mean_absolute_error(y_test, y_pred)),
        "rmse": _round(root_mean_squared_error(y_test, y_pred)),
    }
    return cv_out, test_out


# ---------------------------------------------------------------------------
# Task 5: report models, orchestration, and a single guarded narration call
# ---------------------------------------------------------------------------
_DEFAULT_PRIMARY_METRIC = {"classification": "f1_macro", "regression": "r2"}
_JSON_SAFE = (str, int, float, bool, type(None))

_NARRATION_SYSTEM = (
    "You write a one- to two-sentence plain-English summary of an AutoML "
    "leaderboard. Name the winning model and the metric it won on, and note "
    "any candidate that failed to fit. No markdown, no hype, no numbers you "
    "were not given."
)
_NARRATION_TOOL_DESCRIPTION = "Return the leaderboard narrative as a single short paragraph."


class MetricStat(BaseModel):
    """Cross-validation mean/std for one metric (already rounded upstream)."""

    mean: float
    std: float


class CandidateResult(BaseModel):
    """Outcome for one catalog candidate - metrics on success, ``error`` on failure."""

    name: str
    estimator_class: str
    params: dict[str, Any] = {}
    cv_metrics: dict[str, MetricStat] = {}
    test_metrics: dict[str, float] = {}
    fit_seconds: float = 0.0
    model_path: str | None = None
    error: str | None = None


class LeaderboardRow(BaseModel):
    """One ranked row: successful candidates first (best->worst), failures last."""

    rank: int
    name: str
    primary_metric: str
    cv_mean: float | None = None
    cv_std: float | None = None
    test_score: float | None = None
    failed: bool = False


class TrainingReport(BaseModel):
    """The complete, deterministic result of a training run.

    Every numeric field comes from real scikit-learn output; ``narrative`` is the
    only field a language model may touch and it is never on the critical path.
    """

    dataset_id: str
    problem_type: Literal["classification", "regression"]
    target: str
    n_rows: int
    feature_columns: list[str]
    column_roles: ColumnRoles
    train_rows: int
    test_rows: int
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    cv_splits: int = CV_SPLITS
    stratified: bool = True
    primary_metric: str
    candidates: list[CandidateResult]
    best_model: str | None = None
    best_score: float | None = None
    leaderboard: list[LeaderboardRow]
    narrative: str = ""
    warnings: list[str] = []


class TrainingError(Exception):
    """A training run could not produce any usable model.

    Raised for a bad problem type, a missing target, too few usable rows, or
    every candidate failing to fit. Never an ``HTTPException`` - the router maps
    it to a response.
    """


class _LeaderboardNarration(BaseModel):
    narrative: str


def _resolve_candidate_names(
    recommendation_report: dict | None, problem_type: str
) -> list[str]:
    """Candidate shortlist: Module 4's ``recommendation`` sidecar ``candidates``
    list when present and non-empty, else the fixed default set."""
    if recommendation_report:
        m4 = recommendation_report.get("candidates")
        if isinstance(m4, list) and m4:
            names = [c["name"] for c in m4 if isinstance(c, dict) and c.get("name")]
            if names:
                return names
    return list(DEFAULT_CANDIDATES[problem_type])


def _resolve_primary_metric(
    recommendation_report: dict | None, problem_type: str
) -> str:
    default = _DEFAULT_PRIMARY_METRIC[problem_type]
    if recommendation_report:
        metric = recommendation_report.get("primary_metric")
        if metric in HIGHER_IS_BETTER or metric in LOWER_IS_BETTER:
            return metric
    return default


def _json_safe_params(estimator: Any) -> dict[str, Any]:
    """Shallow, JSON-serialisable view of an estimator's own hyperparameters."""
    return {
        key: value
        for key, value in estimator.get_params(deep=False).items()
        if isinstance(value, _JSON_SAFE)
    }


def _build_leaderboard(
    candidates: list[CandidateResult], primary_metric: str, higher_is_better: bool
) -> list[LeaderboardRow]:
    succeeded = [c for c in candidates if c.error is None]
    failed = [c for c in candidates if c.error is not None]

    def score(candidate: CandidateResult) -> float:
        value = candidate.test_metrics.get(primary_metric)
        if value is not None:
            return value
        return float("-inf") if higher_is_better else float("inf")

    succeeded.sort(key=score, reverse=higher_is_better)

    rows: list[LeaderboardRow] = []
    for rank, candidate in enumerate(succeeded + failed, start=1):
        cv_stat = candidate.cv_metrics.get(primary_metric)
        rows.append(
            LeaderboardRow(
                rank=rank,
                name=candidate.name,
                primary_metric=primary_metric,
                cv_mean=cv_stat.mean if cv_stat is not None else None,
                cv_std=cv_stat.std if cv_stat is not None else None,
                test_score=candidate.test_metrics.get(primary_metric),
                failed=candidate.error is not None,
            )
        )
    return rows


def _leaderboard_summary(
    problem_type: str,
    primary_metric: str,
    best_model: str | None,
    best_score: float | None,
    leaderboard: list[LeaderboardRow],
) -> str:
    lines = [
        f"Problem type: {problem_type}. Primary metric: {primary_metric}.",
        f"Best model: {best_model} (test {primary_metric}={best_score}).",
        "Leaderboard:",
    ]
    for row in leaderboard:
        if row.failed:
            lines.append(f"  {row.rank}. {row.name} - failed to fit")
        else:
            lines.append(
                f"  {row.rank}. {row.name} - cv_mean={row.cv_mean} test={row.test_score}"
            )
    return "\n".join(lines)


def run_training(
    df: pd.DataFrame,
    dataset_id: str,
    understanding_report: dict,
    recommendation_report: dict | None = None,
    client: Any | None = None,
) -> TrainingReport:
    """Fit every shortlisted catalog model, rank them, and persist each pipeline.

    Deterministic end to end: fixed seed, fixed folds, ``n_jobs=1``, metric
    floats rounded upstream. The only optional step is a single ``narrate`` call
    for prose, wrapped so any failure leaves ``narrative == ""``.
    """
    target = understanding_report.get("target_candidate")
    problem_type = understanding_report.get("problem_type")

    if problem_type not in {"classification", "regression"}:
        raise TrainingError(
            f"problem_type must be 'classification' or 'regression', got {problem_type!r}"
        )
    if not target or target not in df.columns:
        raise TrainingError(f"target {target!r} is not a column in the dataset")

    df = df.dropna(subset=[target])
    if len(df) < 10:
        raise TrainingError(
            f"need at least 10 usable rows after dropping a missing target; got {len(df)}"
        )

    feature_columns = [c for c in df.columns if c != target]
    warn_msgs: list[str] = []

    candidate_names = _resolve_candidate_names(recommendation_report, problem_type)
    primary_metric = _resolve_primary_metric(recommendation_report, problem_type)
    valid_names = {entry["name"] for entry in MODEL_CATALOG[problem_type]}
    for name in candidate_names:
        if name not in valid_names:
            warn_msgs.append(
                f"Unknown candidate {name!r}: no catalog entry for {problem_type}; "
                "recorded as a failed candidate"
            )

    base_pre, roles = build_preprocessor(df, target, feature_columns)

    X, y = df[feature_columns], df[target]
    stratified = True
    if problem_type == "classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            stratified = False
            warn_msgs.append(
                "Stratified split failed (a class with < 2 members); "
                "used a non-stratified split"
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    candidates: list[CandidateResult] = []
    for name in candidate_names:
        try:
            if name not in valid_names:
                raise ValueError(
                    f"Unknown model {name!r}; not in the {problem_type} catalog"
                )
            estimator = estimator_from_catalog(name)
            pipe = Pipeline([("pre", clone(base_pre)), ("model", estimator)])
            started = time.perf_counter()
            metrics = evaluate_candidate(
                pipe, X_train, X_test, y_train, y_test, problem_type
            )
            elapsed = time.perf_counter() - started
            path = model_dir(dataset_id) / f"{name}.joblib"
            joblib.dump(pipe, path)
            candidates.append(
                CandidateResult(
                    name=name,
                    estimator_class=estimator.__class__.__name__,
                    params=_json_safe_params(estimator),
                    cv_metrics={
                        k: MetricStat(**v) for k, v in metrics["cv"].items()
                    },
                    test_metrics=metrics["test"],
                    fit_seconds=round(elapsed, 3),
                    model_path=str(path),
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort the run
            candidates.append(
                CandidateResult(
                    name=name,
                    estimator_class="",
                    params={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    if all(c.error is not None for c in candidates):
        raise TrainingError(
            f"All {len(candidates)} candidates failed to fit: "
            + "; ".join(f"{c.name}: {c.error}" for c in candidates)
        )

    higher_is_better = primary_metric in HIGHER_IS_BETTER
    scorable = [
        c
        for c in candidates
        if c.error is None and primary_metric in c.test_metrics
    ]
    best_model: str | None = None
    best_score: float | None = None
    if scorable:
        pick = max if higher_is_better else min
        best = pick(scorable, key=lambda c: c.test_metrics[primary_metric])
        best_model = best.name
        best_score = best.test_metrics[primary_metric]

    leaderboard = _build_leaderboard(candidates, primary_metric, higher_is_better)

    narrative = ""
    if client is not None:
        try:
            result = narrate(
                client,
                system=_NARRATION_SYSTEM,
                user=_leaderboard_summary(
                    problem_type, primary_metric, best_model, best_score, leaderboard
                ),
                schema=_LeaderboardNarration,
                tool_name="submit_narrative",
                tool_description=_NARRATION_TOOL_DESCRIPTION,
                default=_LeaderboardNarration(narrative=""),
            )
            narrative = result.narrative
        except Exception:  # noqa: BLE001 - narration is never fatal
            narrative = ""

    return TrainingReport(
        dataset_id=dataset_id,
        problem_type=problem_type,
        target=target,
        n_rows=len(df),
        feature_columns=feature_columns,
        column_roles=roles,
        train_rows=len(X_train),
        test_rows=len(X_test),
        stratified=stratified,
        primary_metric=primary_metric,
        candidates=candidates,
        best_model=best_model,
        best_score=best_score,
        leaderboard=leaderboard,
        narrative=narrative,
        warnings=warn_msgs,
    )
