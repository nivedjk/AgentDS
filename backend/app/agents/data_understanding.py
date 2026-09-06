"""Data Understanding Agent.

Two ways to investigate a dataset, sharing the same deterministic building
blocks:

- ``quick_stats``: a fast, rule-based, LLM-free pass. No API key needed.
  Used by the ``/quick-stats`` endpoint and as an offline fallback.
- ``DataUnderstandingAgent.run``: an LLM drives an investigative tool-use
  loop over the same deterministic functions (exposed to it as tools), then
  submits a narrative report via a terminal ``submit_report`` tool call.
  Provider is ``AGENTDS_LLM_PROVIDER`` (default ``"ollama"``, needing a
  local Ollama server; ``"anthropic"`` needs ``ANTHROPIC_API_KEY``) — see
  ``app.agents._llm_client``.

The deterministic functions (``list_columns``, ``inspect_column``,
``sample_rows``, ``check_duplicates``, ``get_correlations``) are pure and
pandas-only — no network calls — so they're unit-testable in isolation and
reused as the agent's tool implementations.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from app.agents._llm_client import get_client

MODEL = "claude-sonnet-5"
MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = (
    "You are a data understanding agent. You've been given an overview of a "
    "dataset's columns. Investigate further using the available tools — "
    "inspect columns that look unusual, check duplicates, check correlations "
    "— but only as much as the dataset actually warrants; a small clean "
    "dataset might need 2-3 tool calls, a messy one more. As baseline "
    "hygiene, check duplicates and correlations before concluding, unless "
    "you have good reason to skip them. Ground every specific number in a "
    "tool result — never state a statistic you haven't actually retrieved. "
    "When confident, call submit_report with your conclusions."
)


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class ColumnOverview(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    n_unique: int


class DuplicatesReport(BaseModel):
    n_duplicate_rows: int
    percent: float
    example_rows: list[dict[str, Any]]


class CorrelationPair(BaseModel):
    column_a: str
    column_b: str
    correlation: float


class SubmitReportInput(BaseModel):
    """Schema for the terminal ``submit_report`` tool call."""

    target_candidate: str | None = None
    problem_type: Literal["classification", "regression", "unclear"]
    reasoning: str
    narrative: str
    key_findings: list[str]


class DataUnderstandingReport(BaseModel):
    dataset_id: str
    n_rows: int
    n_columns: int
    columns: list[ColumnOverview]
    duplicates: DuplicatesReport | None = None
    correlations: list[CorrelationPair] | None = None
    target_candidate: str | None = None
    problem_type: Literal["classification", "regression", "unclear"]
    reasoning: str
    narrative: str
    key_findings: list[str]


# ---------------------------------------------------------------------------
# Deterministic tool implementations (pure, pandas-only)
# ---------------------------------------------------------------------------


def _sanitize(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain, JSON-safe Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {str(col): _sanitize(val) for col, val in row.items()}


def list_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Cheap overview of every column: name, dtype, missing count/%, n_unique."""
    n_rows = len(df)
    overview = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        overview.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return overview


def inspect_column(df: pd.DataFrame, column_name: str) -> dict[str, Any]:
    """Full stats for one column.

    Numeric columns: count/mean/median/std/min/max/quartiles/skew.
    Categorical columns: cardinality + top 10 value counts with frequencies.
    Missing count/% is included either way.

    Raises:
        ValueError: if ``column_name`` isn't in the dataset.
    """
    if column_name not in df.columns:
        raise ValueError(
            f"Column {column_name!r} not found. Available columns: {list(df.columns)}"
        )

    series = df[column_name]
    n_rows = len(df)
    missing = int(series.isna().sum())
    result: dict[str, Any] = {
        "name": column_name,
        "dtype": str(series.dtype),
        "missing_count": missing,
        "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0.0,
    }

    non_null = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        result["is_numeric"] = True
        result["count"] = int(non_null.count())
        if not non_null.empty:
            result.update(
                {
                    "mean": _sanitize(non_null.mean()),
                    "median": _sanitize(non_null.median()),
                    "std": _sanitize(non_null.std()) if len(non_null) > 1 else 0.0,
                    "min": _sanitize(non_null.min()),
                    "max": _sanitize(non_null.max()),
                    "q25": _sanitize(non_null.quantile(0.25)),
                    "q75": _sanitize(non_null.quantile(0.75)),
                    "skew": _sanitize(non_null.skew()) if len(non_null) > 2 else 0.0,
                }
            )
    else:
        value_counts = non_null.value_counts().head(10)
        n_non_null = len(non_null)
        result["is_numeric"] = False
        result["cardinality"] = int(non_null.nunique())
        result["top_values"] = [
            {
                "value": _sanitize(val),
                "count": int(cnt),
                "pct": round(cnt / n_non_null * 100, 2) if n_non_null else 0.0,
            }
            for val, cnt in value_counts.items()
        ]

    return result


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """``n`` random rows as a list of dicts (all columns)."""
    if len(df) == 0:
        return []
    n = max(1, min(n, len(df)))
    sample = df.sample(n=n)
    return [_row_to_dict(row) for _, row in sample.iterrows()]


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Fully duplicated rows: count, percent, and up to 3 example rows."""
    n_rows = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    examples = df[dup_mask].head(3)
    return {
        "n_duplicate_rows": n_dup,
        "percent": round(n_dup / n_rows * 100, 2) if n_rows else 0.0,
        "example_rows": [_row_to_dict(row) for _, row in examples.iterrows()],
    }


def get_correlations(df: pd.DataFrame, threshold: float = 0.85) -> list[dict[str, Any]]:
    """Numeric column pairs with |correlation| >= threshold, sorted descending."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr = numeric_df.corr(numeric_only=True)
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.isna(value) or abs(value) < threshold:
                continue
            pairs.append(
                {
                    "column_a": str(cols[i]),
                    "column_b": str(cols[j]),
                    "correlation": round(float(value), 4),
                }
            )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Deterministic report (no LLM) — fast heuristic pass
# ---------------------------------------------------------------------------


def _guess_target_and_problem_type(
    df: pd.DataFrame,
) -> tuple[str | None, Literal["classification", "regression", "unclear"]]:
    if len(df.columns) == 0:
        return None, "unclear"

    name_candidates = [
        c for c in df.columns if str(c).lower() in {"target", "label", "class", "y", "outcome"}
    ]
    target = name_candidates[0] if name_candidates else df.columns[-1]
    target = str(target)
    series = df[target]

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        problem_type: Literal["classification", "regression", "unclear"] = (
            "classification" if n_unique <= 10 else "regression"
        )
    else:
        problem_type = "classification"

    return target, problem_type


def quick_stats(df: pd.DataFrame, dataset_id: str) -> DataUnderstandingReport:
    """Deterministic, LLM-free report — the fast heuristic pass."""
    overview_raw = list_columns(df)
    columns = [ColumnOverview(**c) for c in overview_raw]

    duplicates_raw = check_duplicates(df)
    duplicates = DuplicatesReport(**duplicates_raw)

    correlations_raw = get_correlations(df)
    correlations = [CorrelationPair(**c) for c in correlations_raw]

    target, problem_type = _guess_target_and_problem_type(df)

    key_findings = []
    if duplicates.n_duplicate_rows:
        key_findings.append(
            f"{duplicates.n_duplicate_rows} duplicate rows ({duplicates.percent}%)."
        )
    high_missing = [c for c in overview_raw if c["missing_pct"] > 20]
    if high_missing:
        names = ", ".join(c["name"] for c in high_missing)
        key_findings.append(f"High missingness (>20%) in: {names}.")
    if correlations:
        key_findings.append(
            f"{len(correlations)} highly correlated column pair(s) (|r| >= 0.85)."
        )
    if not key_findings:
        key_findings.append("No major data quality issues detected by the quick heuristic pass.")

    narrative = (
        f"This dataset has {len(df)} rows and {len(df.columns)} columns. "
        f"The likely target column is {target!r}, suggesting a "
        f"'{problem_type}' problem. {duplicates.n_duplicate_rows} duplicate "
        f"row(s) and {len(correlations)} highly correlated column pair(s) "
        "were found. This is a fast, rule-based summary — run the full "
        "analysis for a deeper investigation."
    )

    return DataUnderstandingReport(
        dataset_id=dataset_id,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
        duplicates=duplicates,
        correlations=correlations,
        target_candidate=target,
        problem_type=problem_type,
        reasoning=(
            f"Heuristic: picked {target!r} as the likely-named/last column; "
            f"classified as {problem_type!r} from its dtype and cardinality."
        ),
        narrative=narrative,
        key_findings=key_findings,
    )


# ---------------------------------------------------------------------------
# Agentic loop (Claude tool use)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "inspect_column",
        "description": (
            "Get full statistics for one column. Numeric columns return "
            "count/mean/median/std/min/max/quartiles/skew. Categorical "
            "columns return cardinality and the top 10 value counts with "
            "frequencies. Always includes missing count/percent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "Exact column name to inspect.",
                }
            },
            "required": ["column_name"],
        },
    },
    {
        "name": "sample_rows",
        "description": "Return n random rows from the dataset as a list of dicts (all columns).",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of rows to sample.",
                    "default": 5,
                }
            },
        },
    },
    {
        "name": "check_duplicates",
        "description": (
            "Check for fully duplicated rows in the dataset. Returns count, "
            "percent, and up to 3 example duplicate rows."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_correlations",
        "description": (
            "Find numeric column pairs with absolute correlation at or "
            "above a threshold, sorted descending by magnitude."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Minimum absolute correlation to report.",
                    "default": 0.85,
                }
            },
        },
    },
    {
        "name": "submit_report",
        "description": (
            "Submit your final conclusions about the dataset. Calling this "
            "ends the investigation."
        ),
        "input_schema": SubmitReportInput.model_json_schema(),
    },
]


class DataUnderstandingAgent:
    """Drives an investigative Claude tool-use loop over one dataset."""

    def __init__(self, client: Any | None = None):
        self.client = client if client is not None else get_client()

    def run(self, df: pd.DataFrame, dataset_id: str) -> DataUnderstandingReport:
        overview_raw = list_columns(df)

        first_user_content = (
            "Here is the column overview for this dataset "
            f"({len(df)} rows, {len(df.columns)} columns):\n\n"
            f"{json.dumps(overview_raw, indent=2)}"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": first_user_content}]

        duplicates_result: dict[str, Any] | None = None
        correlations_result: list[dict[str, Any]] | None = None
        submit_input: dict[str, Any] | None = None
        tool_call_count = 0
        forced_once = False

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                if forced_once:
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please call submit_report now with your conclusions.",
                    }
                )
                forced_once = True
                continue

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                if block.name == "submit_report":
                    try:
                        validated = SubmitReportInput.model_validate(block.input)
                    except ValidationError as exc:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Invalid report: {exc}",
                                "is_error": True,
                            }
                        )
                        continue
                    submit_input = validated.model_dump()
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Report received.",
                        }
                    )
                    continue

                tool_call_count += 1
                try:
                    result = self._execute_tool(df, block.name, block.input)
                    if block.name == "check_duplicates":
                        duplicates_result = result
                    elif block.name == "get_correlations":
                        correlations_result = result
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - fed back to Claude, not raised
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            if submit_input is not None:
                break

            if tool_call_count >= MAX_TOOL_CALLS:
                if forced_once:
                    break
                forced_once = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"You've reached the maximum number of investigative "
                            f"tool calls ({MAX_TOOL_CALLS}). Call submit_report now "
                            "with your best current conclusions."
                        ),
                    }
                )

        if submit_input is None:
            submit_input = {
                "target_candidate": None,
                "problem_type": "unclear",
                "reasoning": "Agent did not submit a report within the tool-call budget.",
                "narrative": "Investigation was inconclusive within the available tool-call budget.",
                "key_findings": [],
            }

        return DataUnderstandingReport(
            dataset_id=dataset_id,
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=[ColumnOverview(**c) for c in overview_raw],
            duplicates=DuplicatesReport(**duplicates_result) if duplicates_result else None,
            correlations=(
                [CorrelationPair(**c) for c in correlations_result]
                if correlations_result is not None
                else None
            ),
            **submit_input,
        )

    @staticmethod
    def _execute_tool(df: pd.DataFrame, name: str, tool_input: dict[str, Any]) -> Any:
        if name == "inspect_column":
            return inspect_column(df, tool_input["column_name"])
        if name == "sample_rows":
            return sample_rows(df, tool_input.get("n", 5))
        if name == "check_duplicates":
            return check_duplicates(df)
        if name == "get_correlations":
            return get_correlations(df, tool_input.get("threshold", 0.85))
        raise ValueError(f"Unknown tool: {name}")
