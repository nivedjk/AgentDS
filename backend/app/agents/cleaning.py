"""Cleaning Agent (Module 2).

An LLM drives a tool-use loop of mutating cleaning actions over one dataset,
then submits its conclusions via a terminal ``submit_cleaning_report`` tool.
Provider is ``AGENTDS_LLM_PROVIDER`` (default ``"ollama"``, needing a local
Ollama server; ``"anthropic"`` needs ``ANTHROPIC_API_KEY``) — see
``app.agents._llm_client``.

Unlike Module 1, the action tools mutate a working DataFrame across calls.
The transformation functions (``apply_impute``, ``apply_drop_column``,
``apply_remove_duplicates``, ``apply_flag_outliers``, ``apply_encode_column``,
``apply_scale_column``) are pure — each takes a DataFrame and returns a new
DataFrame plus a structured result dict — so they are unit-testable in
isolation. ``CleaningAgent`` is the only stateful piece: it reassigns
``self.df`` after every successful action and appends an ordered
``CleaningStep`` to ``self.steps``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from app.agents._llm_client import get_client
from app.agents.data_understanding import _sanitize, inspect_column

MODEL = "claude-sonnet-5"
MAX_ACTION_CALLS = 25
MAX_TOTAL_TOOL_CALLS = 40


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class CleaningStep(BaseModel):
    order: int
    tool: str
    reason: str
    params: dict[str, Any]
    result: dict[str, Any]


class SubmitCleaningReportInput(BaseModel):
    """Schema for the terminal ``submit_cleaning_report`` tool call."""

    summary: list[str]
    remaining_issues: list[str]


class CleaningReport(BaseModel):
    dataset_id: str
    cleaned_dataset_id: str | None = None
    target_candidate: str | None = None
    problem_type: Literal["classification", "regression", "unclear"]
    initial_shape: list[int]
    final_shape: list[int]
    initial_missing_cells: int
    final_missing_cells: int
    steps: list[CleaningStep]
    summary: list[str]
    remaining_issues: list[str]


# ---------------------------------------------------------------------------
# Pure transformation functions
# ---------------------------------------------------------------------------


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(
            f"Column {column!r} not found. Current columns: {list(df.columns)}"
        )


def _guard_row_loss(df: pd.DataFrame, rows_after: int, operation: str) -> None:
    """Raise if ``operation`` would drop more than half of ``df``'s rows."""
    n = len(df)
    if n == 0:
        return
    removed = n - rows_after
    if removed > n * 0.5:
        raise ValueError(
            f"Refusing {operation}: it would remove {removed} of {n} rows "
            "(>50%). Choose a less aggressive strategy."
        )


def apply_impute(
    df: pd.DataFrame,
    column: str,
    strategy: Literal["mean", "median", "mode", "constant", "drop_rows"],
    constant_value: str | float | None = None,
) -> tuple[pd.DataFrame, dict]:
    _require_column(df, column)
    series = df[column]
    missing_before = int(series.isna().sum())

    if strategy == "drop_rows":
        rows_after = int(series.notna().sum())
        _guard_row_loss(
            df, rows_after, f"impute_column(column={column!r}, strategy='drop_rows')"
        )
        new_df = df[series.notna()].reset_index(drop=True)
        fill_value: Any = None
    else:
        if strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                f"Column {column!r} is not numeric; cannot impute with {strategy!r}."
            )
        if strategy == "mean":
            fill_value = series.mean()
        elif strategy == "median":
            fill_value = series.median()
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Column {column!r} has no non-null values to compute a mode."
                )
            fill_value = modes.iloc[0]
        elif strategy == "constant":
            if constant_value is None:
                raise ValueError("strategy='constant' requires constant_value.")
            fill_value = constant_value
        else:  # pragma: no cover - schema-constrained
            raise ValueError(f"Unknown strategy {strategy!r}")
        new_df = df.copy()
        new_df[column] = series.fillna(fill_value)

    return new_df, {
        "column": column,
        "strategy": strategy,
        "missing_before": missing_before,
        "missing_after": int(new_df[column].isna().sum()),
        "fill_value": _sanitize(fill_value),
    }


def apply_drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    _require_column(df, column)
    new_df = df.drop(columns=[column])
    return new_df, {
        "column": column,
        "dropped": True,
        "columns_remaining": list(new_df.columns),
    }


def apply_remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows_before = len(df)
    dup_mask = df.duplicated(keep="first")
    rows_after = rows_before - int(dup_mask.sum())
    _guard_row_loss(df, rows_after, "remove_duplicates")
    new_df = df[~dup_mask].reset_index(drop=True)
    return new_df, {
        "rows_before": rows_before,
        "rows_after": len(new_df),
        "rows_removed": rows_before - len(new_df),
    }


def apply_flag_outliers(
    df: pd.DataFrame,
    column: str,
    method: Literal["iqr", "zscore"],
    action: Literal["flag", "cap", "remove"],
) -> tuple[pd.DataFrame, dict]:
    _require_column(df, column)
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric; cannot detect outliers.")

    non_null = series.dropna()
    if method == "iqr":
        q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
    else:  # zscore
        mu, sigma = non_null.mean(), non_null.std()
        spread = sigma
        lower, upper = mu - 3 * sigma, mu + 3 * sigma

    if spread == 0 or pd.isna(spread):
        lower, upper = non_null.min(), non_null.max()

    outlier_mask = series.notna() & ((series < lower) | (series > upper))
    n_affected = int(outlier_mask.sum())
    result: dict[str, Any] = {
        "column": column,
        "method": method,
        "action": action,
        "n_affected": n_affected,
        "bounds": {"lower": _sanitize(lower), "upper": _sanitize(upper)},
    }

    new_df = df.copy()
    if action == "flag":
        new_col = f"{column}_outlier"
        new_df[new_col] = outlier_mask
        result["new_column"] = new_col
    elif action == "cap":
        new_df[column] = series.clip(lower=lower, upper=upper)
    elif action == "remove":
        rows_after = len(df) - n_affected
        # Structurally unreachable via outlier removal alone: IQR fences sit
        # outside the interquartile box (which is >50% of the data by
        # definition), and z-score removal is Chebyshev-bounded to ~11% of any
        # distribution beyond 3 SD. Kept for defense-in-depth, not because it
        # is expected to trip — hence no test exercises this path firing.
        _guard_row_loss(df, rows_after, "flag_outliers(action='remove')")
        new_df = df[~outlier_mask].reset_index(drop=True)
        result["rows_after"] = len(new_df)
    else:  # pragma: no cover - schema-constrained
        raise ValueError(f"Unknown action {action!r}")

    return new_df, result


def apply_encode_column(
    df: pd.DataFrame, column: str, method: Literal["one_hot", "label"]
) -> tuple[pd.DataFrame, dict]:
    _require_column(df, column)
    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column {column!r} is numeric; encoding is for categorical columns."
        )

    if method == "one_hot":
        before_cols = set(df.columns)
        encoded = pd.get_dummies(df, columns=[column], prefix=column)
        new_columns = [c for c in encoded.columns if c not in before_cols]
        encoded[new_columns] = encoded[new_columns].astype(int)
        return encoded, {
            "column": column,
            "method": method,
            "new_columns": new_columns,
            "original_dropped": True,
        }

    # label
    uniques = sorted(df[column].dropna().unique(), key=str)
    value_to_code = {value: code for code, value in enumerate(uniques)}
    new_df = df.copy()
    new_df[column] = df[column].map(value_to_code)
    return new_df, {
        "column": column,
        "method": method,
        "mapping": {str(value): code for value, code in value_to_code.items()},
    }


def apply_scale_column(
    df: pd.DataFrame, column: str, method: Literal["standard", "minmax", "robust"]
) -> tuple[pd.DataFrame, dict]:
    _require_column(df, column)
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column {column!r} is not numeric; cannot scale.")
    non_null = series.dropna()

    if method == "standard":
        mu, sigma = non_null.mean(), non_null.std()
        if sigma == 0 or pd.isna(sigma):
            raise ValueError(
                f"Column {column!r} has zero standard deviation; cannot standard-scale."
            )
        scaled = (series - mu) / sigma
        before = {"mean": _sanitize(mu), "std": _sanitize(sigma)}
    elif method == "minmax":
        lo, hi = non_null.min(), non_null.max()
        if hi == lo:
            raise ValueError(
                f"Column {column!r} has zero range (min == max); cannot minmax-scale."
            )
        scaled = (series - lo) / (hi - lo)
        before = {"min": _sanitize(lo), "max": _sanitize(hi)}
    else:  # robust
        median = non_null.median()
        iqr = non_null.quantile(0.75) - non_null.quantile(0.25)
        if iqr == 0 or pd.isna(iqr):
            raise ValueError(
                f"Column {column!r} has zero IQR; cannot robust-scale."
            )
        scaled = (series - median) / iqr
        before = {"mean": _sanitize(non_null.mean()), "std": _sanitize(non_null.std())}

    new_df = df.copy()
    new_df[column] = scaled
    after_non_null = new_df[column].dropna()
    if method == "minmax":
        after = {"min": _sanitize(after_non_null.min()), "max": _sanitize(after_non_null.max())}
    else:
        after = {"mean": _sanitize(after_non_null.mean()), "std": _sanitize(after_non_null.std())}

    return new_df, {"column": column, "method": method, "before": before, "after": after}


# ---------------------------------------------------------------------------
# Agentic loop (Claude tool use)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a data cleaning agent. You've been given a dataset and the "
    "Module 1 Data Understanding report describing its columns, missing "
    "values, likely target, problem type, and correlations — treat that as "
    "established context and do not re-derive it. Use the tools to clean the "
    "dataset for downstream modelling: impute or drop columns with missing "
    "data, remove duplicate rows, handle outliers, and encode or scale "
    "columns as needed. Every action tool requires a `reason` — give the "
    "specific reason for that choice at the moment you make it. Inspect a "
    "column first when the report doesn't tell you enough to decide. Don't "
    "over-clean: a tidy dataset may need only a few actions. Never drop the "
    "target column. When done, call submit_cleaning_report with a list "
    "summarising what you did and why it mattered, plus any issues you "
    "deliberately left unaddressed."
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "inspect_column",
        "description": (
            "Get full statistics for one column of the CURRENT working "
            "dataset. Numeric: count/mean/median/std/min/max/quartiles/skew. "
            "Categorical: cardinality + top 10 value counts. Always includes "
            "missing count/percent. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_name": {"type": "string", "description": "Exact column name."}
            },
            "required": ["column_name"],
        },
    },
    {
        "name": "impute_column",
        "description": (
            "Fill missing values in a column, or drop rows missing it. "
            "Returns before/after missing counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "strategy": {
                    "type": "string",
                    "enum": ["mean", "median", "mode", "constant", "drop_rows"],
                },
                "reason": {"type": "string", "description": "Why this column and strategy."},
                "constant_value": {
                    "type": ["string", "number", "null"],
                    "description": "Required when strategy is 'constant'.",
                },
            },
            "required": ["column", "strategy", "reason"],
        },
    },
    {
        "name": "drop_column",
        "description": "Remove a column entirely. The target column cannot be dropped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["column", "reason"],
        },
    },
    {
        "name": "remove_duplicates",
        "description": "Drop fully duplicated rows (keep first). Returns rows removed.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "flag_outliers",
        "description": (
            "Detect outliers in a numeric column by IQR or z-score, then "
            "flag (add <column>_outlier bool), cap (winsorize to bounds), or "
            "remove (drop the rows). Returns count affected and the bounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "method": {"type": "string", "enum": ["iqr", "zscore"]},
                "action": {"type": "string", "enum": ["flag", "cap", "remove"]},
                "reason": {"type": "string"},
            },
            "required": ["column", "method", "action", "reason"],
        },
    },
    {
        "name": "encode_column",
        "description": (
            "Encode a categorical column: 'one_hot' (adds <column>_<value> "
            "int columns, drops the original) or 'label' (maps values to "
            "integers in place). Returns the new columns or the label mapping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "method": {"type": "string", "enum": ["one_hot", "label"]},
                "reason": {"type": "string"},
            },
            "required": ["column", "method", "reason"],
        },
    },
    {
        "name": "scale_column",
        "description": (
            "Scale a numeric column: 'standard' ((x-mean)/std), 'minmax' "
            "((x-min)/(max-min)), or 'robust' ((x-median)/IQR). Returns "
            "before/after mean/std or min/max."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "method": {"type": "string", "enum": ["standard", "minmax", "robust"]},
                "reason": {"type": "string"},
            },
            "required": ["column", "method", "reason"],
        },
    },
    {
        "name": "submit_cleaning_report",
        "description": (
            "Submit your final conclusions. Calling this ends the cleaning "
            "session. `summary` is a list of what you did and why it "
            "mattered; `remaining_issues` is a list of things you chose to "
            "leave."
        ),
        "input_schema": SubmitCleaningReportInput.model_json_schema(),
    },
]


_ACTION_DISPATCH = {
    "impute_column": lambda df, i: apply_impute(
        df, i["column"], i["strategy"], i.get("constant_value")
    ),
    "drop_column": lambda df, i: apply_drop_column(df, i["column"]),
    "remove_duplicates": lambda df, i: apply_remove_duplicates(df),
    "flag_outliers": lambda df, i: apply_flag_outliers(
        df, i["column"], i["method"], i["action"]
    ),
    "encode_column": lambda df, i: apply_encode_column(df, i["column"], i["method"]),
    "scale_column": lambda df, i: apply_scale_column(df, i["column"], i["method"]),
}


class CleaningAgent:
    """Drives a mutating Claude tool-use cleaning loop over one dataset."""

    def __init__(self, client: Any | None = None):
        self.client = client if client is not None else get_client()

    def _prepare(
        self, df: pd.DataFrame, dataset_id: str, context_report: dict
    ) -> None:
        self.df = df.copy()
        self.dataset_id = dataset_id
        self.context_report = context_report
        self.target_candidate = context_report.get("target_candidate")
        self.problem_type = context_report.get("problem_type") or "unclear"
        self.steps: list[CleaningStep] = []
        self.initial_shape = [len(self.df), self.df.shape[1]]
        self.initial_missing_cells = int(self.df.isna().sum().sum())

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> Any:
        if name == "inspect_column":
            return inspect_column(self.df, tool_input["column_name"])

        if name == "drop_column" and (
            self.target_candidate is not None
            and tool_input.get("column") == self.target_candidate
        ):
            raise ValueError(
                f"Refusing to drop {self.target_candidate!r}: it is the target "
                "column identified in the Data Understanding report."
            )

        try:
            handler = _ACTION_DISPATCH[name]
        except KeyError:
            raise ValueError(f"Unknown tool: {name}")

        new_df, result = handler(self.df, tool_input)
        self.df = new_df
        self.steps.append(
            CleaningStep(
                order=len(self.steps) + 1,
                tool=name,
                reason=tool_input.get("reason", ""),
                params={k: v for k, v in tool_input.items() if k != "reason"},
                result=result,
            )
        )
        return result

    def run(
        self, df: pd.DataFrame, dataset_id: str, context_report: dict
    ) -> CleaningReport:
        self._prepare(df, dataset_id, context_report)

        first_user_content = (
            "Here is the Module 1 Data Understanding report for this dataset. "
            "Use it as your starting context — do not re-derive it.\n\n"
            f"{json.dumps(context_report, indent=2, default=str)}\n\n"
            f"Current columns ({len(self.df.columns)}): {list(self.df.columns)}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": first_user_content}
        ]

        submit_input: dict[str, Any] | None = None
        action_call_count = 0
        total_tool_call_count = 0
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
                        "content": "Please call submit_cleaning_report now with your conclusions.",
                    }
                )
                forced_once = True
                continue

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                if block.name == "submit_cleaning_report":
                    try:
                        validated = SubmitCleaningReportInput.model_validate(block.input)
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

                total_tool_call_count += 1
                is_action = block.name != "inspect_column"
                try:
                    result = self._execute_tool(block.name, block.input)
                    if is_action:
                        action_call_count += 1
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

            if (
                action_call_count >= MAX_ACTION_CALLS
                or total_tool_call_count >= MAX_TOTAL_TOOL_CALLS
            ):
                if forced_once:
                    break
                forced_once = True
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool-call limit reached (actions "
                            f"{action_call_count}/{MAX_ACTION_CALLS}, total "
                            f"{total_tool_call_count}/{MAX_TOTAL_TOOL_CALLS}). "
                            "Call submit_cleaning_report now with your best "
                            "current conclusions."
                        ),
                    }
                )

        if submit_input is None:
            submit_input = {
                "summary": [
                    "Cleaning halted at the tool-call budget without a final summary."
                ],
                "remaining_issues": [],
            }

        return CleaningReport(
            dataset_id=dataset_id,
            cleaned_dataset_id=None,
            target_candidate=self.target_candidate,
            problem_type=self.problem_type,
            initial_shape=self.initial_shape,
            final_shape=[len(self.df), self.df.shape[1]],
            initial_missing_cells=self.initial_missing_cells,
            final_missing_cells=int(self.df.isna().sum().sum()),
            steps=self.steps,
            **submit_input,
        )
