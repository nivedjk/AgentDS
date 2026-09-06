"""Visualization Agent (Module 3).

Deterministic pandas/numpy builders compute the actual chart data (histogram
bin counts, category counts, correlation values, box five-number summaries,
resampled time series) and embed it into literal Plotly figure dicts. A
single ``narrate`` call then selects/orders the most insightful charts and
attaches a one-line ``insight`` per chart plus an overall ``narrative``. The
LLM never produces or edits chart data.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from app.agents._llm import MODEL, narrate
from app.agents.data_understanding import _sanitize
from app.storage.dataset_store import (
    get_artifact,
    get_cleaning_report,
    get_dataset_path,
)

RANDOM_STATE = 42
MAX_CATEGORIES = 20
MAX_SELECTED = 12
MAX_TARGET_FEATURE_CHARTS = 12
HIST_MAX_BINS = 30
DATETIME_PARSE_THRESHOLD = 0.9
MAX_TEMPORAL_SERIES = 3
CORR_DECIMALS = 4


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class ChartSpec(BaseModel):
    chart_type: str
    title: str
    columns: list[str]
    plotly: dict[str, Any]
    insight: str = ""


class SelectedChart(BaseModel):
    index: int
    insight: str


class VisualizationNarration(BaseModel):
    """Schema for the single ``submit_visualization`` narration tool call."""

    selected: list[SelectedChart]
    narrative: str


class VisualizationReport(BaseModel):
    dataset_id: str
    source: Literal["original", "cleaned"]
    target_column: str | None = None
    problem_type: Literal["classification", "regression", "unclear"]
    n_charts: int
    charts: list[ChartSpec]
    skipped: list[str]
    narrative: str


def _json_safe(obj: Any) -> Any:
    """Recursively convert a figure dict to plain JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return _sanitize(obj)


# ---------------------------------------------------------------------------
# Helper functions for chart building
# ---------------------------------------------------------------------------


def _target_of(understanding_report: dict) -> tuple[str | None, str]:
    """Extract target candidate and problem type from understanding report.

    Returns:
        (target_candidate, problem_type) — problem_type defaults to "unclear" if missing.
    """
    target = understanding_report.get("target_candidate")
    problem_type = understanding_report.get("problem_type") or "unclear"
    return target, problem_type


def _resolve_source_df(dataset_id: str) -> tuple[pd.DataFrame, Literal["original", "cleaned"]]:
    """Resolve the source dataframe, preferring cleaned if available.

    Reads the cleaning report for the dataset. If it has a valid cleaned_dataset_id
    whose file resolves, returns the cleaned DataFrame. Otherwise falls back to
    the original dataset.

    Args:
        dataset_id: The original dataset ID.

    Returns:
        (DataFrame, source_type) — source_type is "cleaned" or "original".
    """
    try:
        cleaning_report = get_cleaning_report(dataset_id)
        if cleaning_report and cleaning_report.get("cleaned_dataset_id"):
            cleaned_id = cleaning_report["cleaned_dataset_id"]
            try:
                cleaned_path = get_dataset_path(cleaned_id)
                df_cleaned = pd.read_csv(cleaned_path)
                return df_cleaned, "cleaned"
            except FileNotFoundError:
                # Cleaned file not found, fall back to original
                pass
    except (FileNotFoundError, ValueError):
        # No cleaning report, fall back to original
        pass

    # Fall back to original
    original_path = get_dataset_path(dataset_id)
    df_original = pd.read_csv(original_path)
    return df_original, "original"


def _hist_trace(series: pd.Series, col: str) -> tuple[dict, list]:
    """Build a histogram trace and return (trace_dict, bin_mids).

    Args:
        series: Non-null numeric series.
        col: Column name.

    Returns:
        (trace_dict, bin_midpoints) where trace is Plotly-ready.
    """
    non_null = series.dropna().values
    n = len(non_null)

    # Compute bin count: min(HIST_MAX_BINS, max(10, int(sqrt(n))))
    bin_count = min(HIST_MAX_BINS, max(10, int(np.sqrt(n))))
    counts, edges = np.histogram(non_null, bins=bin_count)

    # Compute bin midpoints
    bin_mids = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

    trace = {
        "type": "bar",
        "x": bin_mids,
        "y": counts.tolist(),
        "name": col,
    }
    return trace, bin_mids


def _box_trace(series: pd.Series, col: str) -> dict:
    """Build a box plot trace from quartiles and range.

    Args:
        series: Non-null numeric series.
        col: Column name.

    Returns:
        Plotly box trace dict with q1, median, q3, lowerfence, upperfence, mean.
    """
    non_null = series.dropna().values

    q1 = float(np.percentile(non_null, 25))
    median = float(np.percentile(non_null, 50))
    q3 = float(np.percentile(non_null, 75))
    iqr = q3 - q1

    data_min = float(non_null.min())
    data_max = float(non_null.max())

    # Fences clamped to data range
    lower_fence = max(data_min, q1 - 1.5 * iqr)
    upper_fence = min(data_max, q3 + 1.5 * iqr)
    mean_val = float(non_null.mean())

    trace = {
        "type": "box",
        "name": col,
        "q1": [q1],
        "median": [median],
        "q3": [q3],
        "lowerfence": [lower_fence],
        "upperfence": [upper_fence],
        "mean": [mean_val],
    }
    return trace


def build_numeric_charts(df: pd.DataFrame, understanding_report: dict) -> tuple[list[ChartSpec], list[str]]:
    """Build histogram and box charts for numeric columns.

    For each numeric column (except target), creates:
    - A histogram chart showing distribution
    - A box chart showing spread

    Skips columns that are all-null, have < 2 non-null values, or are constant.

    Args:
        df: DataFrame to analyze.
        understanding_report: Dict with "target_candidate" and "problem_type".

    Returns:
        (charts, skipped_reasons) — list of ChartSpec objects and skip messages.
    """
    target, _ = _target_of(understanding_report)
    charts = []
    skipped = []

    # Get numeric columns, plus any all-null object columns (which could be numeric)
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    # Also include all-null non-numeric columns (could be numeric data)
    for col in [c for c in df.columns if c not in numeric_cols]:
        if df[col].isna().all():
            numeric_cols.add(col)

    for col in sorted(numeric_cols):  # sort for determinism
        # Skip the target column
        if col == target:
            continue

        # Guard against duplicate column names (pandas allows them)
        if not isinstance(df[col], pd.Series):
            skipped.append(f"{col}: skipped (duplicate column name)")
            continue

        series = df[col]
        non_null = series.dropna()
        non_null_count = len(non_null)

        # Skip all-null
        if non_null_count == 0:
            skipped.append(f"{col}: skipped numeric charts (all values null)")
            continue

        # Skip < 2 non-null
        if non_null_count < 2:
            skipped.append(f"{col}: skipped numeric charts (fewer than 2 non-null values)")
            continue

        # Skip constant
        if non_null.nunique() == 1:
            skipped.append(f"{col}: skipped numeric charts (constant column)")
            continue

        # Create histogram chart
        hist_trace, bin_mids = _hist_trace(non_null, col)
        hist_fig = {
            "data": [hist_trace],
            "layout": {
                "title": f"Distribution of {col}",
                "xaxis": {"title": col},
                "yaxis": {"title": "Count"},
                "bargap": 0.02,
            },
        }
        hist_spec = ChartSpec(
            chart_type="histogram",
            title=f"Distribution of {col}",
            columns=[col],
            plotly=_json_safe(hist_fig),
        )
        charts.append(hist_spec)

        # Create box chart
        box_trace = _box_trace(non_null, col)
        box_fig = {
            "data": [box_trace],
            "layout": {
                "title": f"Spread of {col}",
            },
        }
        box_spec = ChartSpec(
            chart_type="box",
            title=f"Spread of {col}",
            columns=[col],
            plotly=_json_safe(box_fig),
        )
        charts.append(box_spec)

    return charts, skipped


def build_categorical_charts(df: pd.DataFrame, understanding_report: dict) -> tuple[list[ChartSpec], list[str]]:
    """Build bar charts for categorical columns.

    For each non-numeric column (except target), creates one bar chart showing
    value counts (excluding NaN).

    Skips columns with zero non-null values or > MAX_CATEGORIES distinct values.

    Args:
        df: DataFrame to analyze.
        understanding_report: Dict with "target_candidate" and "problem_type".

    Returns:
        (charts, skipped_reasons) — list of ChartSpec objects and skip messages.
    """
    target, _ = _target_of(understanding_report)
    charts = []
    skipped = []

    # Get non-numeric columns (object, category, bool)
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    for col in df.columns:
        # Skip numeric columns and target
        if col in numeric_cols or col == target:
            continue

        # Guard against duplicate column names
        if not isinstance(df[col], pd.Series):
            skipped.append(f"{col}: skipped (duplicate column name)")
            continue

        # Get value counts (dropna=True excludes NaN)
        vc = df[col].value_counts(dropna=True)

        # Skip all-null / zero non-null
        if len(vc) == 0:
            skipped.append(f"{col}: skipped bar chart (no non-null values)")
            continue

        # Skip high cardinality
        if vc.size > MAX_CATEGORIES:
            skipped.append(f"{col}: skipped bar chart ({vc.size} distinct values exceeds the {MAX_CATEGORIES}-category limit)")
            continue

        # Create bar chart
        x_labels = [str(idx) for idx in vc.index]
        y_counts = vc.values.tolist()

        bar_trace = {
            "type": "bar",
            "x": x_labels,
            "y": y_counts,
            "name": col,
        }
        bar_fig = {
            "data": [bar_trace],
            "layout": {
                "title": f"Counts by {col}",
            },
        }
        bar_spec = ChartSpec(
            chart_type="bar",
            title=f"Counts by {col}",
            columns=[col],
            plotly=_json_safe(bar_fig),
        )
        charts.append(bar_spec)

    return charts, skipped


def _detect_datetime_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect datetime columns in a DataFrame.

    A column qualifies if:
    - It is already datetime64 dtype, or
    - It is object/string and pd.to_datetime(series, errors="coerce") parses
      >= DATETIME_PARSE_THRESHOLD of non-null values AND yields >= 3 distinct timestamps.

    For object columns whose name matches date|time|timestamp|year|month|day (case-insensitive)
    but fails the threshold, append a skip note.

    Args:
        df: DataFrame to analyze.

    Returns:
        (datetime_cols, skip_notes) — lists of datetime column names and skip reasons.
    """
    import re

    datetime_cols = []
    skip_notes = []

    # Check for columns that are already datetime64
    for col in df.select_dtypes(include=['datetime64']).columns:
        datetime_cols.append(col)

    # Check object/string columns (pandas 3.0.5+ uses StringDtype)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        series = df[col]
        non_null_count = series.notna().sum()

        if non_null_count == 0:
            continue

        # Try to parse with utc=True to handle mixed-offset strings gracefully
        try:
            parsed = pd.to_datetime(series, errors='coerce', utc=True)
        except (ValueError, TypeError):
            # Mixed-offset or unparseable column
            if re.search(r'(date|time|timestamp|year|month|day)', col.lower()):
                skip_notes.append(f"{col}: looks temporal but contains mixed-offset or unparseable timestamps")
            continue

        parsed_count = parsed.notna().sum()
        pct = parsed_count / non_null_count

        # Check if >= DATETIME_PARSE_THRESHOLD and >= 3 distinct values
        distinct_count = parsed.nunique()

        if pct >= DATETIME_PARSE_THRESHOLD and distinct_count >= 3:
            datetime_cols.append(col)
        elif re.search(r'(date|time|timestamp|year|month|day)', col.lower()):
            # Temporal-named column that failed threshold
            skip_notes.append(f"{col}: looks temporal but only {pct:.0%} of values parse as dates")

    return datetime_cols, skip_notes


def build_temporal_charts(df: pd.DataFrame, understanding_report: dict) -> tuple[list[ChartSpec], list[str]]:
    """Build temporal line charts for datetime columns.

    For each detected datetime column, creates one line chart with:
    - Trace 1: row count per period
    - Up to MAX_TEMPORAL_SERIES further traces: mean per period of numeric columns

    Args:
        df: DataFrame to analyze.
        understanding_report: Dict with "target_candidate" and "problem_type".

    Returns:
        (charts, skipped_reasons) — list of ChartSpec objects and skip messages.
    """
    datetime_cols, skip_notes = _detect_datetime_columns(df)
    charts = []
    skipped = skip_notes.copy()

    if not datetime_cols:
        skipped.append("temporal charts: no parseable datetime column found")
        return [], skipped

    for dtcol in datetime_cols:
        try:
            # Parse and sort (with utc=True for consistent timezone handling)
            parsed = pd.to_datetime(df[dtcol], errors='coerce', utc=True)
            df_with_dt = df.copy()
            df_with_dt['_parsed_dt'] = parsed

            # Drop all-null rows
            df_with_dt = df_with_dt[df_with_dt['_parsed_dt'].notna()]

            if len(df_with_dt) == 0:
                continue

            # Sort by parsed datetime
            df_with_dt = df_with_dt.sort_values('_parsed_dt')

            # Determine resample frequency
            date_range = df_with_dt['_parsed_dt'].max() - df_with_dt['_parsed_dt'].min()
            days_span = date_range.days

            if days_span > 730:
                freq = 'MS'
            elif days_span > 60:
                freq = 'W'
            elif days_span > 2:
                freq = 'D'
            else:
                freq = 'h'

            # Set index and resample
            indexed = df_with_dt.set_index('_parsed_dt').sort_index()
            resampled = indexed.resample(freq)
        except (TypeError, ValueError) as e:
            skipped.append(f"{dtcol}: skipped temporal chart (unparseable / mixed-offset timestamps)")
            continue

        # Row count per period
        row_counts = resampled.size()
        xs = [idx.strftime("%Y-%m-%dT%H:%M:%S") for idx in row_counts.index]
        ys = row_counts.values.tolist()

        traces = [{
            "type": "scatter",
            "mode": "lines+markers",
            "x": xs,
            "y": ys,
            "name": "row count"
        }]

        # Add numeric series (up to MAX_TEMPORAL_SERIES)
        numeric_cols = list(df_with_dt.select_dtypes(include=[np.number]).columns)
        if dtcol in numeric_cols:
            numeric_cols.remove(dtcol)

        series_added = 0
        for num_col in numeric_cols:
            if series_added >= MAX_TEMPORAL_SERIES:
                break
            if num_col == '_parsed_dt':
                continue

            means = resampled[num_col].mean()
            ys_series = [float(y) if pd.notna(y) else None for y in means.values]

            traces.append({
                "type": "scatter",
                "mode": "lines",
                "x": xs,
                "y": ys_series,
                "name": num_col,
                "yaxis": "y2"
            })
            series_added += 1

        # Build layout
        layout = {
            "title": f"{dtcol} over time",
            "xaxis": {"title": dtcol},
            "yaxis": {"title": "row count"},
        }

        if series_added > 0:
            layout["yaxis2"] = {
                "title": "numeric series",
                "overlaying": "y",
                "side": "right"
            }

        fig = {
            "data": traces,
            "layout": layout
        }

        spec = ChartSpec(
            chart_type="line",
            title=f"{dtcol} over time",
            columns=[dtcol] + [n for n in numeric_cols[:series_added]],
            plotly=_json_safe(fig),
        )
        charts.append(spec)

    return charts, skipped


def build_correlation_chart(df: pd.DataFrame) -> tuple[list[ChartSpec], list[str]]:
    """Build correlation heatmap for numeric columns.

    Computes Pearson correlation and drops zero-variance columns.

    Args:
        df: DataFrame to analyze.

    Returns:
        (charts, skipped_reasons) — list of ChartSpec objects and skip message.
    """
    skipped = []

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        skipped.append("correlation heatmap: fewer than 2 usable numeric columns")
        return [], skipped

    # Compute correlation
    corr_matrix = numeric_df.corr(numeric_only=True).round(CORR_DECIMALS)

    # Drop zero-variance columns
    initial_cols = set(corr_matrix.columns)
    corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Record dropped columns
    dropped = initial_cols - set(corr_matrix.columns)
    for col in sorted(dropped):
        skipped.append(f"correlation heatmap: dropped {col} (zero variance)")

    # Check if we still have >= 2 columns
    if len(corr_matrix.columns) < 2:
        skipped.append("correlation heatmap: fewer than 2 usable numeric columns")
        return [], skipped

    # Convert NaN to None for JSON serialization
    z = [[float(v) if pd.notna(v) else None for v in row] for row in corr_matrix.values]

    cols = corr_matrix.columns.tolist()

    trace = {
        "type": "heatmap",
        "z": z,
        "x": cols,
        "y": cols,
        "zmin": -1,
        "zmax": 1,
        "colorscale": "RdBu",
        "reversescale": True,
    }

    fig = {
        "data": [trace],
        "layout": {
            "title": "Correlation matrix",
        }
    }

    spec = ChartSpec(
        chart_type="heatmap",
        title="Correlation matrix",
        columns=cols,
        plotly=_json_safe(fig),
    )

    return [spec], skipped


def _class_box_trace(values: pd.Series, name: str) -> dict:
    """Precomputed box trace for a single group.

    Args:
        values: Series of numeric values for one group.
        name: Name/label for the group.

    Returns:
        Plotly box trace dict.
    """
    return _box_trace(values, name)


def build_target_relationship_charts(df: pd.DataFrame, understanding_report: dict) -> tuple[list[ChartSpec], list[str]]:
    """Build target relationship charts (univariate and bivariate).

    Args:
        df: DataFrame to analyze.
        understanding_report: Dict with "target_candidate" and "problem_type".

    Returns:
        (charts, skipped_reasons) — list of ChartSpec objects and skip messages.
    """
    target, problem_type = _target_of(understanding_report)
    charts = []
    skipped = []

    # Check if target exists
    if target is None:
        skipped.append("target-relationship charts: no target column identified")
        return [], skipped

    if target not in df.columns:
        skipped.append(f"target-relationship charts: target {target!r} not present in the dataset")
        return [], skipped

    # Univariate target chart
    target_series = df[target]

    if problem_type == "regression" and pd.api.types.is_numeric_dtype(df[target]):
        # Histogram: distribution (only for numeric targets)
        hist_trace, _ = _hist_trace(target_series.dropna(), target)
        hist_fig = {
            "data": [hist_trace],
            "layout": {
                "title": f"Distribution of {target}",
                "xaxis": {"title": target},
                "yaxis": {"title": "Count"},
                "bargap": 0.02,
            },
        }
        hist_spec = ChartSpec(
            chart_type="histogram",
            title=f"Distribution of {target}",
            columns=[target],
            plotly=_json_safe(hist_fig),
        )
        charts.append(hist_spec)
    elif problem_type == "classification" or not pd.api.types.is_numeric_dtype(df[target]) or target_series.nunique(dropna=True) <= MAX_CATEGORIES:
        # Bar chart: class balance
        vc = target_series.value_counts(dropna=True)
        x_labels = [str(idx) for idx in vc.index]
        y_counts = vc.values.tolist()

        bar_trace = {
            "type": "bar",
            "x": x_labels,
            "y": y_counts,
            "name": target,
        }
        bar_fig = {
            "data": [bar_trace],
            "layout": {
                "title": f"Class balance of {target}",
            },
        }
        bar_spec = ChartSpec(
            chart_type="bar",
            title=f"Class balance of {target}",
            columns=[target],
            plotly=_json_safe(bar_fig),
        )
        charts.append(bar_spec)
    else:
        # Histogram: distribution (for numeric non-classification)
        hist_trace, _ = _hist_trace(target_series.dropna(), target)
        hist_fig = {
            "data": [hist_trace],
            "layout": {
                "title": f"Distribution of {target}",
                "xaxis": {"title": target},
                "yaxis": {"title": "Count"},
                "bargap": 0.02,
            },
        }
        hist_spec = ChartSpec(
            chart_type="histogram",
            title=f"Distribution of {target}",
            columns=[target],
            plotly=_json_safe(hist_fig),
        )
        charts.append(hist_spec)

    # Bivariate charts
    treat_as_class = (problem_type == "classification" or
                      not pd.api.types.is_numeric_dtype(df[target]) or
                      target_series.nunique(dropna=True) <= MAX_CATEGORIES)

    if not treat_as_class and problem_type == "unclear":
        skipped.append("target-vs-feature charts: problem type is unclear")
        return charts, skipped

    # Build grouping series
    if treat_as_class:
        g = target_series
    else:
        try:
            g = pd.qcut(target_series, q=4, duplicates='drop')
            if g.nunique() < 2:
                skipped.append("target-vs-feature charts: target quartiles produced fewer than 2 groups")
                return charts, skipped
            g = g.astype(str)
        except Exception:
            skipped.append("target-vs-feature charts: could not quartile the target")
            return charts, skipped

    # Count bivariate charts
    bivariate_count = 0
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if target in numeric_cols:
        numeric_cols.remove(target)

    # Numeric feature columns
    for feat in numeric_cols:
        if bivariate_count >= MAX_TARGET_FEATURE_CHARTS:
            skipped.append(f"target-vs-{feat}: skipped, chart budget of {MAX_TARGET_FEATURE_CHARTS} reached")
            continue

        # Guard against duplicate column names
        if not isinstance(df[feat], pd.Series):
            skipped.append(f"{feat}: skipped (duplicate column name)")
            continue

        feat_series = df[feat]
        non_null_feat = feat_series.dropna()

        if len(non_null_feat) == 0:
            continue

        # Build box traces, one per group
        traces = []
        for group_val in sorted(g.unique()):
            try:
                group_mask = g == group_val
                group_values = feat_series[group_mask].dropna()
                if len(group_values) > 0:
                    trace = _class_box_trace(group_values, str(group_val))
                    traces.append(trace)
            except Exception:
                continue

        if traces:
            fig = {
                "data": traces,
                "layout": {
                    "title": f"{feat} by {target}",
                },
            }
            spec = ChartSpec(
                chart_type="box",
                title=f"{feat} by {target}",
                columns=[feat, target],
                plotly=_json_safe(fig),
            )
            charts.append(spec)
            bivariate_count += 1

    # Categorical feature columns (non-numeric)
    numeric_col_set = set(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in df.columns if c != target and c not in numeric_col_set]

    for feat in categorical_cols:
        if bivariate_count >= MAX_TARGET_FEATURE_CHARTS:
            skipped.append(f"target-vs-{feat}: skipped, chart budget of {MAX_TARGET_FEATURE_CHARTS} reached")
            continue

        # Guard against duplicate column names
        if not isinstance(df[feat], pd.Series):
            skipped.append(f"{feat}: skipped (duplicate column name)")
            continue

        feat_vc = df[feat].nunique()

        if feat_vc > MAX_CATEGORIES:
            skipped.append(f"{feat}: skipped target grouped bar ({feat_vc} distinct values)")
            continue

        # Build crosstab
        try:
            ct = pd.crosstab(df[feat], g)
        except Exception:
            continue

        if ct.empty or len(ct) == 0:
            continue

        # Build traces, one per group column
        traces = []
        for gval in sorted(ct.columns):
            try:
                x_labels = [str(idx) for idx in ct.index]
                y_vals = ct[gval].values.tolist()
                trace = {
                    "type": "bar",
                    "x": x_labels,
                    "y": y_vals,
                    "name": str(gval),
                }
                traces.append(trace)
            except Exception:
                continue

        if traces:
            fig = {
                "data": traces,
                "layout": {
                    "title": f"{feat} vs {target}",
                },
            }
            spec = ChartSpec(
                chart_type="grouped_bar",
                title=f"{feat} vs {target}",
                columns=[feat, target],
                plotly=_json_safe(fig),
            )
            charts.append(spec)
            bivariate_count += 1

    return charts, skipped


def build_chart_candidates(df: pd.DataFrame, understanding_report: dict) -> tuple[list[ChartSpec], list[str]]:
    """Build all chart candidates by running all builders in order.

    Runs: build_target_relationship_charts, build_correlation_chart,
    build_numeric_charts, build_categorical_charts, build_temporal_charts.

    Args:
        df: DataFrame to analyze.
        understanding_report: Dict with "target_candidate" and "problem_type".

    Returns:
        (all_charts, all_skipped) — concatenated lists from all builders.
    """
    all_charts = []
    all_skipped = []

    # Target relationship charts
    charts, skipped = build_target_relationship_charts(df, understanding_report)
    all_charts.extend(charts)
    all_skipped.extend(skipped)

    # Correlation chart
    charts, skipped = build_correlation_chart(df)
    all_charts.extend(charts)
    all_skipped.extend(skipped)

    # Numeric charts
    charts, skipped = build_numeric_charts(df, understanding_report)
    all_charts.extend(charts)
    all_skipped.extend(skipped)

    # Categorical charts
    charts, skipped = build_categorical_charts(df, understanding_report)
    all_charts.extend(charts)
    all_skipped.extend(skipped)

    # Temporal charts
    charts, skipped = build_temporal_charts(df, understanding_report)
    all_charts.extend(charts)
    all_skipped.extend(skipped)

    return all_charts, all_skipped


def _candidate_digest(candidates: list[ChartSpec]) -> list[dict]:
    """Create metadata-only summaries of candidate charts.

    Args:
        candidates: List of ChartSpec objects.

    Returns:
        List of dicts with: index, chart_type, title, columns, summary (no data arrays).
    """
    digests = []

    for i, spec in enumerate(candidates):
        # Generate summary based on chart type and plotly figure
        summary = ""
        fig_data = spec.plotly.get("data", [])

        if spec.chart_type == "histogram" and fig_data:
            xs = fig_data[0].get("x", [])
            ys = fig_data[0].get("y", [])
            summary = f"{len(xs)} bins, {sum(ys)} values"
        elif spec.chart_type == "box" and fig_data:
            median = fig_data[0].get("median", [None])[0]
            q1 = fig_data[0].get("q1", [None])[0]
            q3 = fig_data[0].get("q3", [None])[0]
            summary = f"median {median}, IQR {q1}-{q3}"
        elif spec.chart_type == "bar" and fig_data:
            xs = fig_data[0].get("x", [])
            ys = fig_data[0].get("y", [])
            if xs and ys:
                top_idx = np.argmax(ys)
                summary = f"{len(xs)} categories, top '{xs[top_idx]}'={max(ys)}"
            else:
                summary = f"{len(xs)} categories"
        elif spec.chart_type == "line" and fig_data:
            xs = fig_data[0].get("x", [])
            num_series = len(fig_data)
            summary = f"{len(xs)} time points, {num_series} series"
        elif spec.chart_type == "heatmap" and fig_data:
            x = fig_data[0].get("x", [])
            summary = f"{len(x)}x{len(x)} matrix"
        elif spec.chart_type == "grouped_bar" and fig_data:
            x = fig_data[0].get("x", [])
            num_series = len(fig_data)
            summary = f"{len(x)} groups x {num_series} series"
        else:
            summary = f"{spec.chart_type} chart"

        digests.append({
            "index": i,
            "chart_type": spec.chart_type,
            "title": spec.title,
            "columns": spec.columns,
            "summary": summary,
        })

    return digests


def _fallback_narrative(candidates: list[ChartSpec], target: str | None, problem_type: str) -> str:
    """Generate a deterministic one-liner narrative.

    Args:
        candidates: List of candidate ChartSpec objects.
        target: Target column name or None.
        problem_type: "classification", "regression", or "unclear".

    Returns:
        One-liner narrative string.
    """
    n = len(candidates)
    base = f"{n} candidate charts generated for a {problem_type} problem"
    if target:
        return f"{base} targeting {target}."
    return f"{base}."


def run_visualization(
    df: pd.DataFrame,
    dataset_id: str,
    understanding_report: dict,
    client: Any | None = None,
    source: Literal["original", "cleaned"] = "original",
) -> VisualizationReport:
    """Generate visualization report with LLM narration.

    Args:
        df: DataFrame to analyze.
        dataset_id: Unique dataset identifier.
        understanding_report: Dict with "target_candidate" and "problem_type".
        client: LLM client (if None, ``narrate()`` builds one from AGENTDS_LLM_PROVIDER).
        source: "original" or "cleaned".

    Returns:
        VisualizationReport with selected charts and narrative.
    """
    target, problem_type = _target_of(understanding_report)

    # Build candidates
    candidates, skipped = build_chart_candidates(df, understanding_report)

    # Early exit if no candidates
    if not candidates:
        return VisualizationReport(
            dataset_id=dataset_id,
            source=source,
            target_column=target,
            problem_type=problem_type,
            n_charts=0,
            charts=[],
            skipped=skipped,
            narrative="No chartable columns were found in this dataset.",
        )

    # Build default narrative
    default_narrative = _fallback_narrative(candidates, target, problem_type)
    default = VisualizationNarration(
        selected=[SelectedChart(index=i, insight="") for i in range(min(len(candidates), MAX_SELECTED))],
        narrative=default_narrative,
    )

    # Call LLM to select and narrate charts
    system = (
        "You are a data-visualization narrator. Pick the most insightful charts from a fixed candidate set, "
        "order them best-first, drop redundant/uninformative ones, keep at most N. Do not invent charts or numbers; "
        "refer to charts only by their integer index."
    )

    user_payload = {
        "dataset_id": dataset_id,
        "n_rows": len(df),
        "target": target,
        "problem_type": problem_type,
        "max_select": MAX_SELECTED,
        "candidates": _candidate_digest(candidates),
    }
    user = json.dumps(user_payload, default=str)

    narration = narrate(
        client,
        system=system,
        user=user,
        schema=VisualizationNarration,
        tool_name="submit_visualization",
        tool_description="Submit the selected chart indices, one insight each, and an overall narrative.",
        default=default,
        model=MODEL,
    )

    # Assemble final charts
    final_charts = []
    seen_indices = set()

    for sel in narration.selected:
        if 0 <= sel.index < len(candidates) and sel.index not in seen_indices:
            seen_indices.add(sel.index)
            # Copy candidate and set insight
            chart = candidates[sel.index].model_copy(update={"insight": sel.insight})
            final_charts.append(chart)

    # Fall back to candidates if selection is empty
    if not final_charts:
        final_charts = [
            c.model_copy(update={"insight": ""}) for c in candidates[:MAX_SELECTED]
        ]

    # Cap at MAX_SELECTED
    final_charts = final_charts[:MAX_SELECTED]

    # Use narration narrative or default
    narrative = narration.narrative or default.narrative

    return VisualizationReport(
        dataset_id=dataset_id,
        source=source,
        target_column=target,
        problem_type=problem_type,
        n_charts=len(final_charts),
        charts=final_charts,
        skipped=skipped,
        narrative=narrative,
    )
