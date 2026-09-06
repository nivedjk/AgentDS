"""Report Agent (Module 7).

Deterministic assembly: gather every sidecar Modules 1-6 and 8 wrote for a
dataset, render an ordered set of Markdown sections whose every fact is copied
verbatim from those sidecars, then make ONE narration call for prose only
(executive summary, one intro paragraph per section, conclusion). Returns a
FinalReport: the structured sections plus one rendered Markdown document.

The per-module section renderers are pure functions of a single sidecar dict
(or None) -> ReportSection. None -> status "not_run" with a one-line stub.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

from app.agents._llm import MODEL, narrate  # noqa: F401  (MODEL re-exported)
from app.storage.dataset_store import get_artifact

SECTION_ORDER: list[tuple[str, str, str]] = [
    ("executive_summary", "Executive Summary", "report"),
    ("dataset_overview", "Dataset Overview", "understanding"),
    ("understanding", "Data Understanding", "understanding"),
    ("cleaning", "Cleaning", "cleaning"),
    ("visualization", "Visualizations", "visualization"),
    ("recommendation", "Model Recommendation", "recommendation"),
    ("training", "Training & Evaluation", "training"),
    ("explainability", "Explainability", "explainability"),
    ("whatif", "What-If Analysis", "whatif"),
    ("conclusion", "Conclusion & Limitations", "report"),
]
UPSTREAM_KINDS: list[str] = [
    "understanding", "cleaning", "visualization", "recommendation",
    "training", "explainability", "whatif",
]
PROSE_SECTION_KEYS: list[str] = [k for k, _, m in SECTION_ORDER if m != "report"]


class ReportSection(BaseModel):
    key: str
    title: str
    body_markdown: str
    source_module: str
    status: Literal["ok", "not_run"]


class ReportNarration(BaseModel):
    """Schema for the single narration tool call - prose only."""

    executive_summary: str
    section_intros: dict[str, str] = {}
    conclusion: str


class FinalReport(BaseModel):
    dataset_id: str
    generated_at: str
    source_dataset_id: str
    modules_present: list[str]
    modules_missing: list[str]
    sections: list[ReportSection]
    executive_summary: str
    markdown: str


# ---------------------------------------------------------------------------
# Small Markdown helpers
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    """Render a scalar verbatim; ``None`` becomes an em-dash."""
    if value is None:
        return "—"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    """GitHub-flavoured pipe table. ``"_None._"`` when ``rows`` is empty."""
    if not rows:
        return "_None._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _bullets(items: list[str]) -> str:
    """``"- item"`` lines; ``"_None._"`` when ``items`` is empty."""
    if not items:
        return "_None._"
    return "\n".join(f"- {item}" for item in items)


def _stub(human_name: str) -> str:
    return f"_{human_name} has not been run for this dataset._"


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines())


# ---------------------------------------------------------------------------
# Per-module section renderers
#
# Every value below is read verbatim from the sidecar via .get(...); nothing
# is computed here. Each renderer is total: it never raises on a missing key
# or on artifact=None.
# ---------------------------------------------------------------------------
def render_dataset_overview(
    understanding: dict | None,
    cleaning: dict | None,
    *,
    dataset_id: str,
    source_dataset_id: str,
) -> ReportSection:
    key, title, source_module = "dataset_overview", "Dataset Overview", "understanding"
    if understanding is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Data understanding"),
        )

    cleaned_dataset_id = (
        cleaning.get("cleaned_dataset_id") if cleaning else "— (cleaning not run)"
    )
    rows = [
        ["Dataset ID", dataset_id],
        ["Reported dataset", source_dataset_id],
        ["Rows", understanding.get("n_rows")],
        ["Columns", understanding.get("n_columns")],
        ["Target column", understanding.get("target_candidate")],
        ["Problem type", understanding.get("problem_type")],
        ["Cleaned dataset", cleaned_dataset_id],
    ]
    body = _md_table(["Property", "Value"], rows)
    return ReportSection(
        key=key, title=title, source_module=source_module, status="ok", body_markdown=body,
    )


def render_understanding_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "understanding", "Data Understanding", "understanding"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Data understanding"),
        )

    parts: list[str] = []
    parts.append("**Key findings**\n" + _bullets(artifact.get("key_findings", [])))

    dup = artifact.get("duplicates")
    if dup:
        parts.append(
            f"Duplicate rows: {dup.get('n_duplicate_rows')} ({dup.get('percent')}%)"
        )

    corr_rows = [
        [c.get("column_a"), c.get("column_b"), c.get("correlation")]
        for c in (artifact.get("correlations") or [])
    ]
    parts.append(
        "**Highly correlated pairs**\n"
        + _md_table(["Column A", "Column B", "r"], corr_rows)
    )

    col_rows = [
        [c.get("name"), c.get("dtype"), c.get("missing_pct"), c.get("n_unique")]
        for c in artifact.get("columns", [])
    ]
    parts.append(
        "**Columns**\n" + _md_table(["Name", "Dtype", "Missing %", "Unique"], col_rows)
    )

    narrative = artifact.get("narrative")
    if narrative:
        parts.append(_blockquote(narrative))

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(parts),
    )


def render_cleaning_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "cleaning", "Cleaning", "cleaning"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Cleaning"),
        )

    initial_shape = artifact.get("initial_shape")
    final_shape = artifact.get("final_shape")
    initial_missing = artifact.get("initial_missing_cells")
    final_missing = artifact.get("final_missing_cells")
    shape_line = (
        f"Shape {_fmt(initial_shape)} → {_fmt(final_shape)}; "
        f"missing cells {_fmt(initial_missing)} → {_fmt(final_missing)}."
    )

    def _kv_join(d: dict[str, Any] | None) -> str:
        if not d:
            return "—"
        return "; ".join(f"{k}={v}" for k, v in d.items())

    step_rows = [
        [
            s.get("order"),
            s.get("tool"),
            _kv_join(s.get("params")),
            s.get("reason"),
            _kv_join(s.get("result")),
        ]
        for s in artifact.get("steps", [])
    ]
    steps_table = _md_table(["#", "Tool", "Params", "Reason", "Result"], step_rows)

    body = "\n\n".join([
        shape_line,
        "**Steps**\n" + steps_table,
        "**Summary**\n" + _bullets(artifact.get("summary", [])),
        "**Remaining issues**\n" + _bullets(artifact.get("remaining_issues", [])),
    ])
    return ReportSection(
        key=key, title=title, source_module=source_module, status="ok", body_markdown=body,
    )


def render_visualization_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "visualization", "Visualizations", "visualization"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Visualization"),
        )

    n_charts = artifact.get("n_charts")
    source = artifact.get("source")
    summary_line = f"{_fmt(n_charts)} charts generated from the {_fmt(source)} dataset."

    chart_rows = [
        [
            c.get("title"),
            c.get("chart_type"),
            ", ".join(c.get("columns", [])),
            c.get("insight"),
        ]
        for c in artifact.get("charts", [])
    ]
    charts_table = _md_table(["Title", "Type", "Columns", "Insight"], chart_rows)

    parts = [
        summary_line,
        charts_table,
        "**Skipped**\n" + _bullets(artifact.get("skipped", [])),
    ]
    narrative = artifact.get("narrative")
    if narrative:
        parts.append(_blockquote(narrative))

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(parts),
    )


def render_recommendation_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "recommendation", "Model Recommendation", "recommendation"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Model recommendation"),
        )

    primary_metric = artifact.get("primary_metric")
    basis = "cleaned" if artifact.get("used_cleaned_dataset") else "original"
    summary_line = f"Primary metric: {_fmt(primary_metric)}. Basis: {basis} dataset."

    candidate_rows = [
        [c.get("name"), c.get("library"), c.get("rationale")]
        for c in artifact.get("candidates", [])
    ]
    candidates_table = _md_table(["Model", "Library", "Rationale"], candidate_rows)

    parts = [
        summary_line,
        candidates_table,
        "**Preprocessing recommendations**\n"
        + _bullets(artifact.get("preprocessing_recommendations", [])),
    ]
    reasoning = artifact.get("reasoning")
    if reasoning:
        parts.append(_blockquote(reasoning))

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(parts),
    )


def render_training_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "training", "Training & Evaluation", "training"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Training"),
        )

    summary_line = (
        f"Best model: {_fmt(artifact.get('best_model'))} — "
        f"{_fmt(artifact.get('primary_metric'))} = {_fmt(artifact.get('best_score'))}. "
        f"{_fmt(artifact.get('n_rows'))} rows "
        f"({_fmt(artifact.get('train_rows'))} train / {_fmt(artifact.get('test_rows'))} test), "
        f"{_fmt(artifact.get('cv_splits'))}-fold CV."
    )

    leaderboard_rows = [
        [
            r.get("rank"), r.get("name"), r.get("cv_mean"), r.get("cv_std"),
            r.get("test_score"), r.get("failed"),
        ]
        for r in artifact.get("leaderboard", [])
    ]
    leaderboard_table = _md_table(
        ["Rank", "Model", "CV mean", "CV std", "Test", "Failed"], leaderboard_rows
    )

    error_bullets = _bullets(
        [
            f"{c.get('name')}: {c.get('error')}"
            for c in artifact.get("candidates", [])
            if c.get("error")
        ]
    )

    parts = [
        summary_line,
        "**Leaderboard**\n" + leaderboard_table,
        "**Candidate errors**\n" + error_bullets,
        "**Warnings**\n" + _bullets(artifact.get("warnings", [])),
    ]
    narrative = artifact.get("narrative")
    if narrative:
        parts.append(_blockquote(narrative))

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(parts),
    )


def render_explainability_section(artifact: dict | None) -> ReportSection:
    """Renders the Module 6 ExplainabilityReport sidecar.

    NOTE: real Module 6 field names (verified against
    ``app/agents/explainability.py::ExplainabilityReport``) are
    ``explainer_type`` (not ``method``) and ``feature_importance`` with each
    item ``{"feature", "mean_abs_shap"}`` (not ``global_importances``/
    ``top_features`` with ``{"feature", "importance"}``).
    """
    key, title, source_module = "explainability", "Explainability", "explainability"
    if artifact is None:
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("Explainability"),
        )

    summary_line = (
        f"Method: {_fmt(artifact.get('explainer_type'))}. "
        f"Model explained: {_fmt(artifact.get('model_name'))}."
    )

    feature_rows = [
        [fi.get("feature"), fi.get("mean_abs_shap")]
        for fi in artifact.get("feature_importance", [])
    ]
    features_table = _md_table(["Feature", "Mean |SHAP|"], feature_rows)

    parts = [summary_line, "**Top features**\n" + features_table]
    narrative = artifact.get("narrative", "")
    if narrative:
        parts.append(_blockquote(narrative))

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(parts),
    )


def render_whatif_section(artifact: dict | None) -> ReportSection:
    key, title, source_module = "whatif", "What-If Analysis", "whatif"
    if not artifact or not artifact.get("experiments"):
        return ReportSection(
            key=key, title=title, source_module=source_module,
            status="not_run", body_markdown=_stub("What-if analysis"),
        )

    blocks: list[str] = []
    for i, exp in enumerate(artifact["experiments"], start=1):
        question = exp.get("question", f"Experiment {i}")
        summary = exp.get("summary") or exp.get("narrative", "")
        delta_rows = [
            [d.get("label", d.get("feature")), d.get("change", d.get("delta"))]
            for d in exp.get("deltas", [])
        ]
        deltas_table = _md_table(["Label", "Change"], delta_rows)
        block = f"### {question}\n\n{summary}\n\n{deltas_table}"
        blocks.append(block)

    return ReportSection(
        key=key, title=title, source_module=source_module,
        status="ok", body_markdown="\n\n".join(blocks),
    )


# ---------------------------------------------------------------------------
# Digest + prose layer
# ---------------------------------------------------------------------------
def build_report_digest(artifacts: dict[str, dict | None]) -> dict:
    """Collapse the raw sidecars into a compact digest for the single narrate() call.

    Every array is collapsed to a count or a short name list. No plotly dicts,
    column lists, leaderboards, or importance arrays leak into the digest.
    """
    modules_present = [k for k in UPSTREAM_KINDS if artifacts.get(k) is not None]
    modules_missing = [k for k in UPSTREAM_KINDS if k not in modules_present]
    digest: dict[str, Any] = {
        "modules_present": modules_present,
        "modules_missing": modules_missing,
    }

    understanding = artifacts.get("understanding")
    if understanding is not None:
        duplicates = understanding.get("duplicates") or {}
        digest["understanding"] = {
            "n_rows": understanding.get("n_rows"),
            "n_columns": understanding.get("n_columns"),
            "target_candidate": understanding.get("target_candidate"),
            "problem_type": understanding.get("problem_type"),
            "n_duplicate_rows": duplicates.get("n_duplicate_rows"),
            "n_high_corr_pairs": len(understanding.get("correlations") or []),
            "key_findings": understanding.get("key_findings", []),
        }

    cleaning = artifacts.get("cleaning")
    if cleaning is not None:
        initial_missing = cleaning.get("initial_missing_cells")
        final_missing = cleaning.get("final_missing_cells")
        missing_cells_removed = None
        if initial_missing is not None and final_missing is not None:
            missing_cells_removed = initial_missing - final_missing
        digest["cleaning"] = {
            "n_steps": len(cleaning.get("steps", [])),
            "initial_shape": cleaning.get("initial_shape"),
            "final_shape": cleaning.get("final_shape"),
            "missing_cells_removed": missing_cells_removed,
            "n_remaining_issues": len(cleaning.get("remaining_issues", [])),
        }

    visualization = artifacts.get("visualization")
    if visualization is not None:
        charts = visualization.get("charts", [])
        digest["visualization"] = {
            "n_charts": visualization.get("n_charts"),
            "source": visualization.get("source"),
            "chart_types": sorted({c.get("chart_type") for c in charts}),
        }

    recommendation = artifacts.get("recommendation")
    if recommendation is not None:
        candidates = recommendation.get("candidates", [])
        digest["recommendation"] = {
            "primary_metric": recommendation.get("primary_metric"),
            "n_candidates": len(candidates),
            "candidate_names": [c.get("name") for c in candidates],
        }

    training = artifacts.get("training")
    if training is not None:
        leaderboard = training.get("leaderboard", [])
        digest["training"] = {
            "best_model": training.get("best_model"),
            "primary_metric": training.get("primary_metric"),
            "best_score": training.get("best_score"),
            "n_candidates": len(leaderboard),
            "n_failed": sum(1 for r in leaderboard if r.get("failed")),
        }

    explainability = artifacts.get("explainability")
    if explainability is not None:
        top_features = explainability.get("feature_importance", [])
        digest["explainability"] = {
            "method": explainability.get("explainer_type"),
            "top_feature_names": [f.get("feature") for f in top_features[:5]],
        }

    whatif = artifacts.get("whatif")
    if whatif is not None:
        experiments = whatif.get("experiments", [])
        digest["whatif"] = {
            "n_experiments": len(experiments),
            "questions": [e.get("question") for e in experiments[:5]],
        }

    return digest


def _template_executive_summary(digest: dict) -> str:
    understanding = digest.get("understanding")
    if not understanding:
        return "This dataset has not yet been analysed beyond upload."
    n_rows = understanding.get("n_rows")
    n_columns = understanding.get("n_columns")
    problem_type = understanding.get("problem_type")
    parts = [f"This dataset has {_fmt(n_rows)} rows and {_fmt(n_columns)} columns."]
    if problem_type and problem_type != "unclear":
        parts.append(f"It is framed as a {problem_type} problem.")
    training = digest.get("training")
    if training:
        parts.append(
            f"The best-performing model was {_fmt(training.get('best_model'))} "
            f"({_fmt(training.get('primary_metric'))} = {_fmt(training.get('best_score'))})."
        )
    return " ".join(parts)


def _template_section_intros(digest: dict) -> dict[str, str]:
    missing = set(digest.get("modules_missing", []))
    intros: dict[str, str] = {}

    def _set(key: str, module: str, text: str) -> None:
        intros[key] = "" if module in missing else text

    understanding = digest.get("understanding")
    _set(
        "dataset_overview", "understanding",
        f"The dataset has {_fmt(understanding.get('n_rows') if understanding else None)} rows.",
    )
    _set(
        "understanding", "understanding",
        f"Data understanding found {_fmt(understanding.get('n_duplicate_rows') if understanding else None)} "
        "duplicate rows and "
        f"{_fmt(understanding.get('n_high_corr_pairs') if understanding else None)} highly correlated pairs.",
    )
    cleaning = digest.get("cleaning")
    _set(
        "cleaning", "cleaning",
        f"Cleaning applied {_fmt(cleaning.get('n_steps') if cleaning else None)} steps, removing "
        f"{_fmt(cleaning.get('missing_cells_removed') if cleaning else None)} missing cells.",
    )
    visualization = digest.get("visualization")
    _set(
        "visualization", "visualization",
        f"{_fmt(visualization.get('n_charts') if visualization else None)} charts were generated.",
    )
    recommendation = digest.get("recommendation")
    _set(
        "recommendation", "recommendation",
        f"{_fmt(recommendation.get('n_candidates') if recommendation else None)} candidate models were "
        f"recommended, targeting {_fmt(recommendation.get('primary_metric') if recommendation else None)}.",
    )
    training = digest.get("training")
    _set(
        "training", "training",
        f"{_fmt(training.get('n_candidates') if training else None)} candidates were trained; "
        f"the best was {_fmt(training.get('best_model') if training else None)}.",
    )
    explainability = digest.get("explainability")
    _set(
        "explainability", "explainability",
        f"The top features were "
        f"{', '.join(explainability.get('top_feature_names', [])) if explainability else '—'}.",
    )
    whatif = digest.get("whatif")
    _set(
        "whatif", "whatif",
        f"{_fmt(whatif.get('n_experiments') if whatif else None)} what-if experiments were run.",
    )
    return intros


def _template_conclusion(digest: dict) -> str:
    missing = digest.get("modules_missing", [])
    if not missing:
        return "All modules ran to completion. Results should still be validated before production use."
    return (
        "This report covers the modules that have run so far; "
        f"the following have not been run: {', '.join(missing)}."
    )


def _default_narration(digest: dict) -> ReportNarration:
    return ReportNarration(
        executive_summary=_template_executive_summary(digest),
        section_intros=_template_section_intros(digest),
        conclusion=_template_conclusion(digest),
    )


def assemble_markdown(sections: list[ReportSection], prose: ReportNarration) -> str:
    """Weave the LLM intros between the deterministic section bodies, in order."""
    blocks: list[str] = []
    for section in sections:
        block_parts = [f"## {section.title}"]
        if section.status == "not_run":
            block_parts.append("_This module has not been run for this dataset._")
        intro = prose.section_intros.get(section.key, "").strip()
        if intro:
            block_parts.append(intro)
        body = section.body_markdown.strip()
        if body:
            block_parts.append(body)
        blocks.append("\n\n".join(block_parts))
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
REPORT_SYSTEM = (
    "You are writing prose for a data-science project report. You are given a "
    "compact digest of facts that have already been computed deterministically "
    "(row/column counts, model names, metrics, feature names). Write only "
    "prose that restates or lightly interprets those facts - never invent a "
    "number, name, or claim that is not present in the digest. Return an "
    "executive_summary (2-4 sentences), one short section_intros paragraph per "
    "listed section key (omit or leave empty for a section whose module is in "
    "modules_missing), and a conclusion (2-4 sentences) noting limitations for "
    "any missing modules."
)
_REPORT_TOOL_DESCRIPTION = (
    "Submit the report prose: executive_summary, section_intros (one per section "
    "key), and conclusion."
)


def run_report(dataset_id: str, client: Any | None = None) -> FinalReport:
    """Gather every sidecar, render deterministic sections, and add one narration pass."""
    artifacts = {k: get_artifact(dataset_id, k) for k in UPSTREAM_KINDS}

    cleaning = artifacts["cleaning"]
    source_dataset_id = (
        (cleaning.get("cleaned_dataset_id") or dataset_id) if cleaning else dataset_id
    )

    deterministic_sections = [
        render_dataset_overview(
            artifacts["understanding"], cleaning,
            dataset_id=dataset_id, source_dataset_id=source_dataset_id,
        ),
        render_understanding_section(artifacts["understanding"]),
        render_cleaning_section(cleaning),
        render_visualization_section(artifacts["visualization"]),
        render_recommendation_section(artifacts["recommendation"]),
        render_training_section(artifacts["training"]),
        render_explainability_section(artifacts["explainability"]),
        render_whatif_section(artifacts["whatif"]),
    ]

    digest = build_report_digest(artifacts)
    default = _default_narration(digest)

    if client is None:
        prose = default
    else:
        try:
            prose = narrate(
                client,
                system=REPORT_SYSTEM,
                user=json.dumps(digest, indent=2, default=str),
                schema=ReportNarration,
                tool_name="submit_report_prose",
                tool_description=_REPORT_TOOL_DESCRIPTION,
                default=default,
                max_tokens=2048,
            )
        except Exception:  # noqa: BLE001 - narration is never fatal
            prose = default

    exec_section = ReportSection(
        key="executive_summary", title="Executive Summary",
        body_markdown=prose.executive_summary, source_module="report", status="ok",
    )
    conclusion_section = ReportSection(
        key="conclusion", title="Conclusion & Limitations",
        body_markdown=prose.conclusion, source_module="report", status="ok",
    )
    sections = [exec_section, *deterministic_sections, conclusion_section]

    generated_at = datetime.now(timezone.utc).isoformat()
    header = f"# AgentDS Final Report — {dataset_id}\n\n_Generated {generated_at} (UTC)_\n\n"
    markdown = header + assemble_markdown(sections, prose)

    modules_present = digest["modules_present"]
    modules_missing = digest["modules_missing"]

    return FinalReport(
        dataset_id=dataset_id,
        generated_at=generated_at,
        source_dataset_id=source_dataset_id,
        modules_present=modules_present,
        modules_missing=modules_missing,
        sections=sections,
        executive_summary=prose.executive_summary,
        markdown=markdown,
    )
