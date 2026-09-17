"""Data Understanding Agent endpoints.

- POST /datasets/{dataset_id}/analyze: agentic investigation driven by an
  LLM tool-use loop. Provider is AGENTDS_LLM_PROVIDER (default "ollama");
  503s if that provider isn't reachable/configured right now — see
  app.agents._llm_client.check_llm_available.
- POST /datasets/{dataset_id}/quick-stats: deterministic, LLM-free fallback
  — no LLM provider required either way. Same fast heuristic report as
  before, for offline use or if live API access has issues during a demo.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available
from app.agents.data_understanding import (
    DataUnderstandingAgent,
    DataUnderstandingReport,
    quick_stats,
)
from app.storage.dataset_store import get_dataset_path, get_report, save_report

router = APIRouter(prefix="/datasets", tags=["data-understanding"])


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    try:
        path = get_dataset_path(dataset_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No dataset found for id {dataset_id!r}."
        )
    return pd.read_csv(path)


@router.post("/{dataset_id}/quick-stats", response_model=DataUnderstandingReport)
def quick_stats_endpoint(dataset_id: str) -> DataUnderstandingReport:
    df = load_dataframe(dataset_id)
    report = quick_stats(df, dataset_id)
    save_report(dataset_id, report.model_dump())
    return report


@router.post("/{dataset_id}/analyze", response_model=DataUnderstandingReport)
def analyze_endpoint(dataset_id: str) -> DataUnderstandingReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc} Use /quick-stats for an offline heuristic report instead.",
        )
    df = load_dataframe(dataset_id)
    agent = DataUnderstandingAgent()
    report = agent.run(df, dataset_id)
    save_report(dataset_id, report.model_dump())
    return report


@router.get("/{dataset_id}/understanding", response_model=DataUnderstandingReport)
def get_understanding_endpoint(dataset_id: str) -> DataUnderstandingReport:
    """A previously-computed report (from /analyze or /quick-stats, whichever
    ran and was cached last), or 404 if neither has ever run for this dataset.
    Lets the frontend show whatever is already cached - agentic or heuristic -
    instead of unconditionally overwriting it with a fresh /quick-stats call."""
    cached = get_report(dataset_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No understanding report found for id {dataset_id!r}.",
        )
    return DataUnderstandingReport(**cached)
