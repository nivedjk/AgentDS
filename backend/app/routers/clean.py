"""Cleaning Agent endpoint.

POST /datasets/{dataset_id}/clean — agentic cleaning driven by an LLM
tool-use loop. Seeds the agent with the cached Module 1 report (running
Module 1 first if none is cached), runs the loop, writes the cleaned
dataset under a new id, and returns a CleaningReport. Provider is
AGENTDS_LLM_PROVIDER (default "ollama"); 503s if that provider isn't
reachable/configured right now — see app.agents._llm_client.check_llm_available.
"""

from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available
from app.agents.cleaning import CleaningAgent, CleaningReport
from app.agents.data_understanding import DataUnderstandingAgent
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import (
    get_cleaning_report,
    get_report,
    save_cleaning_report,
    save_dataset,
    save_report,
)

router = APIRouter(prefix="/datasets", tags=["cleaning"])


@router.get("/{dataset_id}/clean")
def get_clean_endpoint(dataset_id: str) -> dict:
    """Retrieve previously-saved cleaning report from disk, or 404."""
    report = get_cleaning_report(dataset_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No cleaning report found for id {dataset_id!r}.")
    return report


@router.post("/{dataset_id}/clean", response_model=CleaningReport)
def clean_endpoint(dataset_id: str) -> CleaningReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    df = load_dataframe(dataset_id)  # raises HTTPException(404) if missing

    context_report = get_report(dataset_id)
    if context_report is None:
        understanding = DataUnderstandingAgent().run(df, dataset_id)
        context_report = understanding.model_dump()
        save_report(dataset_id, context_report)

    agent = CleaningAgent()
    report = agent.run(df, dataset_id, context_report)

    cleaned_id = save_dataset(
        f"{dataset_id}_cleaned.csv", agent.df.to_csv(index=False).encode()
    )
    report.cleaned_dataset_id = cleaned_id
    save_cleaning_report(dataset_id, report.model_dump())
    return report
