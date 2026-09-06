"""Explainability Agent endpoint (Module 6).

POST /datasets/{dataset_id}/explain - loads the best fitted pipeline from
Module 5's training report, reproduces its exact test split, and computes
SHAP-based global feature importance + per-prediction explanations. One
narration call at the end for prose only. Follows the Module 3/4 gating
pattern (503 if the LLM provider isn't available), not Module 5's
no-LLM-required one. Provider is AGENTDS_LLM_PROVIDER (default "ollama") —
see app.agents._llm_client.check_llm_available.

GET /datasets/{dataset_id}/explain - return the cached ExplainabilityReport
(404 if the stage has never been run). No LLM required; lets the UI show a
prior result on page load instead of recomputing.
"""

from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available, get_client
from app.agents.explainability import (
    ExplainabilityError,
    ExplainabilityReport,
    run_explainability,
)
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import get_artifact, get_cleaning_report, save_artifact

router = APIRouter(prefix="/datasets", tags=["explainability"])


@router.post("/{dataset_id}/explain", response_model=ExplainabilityReport)
def explain_endpoint(dataset_id: str) -> ExplainabilityReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    original_df = load_dataframe(dataset_id)  # raises HTTPException(404) if missing

    df = original_df
    cr = get_cleaning_report(dataset_id)
    if cr and cr.get("cleaned_dataset_id"):
        try:
            df = load_dataframe(cr["cleaned_dataset_id"])
        except HTTPException:
            df = original_df

    client = get_client()

    try:
        report = run_explainability(df, dataset_id, client=client)
    except ExplainabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    save_artifact(dataset_id, "explainability", report.model_dump())
    return report


@router.get("/{dataset_id}/explain", response_model=ExplainabilityReport)
def get_explainability_endpoint(dataset_id: str) -> ExplainabilityReport:
    cached = get_artifact(dataset_id, "explainability")
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Explainability has not been run for this dataset. POST to build it.",
        )
    return ExplainabilityReport(**cached)
