"""Visualization Agent endpoint.

POST /datasets/{dataset_id}/visualize — deterministic chart builders + one
narration call. Operates on the cleaned dataset when Module 2 produced one,
else the original. Regenerates the Module 1 report if none is cached.
Provider is AGENTDS_LLM_PROVIDER (default "ollama"); 503s if that provider
isn't reachable/configured right now — see app.agents._llm_client.check_llm_available.

GET /datasets/{dataset_id}/visualize — return the cached VisualizationReport
(404 if the stage has never been run). No LLM required; lets the UI show a
prior result on page load instead of recomputing.
"""

from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available
from app.agents.data_understanding import DataUnderstandingAgent
from app.agents.visualization import (
    VisualizationReport,
    _resolve_source_df,
    run_visualization,
)
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import get_artifact, save_artifact

router = APIRouter(prefix="/datasets", tags=["visualization"])


@router.post("/{dataset_id}/visualize", response_model=VisualizationReport)
def visualize_endpoint(dataset_id: str) -> VisualizationReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    original_df = load_dataframe(dataset_id)  # raises HTTPException(404) if missing

    understanding = get_artifact(dataset_id, "understanding")
    if understanding is None:
        understanding = DataUnderstandingAgent().run(original_df, dataset_id).model_dump()
        save_artifact(dataset_id, "understanding", understanding)

    df, source = _resolve_source_df(dataset_id)
    report = run_visualization(df, dataset_id, understanding, source=source)
    save_artifact(dataset_id, "visualization", report.model_dump())
    return report


@router.get("/{dataset_id}/visualize", response_model=VisualizationReport)
def get_visualization_endpoint(dataset_id: str) -> VisualizationReport:
    cached = get_artifact(dataset_id, "visualization")
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Visualization has not been run for this dataset. POST to build it.",
        )
    return VisualizationReport(**cached)
