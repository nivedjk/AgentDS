"""Model Recommendation Agent endpoint.

POST /datasets/{dataset_id}/recommend - deterministically profile the (cleaned)
dataset, then make ONE LLM reasoning call to pick 3-4 candidate models from a
fixed catalog, a rationale for each, the preprocessing needed, and a primary
metric. No training, no metric numbers. Provider is AGENTDS_LLM_PROVIDER
(default "ollama"); 503s if that provider isn't reachable/configured right
now — see app.agents._llm_client.check_llm_available.

GET /datasets/{dataset_id}/recommend - return the cached RecommendationReport
(404 if the stage has never been run). No LLM required; lets the UI show a
prior result on page load instead of recomputing.
"""

from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available
from app.agents.data_understanding import DataUnderstandingAgent
from app.agents.recommendation import (
    RecommendationReport,
    _resolve_source_df,
    build_modeling_profile,
    recommend_models,
)
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import get_artifact, get_report, save_artifact, save_report

router = APIRouter(prefix="/datasets", tags=["recommendation"])


@router.post("/{dataset_id}/recommend", response_model=RecommendationReport)
def recommend_endpoint(dataset_id: str) -> RecommendationReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        source_df, source_id, used_cleaned = _resolve_source_df(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No dataset found for id {dataset_id!r}.")

    understanding = get_report(dataset_id)
    if understanding is None:
        original_df = load_dataframe(dataset_id)
        understanding = DataUnderstandingAgent().run(original_df, dataset_id).model_dump()
        save_report(dataset_id, understanding)

    problem_type = understanding.get("problem_type")
    if problem_type not in ("classification", "regression"):
        raise HTTPException(
            status_code=422,
            detail=f"Module 1 problem type is {problem_type!r}; cannot recommend models.",
        )

    target = understanding.get("target_candidate")
    if target not in source_df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column {target!r} is not present in the dataset to be modeled.",
        )

    profile = build_modeling_profile(source_df, understanding)
    report = recommend_models(profile, problem_type)
    report.dataset_id = dataset_id
    report.source_dataset_id = source_id
    report.used_cleaned_dataset = used_cleaned

    save_artifact(dataset_id, "recommendation", report.model_dump())
    return report


@router.get("/{dataset_id}/recommend", response_model=RecommendationReport)
def get_recommendation_endpoint(dataset_id: str) -> RecommendationReport:
    cached = get_artifact(dataset_id, "recommendation")
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Recommendation has not been run for this dataset. POST to build it.",
        )
    return RecommendationReport(**cached)
