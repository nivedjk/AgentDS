"""Training & AutoML Agent endpoint (Module 5).

POST /datasets/{dataset_id}/train - deterministically fit every shortlisted
catalog model on the (cleaned, if available) dataset, rank them by a primary
metric, and persist both a training sidecar and the fitted pipelines. No LLM
provider is required: narration is a single best-effort call (provider
selected by AGENTDS_LLM_PROVIDER, same as every other module) that never
affects the response's status code — if the provider isn't available right
now, or fails mid-call, training still succeeds with an empty narrative.

GET /datasets/{dataset_id}/train - return the cached TrainingReport (404 if
the stage has never been run). No LLM required; lets the UI show a prior
result on page load instead of recomputing.
"""

from fastapi import APIRouter, HTTPException

from app.agents._llm_client import LLMUnavailableError, check_llm_available, get_client
from app.agents.data_understanding import quick_stats
from app.agents.training import TrainingError, TrainingReport, run_training
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import get_artifact, get_cleaning_report, get_report, save_artifact

router = APIRouter(prefix="/datasets", tags=["training"])


@router.post("/{dataset_id}/train", response_model=TrainingReport)
def train_endpoint(dataset_id: str) -> TrainingReport:
    original_df = load_dataframe(dataset_id)

    df = original_df
    cr = get_cleaning_report(dataset_id)
    if cr and cr.get("cleaned_dataset_id"):
        try:
            df = load_dataframe(cr["cleaned_dataset_id"])
        except HTTPException:
            df = original_df

    understanding = get_report(dataset_id)
    if understanding is None:
        understanding = quick_stats(df, dataset_id).model_dump()

    recommendation = get_artifact(dataset_id, "recommendation")

    try:
        check_llm_available()
        client = get_client()
    except LLMUnavailableError:
        client = None

    try:
        report = run_training(df, dataset_id, understanding, recommendation, client=client)
    except TrainingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    save_artifact(dataset_id, "training", report.model_dump())
    return report


@router.get("/{dataset_id}/train", response_model=TrainingReport)
def get_training_endpoint(dataset_id: str) -> TrainingReport:
    cached = get_artifact(dataset_id, "training")
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="Training has not been run for this dataset. POST to build it.",
        )
    return TrainingReport(**cached)
