"""Dataset artifact-presence status endpoint.

GET /datasets/{dataset_id}/artifacts - a tiny, standalone probe of which
module sidecars exist for a dataset. Used by the frontend project overview.
No API key, no LLM call.
"""

from fastapi import APIRouter, HTTPException

from app.agents.report import UPSTREAM_KINDS
from app.storage.dataset_store import get_artifact, get_dataset_path

router = APIRouter(prefix="/datasets", tags=["status"])


@router.get("/{dataset_id}/artifacts")
def get_artifacts_status(dataset_id: str) -> dict:
    try:
        get_dataset_path(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No dataset found for id {dataset_id!r}.")

    present = [k for k in UPSTREAM_KINDS if get_artifact(dataset_id, k) is not None]
    missing = [k for k in UPSTREAM_KINDS if k not in present]
    return {"present": present, "missing": missing}
