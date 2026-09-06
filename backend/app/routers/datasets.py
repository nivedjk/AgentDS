"""Dataset upload + listing endpoints.

Accepts a raw .csv file, persists it to local disk under a UUID, and hands back
an id the rest of the pipeline can use to locate it later.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.storage.dataset_store import list_datasets, save_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dataset_id = save_dataset(filename, content)
    return {"dataset_id": dataset_id, "filename": filename}


@router.get("")
def get_datasets() -> list[dict]:
    """Every stored dataset (uploads and derived ones, e.g. cleaned copies),
    newest first. Used by the frontend's dataset picker. No key required."""
    return list_datasets()
