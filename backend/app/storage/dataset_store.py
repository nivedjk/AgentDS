"""Local-disk dataset storage.

Deliberately dependency-free: no database, no object store. Files land under a
configurable data directory and are addressed by a generated UUID.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_DIR = "./data/uploads"


def get_data_dir() -> Path:
    """Resolve the upload directory.

    Read from the ``AGENTDS_DATA_DIR`` env var on every call so tests (and
    deployments) can point it elsewhere without reimporting this module.
    """
    return Path(os.environ.get("AGENTDS_DATA_DIR", DEFAULT_DATA_DIR))


def save_dataset(filename: str, content: bytes) -> str:
    """Persist ``content`` under a fresh UUID and return that id.

    The original ``filename`` is kept (alongside the upload timestamp) in a
    ``{id}.meta.json`` sidecar so ``list_datasets`` can show it later - only
    its extension is used for the stored data file's own name.
    """
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())
    suffix = Path(filename).suffix or ".csv"
    (data_dir / f"{dataset_id}{suffix}").write_bytes(content)
    (data_dir / f"{dataset_id}.meta.json").write_text(
        json.dumps(
            {
                "filename": filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return dataset_id


def list_datasets() -> list[dict]:
    """List every stored dataset (original uploads and derived ones, e.g.
    Module 2's ``{id}_cleaned.csv``), newest first.

    Each entry is ``{"dataset_id", "filename", "uploaded_at"}``. A data file
    with no ``.meta.json`` (only possible if it predates this sidecar) falls
    back to its on-disk name and a ``None`` timestamp rather than raising.
    """
    data_dir = get_data_dir()
    if not data_dir.is_dir():
        return []

    entries: list[dict] = []
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        dataset_id = path.stem
        meta_path = data_dir / f"{dataset_id}.meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            filename = meta.get("filename", path.name)
            uploaded_at = meta.get("uploaded_at")
        else:
            filename = path.name
            uploaded_at = None
        entries.append(
            {"dataset_id": dataset_id, "filename": filename, "uploaded_at": uploaded_at}
        )

    entries.sort(key=lambda e: e["uploaded_at"] or "", reverse=True)
    return entries


def get_dataset_path(dataset_id: str) -> Path:
    """Return the on-disk path for ``dataset_id`` (a .csv / .parquet data file).

    Raises:
        FileNotFoundError: if no stored data file matches the id. Report
        sidecars (``*.report.json``) and model files (``*.joblib``) are never
        returned.
    """
    data_dir = get_data_dir()

    exact = data_dir / f"{dataset_id}.csv"
    if exact.is_file():
        return exact

    for match in sorted(data_dir.glob(f"{dataset_id}*")):
        if match.is_file() and match.suffix.lower() in {".csv", ".parquet"}:
            return match

    raise FileNotFoundError(f"No dataset found for id {dataset_id!r}")


def artifact_path(dataset_id: str, kind: str) -> Path:
    """Sidecar path for a cached module report, e.g. kind='visualization'."""
    return get_data_dir() / f"{dataset_id}.{kind}.report.json"


def save_artifact(dataset_id: str, kind: str, payload: dict) -> None:
    path = artifact_path(dataset_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def get_artifact(dataset_id: str, kind: str) -> dict | None:
    path = artifact_path(dataset_id, kind)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_dir(dataset_id: str) -> Path:
    path = get_data_dir() / "models" / dataset_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_report(dataset_id: str, report: dict) -> None:
    """Cache a Module 1 DataUnderstandingReport dict (kind='understanding')."""
    save_artifact(dataset_id, "understanding", report)


def get_report(dataset_id: str) -> dict | None:
    return get_artifact(dataset_id, "understanding")


def save_cleaning_report(dataset_id: str, report: dict) -> None:
    """Cache a Module 2 CleaningReport dict (kind='cleaning')."""
    save_artifact(dataset_id, "cleaning", report)


def get_cleaning_report(dataset_id: str) -> dict | None:
    return get_artifact(dataset_id, "cleaning")
