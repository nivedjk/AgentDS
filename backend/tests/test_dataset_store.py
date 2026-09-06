"""Tests for the JSON report sidecars in dataset_store."""

from pathlib import Path

import pytest

from app.storage.dataset_store import (
    artifact_path,
    get_artifact,
    get_cleaning_report,
    get_data_dir,
    get_dataset_path,
    get_report,
    list_datasets,
    model_dir,
    save_artifact,
    save_cleaning_report,
    save_dataset,
    save_report,
)


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))


def test_save_and_get_report_round_trips():
    payload = {"dataset_id": "ds-1", "problem_type": "classification", "columns": [1, 2, 3]}
    save_report("ds-1", payload)
    assert get_report("ds-1") == payload


def test_get_report_returns_none_when_absent():
    assert get_report("never-saved") is None


def test_save_and_get_cleaning_report_round_trips():
    payload = {"dataset_id": "ds-2", "steps": [], "summary": ["did nothing"]}
    save_cleaning_report("ds-2", payload)
    assert get_cleaning_report("ds-2") == payload


def test_get_cleaning_report_returns_none_when_absent():
    assert get_cleaning_report("never-saved") is None


def test_report_and_cleaning_report_use_separate_files():
    save_report("ds-3", {"kind": "understanding"})
    save_cleaning_report("ds-3", {"kind": "cleaning"})
    assert get_report("ds-3") == {"kind": "understanding"}
    assert get_cleaning_report("ds-3") == {"kind": "cleaning"}


def get_data_dir_for_test():
    d = get_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_artifact_path_shape():
    assert artifact_path("ds", "visualization").name == "ds.visualization.report.json"


def test_save_and_get_artifact_round_trips():
    payload = {"dataset_id": "ds-v", "charts": [], "narrative": "n"}
    save_artifact("ds-v", "visualization", payload)
    assert get_artifact("ds-v", "visualization") == payload


def test_get_artifact_returns_none_when_absent():
    assert get_artifact("never", "visualization") is None


def test_named_shims_delegate_to_generic_helpers():
    save_report("ds-1", {"kind": "understanding"})
    save_cleaning_report("ds-1", {"kind": "cleaning"})
    assert get_artifact("ds-1", "understanding") == {"kind": "understanding"}
    assert get_artifact("ds-1", "cleaning") == {"kind": "cleaning"}
    save_artifact("ds-2", "understanding", {"x": 1})
    assert get_report("ds-2") == {"x": 1}
    save_artifact("ds-2", "cleaning", {"y": 2})
    assert get_cleaning_report("ds-2") == {"y": 2}


def test_model_dir_creates_and_returns_directory(tmp_path):
    p = model_dir("ds-m")
    assert p.is_dir()
    assert p.name == "ds-m"
    assert p.parent.name == "models"


def test_get_dataset_path_ignores_report_sidecar():
    save_artifact("orphan", "understanding", {"a": 1})   # only a sidecar, no CSV
    with pytest.raises(FileNotFoundError):
        get_dataset_path("orphan")


def test_get_dataset_path_ignores_joblib():
    (get_data_dir_for_test() / "m1.joblib").write_bytes(b"x")  # helper above
    with pytest.raises(FileNotFoundError):
        get_dataset_path("m1")


def test_get_dataset_path_still_finds_csv_beside_a_sidecar():
    ds_id = save_dataset("people.csv", b"a,b\n1,2\n")
    save_artifact(ds_id, "understanding", {"a": 1})
    assert get_dataset_path(ds_id).suffix == ".csv"


def test_save_dataset_writes_a_meta_sidecar_with_filename_and_timestamp():
    ds_id = save_dataset("orders.csv", b"a,b\n1,2\n")
    meta_path = get_data_dir() / f"{ds_id}.meta.json"
    assert meta_path.is_file()
    import json as _json

    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["filename"] == "orders.csv"
    assert meta["uploaded_at"]


def test_list_datasets_is_empty_when_no_data_dir_yet():
    assert list_datasets() == []


def test_list_datasets_falls_back_when_meta_sidecar_is_missing():
    d = get_data_dir_for_test()
    (d / "legacy-id.csv").write_bytes(b"a,b\n1,2\n")  # no .meta.json beside it

    entries = list_datasets()
    assert entries == [
        {"dataset_id": "legacy-id", "filename": "legacy-id.csv", "uploaded_at": None}
    ]


def test_list_datasets_excludes_sidecars_and_models():
    ds_id = save_dataset("people.csv", b"a,b\n1,2\n")
    save_artifact(ds_id, "understanding", {"a": 1})
    model_dir(ds_id)  # creates data/uploads/models/<id>/, must not be listed

    entries = list_datasets()
    assert [e["dataset_id"] for e in entries] == [ds_id]
