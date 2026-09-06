"""End-to-end tests for the dataset upload endpoint."""

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with dataset storage redirected to a temp directory."""
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    from app.main import app

    return TestClient(app)


def test_health_check(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_accepts_csv(client):
    csv_bytes = b"name,age\nAda,36\nGrace,45\n"
    resp = client.post(
        "/datasets/upload",
        files={"file": ("people.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "people.csv"
    assert body["dataset_id"]

    # The stored file should be retrievable by its id and byte-identical.
    from app.storage.dataset_store import get_dataset_path

    stored = get_dataset_path(body["dataset_id"])
    assert stored.read_bytes() == csv_bytes


def test_upload_rejects_non_csv(client):
    resp = client.post(
        "/datasets/upload",
        files={"file": ("notes.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert resp.status_code == 400


def test_get_dataset_path_raises_for_unknown_id(client):
    from app.storage.dataset_store import get_dataset_path

    with pytest.raises(FileNotFoundError):
        get_dataset_path("does-not-exist")


def test_list_datasets_endpoint_returns_uploaded_file_newest_first(client):
    assert client.get("/datasets").json() == []

    first = client.post(
        "/datasets/upload",
        files={"file": ("a.csv", io.BytesIO(b"x\n1\n"), "text/csv")},
    ).json()
    second = client.post(
        "/datasets/upload",
        files={"file": ("b.csv", io.BytesIO(b"x\n2\n"), "text/csv")},
    ).json()

    resp = client.get("/datasets")
    assert resp.status_code == 200
    body = resp.json()
    assert [d["dataset_id"] for d in body] == [second["dataset_id"], first["dataset_id"]]
    assert body[0]["filename"] == "b.csv"
    assert body[0]["uploaded_at"]  # non-empty ISO timestamp
