"""Tests for GET /datasets/{id}/artifacts."""

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def status_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDS_DATA_DIR", str(tmp_path / "uploads"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # endpoint needs no key
    from app.main import app

    tc = TestClient(app)
    csv = b"a,b,target\n1,2,0\n3,4,1\n"
    ds = tc.post("/datasets/upload",
                 files={"file": ("x.csv", io.BytesIO(csv), "text/csv")}).json()["dataset_id"]
    return tc, ds


def test_artifacts_all_missing_after_upload(status_client):
    tc, ds = status_client
    resp = tc.get(f"/datasets/{ds}/artifacts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["present"] == []
    assert body["missing"] == ["understanding", "cleaning", "visualization",
                               "recommendation", "training", "explainability", "whatif"]


def test_artifacts_partitions_present_and_missing(status_client):
    tc, ds = status_client
    from app.storage.dataset_store import save_artifact
    save_artifact(ds, "understanding", {"n_rows": 2})
    save_artifact(ds, "cleaning", {"summary": []})
    body = tc.get(f"/datasets/{ds}/artifacts").json()
    assert body["present"] == ["understanding", "cleaning"]
    assert "understanding" not in body["missing"]
    assert "training" in body["missing"]


def test_artifacts_unknown_dataset_404(status_client):
    tc, _ = status_client
    assert tc.get("/datasets/does-not-exist/artifacts").status_code == 404
