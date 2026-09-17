"""Tests for pipeline progress timing, stage duration tracking, truth-based progress, and SSE streaming."""

import time
from fastapi.testclient import TestClient

from app.main import app
from app.routers.pipeline import (
    _PIPELINE_STATUS,
    STAGE_KEYS,
    _complete_stage,
    _fail_stage,
    _get_initial_status,
    _recalculate_progress,
    _set_stage_status,
    _skip_stage,
    _start_stage,
    _update_stage_operation,
)
from app.storage.dataset_store import save_dataset

client = TestClient(app)


def test_initial_pipeline_status_structure():
    status = _get_initial_status("test-dataset-id")
    assert status["status"] == "idle"
    assert status["pipeline_id"] == "test-dataset-id"
    assert status["dataset_id"] == "test-dataset-id"
    assert status["current_stage"] is None
    assert status["current_stage_index"] is None
    assert status["current_operation"] is None
    assert status["current_stage_started_at"] is None
    assert status["total_stages"] == 7
    assert status["completed_stages_count"] == 0
    assert status["progress_percent"] == 0.0
    assert status["started_at"] is None
    assert status["finished_at"] is None
    assert status["total_duration_seconds"] is None
    assert len(status["stages"]) == 7
    for idx, k in enumerate(STAGE_KEYS):
        stage = status["stages"][k]
        assert stage["status"] == "pending"
        assert stage["stage_index"] == idx + 1
        assert stage["started_at"] is None
        assert stage["finished_at"] is None
        assert stage["duration_seconds"] is None
        assert stage["current_operation"] is None
        assert stage["error"] is None


def test_stage_status_transitions_and_timing():
    dataset_id = "test-timing-id"
    _PIPELINE_STATUS[dataset_id] = _get_initial_status(dataset_id)

    # 1. Start stage with descriptive sub-operation
    _start_stage(dataset_id, "understanding", "Analyzing dataset schema & statistics")
    status = _PIPELINE_STATUS[dataset_id]
    assert status["current_stage"] == "understanding"
    assert status["current_stage_index"] == 1
    assert status["current_operation"] == "Analyzing dataset schema & statistics"
    assert status["current_stage_started_at"] is not None
    assert status["completed_stages_count"] == 0
    assert status["progress_percent"] == 0.0

    stage = status["stages"]["understanding"]
    assert stage["status"] == "running"
    assert stage["current_operation"] == "Analyzing dataset schema & statistics"
    assert stage["started_at"] is not None
    assert stage["finished_at"] is None
    assert stage["duration_seconds"] is None

    # 2. Substage update
    _update_stage_operation(dataset_id, "understanding", "Generating numerical summary")
    assert _PIPELINE_STATUS[dataset_id]["current_operation"] == "Generating numerical summary"
    assert _PIPELINE_STATUS[dataset_id]["stages"]["understanding"]["current_operation"] == "Generating numerical summary"

    # Sleep slightly to have measurable duration
    time.sleep(0.05)

    # 3. Complete stage
    _complete_stage(dataset_id, "understanding", "Data understanding report generated")
    status = _PIPELINE_STATUS[dataset_id]
    assert status["current_stage"] is None
    assert status["current_stage_index"] is None
    assert status["current_operation"] is None
    assert status["current_stage_started_at"] is None
    assert status["completed_stages_count"] == 1
    assert status["progress_percent"] == 14.3

    stage = status["stages"]["understanding"]
    assert stage["status"] == "completed"
    assert stage["finished_at"] is not None
    assert stage["duration_seconds"] is not None
    assert stage["duration_seconds"] >= 0.04
    assert stage["current_operation"] == "Data understanding report generated"


def test_fail_and_skip_stage_transitions():
    dataset_id = "test-fail-skip-id"
    _PIPELINE_STATUS[dataset_id] = _get_initial_status(dataset_id)

    # Start and fail cleaning
    _start_stage(dataset_id, "cleaning", "Cleaning missing values")
    _fail_stage(dataset_id, "cleaning", "Imputation failed on column A")
    status = _PIPELINE_STATUS[dataset_id]
    assert status["stages"]["cleaning"]["status"] == "failed"
    assert status["stages"]["cleaning"]["error"] == "Imputation failed on column A"
    assert status["current_stage"] is None

    # Skip recommendation
    _skip_stage(dataset_id, "recommendation", "Problem type is text clustering")
    status = _PIPELINE_STATUS[dataset_id]
    assert status["stages"]["recommendation"]["status"] == "skipped"
    assert status["stages"]["recommendation"]["error"] == "Problem type is text clustering"
    # Skipped counts toward progress completion
    assert status["completed_stages_count"] == 1
    assert status["progress_percent"] == 14.3


def test_discrete_truth_based_progress_calculation():
    status = _get_initial_status("test-discrete-id")
    assert status["progress_percent"] == 0.0

    # Running stage should NOT increment percentage
    status["stages"]["understanding"]["status"] = "running"
    _recalculate_progress(status)
    assert status["completed_stages_count"] == 0
    assert status["progress_percent"] == 0.0

    expected_percentages = [14.3, 28.6, 42.9, 57.1, 71.4, 85.7, 100.0]
    for idx, k in enumerate(STAGE_KEYS):
        status["stages"][k]["status"] = "completed"
        _recalculate_progress(status)
        assert status["completed_stages_count"] == idx + 1
        assert status["progress_percent"] == expected_percentages[idx]


def test_pipeline_status_api_endpoint(tmp_path, monkeypatch):
    csv_bytes = b"feature_a,target\n1,0\n2,1\n3,0\n"
    ds_id = save_dataset("sample.csv", csv_bytes)

    resp = client.get(f"/datasets/{ds_id}/pipeline/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_id"] == ds_id
    assert data["total_stages"] == 7
    assert "progress_percent" in data
    assert "stages" in data
    assert "understanding" in data["stages"]
    assert data["stages"]["understanding"]["status"] == "pending"


import pytest
from app.routers.pipeline import stream_pipeline_events


@pytest.mark.anyio
async def test_pipeline_events_sse_endpoint():
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    ds_id = save_dataset("sample_sse.csv", csv_bytes)

    res = await stream_pipeline_events(ds_id)
    assert res.status_code == 200
    assert res.media_type == "text/event-stream"

    gen = res.body_iterator
    first_chunk = await gen.__anext__()
    assert "event: SNAPSHOT" in first_chunk
    assert ds_id in first_chunk
    await gen.aclose()
