"""Chronological Multi-Agent Pipeline Orchestrator.

Runs all pipeline modules (1-7) sequentially in the background upon dataset upload
or on demand. Persists each artifact as it completes so that downstream modules
and user frontend views immediately get results with zero wait time.

Backend is the single source of truth for all pipeline state, stage lifecycles,
truth-based progress percentage, and timestamps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.agents._llm_client import check_llm_available, get_client
from app.agents.cleaning import CleaningAgent
from app.agents.data_understanding import DataUnderstandingAgent, quick_stats
from app.agents.explainability import run_explainability
from app.agents.recommendation import (
    _resolve_source_df as _resolve_recommendation_source_df,
    build_modeling_profile,
    recommend_models,
)
from app.agents.report import run_report
from app.agents.training import run_training
from app.agents.visualization import (
    _resolve_source_df as _resolve_viz_source_df,
    run_visualization,
)
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import (
    get_artifact,
    get_cleaning_report,
    get_dataset_path,
    get_report,
    save_artifact,
    save_cleaning_report,
    save_dataset,
    save_report,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["pipeline"])

STAGE_KEYS = [
    "understanding",
    "cleaning",
    "visualization",
    "recommendation",
    "training",
    "explainability",
    "report",
]

STAGE_NAMES = {
    "understanding": "Data Understanding",
    "cleaning": "Data Cleaning",
    "visualization": "Visualization",
    "recommendation": "Model Recommendation",
    "training": "Training & Evaluation",
    "explainability": "Explainability",
    "report": "Final Report",
}

STAGE_INDEX_MAP = {k: idx + 1 for idx, k in enumerate(STAGE_KEYS)}

# In-memory tracking of running / completed pipelines
_PIPELINE_STATUS: Dict[str, Dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()

# SSE event queues per dataset
_EVENT_QUEUES: Dict[str, List[asyncio.Queue]] = {}
_QUEUES_LOCK = threading.Lock()
_EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _EVENT_LOOP
    _EVENT_LOOP = loop


def _recalculate_progress(status_dict: Dict[str, Any]) -> None:
    """Truth-based stage progress percentage: (completed_count / 7) * 100."""
    completed_count = sum(
        1 for s in status_dict["stages"].values() if s.get("status") in ("completed", "skipped")
    )
    status_dict["completed_stages_count"] = completed_count
    status_dict["progress_percent"] = round((completed_count / len(STAGE_KEYS)) * 100, 1)


def _get_initial_status(dataset_id: str) -> Dict[str, Any]:
    status = {
        "pipeline_id": dataset_id,
        "dataset_id": dataset_id,
        "status": "idle",  # idle | running | completed | failed
        "current_stage": None,
        "current_stage_index": None,
        "current_operation": None,
        "current_stage_started_at": None,
        "total_stages": len(STAGE_KEYS),
        "completed_stages_count": 0,
        "progress_percent": 0.0,
        "stages": {
            k: {
                "name": STAGE_NAMES[k],
                "stage_index": idx + 1,
                "status": "pending",  # pending | running | completed | failed | skipped
                "current_operation": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
            }
            for idx, k in enumerate(STAGE_KEYS)
        },
        "error": None,
        "started_at": None,
        "finished_at": None,
        "total_duration_seconds": None,
    }
    _recalculate_progress(status)
    return status


def _notify_listeners(dataset_id: str, event_type: str, data: Dict[str, Any]) -> None:
    """Broadcast state events to SSE listeners if an event loop is running."""
    with _QUEUES_LOCK:
        queues = list(_EVENT_QUEUES.get(dataset_id, []))
    if not queues:
        return

    payload = {
        "event": event_type,
        "dataset_id": dataset_id,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for q in queues:
        try:
            if _EVENT_LOOP and _EVENT_LOOP.is_running():
                _EVENT_LOOP.call_soon_threadsafe(q.put_nowait, payload)
            else:
                q.put_nowait(payload)
        except Exception:
            pass


def _start_stage(dataset_id: str, stage_key: str, operation: Optional[str] = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with _STATUS_LOCK:
        info = _PIPELINE_STATUS.setdefault(dataset_id, _get_initial_status(dataset_id))
        info["current_stage"] = stage_key
        info["current_stage_index"] = STAGE_INDEX_MAP.get(stage_key)
        info["current_operation"] = operation
        info["current_stage_started_at"] = now_iso

        st = info["stages"][stage_key]
        st["status"] = "running"
        st["current_operation"] = operation
        st["started_at"] = now_iso
        st["finished_at"] = None
        st["duration_seconds"] = None
        st["error"] = None

        _recalculate_progress(info)
        snapshot = dict(info)
    _notify_listeners(dataset_id, "STAGE_STARTED", snapshot)


def _update_stage_operation(dataset_id: str, stage_key: str, operation: str) -> None:
    with _STATUS_LOCK:
        info = _PIPELINE_STATUS.get(dataset_id)
        if info:
            info["current_operation"] = operation
            if stage_key in info["stages"]:
                info["stages"][stage_key]["current_operation"] = operation
            snapshot = dict(info)
        else:
            snapshot = None
    if snapshot:
        _notify_listeners(dataset_id, "SUBSTAGE_UPDATE", snapshot)


def _complete_stage(dataset_id: str, stage_key: str, final_operation: Optional[str] = None) -> None:
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    with _STATUS_LOCK:
        info = _PIPELINE_STATUS.setdefault(dataset_id, _get_initial_status(dataset_id))
        st = info["stages"][stage_key]
        st["status"] = "completed"
        st["finished_at"] = now_iso
        st["current_operation"] = final_operation

        if st.get("started_at"):
            try:
                s_dt = datetime.fromisoformat(st["started_at"])
                st["duration_seconds"] = round((now_dt - s_dt).total_seconds(), 2)
            except Exception:
                pass

        if info.get("current_stage") == stage_key:
            info["current_stage"] = None
            info["current_stage_index"] = None
            info["current_operation"] = None
            info["current_stage_started_at"] = None

        _recalculate_progress(info)
        snapshot = dict(info)
    _notify_listeners(dataset_id, "STAGE_COMPLETED", snapshot)


def _fail_stage(dataset_id: str, stage_key: str, error: str) -> None:
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    with _STATUS_LOCK:
        info = _PIPELINE_STATUS.setdefault(dataset_id, _get_initial_status(dataset_id))
        st = info["stages"][stage_key]
        st["status"] = "failed"
        st["finished_at"] = now_iso
        st["error"] = error

        if st.get("started_at"):
            try:
                s_dt = datetime.fromisoformat(st["started_at"])
                st["duration_seconds"] = round((now_dt - s_dt).total_seconds(), 2)
            except Exception:
                pass

        if info.get("current_stage") == stage_key:
            info["current_stage"] = None
            info["current_stage_index"] = None
            info["current_operation"] = None
            info["current_stage_started_at"] = None

        _recalculate_progress(info)
        snapshot = dict(info)
    _notify_listeners(dataset_id, "STAGE_FAILED", snapshot)


def _skip_stage(dataset_id: str, stage_key: str, reason: str) -> None:
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    with _STATUS_LOCK:
        info = _PIPELINE_STATUS.setdefault(dataset_id, _get_initial_status(dataset_id))
        st = info["stages"][stage_key]
        st["status"] = "skipped"
        st["finished_at"] = now_iso
        st["error"] = reason

        if info.get("current_stage") == stage_key:
            info["current_stage"] = None
            info["current_stage_index"] = None
            info["current_operation"] = None
            info["current_stage_started_at"] = None

        _recalculate_progress(info)
        snapshot = dict(info)
    _notify_listeners(dataset_id, "STAGE_SKIPPED", snapshot)


# Backwards compatible alias for tests
def _set_stage_status(dataset_id: str, stage: str, status: str, error: Optional[str] = None) -> None:
    if status == "running":
        _start_stage(dataset_id, stage)
    elif status in ("done", "completed"):
        _complete_stage(dataset_id, stage)
    elif status in ("error", "failed"):
        _fail_stage(dataset_id, stage, error or "Error occurred")
    elif status == "skipped":
        _skip_stage(dataset_id, stage, error or "Skipped")


def execute_pipeline(dataset_id: str) -> None:
    """Execute all pipeline stages sequentially for dataset_id."""
    with _STATUS_LOCK:
        if dataset_id in _PIPELINE_STATUS and _PIPELINE_STATUS[dataset_id].get("status") == "running":
            return
        status_info = _get_initial_status(dataset_id)
        status_info["status"] = "running"
        status_info["started_at"] = datetime.now(timezone.utc).isoformat()
        _PIPELINE_STATUS[dataset_id] = status_info
        snapshot = dict(status_info)

    _notify_listeners(dataset_id, "PIPELINE_STARTED", snapshot)
    logger.info("Starting chronological pipeline execution for dataset %s", dataset_id)

    try:
        df = load_dataframe(dataset_id)
    except Exception as exc:
        now_dt = datetime.now(timezone.utc)
        with _STATUS_LOCK:
            _PIPELINE_STATUS[dataset_id]["status"] = "failed"
            _PIPELINE_STATUS[dataset_id]["error"] = str(exc)
            _PIPELINE_STATUS[dataset_id]["finished_at"] = now_dt.isoformat()
            if _PIPELINE_STATUS[dataset_id].get("started_at"):
                try:
                    s_dt = datetime.fromisoformat(_PIPELINE_STATUS[dataset_id]["started_at"])
                    _PIPELINE_STATUS[dataset_id]["total_duration_seconds"] = round((now_dt - s_dt).total_seconds(), 2)
                except Exception:
                    pass
            snapshot = dict(_PIPELINE_STATUS[dataset_id])
        _notify_listeners(dataset_id, "PIPELINE_FAILED", snapshot)
        return

    # Check LLM client availability once
    try:
        check_llm_available()
        llm_available = True
        client = get_client()
    except Exception:
        llm_available = False
        client = None

    # --- Stage 1: Data Understanding ---
    _start_stage(dataset_id, "understanding", "Analyzing dataset schema, distributions & statistics")
    try:
        if llm_available:
            try:
                agent = DataUnderstandingAgent()
                understanding_report = agent.run(df, dataset_id)
            except Exception as exc:
                logger.warning("LLM DataUnderstandingAgent failed, falling back to quick_stats: %s", exc)
                understanding_report = quick_stats(df, dataset_id)
        else:
            understanding_report = quick_stats(df, dataset_id)

        understanding_dict = understanding_report.model_dump()
        save_report(dataset_id, understanding_dict)
        save_artifact(dataset_id, "understanding", understanding_dict)
        _complete_stage(dataset_id, "understanding", "Data understanding report generated")
    except Exception as exc:
        logger.exception("Error in Data Understanding for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "understanding", str(exc))

    # --- Stage 2: Data Cleaning ---
    _start_stage(dataset_id, "cleaning", "Applying automated data cleaning tools and deduplication")
    try:
        understanding_dict = get_report(dataset_id) or quick_stats(df, dataset_id).model_dump()
        if llm_available:
            try:
                clean_agent = CleaningAgent()
                cleaning_report = clean_agent.run(df, dataset_id, understanding_dict)
                cleaned_id = save_dataset(
                    f"{dataset_id}_cleaned.csv", clean_agent.df.to_csv(index=False).encode()
                )
                cleaning_report.cleaned_dataset_id = cleaned_id
                save_cleaning_report(dataset_id, cleaning_report.model_dump())
                _complete_stage(dataset_id, "cleaning", "Cleaned dataset produced and saved")
            except Exception as exc:
                logger.warning("Cleaning agent failed (%s); skipping mutation", exc)
                _fail_stage(dataset_id, "cleaning", str(exc))
        else:
            _complete_stage(dataset_id, "cleaning", "No LLM key; dataset passed as-is")
    except Exception as exc:
        logger.exception("Error in Cleaning for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "cleaning", str(exc))

    # --- Stage 3: Visualization ---
    _start_stage(dataset_id, "visualization", "Computing chart distributions & generating Plotly visualizations")
    try:
        understanding_dict = get_artifact(dataset_id, "understanding") or quick_stats(df, dataset_id).model_dump()
        viz_df, source = _resolve_viz_source_df(dataset_id)
        viz_report = run_visualization(viz_df, dataset_id, understanding_dict, source=source)
        save_artifact(dataset_id, "visualization", viz_report.model_dump())
        _complete_stage(dataset_id, "visualization", f"{len(viz_report.charts)} interactive charts generated")
    except Exception as exc:
        logger.exception("Error in Visualization for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "visualization", str(exc))

    # --- Stage 4: Model Recommendation ---
    _start_stage(dataset_id, "recommendation", "Analyzing target candidate & recommending modeling strategy")
    try:
        source_df, source_id, used_cleaned = _resolve_recommendation_source_df(dataset_id)
        understanding_dict = get_report(dataset_id) or quick_stats(source_df, dataset_id).model_dump()
        problem_type = understanding_dict.get("problem_type")

        if problem_type in ("classification", "regression"):
            profile = build_modeling_profile(source_df, understanding_dict)
            rec_report = recommend_models(profile, problem_type)
            rec_report.dataset_id = dataset_id
            rec_report.source_dataset_id = source_id
            rec_report.used_cleaned_dataset = used_cleaned
            save_artifact(dataset_id, "recommendation", rec_report.model_dump())
            _complete_stage(dataset_id, "recommendation", f"{len(rec_report.candidates)} model candidates recommended")
        else:
            _skip_stage(dataset_id, "recommendation", f"Problem type is {problem_type}")
    except Exception as exc:
        logger.exception("Error in Recommendation for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "recommendation", str(exc))

    # --- Stage 5: Training & Evaluation ---
    _start_stage(dataset_id, "training", "Fitting model candidates with cross-validation & evaluation")
    try:
        cr = get_cleaning_report(dataset_id)
        train_df = df
        if cr and cr.get("cleaned_dataset_id"):
            try:
                train_df = load_dataframe(cr["cleaned_dataset_id"])
            except Exception:
                train_df = df

        understanding_dict = get_report(dataset_id) or quick_stats(train_df, dataset_id).model_dump()
        recommendation_dict = get_artifact(dataset_id, "recommendation")

        train_report = run_training(train_df, dataset_id, understanding_dict, recommendation_dict, client=client)
        save_artifact(dataset_id, "training", train_report.model_dump())
        best_name = train_report.best_model or "Candidate model"
        _complete_stage(dataset_id, "training", f"Training completed. Best model: {best_name}")
    except Exception as exc:
        logger.exception("Error in Training for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "training", str(exc))

    # --- Stage 6: Explainability ---
    _start_stage(dataset_id, "explainability", "Computing SHAP values & feature importance explanations")
    try:
        cr = get_cleaning_report(dataset_id)
        explain_df = df
        if cr and cr.get("cleaned_dataset_id"):
            try:
                explain_df = load_dataframe(cr["cleaned_dataset_id"])
            except Exception:
                explain_df = df

        if get_artifact(dataset_id, "training") is not None:
            explain_report = run_explainability(explain_df, dataset_id, client=client)
            save_artifact(dataset_id, "explainability", explain_report.model_dump())
            _complete_stage(dataset_id, "explainability", f"SHAP explanations generated for {len(explain_report.feature_importance)} features")
        else:
            _skip_stage(dataset_id, "explainability", "No training artifact")
    except Exception as exc:
        logger.exception("Error in Explainability for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "explainability", str(exc))

    # --- Stage 7: Final Report ---
    _start_stage(dataset_id, "report", "Assembling comprehensive multi-module final report")
    try:
        final_report = run_report(dataset_id, client=client)
        save_artifact(dataset_id, "final", final_report.model_dump())
        _complete_stage(dataset_id, "report", "Final comprehensive report assembled and cached")
    except Exception as exc:
        logger.exception("Error in Report for %s: %s", dataset_id, exc)
        _fail_stage(dataset_id, "report", str(exc))

    now_dt = datetime.now(timezone.utc)
    with _STATUS_LOCK:
        status_dict = _PIPELINE_STATUS.get(dataset_id, {})
        has_failed_stages = any(s["status"] == "failed" for s in status_dict.get("stages", {}).values())
        status_dict["status"] = "failed" if has_failed_stages else "completed"
        status_dict["current_stage"] = None
        status_dict["current_stage_index"] = None
        status_dict["current_operation"] = None
        status_dict["current_stage_started_at"] = None
        status_dict["finished_at"] = now_dt.isoformat()
        if status_dict.get("started_at"):
            try:
                s_dt = datetime.fromisoformat(status_dict["started_at"])
                status_dict["total_duration_seconds"] = round((now_dt - s_dt).total_seconds(), 2)
            except Exception:
                pass
        _recalculate_progress(status_dict)
        snapshot = dict(status_dict)

    event_name = "PIPELINE_FAILED" if has_failed_stages else "PIPELINE_COMPLETED"
    _notify_listeners(dataset_id, event_name, snapshot)
    logger.info("Finished chronological pipeline execution for dataset %s with status %s", dataset_id, status_dict["status"])


def start_pipeline_background(dataset_id: str) -> None:
    """Start the pipeline in a separate daemon thread."""
    thread = threading.Thread(target=execute_pipeline, args=(dataset_id,), daemon=True)
    thread.start()


@router.post("/{dataset_id}/pipeline/run")
def trigger_pipeline(dataset_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Manually trigger or restart the chronological pipeline in the background."""
    try:
        get_dataset_path(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No dataset found for id {dataset_id!r}.")

    start_pipeline_background(dataset_id)
    return get_pipeline_status(dataset_id)


@router.get("/{dataset_id}/pipeline/status")
def get_pipeline_status(dataset_id: str) -> Dict[str, Any]:
    """Get the single source of truth pipeline status and stage breakdown."""
    try:
        get_dataset_path(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No dataset found for id {dataset_id!r}.")

    with _STATUS_LOCK:
        status_info = _PIPELINE_STATUS.get(dataset_id)
        if status_info is None:
            status_info = _get_initial_status(dataset_id)
            # Sync with existing on-disk artifacts if pipeline ran previously
            for k in STAGE_KEYS:
                kind = "final" if k == "report" else k
                if get_artifact(dataset_id, kind) is not None:
                    status_info["stages"][k]["status"] = "completed"
            _recalculate_progress(status_info)
            if status_info["completed_stages_count"] == len(STAGE_KEYS):
                status_info["status"] = "completed"
            _PIPELINE_STATUS[dataset_id] = status_info

        return status_info


@router.get("/{dataset_id}/pipeline/events")
async def stream_pipeline_events(dataset_id: str) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint for live pipeline event push."""
    try:
        get_dataset_path(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No dataset found for id {dataset_id!r}.")

    # Record event loop for threadsafe queues
    try:
        loop = asyncio.get_running_loop()
        _set_event_loop(loop)
    except Exception:
        pass

    queue: asyncio.Queue = asyncio.Queue()
    with _QUEUES_LOCK:
        _EVENT_QUEUES.setdefault(dataset_id, []).append(queue)

    # Send initial snapshot immediately
    initial_snapshot = get_pipeline_status(dataset_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        # First event: SNAPSHOT
        yield f"event: SNAPSHOT\ndata: {json.dumps(initial_snapshot)}\n\n"

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    evt_type = msg.get("event", "UPDATE")
                    data_str = json.dumps(msg.get("data", {}))
                    yield f"event: {evt_type}\ndata: {data_str}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive ping
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            with _QUEUES_LOCK:
                if dataset_id in _EVENT_QUEUES and queue in _EVENT_QUEUES[dataset_id]:
                    _EVENT_QUEUES[dataset_id].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
