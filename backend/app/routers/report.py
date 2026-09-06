"""Report Agent endpoint (Module 7).

POST /datasets/{dataset_id}/report - build the final report (deterministic
section bodies + one narration call for prose) and cache it as the ``final``
sidecar. Regenerates the Module 1 sidecar first if it is missing, so a
dataset that has only been uploaded still produces a complete report.
Provider is AGENTDS_LLM_PROVIDER (default "ollama"); 503s if that provider
isn't reachable/configured right now — see app.agents._llm_client.check_llm_available.

GET /datasets/{dataset_id}/report - return the cached FinalReport. No LLM
required; works fully offline.

GET /datasets/{dataset_id}/report/pdf and .../report/docx - render the cached
FinalReport as a downloadable PDF/Word document. No LLM required (formatting
an already-built report needs no LLM call), same offline contract as GET
/report.
"""

from fastapi import APIRouter, HTTPException, Response

from app.agents._llm_client import LLMUnavailableError, check_llm_available, get_client
from app.agents.data_understanding import DataUnderstandingAgent
from app.agents.report import FinalReport, run_report
from app.agents.report_export import render_docx, render_pdf
from app.routers.analyze import load_dataframe
from app.storage.dataset_store import get_artifact, save_artifact, save_report

router = APIRouter(prefix="/datasets", tags=["report"])

_NO_REPORT_DETAIL = "No report has been generated for this dataset. POST to build one."


@router.post("/{dataset_id}/report", response_model=FinalReport)
def post_report(dataset_id: str) -> FinalReport:
    try:
        check_llm_available()
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    df = load_dataframe(dataset_id)

    if get_artifact(dataset_id, "understanding") is None:
        report = DataUnderstandingAgent().run(df, dataset_id)
        save_report(dataset_id, report.model_dump())

    final_report = run_report(dataset_id, client=get_client())
    save_artifact(dataset_id, "final", final_report.model_dump())
    return final_report


@router.get("/{dataset_id}/report", response_model=FinalReport)
def get_report_endpoint(dataset_id: str) -> FinalReport:
    cached = get_artifact(dataset_id, "final")
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="No report has been generated for this dataset. POST to build one.",
        )
    return FinalReport(**cached)


@router.get("/{dataset_id}/report/pdf")
def get_report_pdf(dataset_id: str) -> Response:
    cached = get_artifact(dataset_id, "final")
    if cached is None:
        raise HTTPException(status_code=404, detail=_NO_REPORT_DETAIL)

    pdf_bytes = render_pdf(FinalReport(**cached))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}_report.pdf"'},
    )


@router.get("/{dataset_id}/report/docx")
def get_report_docx(dataset_id: str) -> Response:
    cached = get_artifact(dataset_id, "final")
    if cached is None:
        raise HTTPException(status_code=404, detail=_NO_REPORT_DETAIL)

    docx_bytes = render_docx(FinalReport(**cached))
    return Response(
        content=docx_bytes,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}_report.docx"'},
    )
