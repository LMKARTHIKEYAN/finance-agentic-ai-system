"""FastAPI routes for the autonomous finance service."""

from __future__ import annotations

from typing import Any
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.autonomous.hybrid_runtime_router import HybridRuntimeRouter
from src.pipelines.monthly import run_monthly_pipeline
from src.tools.pdf_report_tool import PdfReportTool


router = APIRouter(prefix="/api/v1/autonomous", tags=["autonomous-finance"])


class AutonomousAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def get_autonomous_service() -> HybridRuntimeRouter:
    from src.api.dependencies import get_hybrid_runtime_router
    return get_hybrid_runtime_router()


@router.post("/ask")
def ask_autonomous_finance(
    request: AutonomousAskRequest,
    service: HybridRuntimeRouter = Depends(get_autonomous_service),
) -> dict[str, Any]:
    try:
        result = service.run(request.question)
        response = result.response
        payload = dict(response) if isinstance(response, dict) else dict(response.__dict__)
        payload.update({
            "execution_mode": result.execution_mode,
            "runtime": result.runtime,
            "langgraph_shadow": {
                "executed": result.shadow_executed,
                "succeeded": (
                    result.shadow_executed
                    and result.shadow_error is None
                    and bool((result.shadow_result or {}).get("final_answer"))
                ),
                "error": result.shadow_error,
                "summary": _safe_shadow_summary(result.shadow_result),
            },
        })
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _safe_shadow_summary(shadow_result: Any) -> dict[str, Any] | None:
    """Expose graph audit metadata without trusted context or raw payloads."""

    if not isinstance(shadow_result, dict):
        return None
    trace = []
    selected_tools = []
    for item in shadow_result.get("execution_trace", []):
        if not isinstance(item, dict):
            continue
        safe = {
            key: item.get(key)
            for key in ("step", "node", "action", "tool_name", "evidence_id")
            if item.get(key) is not None
        }
        trace.append(safe)
        if safe.get("tool_name") and safe.get("node") == "supervisor":
            selected_tools.append(safe["tool_name"])
    validation = shadow_result.get("validation") or {}
    review = shadow_result.get("review") or {}
    return {
        "completed": bool(shadow_result.get("completed")),
        "step_count": int(shadow_result.get("step_count", 0)),
        "selected_tools": selected_tools,
        "evidence_count": len(shadow_result.get("evidence", [])),
        "validation_passed": bool(validation.get("passed")),
        "reviewer_decision": review.get("decision"),
        "final_answer": shadow_result.get("final_answer"),
        "pending_question": shadow_result.get("pending_question"),
        "approval_request": shadow_result.get("approval_request"),
        "execution_trace": trace,
    }


@router.get("/monthly-report/{year}/{month}")
def download_monthly_report(
    year: int,
    month: int,
) -> FileResponse:
    if not 2020 <= year <= 2100 or not 1 <= month <= 12:
        raise HTTPException(status_code=400, detail="Invalid report year or month.")
    from src.api.dependencies import get_reporting_repository
    if month == 12:
        as_of = date(year + 1, 1, 2)
    else:
        as_of = date(year, month + 1, 2)
    try:
        report = run_monthly_pipeline(get_reporting_repository(), as_of=as_of)
        if report.period_key != f"{year:04d}-{month:02d}":
            raise ValueError(
                f"Requested month is not available as a complete reporting period; generated {report.period_key}."
            )
        output = Path("output/pdf") / f"monthly_finance_report_{report.period_key}.pdf"
        PdfReportTool().create(report, output)
        return FileResponse(
            output,
            media_type="application/pdf",
            filename=output.name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
