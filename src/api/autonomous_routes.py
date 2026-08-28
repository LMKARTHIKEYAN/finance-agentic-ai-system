"""FastAPI routes for the autonomous finance service."""

from __future__ import annotations

from typing import Any
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.services.autonomous_finance_service import AutonomousFinanceService
from src.pipelines.monthly import run_monthly_pipeline
from src.tools.pdf_report_tool import PdfReportTool


router = APIRouter(prefix="/api/v1/autonomous", tags=["autonomous-finance"])


class AutonomousAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def get_autonomous_service() -> AutonomousFinanceService:
    from src.api.dependencies import get_autonomous_finance_service
    return get_autonomous_finance_service()


@router.post("/ask")
def ask_autonomous_finance(
    request: AutonomousAskRequest,
    service: AutonomousFinanceService = Depends(get_autonomous_service),
) -> dict[str, Any]:
    try:
        return service.ask(request.question).__dict__
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
