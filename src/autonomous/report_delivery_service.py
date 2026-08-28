"""Approval-gated generation and delivery of autonomous finance reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

from src.autonomous.release_controller import ReleaseController
from src.autonomous.tools.reporting_tools import generate_pdf, send_report_email
from src.services.report_execution_service import ReportExecutionService


@dataclass(frozen=True)
class DeliveryResult:
    report_type: str
    period_key: str
    pdf_path: str
    email_status: str
    release_reason: str


class AutonomousReportDeliveryService:
    """Build, validate, release, and audit one scheduled report."""

    def __init__(
        self,
        *,
        report_builder: Callable[..., Any],
        repository: Any,
        pdf_tool: Any,
        email_tool: Any,
        release_controller: ReleaseController,
        execution_service: ReportExecutionService,
        output_directory: str | Path = "output/pdf",
    ) -> None:
        self.report_builder = report_builder
        self.repository = repository
        self.pdf_tool = pdf_tool
        self.email_tool = email_tool
        self.release_controller = release_controller
        self.execution_service = execution_service
        self.output_directory = Path(output_directory)

    def deliver(
        self,
        *,
        as_of: date,
        recipients: Iterable[str],
        human_approved: bool,
    ) -> DeliveryResult:
        report = self.report_builder(self.repository, as_of=as_of)
        execution = self.execution_service.start(report.report_type, report.period_key)
        pdf_path = self.output_directory / f"{report.report_type}_finance_report_{report.period_key}.pdf"
        try:
            generated = generate_pdf(pdf_tool=self.pdf_tool, report=report, output_path=pdf_path)
            decision = self.release_controller.evaluate(
                recipients=recipients,
                validated=bool(report.validated),
                pdf_path=generated["pdf_path"],
                human_approved=human_approved,
            )
            if not decision.approved:
                self.execution_service.complete(execution.execution_id, "NOT_SENT")
                return DeliveryResult(
                    report.report_type,
                    report.period_key,
                    generated["pdf_path"],
                    "NOT_SENT",
                    decision.reason,
                )
            result = send_report_email(
                email_tool=self.email_tool,
                recipients=decision.recipients,
                subject=report.title,
                body=report.markdown,
                pdf_path=generated["pdf_path"],
            )
            self.execution_service.complete(execution.execution_id, result["email_status"])
            return DeliveryResult(
                report.report_type,
                report.period_key,
                generated["pdf_path"],
                result["email_status"],
                decision.reason,
            )
        except Exception as exc:
            self.execution_service.fail(execution.execution_id, str(exc))
            raise
