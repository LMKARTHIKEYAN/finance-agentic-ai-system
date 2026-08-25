"""Run and deliver due daily, weekly, and monthly finance reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from src.agents.reporting.notification_agent import NotificationAgent
from src.pipelines import ScheduledReport
from src.services.report_execution_service import ReportExecutionService


@dataclass(frozen=True)
class ScheduleConfig:
    weekly_weekday: int = 0
    monthly_day: int = 5
    test_mode: bool = True


class FinanceReportScheduler:
    """Coordinate pipeline execution, duplicate control, and email release."""

    def __init__(
        self, *, repository: Any, email_tool: Any,
        execution_service: ReportExecutionService, recipients: list[str],
        daily_pipeline: Callable[..., ScheduledReport],
        weekly_pipeline: Callable[..., ScheduledReport],
        monthly_pipeline: Callable[..., ScheduledReport],
        attachment_builder: Callable[[ScheduledReport], list[Any]] | None = None,
        config: ScheduleConfig | None = None,
    ) -> None:
        self.repository = repository
        self.email_tool = email_tool
        self.executions = execution_service
        self.recipients = recipients
        self.pipelines = {
            "daily": daily_pipeline, "weekly": weekly_pipeline, "monthly": monthly_pipeline,
        }
        self.config = config or ScheduleConfig()
        self.attachment_builder = attachment_builder
        self.notifications = NotificationAgent()

    def run_due(self, run_date: date | None = None) -> dict[str, str]:
        today = run_date or date.today()
        due = ["daily"]
        if today.weekday() == self.config.weekly_weekday:
            due.append("weekly")
        if today.day == self.config.monthly_day:
            due.append("monthly")
        return {report_type: self.run_one(report_type, as_of=today) for report_type in due}

    def run_one(self, report_type: str, *, as_of: date | None = None) -> str:
        if report_type not in self.pipelines:
            raise ValueError("report_type must be daily, weekly, or monthly.")
        report = self.pipelines[report_type](self.repository, as_of=as_of)
        execution = self.executions.start(report.report_type, report.period_key)
        try:
            notification = self.notifications.prepare(
                report, self.recipients, test_mode=self.config.test_mode,
            )
            attachments = (
                self.attachment_builder(report)
                if self.attachment_builder is not None else []
            )
            email_body = notification.body
            if attachments:
                email_body = (
                    f"Please find attached the {report.title}.\n\n"
                    f"Report period: {report.period_key}\n"
                    f"Data cutoff: {report.data_cutoff}\n"
                    "Revenue basis: completed-order commission_amount\n\n"
                    "This automated report was generated from validated Snowflake data."
                )
            status = self.email_tool.send(
                recipients=notification.recipients,
                subject=notification.subject,
                markdown_body=email_body,
                attachments=attachments,
            )
            self.executions.complete(execution.execution_id, status)
            return status
        except Exception as exc:
            self.executions.fail(execution.execution_id, str(exc))
            raise
