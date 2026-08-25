"""Release controls and email content for scheduled finance reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.pipelines import ScheduledReport


@dataclass(frozen=True)
class Notification:
    recipients: tuple[str, ...]
    subject: str
    body: str


class NotificationAgent:
    """Allow delivery only for non-empty, validated reports."""

    def prepare(self, report: ScheduledReport, recipients: Iterable[str], *, test_mode: bool = True) -> Notification:
        if not isinstance(report, ScheduledReport):
            raise TypeError("report must be a ScheduledReport.")
        if not report.validated:
            raise ValueError("An unvalidated report cannot be emailed.")
        if not report.markdown.strip():
            raise ValueError("An empty report cannot be emailed.")
        normalized = tuple(dict.fromkeys(value.strip() for value in recipients if value.strip()))
        if not normalized:
            raise ValueError("At least one report recipient is required.")
        prefix = "TEST - " if test_mode else ""
        return Notification(normalized, prefix + report.title, report.markdown)

    def failure(self, report_type: str, period_key: str, error: Exception, recipients: Iterable[str]) -> Notification:
        normalized = tuple(dict.fromkeys(value.strip() for value in recipients if value.strip()))
        return Notification(
            normalized,
            f"FAILED - {report_type.title()} finance report - {period_key}",
            f"The automated finance report failed and was not released.\n\nError: {error}",
        )
