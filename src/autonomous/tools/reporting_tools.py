"""Approved PDF generation and email-delivery adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def generate_pdf(*, pdf_tool: Any, report: Any, output_path: str | Path) -> dict[str, Any]:
    path = pdf_tool.create(report, output_path)
    return {"pdf_path": str(path), "summary": "Validated PDF report generated."}


def send_report_email(
    *,
    email_tool: Any,
    recipients: Iterable[str],
    subject: str,
    body: str,
    pdf_path: str | Path,
) -> dict[str, Any]:
    status = email_tool.send(
        recipients=recipients,
        subject=subject,
        markdown_body=body,
        attachments=(pdf_path,),
    )
    return {"email_status": status, "summary": "Report email released."}
