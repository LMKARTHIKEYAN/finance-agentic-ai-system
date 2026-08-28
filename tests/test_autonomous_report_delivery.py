from datetime import date
from pathlib import Path

import pytest

from src.autonomous.release_controller import ReleaseController
from src.autonomous.report_delivery_service import AutonomousReportDeliveryService
from src.pipelines import ScheduledReport
from src.services.report_execution_service import ReportExecutionService


class FakePdfTool:
    def create(self, report, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-test")
        return path


class FakeEmailTool:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        return "SENT"


def _report_builder(repository, *, as_of):
    return ScheduledReport(
        "monthly", "2026-08", "August management report", "Validated report",
        as_of.isoformat(), validated=True,
    )


def _service(tmp_path, email):
    return AutonomousReportDeliveryService(
        report_builder=_report_builder,
        repository=object(),
        pdf_tool=FakePdfTool(),
        email_tool=email,
        release_controller=ReleaseController(["cfo@example.com"]),
        execution_service=ReportExecutionService(tmp_path / "executions.db"),
        output_directory=tmp_path / "pdf",
    )


def test_delivery_stops_without_human_approval(tmp_path):
    email = FakeEmailTool()
    result = _service(tmp_path, email).deliver(
        as_of=date(2026, 9, 1), recipients=["cfo@example.com"], human_approved=False,
    )
    assert result.email_status == "NOT_SENT"
    assert not email.calls
    assert Path(result.pdf_path).is_file()


def test_delivery_sends_after_all_release_controls_pass(tmp_path):
    email = FakeEmailTool()
    service = _service(tmp_path, email)
    result = service.deliver(
        as_of=date(2026, 9, 1), recipients=["cfo@example.com"], human_approved=True,
    )
    assert result.email_status == "SENT"
    assert len(email.calls) == 1
    with pytest.raises(ValueError, match="already delivered"):
        service.deliver(
            as_of=date(2026, 9, 1), recipients=["cfo@example.com"], human_approved=True,
        )
