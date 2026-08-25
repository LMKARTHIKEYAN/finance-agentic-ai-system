from datetime import date

from src.pipelines import ScheduledReport
from src.scheduler.scheduler import FinanceReportScheduler, ScheduleConfig
from src.services.report_execution_service import ReportExecutionService


class Email:
    def send(self, **kwargs): return "SENT"


def pipeline(repository, *, as_of=None):
    return ScheduledReport("daily", "2026-08-22", "Daily Report", "Report body", "2026-08-22")


def test_scheduler_delivers_and_blocks_duplicate(tmp_path) -> None:
    scheduler = FinanceReportScheduler(
        repository=object(), email_tool=Email(),
        execution_service=ReportExecutionService(tmp_path / "history.db"),
        recipients=["manager@example.com"], daily_pipeline=pipeline,
        weekly_pipeline=pipeline, monthly_pipeline=pipeline,
        config=ScheduleConfig(test_mode=True),
    )
    assert scheduler.run_one("daily", as_of=date(2026, 8, 23)) == "SENT"
    try:
        scheduler.run_one("daily", as_of=date(2026, 8, 23))
    except ValueError as exc:
        assert "already delivered" in str(exc)
    else:
        raise AssertionError("duplicate delivery was not blocked")
