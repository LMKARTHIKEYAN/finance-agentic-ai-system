"""Weekly KPI reporting pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.finance.kpi_agent import KPIAgent
from src.agents.reporting.commentary_agent import CommentaryAgent
from src.agents.reporting.report_agent import ReportAgent
from src.pipelines import ScheduledReport
from src.pipelines.daily import DAILY_KPIS
from src.services.reporting_calendar_service import ReportingCalendarService


def run_weekly_pipeline(repository: Any, *, as_of: date | None = None) -> ScheduledReport:
    """Generate KPIs for the latest completed Monday-to-Sunday week."""
    latest = repository.latest_order_date()
    if latest is None:
        raise ValueError("No order data is available for a weekly report.")
    period = ReportingCalendarService().weekly(latest, as_of)
    orders = repository.get_orders(period.start_date, period.end_date)
    if orders.empty:
        raise ValueError(f"No order data is available for {period.period_key}.")

    operations = OperationsAnalysisAgent().analyze(
        orders, start_date=period.start_date.isoformat(),
        end_date=period.end_date.isoformat(), group_by="week",
    )
    kpis = KPIAgent().analyze(DAILY_KPIS, operations_result=operations)
    commentary = CommentaryAgent().analyze(kpis)
    title = f"Weekly Finance KPI Report - {period.start_date:%d %b} to {period.end_date:%d %b %Y}"
    report = ReportAgent().analyze(
        commentary, operations_result=operations, kpi_result=kpis,
        report_title=title, report_type="weekly",
    )
    return ScheduledReport(
        "weekly", period.period_key, title, report.markdown_report,
        period.data_cutoff.isoformat(), results={
            "operations": operations, "kpis": kpis, "commentary": commentary,
            "report": report,
        },
    )
