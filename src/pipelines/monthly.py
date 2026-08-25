"""Monthly P&L, variance, GP decomposition, and forecast pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.gp_variance_agent import GrossProfitVarianceAgent
from src.agents.finance.kpi_agent import KPIAgent
from src.agents.finance.pnl_agent import PnlAgent
from src.agents.finance.variance_agent import RevenueVarianceAgent
from src.agents.reporting.commentary_agent import CommentaryAgent
from src.agents.reporting.pnl_commentary_agent import PnlCommentaryAgent
from src.agents.reporting.report_agent import ReportAgent
from src.pipelines import ScheduledReport
from src.services.reporting_calendar_service import ReportingCalendarService


MONTHLY_KPIS = [
    "total orders", "completed orders", "cancelled orders", "total revenue",
    "average order value", "budget orders", "budget revenue", "budget aov",
    "revenue variance", "forecast orders", "forecast revenue", "forecast aov",
]


def run_monthly_pipeline(
    repository: Any, *, as_of: date | None = None,
    rolling_window: int = 3, forecast_periods: int = 3,
) -> ScheduledReport:
    """Generate the latest closed-month FP&A management pack."""
    latest = repository.latest_order_date()
    if latest is None:
        raise ValueError("No order data is available for a monthly report.")
    period = ReportingCalendarService().monthly(latest, as_of)
    month = period.end_date.strftime("%Y-%m")
    history_start = period.start_date.replace(day=1) - timedelta(days=1)
    for _ in range(rolling_window - 1):
        history_start = history_start.replace(day=1) - timedelta(days=1)
    history_start = history_start.replace(day=1)

    target_orders = repository.get_orders(period.start_date, period.end_date)
    history_orders = repository.get_orders(history_start, period.end_date)
    budget = repository.get_budget(month, month)
    expenses = repository.get_corporate_expenses(month, month)
    budget_expenses = repository.get_budget_corporate_expenses(month, month)
    for name, frame in {
        "orders": target_orders, "budget": budget, "corporate expenses": expenses,
        "budget corporate expenses": budget_expenses,
    }.items():
        if frame.empty:
            raise ValueError(f"Monthly {name} data is empty for {month}.")

    operations = OperationsAnalysisAgent().analyze(target_orders, group_by="month")
    historical = OperationsAnalysisAgent().analyze(history_orders, group_by="month")
    budget_result = BudgetAgent().analyze(budget, start_month=month, end_month=month)
    variance = RevenueVarianceAgent().analyze(operations, budget_result)
    forecast = ForecastAgent().analyze(
        historical.period_summary, frequency="month",
        rolling_window=rolling_window, forecast_periods=forecast_periods,
    )
    pnl = PnlAgent().analyze(
        target_orders, expenses, budget, budget_expenses,
        start_month=month, end_month=month,
    )
    gp = GrossProfitVarianceAgent().analyze(
        target_orders, budget, start_month=month, end_month=month,
    )
    if gp.reconciliation_status != "PASS":
        raise ValueError("GP decomposition reconciliation failed.")
    kpis = KPIAgent().analyze(
        MONTHLY_KPIS, operations_result=operations, budget_result=budget_result,
        revenue_variance_result=variance, forecast_result=forecast,
        forecast_period=forecast.forecast_summary[0]["forecast_period"],
    )
    commentary = CommentaryAgent().analyze(kpis, variance, forecast)
    pnl_commentary = PnlCommentaryAgent().analyze(pnl)
    title = f"Monthly FP&A Management Report - {period.end_date:%B %Y}"
    report = ReportAgent().analyze(
        commentary, operations_result=operations, kpi_result=kpis,
        budget_result=budget_result, forecast_result=forecast,
        variance_result=variance, pnl_result=pnl,
        pnl_commentary_result=pnl_commentary, report_title=title,
        report_type="monthly",
    )
    gp_markdown = (
        "\n\n## GP% Decomposition\n\n"
        f"- Budget GP%: {gp.budget_gp_percentage:.2f}%\n"
        f"- Actual GP%: {gp.actual_gp_percentage:.2f}%\n"
        f"- Mix effect: {gp.mix_effect_percentage_points:.2f} pp\n"
        f"- Price effect: {gp.price_effect_percentage_points:.2f} pp\n"
        f"- Cost effect: {gp.cost_effect_percentage_points:.2f} pp\n"
        f"- Reconciliation: {gp.reconciliation_status}\n"
    )
    return ScheduledReport(
        "monthly", period.period_key, title, report.markdown_report + gp_markdown,
        period.data_cutoff.isoformat(), results={
            "operations": operations, "budget": budget_result, "variance": variance,
            "forecast": forecast, "pnl": pnl, "gp_decomposition": gp,
            "kpis": kpis, "commentary": commentary, "report": report,
        },
    )
