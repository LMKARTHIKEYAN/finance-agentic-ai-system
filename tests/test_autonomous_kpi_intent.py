from datetime import date

import pandas as pd

from src.autonomous.goal_builder import GoalBuilder
from src.ui.autonomous_app import _format_kpi_display
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.finance_tools import calculate_daily_kpi_extreme


def test_explicit_kpi_request_builds_kpi_criterion_not_pnl() -> None:
    goal = GoalBuilder().build(
        "show me kpi for july 2026",
        reference_date=date(2026, 8, 28),
    )

    keys = {criterion.key for criterion in goal.criteria}
    assert "kpi_analysis" in keys
    assert "pnl_analysis" not in keys
    assert goal.reporting_scope.start_date == date(2026, 7, 1)
    assert goal.reporting_scope.end_date == date(2026, 7, 31)


def test_kpi_display_formats_finance_units() -> None:
    assert _format_kpi_display(1250, "count") == "1,250"
    assert _format_kpi_display(91.77, "percentage") == "91.77%"
    assert _format_kpi_display(533000, "currency") == "INR 533,000.00"


def test_lowest_order_day_request_uses_daily_analysis_not_monthly_kpi() -> None:
    goal = GoalBuilder().build(
        "July 2026 month which day KPI order is low",
        reference_date=date(2026, 8, 28),
    )
    keys = {criterion.key for criterion in goal.criteria}
    assert "daily_kpi_extreme" in keys
    assert "kpi_analysis" not in keys


def test_exact_date_reason_request_uses_daily_root_cause_scope() -> None:
    goal = GoalBuilder().build(
        "Why 2026-06-25 is low in orders Tata Ace, any reason?",
        reference_date=date(2026, 8, 28),
    )
    keys = {criterion.key for criterion in goal.criteria}
    assert "daily_order_root_cause" in keys
    assert "pnl_analysis" not in keys
    assert goal.reporting_scope.start_date == date(2026, 6, 25)
    assert goal.reporting_scope.end_date == date(2026, 6, 25)
    assert goal.reporting_scope.category == "Tata Ace"


def test_daily_high_revenue_low_orders_tool_always_returns_result() -> None:
    frame = pd.DataFrame(
        {
            "order_date": ["2026-06-01", "2026-06-01", "2026-06-02"],
            "order_status": ["Completed", "Cancelled", "Completed"],
            "commission_amount": [100.0, 0.0, 150.0],
        }
    )
    result = calculate_daily_kpi_extreme(
        finance_context=FinanceDataContext(operations_data=frame),
        extreme="highest",
        analysis_mode="high_revenue_low_orders",
    )
    assert result.status == "completed"
    assert result.payload["highest_revenue_day"]["date"] == "2026-06-02"
    assert result.payload["lowest_order_day"]["date"] == "2026-06-02"
