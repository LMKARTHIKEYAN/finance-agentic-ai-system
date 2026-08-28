from __future__ import annotations

import pandas as pd
import pytest

from src.autonomous.tools.action_tracker_tool import prepare_management_action
from src.autonomous.tools.anomaly_detection_tool import detect_anomalies
from src.autonomous.tools.category_profitability_tool import analyze_category_profitability
from src.autonomous.tools.customer_route_tool import analyze_customers_and_routes
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.drilldown_tool import analyze_drilldown
from src.autonomous.tools.driver_forecast_tool import forecast_from_drivers
from src.autonomous.tools.forecast_accuracy_tool import analyze_forecast_accuracy
from src.autonomous.tools.period_comparison_tool import compare_periods
from src.autonomous.tools.profitability_alert_tool import detect_profitability_alerts
from src.autonomous.tools.root_cause_tool import identify_operational_drivers
from src.autonomous.tools.scenario_analysis_tool import analyze_scenario
from src.autonomous.tools.trend_analysis_tool import analyze_trends
from src.autonomous.goal_builder import GoalBuilder


def _context() -> FinanceDataContext:
    rows = []
    for month in range(1, 5):
        for day in range(1, 6):
            rows.append({
                "order_date": f"2026-{month:02d}-{day:02d}",
                "order_status": "Cancelled" if day == 5 else "Completed",
                "commission_amount": 100 + month * 10 + day,
                "incentive": 5.0, "goodwill": 2.0, "dry_run": 1.0, "surge": 3.0,
                "vehicle_category": "2W" if day % 2 else "3W",
                "pickup_cluster": "North" if day < 4 else "South",
                "drop_cluster": "East" if day % 2 else "West",
            })
    return FinanceDataContext(
        operations_data=pd.DataFrame(rows),
        corporate_expenses_data=pd.DataFrame([{"sales_marketing": 100.0, "other_opex": 50.0}]),
    )


def test_all_read_only_analysis_tools_return_structured_results():
    context = _context()
    assert analyze_trends(finance_context=context, frequency="monthly").status == "completed"
    assert compare_periods(finance_context=context, current_start="2026-03-01", current_end="2026-04-30", comparison_start="2026-01-01", comparison_end="2026-02-28").payload["metrics"]
    assert analyze_drilldown(finance_context=context, dimension="pickup_cluster").payload["rows"]
    assert identify_operational_drivers(finance_context=context).payload["drivers"]
    assert detect_anomalies(finance_context=context).status == "completed"
    assert analyze_category_profitability(finance_context=context).payload["rows"]
    customer_route = analyze_customers_and_routes(finance_context=context).payload
    assert customer_route["customer_analysis_available"] is False
    assert customer_route["routes"]
    route_only = analyze_customers_and_routes(finance_context=context, analysis_scope="route").payload
    assert route_only["customer_data_requirement"] is None
    assert route_only["route_summary"]["route_count"] == len(route_only["routes"])
    assert analyze_scenario(finance_context=context, order_change_percentage=-10).payload["scenario"]
    assert forecast_from_drivers(finance_context=context).payload["forecast"]
    alert_result = detect_profitability_alerts(finance_context=context)
    assert alert_result.status == "completed"
    assert alert_result.payload["monitoring_summary"]["categories_evaluated"] == 2


def test_forecast_accuracy_and_action_draft_contracts():
    accuracy = analyze_forecast_accuracy(
        actuals=[{"period": "2026-01", "actual_revenue": 100.0}],
        forecasts=[{"period": "2026-01", "forecast_revenue": 90.0}],
    )
    assert accuracy.payload["summary"]["mape"] == 10.0
    action = prepare_management_action(
        issue="High cancellation", recommended_action="Review clusters",
        owner="Operations", due_date="2026-09-15",
    )
    assert action.payload["action"]["status"] == "draft"
    assert action.payload["action"]["requires_human_approval"] is True


@pytest.mark.parametrize(
    ("question", "criterion"),
    [
        ("Show daily trend for July 2026", "trend_analysis"),
        ("Compare July 2026 versus previous month", "period_comparison"),
        ("Show breakdown by pickup cluster for July 2026", "drilldown_analysis"),
        ("Explain why orders declined in July 2026", "root_cause"),
        ("Show KPI anomalies for July 2026", "anomaly_detection"),
        ("Show category profitability for July 2026", "category_profitability"),
        ("Show route profitability analysis for July 2026", "customer_route_analysis"),
        ("Show forecast accuracy and MAPE for July 2026", "forecast_accuracy"),
        ("What if orders fall by 10% in July 2026", "scenario_analysis"),
        ("Show driver-based forecast from July 2026", "driver_forecast"),
        ("Show profitability alerts for July 2026", "profitability_alerts"),
        ("Create management action tracker for July 2026", "management_action"),
    ],
)
def test_goal_builder_routes_new_analysis_tools(question, criterion):
    keys = {item.key for item in GoalBuilder().build(question).criteria}
    assert criterion in keys
