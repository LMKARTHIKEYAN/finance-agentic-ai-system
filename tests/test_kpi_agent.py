import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.analytics.operations_analysis_agent import (
    OperationsAnalysisAgent,
)
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.finance_rules_agent import (
    FinanceRulesAgent,
)
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.kpi_agent import KPIAgent
from src.agents.finance.scenario_agent import ScenarioAgent
from src.agents.finance.variance_agent import (
    RevenueVarianceAgent,
)


def main():
    orders_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "operations",
        "sample_orders.csv",
    )

    budget_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "planning",
        "sample_budget.csv",
    )

    assumptions_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "assumptions",
        "business_assumptions.csv",
    )

    orders_data = pd.read_csv(orders_path)
    budget_data = pd.read_csv(budget_path)
    assumptions_data = pd.read_csv(assumptions_path)

    operations_agent = OperationsAnalysisAgent()
    budget_agent = BudgetAgent()
    variance_agent = RevenueVarianceAgent()
    forecast_agent = ForecastAgent()
    scenario_agent = ScenarioAgent()
    finance_rules_agent = FinanceRulesAgent()
    kpi_agent = KPIAgent()

    actual_result = operations_agent.analyze(
        data=orders_data,
        start_date="2026-04-01",
        end_date="2026-06-30",
        group_by="month",
    )

    budget_result = budget_agent.analyze(
        data=budget_data,
        start_month="2026-04",
        end_month="2026-06",
        group_by="month",
    )

    variance_result = variance_agent.analyze(
        actual_result=actual_result,
        budget_result=budget_result,
    )

    historical_result = operations_agent.analyze(
        data=orders_data,
        start_date="2025-07-01",
        end_date="2026-06-30",
        group_by="month",
    )

    forecast_result = forecast_agent.analyze(
        period_summary=historical_result.period_summary,
        frequency="month",
        rolling_window=3,
        forecast_periods=6,
    )

    scenario_result = scenario_agent.analyze(
        forecast_result=forecast_result,
        assumptions=assumptions_data,
        scenario_name="Management Case",
    )

    finance_rules_result = finance_rules_agent.analyze(
        operations_result=actual_result,
        budget_result=budget_result,
        revenue_variance_result=variance_result,
        forecast_result=forecast_result,
        scenario_result=scenario_result,
    )

    print("\nOverall KPIs")

    overall_result = kpi_agent.analyze(
        requested_kpis=[
            "Total Revenue",
            "Fulfillment",
            "Revenue Variance",
            "Forecast Revenue",
            "Finance Rules Status",
        ],
        operations_result=actual_result,
        revenue_variance_result=variance_result,
        forecast_result=forecast_result,
        finance_rules_result=finance_rules_result,
        forecast_period="2026-07",
    )

    for kpi in overall_result.selected_kpis:
        print(kpi)

    print("\n2W Vehicle KPIs")

    vehicle_result = kpi_agent.analyze(
        requested_kpis=[
            "Total Orders",
            "Completed Orders",
            "Cancelled Orders",
            "Fulfillment",
            "Cancellation",
            "Total Revenue",
            "AOV",
        ],
        operations_result=actual_result,
        dimension="vehicle_category",
        dimension_value="2W",
    )

    for kpi in vehicle_result.selected_kpis:
        print(kpi)

    print("\nCluster KPIs")

    print("\nAvailable Clusters")
    for row in actual_result.cluster_summary:
         print(row["pickup_cluster"])

    cluster_result = kpi_agent.analyze(
        requested_kpis=[
            "Total Orders",
            "Completed Orders",
            "Fulfillment",
            "Total Revenue",
            "AOV",
        ],
        operations_result=actual_result,
        dimension="pickup_cluster",
        dimension_value="Ambattur",
    )

    for kpi in cluster_result.selected_kpis:
        print(kpi)

    print("\nJune 2026 KPIs")

    period_result = kpi_agent.analyze(
        requested_kpis=[
            "Total Orders",
            "Completed Orders",
            "Fulfillment",
            "Total Revenue",
            "AOV",
        ],
        operations_result=actual_result,
        dimension="period",
        dimension_value="2026-06",
    )

    for kpi in period_result.selected_kpis:
        print(kpi)

    print("\nUnavailable KPIs")
    print(vehicle_result.unavailable_kpis)

    print("\nUnknown KPIs")
    print(vehicle_result.unknown_kpis)


if __name__ == "__main__":
    main()