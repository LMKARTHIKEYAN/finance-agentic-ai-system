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
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.scenario_agent import ScenarioAgent


def main():
    orders_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "operations",
        "sample_orders.csv",
    )

    assumptions_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "assumptions",
        "business_assumptions.csv",
    )

    orders_data = pd.read_csv(orders_path)
    assumptions_data = pd.read_csv(assumptions_path)

    operations_agent = OperationsAnalysisAgent()
    forecast_agent = ForecastAgent()
    scenario_agent = ScenarioAgent()

    actual_result = operations_agent.analyze(
        data=orders_data,
        start_date="2025-07-01",
        end_date="2026-06-30",
        group_by="month",
    )

    forecast_result = forecast_agent.analyze(
        period_summary=actual_result.period_summary,
        frequency="month",
        rolling_window=3,
        forecast_periods=6,
    )

    scenario_result = scenario_agent.analyze(
        forecast_result=forecast_result,
        assumptions=assumptions_data,
        scenario_name="Management Case",
    )

    print("\nScenario Summary")
    print(f"Scenario: {scenario_result.scenario_name}")
    print(f"Base Method: {scenario_result.base_method}")
    print(
        "Applied Assumptions: "
        f"{scenario_result.applied_assumption_count}"
    )

    print("\nAdjusted Forecast")

    for row in scenario_result.adjusted_forecast:
        print(row)

    print("\nApplied Assumptions")

    for row in scenario_result.applied_assumptions:
        print(row)

    print("\nUnapplied Assumptions")

    for row in scenario_result.unapplied_assumptions:
        print(row)


if __name__ == "__main__":
    main()