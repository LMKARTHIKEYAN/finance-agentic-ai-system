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
from src.agents.finance.finance_rules_agent import FinanceRulesAgent
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.scenario_agent import ScenarioAgent
from src.agents.finance.variance_agent import RevenueVarianceAgent


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
    rules_agent = FinanceRulesAgent(
        variance_tolerance=0.05,
        maximum_forecast_growth_percentage=50.0,
    )

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

    rules_result = rules_agent.analyze(
        operations_result=actual_result,
        budget_result=budget_result,
        revenue_variance_result=variance_result,
        forecast_result=forecast_result,
        scenario_result=scenario_result,
    )

    print("\nFinance Rules Result")
    print(f"Overall Status: {rules_result.overall_status}")
    print(f"Rules Checked: {rules_result.rules_checked}")
    print(f"Passed Rules: {rules_result.passed_rules}")
    print(f"Warnings: {rules_result.warning_count}")
    print(f"Errors: {rules_result.error_count}")

    print("\nIssues")

    if not rules_result.issues:
        print("No finance-rule issues found.")

    for issue in rules_result.issues:
        print(issue)


if __name__ == "__main__":
    main()