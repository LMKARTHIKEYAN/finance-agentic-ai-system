import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.variance_agent import (
    RevenueVarianceAgent,
    GPProductVarianceAgent,
    GPPortfolioVarianceAgent,
)


def test_revenue_variance_agent():
    orders_path = os.path.join(PROJECT_ROOT, "data", "operations", "sample_orders.csv")
    budget_path = os.path.join(PROJECT_ROOT, "data", "planning", "sample_budget.csv")

    orders_data = pd.read_csv(orders_path)
    budget_data = pd.read_csv(budget_path)

    operations_agent = OperationsAnalysisAgent()
    budget_agent = BudgetAgent()
    revenue_variance_agent = RevenueVarianceAgent()

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

    result = revenue_variance_agent.analyze(
        actual_result=actual_result,
        budget_result=budget_result,
    )

    print("\nRevenue Variance Result")
    print(result)


def test_gp_product_variance_agent():
    sample_product_data = [
        {
            "category": "Beverages",
            "product": "Cola",
            "base_volume": 100,
            "actual_volume": 120,
            "base_price": 100,
            "actual_price": 110,
            "base_cogs_per_unit": 60,
            "actual_cogs_per_unit": 70,
        },
        {
            "category": "Snacks",
            "product": "Chips",
            "base_volume": 200,
            "actual_volume": 180,
            "base_price": 50,
            "actual_price": 55,
            "base_cogs_per_unit": 30,
            "actual_cogs_per_unit": 35,
        },
    ]

    agent = GPProductVarianceAgent()
    result = agent.analyze(sample_product_data)

    print("\nGP Product Variance Result")
    print(result)


def test_gp_portfolio_variance_agent():
    sample_product_data = [
        {
            "category": "Beverages",
            "product": "Cola",
            "base_volume": 100,
            "actual_volume": 120,
            "base_price": 100,
            "actual_price": 110,
            "base_cogs_per_unit": 60,
            "actual_cogs_per_unit": 70,
        },
        {
            "category": "Snacks",
            "product": "Chips",
            "base_volume": 200,
            "actual_volume": 180,
            "base_price": 50,
            "actual_price": 55,
            "base_cogs_per_unit": 30,
            "actual_cogs_per_unit": 35,
        },
    ]

    agent = GPPortfolioVarianceAgent()
    result = agent.analyze(sample_product_data)

    print("\nGP Portfolio Variance Result")
    print(result)


def main():
    test_revenue_variance_agent()
    test_gp_product_variance_agent()
    test_gp_portfolio_variance_agent()


if __name__ == "__main__":
    main()