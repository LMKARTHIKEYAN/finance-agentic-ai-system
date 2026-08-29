from datetime import date

import pandas as pd

from src.autonomous.langgraph_runtime.context_factory import ComplexFinanceContextFactory


class Repository:
    def __init__(self):
        self.order_period = None

    def get_orders(self, start, end):
        self.order_period = (start, end)
        return pd.DataFrame({
            "order_date": [start], "order_id": ["1"],
            "order_status": ["Completed"], "vehicle_category": ["Tata Ace"],
            "commission_amount": [100.0], "incentive": [0.0],
            "goodwill": [0.0], "dry_run": [0.0], "surge": [0.0],
        })

    def get_budget(self, start, end):
        return pd.DataFrame({"month": [start], "vehicle_category": ["Tata Ace"], "budget_revenue": [90.0]})

    def get_corporate_expenses(self, start, end):
        return pd.DataFrame({"month": [start], "other_opex": [1.0]})

    def get_budget_corporate_expenses(self, start, end):
        return pd.DataFrame({"month": [start], "other_opex": [1.0]})


def test_factory_loads_both_explicit_comparison_months():
    repository = Repository()
    result = ComplexFinanceContextFactory(repository=repository)(
        "Why did Tata Ace revenue change in July 2026 compared with June 2026?"
    )
    assert repository.order_period == (date(2026, 6, 1), date(2026, 7, 31))
    assert result["loaded_period"] == "2026-06-01 to 2026-07-31"
    assert result["revenue_definition"] == "commission_amount"
