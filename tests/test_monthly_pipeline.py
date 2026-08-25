from datetime import date

import pandas as pd

from src.pipelines.monthly import run_monthly_pipeline


class Repository:
    def __init__(self):
        self.orders = pd.DataFrame([
            {"order_id": str(index), "order_date": f"15-{month:02d}-2026", "pickup_cluster": "Chennai", "vehicle_category": "2W", "order_status": "Completed", "fare": 100 + index, "commission_amount": 100 + index, "partner_payout": 50, "incentive": 5, "goodwill": 0, "dry_run": 0, "surge": 0}
            for index, month in enumerate((4, 5, 6, 7), 1)
        ])

    def latest_order_date(self): return date(2026, 8, 22)
    def get_orders(self, start_date, end_date):
        dates = pd.to_datetime(self.orders["order_date"], dayfirst=True).dt.date
        return self.orders.loc[(dates >= start_date) & (dates <= end_date)].copy()
    def get_budget(self, start_month, end_month):
        return pd.DataFrame([{"month": "2026-07", "vehicle_category": "2W", "budget_orders": 1, "budget_revenue": 105, "budget_cogs": 55}])
    def get_corporate_expenses(self, start_month, end_month):
        return pd.DataFrame([{"month": "2026-07", "sales_marketing": 5, "other_opex": 5, "depreciation": 1, "interest": 1}])
    def get_budget_corporate_expenses(self, start_month, end_month):
        return pd.DataFrame([{"month": "2026-07", "sales_marketing": 5, "other_opex": 5, "depreciation": 1, "interest": 1}])


def test_monthly_pipeline_builds_closed_month_pack() -> None:
    result = run_monthly_pipeline(Repository(), as_of=date(2026, 8, 23))
    assert result.period_key == "2026-07"
    assert result.results["gp_decomposition"].reconciliation_status == "PASS"
    assert "GP% Decomposition" in result.markdown
