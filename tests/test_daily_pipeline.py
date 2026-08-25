from datetime import date

import pandas as pd

from src.pipelines.daily import run_daily_pipeline


class Repository:
    def latest_order_date(self):
        return date(2026, 8, 22)

    def get_orders(self, start_date, end_date):
        return pd.DataFrame([
            {"order_id": "1", "order_date": "22-08-2026", "pickup_cluster": "Chennai", "order_status": "Completed", "vehicle_category": "2W", "fare": 100, "commission_amount": 10},
            {"order_id": "2", "order_date": "22-08-2026", "pickup_cluster": "Chennai", "order_status": "Cancelled", "vehicle_category": "2W", "fare": 50, "commission_amount": 0},
        ])


def test_daily_report_uses_latest_complete_day() -> None:
    result = run_daily_pipeline(Repository(), as_of=date(2026, 8, 23))
    assert result.period_key == "2026-08-22"
    assert result.validated is True
    assert result.results["operations"].total_revenue == 10.0
    assert result.results["operations"].average_order_value == 10.0
    assert "Daily Finance KPI Report" in result.markdown
