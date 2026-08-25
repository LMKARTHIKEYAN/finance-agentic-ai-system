from datetime import date

import pandas as pd

from src.pipelines.weekly import run_weekly_pipeline


class Repository:
    def latest_order_date(self):
        return date(2026, 8, 22)

    def get_orders(self, start_date, end_date):
        return pd.DataFrame([
            {"order_id": "1", "order_date": "10-08-2026", "pickup_cluster": "Chennai", "order_status": "Completed", "vehicle_category": "2W", "fare": 100, "commission_amount": 10},
            {"order_id": "2", "order_date": "16-08-2026", "pickup_cluster": "Chennai", "order_status": "Completed", "vehicle_category": "2W", "fare": 120, "commission_amount": 12},
        ])


def test_weekly_report_uses_last_completed_week() -> None:
    result = run_weekly_pipeline(Repository(), as_of=date(2026, 8, 23))
    assert result.period_key == "2026-08-10_2026-08-16"
    assert "Weekly Finance KPI Report" in result.markdown
