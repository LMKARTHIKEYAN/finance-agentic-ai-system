import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.finance.forecast_agent import ForecastAgent


def main():
    orders_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "operations",
        "sample_orders.csv",
    )

    orders_data = pd.read_csv(orders_path)

    operations_agent = OperationsAnalysisAgent()
    forecast_agent = ForecastAgent()

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
        forecast_periods=3,
    )

    print("\nForecast Result")
    print(forecast_result)

    print("\nFuture Forecast")

    for row in forecast_result.forecast_summary:
        print(row)


if __name__ == "__main__":
    main()