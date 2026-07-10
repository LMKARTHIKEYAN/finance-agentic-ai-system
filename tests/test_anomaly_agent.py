import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.analytics.anomaly_agent import AnomalyAgent
from src.agents.analytics.operations_analysis_agent import (
    OperationsAnalysisAgent,
)


def main():
    orders_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "operations",
        "sample_orders.csv",
    )

    orders_data = pd.read_csv(orders_path)

    operations_agent = OperationsAnalysisAgent()

    anomaly_agent = AnomalyAgent(
        percentage_change_threshold=15.0,
        z_score_threshold=2.0,
        high_percentage_threshold=30.0,
        critical_percentage_threshold=50.0,
        high_z_score_threshold=3.0,
        critical_z_score_threshold=3.5,
    )

    operations_result = operations_agent.analyze(
        data=orders_data,
        start_date="2025-07-01",
        end_date="2026-06-30",
        group_by="month",
    )

    anomaly_result = anomaly_agent.analyze(
        operations_result=operations_result,
        levels=[
            "period",
            "vehicle_category",
            "pickup_cluster",
        ],
        metrics=[
            "total_orders",
            "completed_orders",
            "total_revenue",
            "average_order_value",
            "fulfillment_percentage",
            "cancellation_percentage",
        ],
    )

    print("\nAnomaly Summary")
    print(f"Overall Status: {anomaly_result.overall_status}")
    print(f"Levels Checked: {anomaly_result.analysis_levels}")
    print(f"Rows Checked: {anomaly_result.rows_checked}")
    print(f"Metrics Checked: {anomaly_result.metrics_checked}")
    print(f"Anomalies Found: {anomaly_result.anomaly_count}")
    print(
        "High-Priority Anomalies: "
        f"{anomaly_result.high_priority_count}"
    )

    print("\nAnomaly Findings")

    if not anomaly_result.findings:
        print("No anomalies detected.")

    for finding in anomaly_result.findings:
        print(finding)


if __name__ == "__main__":
    main()