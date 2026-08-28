"""Robust daily KPI anomaly detection using median absolute deviation."""

from __future__ import annotations

import numpy as np

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


def detect_anomalies(*, finance_context: FinanceDataContext, threshold: float = 3.5) -> ToolResult:
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    frame = prepared_orders(finance_context)
    frame["date"] = frame["order_date"].dt.normalize()
    daily = aggregate(frame, ["date"])
    metrics = ("total_orders", "revenue", "aov", "cancellation_percentage", "direct_cost", "gp_percentage")
    anomalies = []
    for metric in metrics:
        values = daily[metric].astype(float)
        median = float(values.median())
        mad = float(np.median(np.abs(values - median)))
        scores = 0.6745 * (values - median) / mad if mad else values * 0
        for index in daily.index[np.abs(scores) > threshold]:
            anomalies.append({"date": str(daily.loc[index, "date"]), "metric": metric, "value": round(float(values.loc[index]), 2), "robust_z_score": round(float(scores.loc[index]), 2), "direction": "high" if scores.loc[index] > 0 else "low"})
    return ToolResult(call_id="anomaly-detection", tool_name="detect_anomalies", status="completed", payload={"method": "median_absolute_deviation", "threshold": threshold, "anomaly_count": len(anomalies), "anomalies": anomalies, "daily_kpis": safe_records(daily), "revenue_definition": "commission_amount"})
