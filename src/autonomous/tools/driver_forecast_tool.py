"""Driver-based forecast using completed orders, AOV, and direct cost per order."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


def forecast_from_drivers(*, finance_context: FinanceDataContext, periods: int = 3) -> ToolResult:
    if periods < 1 or periods > 24:
        raise ValueError("periods must be between 1 and 24.")
    frame = prepared_orders(finance_context)
    frame["period"] = frame["order_date"].dt.to_period("M").dt.start_time
    history = aggregate(frame, ["period"]).sort_values("period")
    if len(history) < 3:
        return ToolResult(call_id="driver-forecast", tool_name="forecast_from_drivers", status="completed", payload={"available": False, "reason": "At least three months of history are required.", "forecast": []})
    x = np.arange(len(history), dtype=float)
    def project(column: str) -> list[float]:
        slope, intercept = np.polyfit(x, history[column].astype(float), 1)
        return [max(0.0, float(intercept + slope * step)) for step in range(len(history), len(history) + periods)]
    orders = project("completed_orders")
    aov = project("aov")
    cost_per_order = (history["direct_cost"] / history["completed_orders"].replace(0, np.nan)).fillna(0)
    slope, intercept = np.polyfit(x, cost_per_order.astype(float), 1)
    last_period = pd.Period(str(history.iloc[-1]["period"])[:7], freq="M")
    rows = []
    for index in range(periods):
        period = (last_period + index + 1).strftime("%Y-%m")
        cost_unit = max(0.0, float(intercept + slope * (len(history) + index)))
        revenue = orders[index] * aov[index]
        cost = orders[index] * cost_unit
        rows.append({"period": period, "completed_orders": round(orders[index]), "aov": round(aov[index], 2), "revenue": round(revenue, 2), "direct_cost_per_order": round(cost_unit, 2), "direct_cost": round(cost, 2), "gross_profit": round(revenue - cost, 2), "gp_percentage": round((revenue - cost) / revenue * 100 if revenue else 0.0, 2)})
    return ToolResult(call_id="driver-forecast", tool_name="forecast_from_drivers", status="completed", payload={"available": True, "method": "linear driver projection", "historical": safe_records(history), "forecast": rows, "revenue_definition": "commission_amount"})
