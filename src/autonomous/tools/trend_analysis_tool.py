"""Daily, weekly, and monthly operational trend analysis."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


def analyze_trends(*, finance_context: FinanceDataContext, frequency: str = "daily") -> ToolResult:
    aliases = {"daily": "D", "weekly": "W-MON", "monthly": "M"}
    if frequency not in aliases:
        raise ValueError("frequency must be daily, weekly, or monthly.")
    frame = prepared_orders(finance_context)
    frame["period"] = frame["order_date"].dt.to_period(aliases[frequency]).dt.start_time
    trend = aggregate(frame, ["period"]).sort_values("period")
    for metric in ("total_orders", "revenue", "aov", "fulfillment_percentage", "cancellation_percentage"):
        trend[f"{metric}_change_percentage"] = trend[metric].pct_change().mul(100).round(2)
    highest_orders = trend.loc[trend["total_orders"].idxmax()]
    lowest_orders = trend.loc[trend["total_orders"].idxmin()]
    highest_revenue = trend.loc[trend["revenue"].idxmax()]
    lowest_revenue = trend.loc[trend["revenue"].idxmin()]
    first = trend.iloc[0]
    last = trend.iloc[-1]
    summary = {
        "average_orders": round(float(trend["total_orders"].mean()), 2),
        "average_revenue": round(float(trend["revenue"].mean()), 2),
        "highest_order_period": str(highest_orders["period"]),
        "highest_orders": int(highest_orders["total_orders"]),
        "lowest_order_period": str(lowest_orders["period"]),
        "lowest_orders": int(lowest_orders["total_orders"]),
        "highest_revenue_period": str(highest_revenue["period"]),
        "highest_revenue": round(float(highest_revenue["revenue"]), 2),
        "lowest_revenue_period": str(lowest_revenue["period"]),
        "lowest_revenue": round(float(lowest_revenue["revenue"]), 2),
        "first_to_last_order_change_percentage": round((float(last["total_orders"]) - float(first["total_orders"])) / float(first["total_orders"]) * 100, 2) if first["total_orders"] else None,
        "first_to_last_revenue_change_percentage": round((float(last["revenue"]) - float(first["revenue"])) / float(first["revenue"]) * 100, 2) if first["revenue"] else None,
    }
    return ToolResult(
        call_id="trend-analysis", tool_name="analyze_trends", status="completed",
        payload={"frequency": frequency, "summary": summary, "trend": safe_records(trend), "revenue_definition": "commission_amount"},
    )
