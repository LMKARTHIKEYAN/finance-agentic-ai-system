"""Dimension-based operational and financial drill-down."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


ALLOWED_DIMENSIONS = {"vehicle_category", "pickup_cluster", "drop_cluster", "order_status", "weekday"}


def analyze_drilldown(*, finance_context: FinanceDataContext, dimension: str) -> ToolResult:
    if dimension not in ALLOWED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {sorted(ALLOWED_DIMENSIONS)}")
    frame = prepared_orders(finance_context)
    if dimension == "weekday":
        frame["weekday"] = frame["order_date"].dt.day_name()
    if dimension not in frame.columns:
        return ToolResult(call_id="drilldown", tool_name="analyze_drilldown", status="completed", payload={"dimension": dimension, "available": False, "reason": f"Column {dimension} is not available.", "rows": []})
    result = aggregate(frame, [dimension]).sort_values("revenue", ascending=False)
    return ToolResult(call_id="drilldown", tool_name="analyze_drilldown", status="completed", payload={"dimension": dimension, "available": True, "rows": safe_records(result), "revenue_definition": "commission_amount"})
