"""Evidence-scoped operational root-cause indicators."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


def identify_operational_drivers(*, finance_context: FinanceDataContext, metric: str = "orders") -> ToolResult:
    frame = prepared_orders(finance_context)
    frame["weekday"] = frame["order_date"].dt.day_name()
    dimensions = [name for name in ("vehicle_category", "pickup_cluster", "drop_cluster", "weekday") if name in frame.columns]
    drivers = []
    sort_metric = "revenue" if metric == "revenue" else "total_orders"
    for dimension in dimensions:
        rows = aggregate(frame, [dimension]).sort_values(sort_metric)
        if not rows.empty:
            drivers.append({"dimension": dimension, "lowest": rows.iloc[0].to_dict(), "highest": rows.iloc[-1].to_dict()})
    return ToolResult(call_id="root-cause", tool_name="identify_operational_drivers", status="completed", payload={"metric": metric, "drivers": drivers, "cause_status": "Observed associations only; management evidence is required to prove causation.", "revenue_definition": "commission_amount"})
