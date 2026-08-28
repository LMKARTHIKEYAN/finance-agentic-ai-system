"""Customer concentration, retention, and route profitability analysis."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


CUSTOMER_COLUMNS = ("customer_id", "customer_name", "client_id")


def analyze_customers_and_routes(*, finance_context: FinanceDataContext, analysis_scope: str = "both") -> ToolResult:
    if analysis_scope not in {"customer", "route", "both"}:
        raise ValueError("analysis_scope must be customer, route, or both.")
    frame = prepared_orders(finance_context)
    customer_column = next((name for name in CUSTOMER_COLUMNS if name in frame.columns), None)
    customer_rows = []
    customer_reason = None
    if customer_column and analysis_scope in {"customer", "both"}:
        customer_rows = safe_records(aggregate(frame, [customer_column]).sort_values("revenue", ascending=False))
    elif analysis_scope in {"customer", "both"}:
        customer_reason = "Customer analysis requires customer_id, customer_name, or client_id in ORDERS."
    route_rows = []
    if analysis_scope in {"route", "both"} and {"pickup_cluster", "drop_cluster"} <= set(frame.columns):
        route_rows = safe_records(aggregate(frame, ["pickup_cluster", "drop_cluster"]).sort_values("gross_profit", ascending=False))
    route_summary = {}
    if route_rows:
        top = route_rows[0]
        bottom = route_rows[-1]
        highest_margin = max(route_rows, key=lambda row: float(row.get("gp_percentage") or 0))
        lowest_margin = min(route_rows, key=lambda row: float(row.get("gp_percentage") or 0))
        route_summary = {
            "route_count": len(route_rows),
            "loss_making_route_count": sum(float(row.get("gross_profit") or 0) < 0 for row in route_rows),
            "highest_gross_profit_route": top,
            "lowest_gross_profit_route": bottom,
            "highest_gp_percentage_route": highest_margin,
            "lowest_gp_percentage_route": lowest_margin,
        }
    return ToolResult(call_id="customer-route", tool_name="analyze_customers_and_routes", status="completed", payload={"analysis_scope": analysis_scope, "customer_analysis_available": customer_column is not None, "customer_data_requirement": customer_reason, "customers": customer_rows, "route_analysis_available": bool(route_rows), "route_summary": route_summary, "routes": route_rows, "profitability_basis": "gross profit after order direct costs; corporate expenses are not allocated to routes", "revenue_definition": "commission_amount"})
