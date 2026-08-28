"""Deterministic FP&A what-if scenario calculations."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders
from src.autonomous.tools.data_tools import FinanceDataContext


def analyze_scenario(
    *, finance_context: FinanceDataContext, order_change_percentage: float = 0.0,
    aov_change_amount: float = 0.0, direct_cost_change_percentage: float = 0.0,
    target_gp_percentage: float | None = None,
) -> ToolResult:
    base = aggregate(prepared_orders(finance_context), []).iloc[0].to_dict()
    completed = float(base["completed_orders"]) * (1 + order_change_percentage / 100)
    aov = float(base["aov"]) + aov_change_amount
    revenue = completed * aov
    direct_cost = float(base["direct_cost"]) * (1 + order_change_percentage / 100) * (1 + direct_cost_change_percentage / 100)
    gp = revenue - direct_cost
    gp_pct = gp / revenue * 100 if revenue else 0.0
    required_revenue = direct_cost / (1 - target_gp_percentage / 100) if target_gp_percentage is not None and target_gp_percentage < 100 else None
    return ToolResult(call_id="scenario-analysis", tool_name="analyze_scenario", status="completed", payload={"base": base, "assumptions": {"order_change_percentage": order_change_percentage, "aov_change_amount": aov_change_amount, "direct_cost_change_percentage": direct_cost_change_percentage, "target_gp_percentage": target_gp_percentage}, "scenario": {"completed_orders": round(completed), "aov": round(aov, 2), "revenue": round(revenue, 2), "direct_cost": round(direct_cost, 2), "gross_profit": round(gp, 2), "gp_percentage": round(gp_pct, 2), "required_revenue_for_target_gp": round(required_revenue, 2) if required_revenue is not None else None}, "revenue_definition": "commission_amount"})
