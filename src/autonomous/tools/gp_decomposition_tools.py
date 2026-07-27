"""Deterministic Product- and Portfolio-Level GP% tool wrapper."""

from __future__ import annotations

from typing import Any

from src.agents.finance.gp_variance_agent import GrossProfitVarianceAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import (
    FinanceDataContext,
    serialize_tool_payload,
)


def calculate_validated_gp_decomposition(
    context: FinanceDataContext,
    *,
    start_month: str | None = None,
    end_month: str | None = None,
    agent: Any | None = None,
) -> ToolResult:
    """Run the approved GP% decomposition without duplicating formulas."""

    if not isinstance(context, FinanceDataContext):
        raise TypeError("context must be FinanceDataContext.")

    gp_agent = (
        agent if agent is not None else GrossProfitVarianceAgent()
    )
    result = gp_agent.analyze(
        orders_data=context.require_dataframe("operations_data"),
        budget_data=context.require_dataframe("budget_data"),
        start_month=start_month,
        end_month=end_month,
    )
    payload = serialize_tool_payload(result)

    # Make both required analysis levels explicit without changing values.
    payload["product_level"] = payload.get("category_analysis", [])
    payload["portfolio_level"] = {
        key: value
        for key, value in payload.items()
        if key not in {
            "category_analysis",
            "product_level",
            "available_months",
            "excluded_actual_categories",
            "excluded_budget_categories",
        }
    }

    return ToolResult(
        call_id="calculate_validated_gp_decomposition",
        tool_name="calculate_validated_gp_decomposition",
        status="completed",
        payload=payload,
    )
