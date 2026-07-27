"""Deterministic revenue-variance tool wrapper."""

from __future__ import annotations

from typing import Any

from src.agents.finance.variance_agent import RevenueVarianceAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import (
    FinanceDataContext,
    serialize_tool_payload,
)


def calculate_validated_revenue_variance(
    context: FinanceDataContext,
    *,
    agent: Any | None = None,
) -> ToolResult:
    """Calculate revenue variance using existing deterministic results."""

    if not isinstance(context, FinanceDataContext):
        raise TypeError("context must be FinanceDataContext.")

    variance_agent = (
        agent if agent is not None else RevenueVarianceAgent()
    )
    result = variance_agent.analyze(
        actual_result=context.require_result("operations_result"),
        budget_result=context.require_result("budget_result"),
    )

    return ToolResult(
        call_id="calculate_validated_revenue_variance",
        tool_name="calculate_validated_revenue_variance",
        status="completed",
        payload=serialize_tool_payload(result),
    )
