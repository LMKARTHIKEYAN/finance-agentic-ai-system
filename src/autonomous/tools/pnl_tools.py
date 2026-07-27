"""Deterministic P&L and P&L-variance tool wrapper."""

from __future__ import annotations

from typing import Any

from src.agents.finance.pnl_agent import PnlAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import (
    FinanceDataContext,
    serialize_tool_payload,
)


def generate_validated_pnl_analysis(
    context: FinanceDataContext,
    *,
    start_month: str | None = None,
    end_month: str | None = None,
    agent: Any | None = None,
) -> ToolResult:
    """Generate Actual, Budget, and variance P&L using PnlAgent."""

    if not isinstance(context, FinanceDataContext):
        raise TypeError("context must be FinanceDataContext.")

    pnl_agent = agent if agent is not None else PnlAgent()
    result = pnl_agent.analyze(
        orders_data=context.require_dataframe("operations_data"),
        corporate_expenses_data=context.require_dataframe(
            "corporate_expenses_data"
        ),
        budget_data=context.require_dataframe("budget_data"),
        budget_corporate_expenses_data=context.require_dataframe(
            "budget_corporate_expenses_data"
        ),
        start_month=start_month,
        end_month=end_month,
    )

    payload = serialize_tool_payload(result)
    payload["pnl_summary"] = payload.get("summary", {})

    return ToolResult(
        call_id="generate_validated_pnl_analysis",
        tool_name="generate_validated_pnl_analysis",
        status="completed",
        payload=payload,
    )
