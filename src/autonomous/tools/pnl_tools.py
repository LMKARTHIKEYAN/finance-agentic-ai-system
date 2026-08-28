"""Deterministic P&L and P&L-variance tool wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd

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
    orders_data = context.require_dataframe("operations_data")
    result = pnl_agent.analyze(
        orders_data=orders_data,
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
    payload["actual_direct_cost_breakdown"] = _actual_direct_cost_breakdown(
        orders_data
    )
    payload["direct_cost_definition"] = (
        "incentive + goodwill + dry_run + surge"
    )
    payload["revenue_definition"] = "commission_amount"

    return ToolResult(
        call_id="generate_validated_pnl_analysis",
        tool_name="generate_validated_pnl_analysis",
        status="completed",
        payload=payload,
    )


def _actual_direct_cost_breakdown(orders_data: pd.DataFrame) -> dict[str, float]:
    completed = orders_data
    if "order_status" in completed.columns:
        completed = completed.loc[
            completed["order_status"].astype(str).str.casefold().eq("completed")
        ]
    breakdown = {
        column: round(
            float(
                pd.to_numeric(
                    completed[column]
                    if column in completed.columns
                    else pd.Series(dtype="float64"),
                    errors="coerce",
                ).fillna(0).sum()
            ),
            2,
        )
        for column in ("incentive", "goodwill", "dry_run", "surge")
    }
    breakdown["total_direct_cost"] = round(sum(breakdown.values()), 2)
    return breakdown
