"""Deterministic KPI tool wrapper."""

from __future__ import annotations

from typing import Any

from src.agents.finance.kpi_agent import KPIAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import (
    FinanceDataContext,
    serialize_tool_payload,
)


def calculate_validated_kpis(
    context: FinanceDataContext,
    requested_kpis: list[str],
    *,
    dimension: str | None = None,
    dimension_value: str | None = None,
    forecast_period: str | None = None,
    scenario_period: str | None = None,
    agent: Any | None = None,
) -> ToolResult:
    """Select KPIs using the existing deterministic KPIAgent."""

    if not isinstance(context, FinanceDataContext):
        raise TypeError("context must be FinanceDataContext.")
    if not isinstance(requested_kpis, list) or not requested_kpis:
        raise ValueError("requested_kpis must be a non-empty list.")

    kpi_agent = agent if agent is not None else KPIAgent()
    result = kpi_agent.analyze(
        requested_kpis=list(requested_kpis),
        operations_result=context.operations_result,
        budget_result=context.budget_result,
        revenue_variance_result=context.revenue_variance_result,
        gp_portfolio_result=context.gp_portfolio_result,
        forecast_result=context.forecast_result,
        scenario_result=context.scenario_result,
        finance_rules_result=context.finance_rules_result,
        forecast_period=forecast_period,
        scenario_period=scenario_period,
        dimension=dimension,
        dimension_value=dimension_value,
    )

    return ToolResult(
        call_id="calculate_validated_kpis",
        tool_name="calculate_validated_kpis",
        status="completed",
        payload=serialize_tool_payload(result),
    )
