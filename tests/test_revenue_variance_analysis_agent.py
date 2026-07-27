"""Tests for the controlled revenue-variance specialist agent."""

from typing import Any

import pytest

from src.autonomous.agents.revenue_variance_analysis_agent import (
    RevenueVarianceAnalysisAgent,
    RevenueVarianceAnalysisAgentError,
)
from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext


def _step(**arguments: Any) -> PlanStep:
    return PlanStep(
        step_id="revenue-variance",
        capability="revenue_variance",
        arguments={
            "agent_name": "revenue_variance_agent",
            "tool_name": "calculate_validated_revenue_variance",
            **arguments,
        },
    )


def test_revenue_variance_agent_calls_assigned_tool() -> None:
    calls: list[FinanceDataContext] = []

    def tool(context: FinanceDataContext) -> ToolResult:
        calls.append(context)
        return ToolResult(
            call_id="variance-call",
            tool_name="calculate_validated_revenue_variance",
            status="completed",
            payload={"variance": 25},
        )

    context = FinanceDataContext()
    result = RevenueVarianceAnalysisAgent(tool=tool).execute(
        _step(),
        context,
    )

    assert result.payload == {"variance": 25}
    assert calls == [context]


def test_revenue_variance_agent_rejects_wrong_tool() -> None:
    with pytest.raises(ValueError, match="tool_name"):
        RevenueVarianceAnalysisAgent().execute(
            _step(tool_name="generate_validated_pnl_analysis"),
            FinanceDataContext(),
        )


def test_revenue_variance_agent_rejects_extra_arguments() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        RevenueVarianceAnalysisAgent().execute(
            _step(formula="actual - budget"),
            FinanceDataContext(),
        )


def test_revenue_variance_agent_requires_finance_context() -> None:
    with pytest.raises(TypeError, match="FinanceDataContext"):
        RevenueVarianceAnalysisAgent().execute(
            _step(),
            {},  # type: ignore[arg-type]
        )


def test_revenue_variance_agent_wraps_tool_failure() -> None:
    def tool(context: FinanceDataContext) -> ToolResult:
        raise RuntimeError("private detail")

    with pytest.raises(
        RevenueVarianceAnalysisAgentError,
        match="execution failed",
    ) as error:
        RevenueVarianceAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )

    assert "private detail" not in str(error.value)
