"""Tests for the controlled P&L specialist agent."""

from typing import Any

import pytest

from src.autonomous.agents.pnl_analysis_agent import (
    PnLAnalysisAgent,
    PnLAnalysisAgentError,
)
from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext


def _step(**arguments: Any) -> PlanStep:
    return PlanStep(
        step_id="pnl",
        capability="pnl_analysis",
        arguments={
            "agent_name": "pnl_agent",
            "tool_name": "generate_validated_pnl_analysis",
            **arguments,
        },
    )


def test_pnl_agent_delegates_months_to_deterministic_tool() -> None:
    calls: list[tuple[Any, ...]] = []

    def tool(context: FinanceDataContext, **kwargs: Any) -> ToolResult:
        calls.append((context, kwargs))
        return ToolResult(
            call_id="pnl-call",
            tool_name="generate_validated_pnl_analysis",
            status="completed",
            payload={"net_profit": 75},
        )

    context = FinanceDataContext()
    result = PnLAnalysisAgent(tool=tool).execute(
        _step(start_month="2025-04", end_month="2025-04"),
        context,
    )

    assert result.payload == {"net_profit": 75}
    assert calls == [
        (
            context,
            {"start_month": "2025-04", "end_month": "2025-04"},
        )
    ]


def test_pnl_agent_rejects_wrong_capability_without_execution() -> None:
    called = False

    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        nonlocal called
        called = True
        raise AssertionError

    step = _step().model_copy(update={"capability": "gp_decomposition"})
    with pytest.raises(ValueError, match="capability"):
        PnLAnalysisAgent(tool=tool).execute(step, FinanceDataContext())

    assert called is False


def test_pnl_agent_rejects_unapproved_arguments() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        PnLAnalysisAgent().execute(
            _step(income_tax_rate=0.10),
            FinanceDataContext(),
        )


def test_pnl_agent_rejects_mismatched_tool_result() -> None:
    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        return ToolResult(
            call_id="wrong",
            tool_name="calculate_validated_kpis",
            status="completed",
        )

    with pytest.raises(PnLAnalysisAgentError, match="tool identity"):
        PnLAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )


def test_pnl_agent_wraps_deterministic_tool_failure() -> None:
    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        raise ValueError("missing operations rows")

    with pytest.raises(PnLAnalysisAgentError, match="execution failed"):
        PnLAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )
