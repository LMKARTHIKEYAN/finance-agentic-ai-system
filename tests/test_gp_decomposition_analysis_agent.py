"""Tests for the controlled GP% decomposition specialist agent."""

from typing import Any

import pytest

from src.autonomous.agents.gp_decomposition_analysis_agent import (
    GPDecompositionAnalysisAgent,
    GPDecompositionAnalysisAgentError,
)
from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext


def _step(**arguments: Any) -> PlanStep:
    return PlanStep(
        step_id="gp",
        capability="gp_decomposition",
        arguments={
            "agent_name": "gp_decomposition_agent",
            "tool_name": "calculate_validated_gp_decomposition",
            **arguments,
        },
    )


def test_gp_agent_delegates_months_and_preserves_both_levels() -> None:
    calls: list[tuple[Any, ...]] = []
    payload = {
        "product_level": [{"category": "A"}],
        "portfolio_level": {"actual_gp_pct": 40.0},
    }

    def tool(context: FinanceDataContext, **kwargs: Any) -> ToolResult:
        calls.append((context, kwargs))
        return ToolResult(
            call_id="gp-call",
            tool_name="calculate_validated_gp_decomposition",
            status="completed",
            payload=payload,
        )

    context = FinanceDataContext()
    result = GPDecompositionAnalysisAgent(tool=tool).execute(
        _step(start_month="2025-04", end_month="2025-04"),
        context,
    )

    assert result.payload["product_level"] == [{"category": "A"}]
    assert result.payload["portfolio_level"]["actual_gp_pct"] == 40.0
    assert calls == [
        (
            context,
            {"start_month": "2025-04", "end_month": "2025-04"},
        )
    ]


def test_gp_agent_rejects_wrong_tool_without_execution() -> None:
    called = False

    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(ValueError, match="tool_name"):
        GPDecompositionAnalysisAgent(tool=tool).execute(
            _step(tool_name="calculate_validated_revenue_variance"),
            FinanceDataContext(),
        )

    assert called is False


def test_gp_agent_rejects_formula_override() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        GPDecompositionAnalysisAgent().execute(
            _step(revenue_field="different_revenue"),
            FinanceDataContext(),
        )


def test_gp_agent_rejects_invalid_tool_result() -> None:
    def tool(*args: Any, **kwargs: Any) -> Any:
        return {"product_level": [], "portfolio_level": {}}

    with pytest.raises(
        GPDecompositionAnalysisAgentError,
        match="invalid result",
    ):
        GPDecompositionAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )


def test_gp_agent_wraps_tool_failure() -> None:
    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        raise RuntimeError("raw dataframe details")

    with pytest.raises(
        GPDecompositionAnalysisAgentError,
        match="execution failed",
    ) as error:
        GPDecompositionAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )

    assert "raw dataframe details" not in str(error.value)
