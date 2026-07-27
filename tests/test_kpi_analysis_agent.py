"""Tests for the controlled KPI specialist agent."""

from typing import Any

import pytest

from src.autonomous.agents.kpi_analysis_agent import (
    KPIAnalysisAgent,
    KPIAnalysisAgentError,
)
from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext


def _step(**arguments: Any) -> PlanStep:
    return PlanStep(
        step_id="kpi",
        capability="kpi_analysis",
        arguments={
            "agent_name": "kpi_agent",
            "tool_name": "calculate_validated_kpis",
            "requested_kpis": ["revenue", "gross_profit_pct"],
            **arguments,
        },
    )


def test_kpi_agent_calls_only_assigned_tool_with_arguments() -> None:
    calls: list[tuple[Any, ...]] = []

    def tool(
        context: FinanceDataContext,
        requested_kpis: list[str],
        **kwargs: Any,
    ) -> ToolResult:
        calls.append((context, requested_kpis, kwargs))
        return ToolResult(
            call_id="kpi-call",
            tool_name="calculate_validated_kpis",
            status="completed",
            payload={"revenue": 100},
        )

    context = FinanceDataContext()
    result = KPIAnalysisAgent(tool=tool).execute(
        _step(dimension="category", dimension_value="A"),
        context,
    )

    assert result.status == "completed"
    assert calls == [
        (
            context,
            ["revenue", "gross_profit_pct"],
            {"dimension": "category", "dimension_value": "A"},
        )
    ]


def test_kpi_agent_rejects_wrong_tool_without_execution() -> None:
    called = False

    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(ValueError, match="tool_name"):
        KPIAnalysisAgent(tool=tool).execute(
            _step(tool_name="execute_python"),
            FinanceDataContext(),
        )

    assert called is False


def test_kpi_agent_rejects_missing_kpis() -> None:
    with pytest.raises(ValueError, match="requested_kpis"):
        KPIAnalysisAgent().execute(
            _step(requested_kpis=[]),
            FinanceDataContext(),
        )


def test_kpi_agent_rejects_unsupported_arguments() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        KPIAnalysisAgent().execute(
            _step(raw_dataframe="forbidden"),
            FinanceDataContext(),
        )


def test_kpi_agent_wraps_tool_failure() -> None:
    def tool(*args: Any, **kwargs: Any) -> ToolResult:
        raise RuntimeError("internal finance detail")

    with pytest.raises(
        KPIAnalysisAgentError,
        match="tool execution failed",
    ) as error:
        KPIAnalysisAgent(tool=tool).execute(
            _step(),
            FinanceDataContext(),
        )

    assert "internal finance detail" not in str(error.value)
