"""Controlled specialist agent for deterministic revenue variance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.revenue_variance_tools import (
    calculate_validated_revenue_variance,
)


class RevenueVarianceAnalysisAgentError(RuntimeError):
    """Raised when the revenue-variance specialist cannot complete."""


class RevenueVarianceAnalysisAgent:
    """Execute one revenue-variance step through the approved Python tool."""

    name = "Revenue Variance Analysis Agent"
    capability = "revenue_variance"
    tool_name = "calculate_validated_revenue_variance"

    def __init__(
        self,
        *,
        tool: Callable[..., ToolResult] = (
            calculate_validated_revenue_variance
        ),
    ) -> None:
        if not callable(tool):
            raise TypeError("tool must be callable.")
        self._tool = tool

    def execute(
        self,
        step: PlanStep,
        context: FinanceDataContext,
    ) -> ToolResult:
        """Execute the assigned deterministic revenue-variance calculation."""

        arguments = _validate_step(step, self.capability, self.tool_name)
        if not isinstance(context, FinanceDataContext):
            raise TypeError("context must be FinanceDataContext.")
        _reject_unknown_arguments(
            arguments,
            {"agent_name", "tool_name"},
        )

        try:
            result = self._tool(context)
        except Exception as exc:
            raise RevenueVarianceAnalysisAgentError(
                "Revenue-variance tool execution failed."
            ) from exc
        return _validate_result(result, self.tool_name)


def _validate_step(
    step: object,
    capability: str,
    tool_name: str,
) -> dict[str, Any]:
    if not isinstance(step, PlanStep):
        raise TypeError("step must be a PlanStep.")
    if step.capability != capability:
        raise ValueError(f"step capability must be {capability!r}.")
    if step.arguments.get("tool_name") != tool_name:
        raise ValueError(f"step tool_name must be {tool_name!r}.")
    return dict(step.arguments)


def _reject_unknown_arguments(
    arguments: dict[str, Any],
    allowed: set[str],
) -> None:
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"unsupported step arguments: {sorted(unknown)}.")


def _validate_result(result: object, tool_name: str) -> ToolResult:
    if not isinstance(result, ToolResult):
        raise RevenueVarianceAnalysisAgentError(
            "Revenue-variance tool returned an invalid result."
        )
    if result.tool_name != tool_name:
        raise RevenueVarianceAnalysisAgentError(
            "Revenue-variance tool returned an unexpected tool identity."
        )
    return result
