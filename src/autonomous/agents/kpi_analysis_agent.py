"""Controlled specialist agent for deterministic KPI analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.kpi_tools import calculate_validated_kpis


class KPIAnalysisAgentError(RuntimeError):
    """Raised when the KPI specialist cannot complete its assigned step."""


class KPIAnalysisAgent:
    """Execute one validated KPI step through the approved Python tool."""

    name = "KPI Analysis Agent"
    capability = "kpi_analysis"
    tool_name = "calculate_validated_kpis"

    def __init__(
        self,
        *,
        tool: Callable[..., ToolResult] = calculate_validated_kpis,
    ) -> None:
        if not callable(tool):
            raise TypeError("tool must be callable.")
        self._tool = tool

    def execute(
        self,
        step: PlanStep,
        context: FinanceDataContext,
    ) -> ToolResult:
        """Execute the assigned deterministic KPI calculation."""

        arguments = _validate_step(step, self.capability, self.tool_name)
        if not isinstance(context, FinanceDataContext):
            raise TypeError("context must be FinanceDataContext.")

        requested_kpis = arguments.get("requested_kpis")
        if not isinstance(requested_kpis, list) or not requested_kpis:
            raise ValueError(
                "requested_kpis must be a non-empty list."
            )
        optional_arguments = {
            name: arguments[name]
            for name in (
                "dimension",
                "dimension_value",
                "forecast_period",
                "scenario_period",
            )
            if name in arguments
        }
        _reject_unknown_arguments(
            arguments,
            {
                "agent_name",
                "tool_name",
                "requested_kpis",
                *optional_arguments,
            },
        )

        try:
            result = self._tool(
                context,
                list(requested_kpis),
                **optional_arguments,
            )
        except Exception as exc:
            raise KPIAnalysisAgentError(
                "KPI analysis tool execution failed."
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
        raise KPIAnalysisAgentError(
            "KPI analysis tool returned an invalid result."
        )
    if result.tool_name != tool_name:
        raise KPIAnalysisAgentError(
            "KPI analysis tool returned an unexpected tool identity."
        )
    return result
