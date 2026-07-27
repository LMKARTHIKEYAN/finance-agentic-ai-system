"""Controlled specialist agent for deterministic P&L analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.autonomous.schemas import PlanStep, ToolResult
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.pnl_tools import generate_validated_pnl_analysis


class PnLAnalysisAgentError(RuntimeError):
    """Raised when the P&L specialist cannot complete its assigned step."""


class PnLAnalysisAgent:
    """Execute one validated P&L step through the approved Python tool."""

    name = "P&L Analysis Agent"
    capability = "pnl_analysis"
    tool_name = "generate_validated_pnl_analysis"

    def __init__(
        self,
        *,
        tool: Callable[..., ToolResult] = generate_validated_pnl_analysis,
    ) -> None:
        if not callable(tool):
            raise TypeError("tool must be callable.")
        self._tool = tool

    def execute(
        self,
        step: PlanStep,
        context: FinanceDataContext,
    ) -> ToolResult:
        """Execute the assigned deterministic P&L calculation."""

        arguments = _validate_step(step, self.capability, self.tool_name)
        if not isinstance(context, FinanceDataContext):
            raise TypeError("context must be FinanceDataContext.")
        optional_arguments = {
            name: arguments[name]
            for name in ("start_month", "end_month")
            if name in arguments
        }
        _reject_unknown_arguments(
            arguments,
            {"agent_name", "tool_name", *optional_arguments},
        )

        try:
            result = self._tool(context, **optional_arguments)
        except Exception as exc:
            raise PnLAnalysisAgentError(
                "P&L analysis tool execution failed."
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
        raise PnLAnalysisAgentError(
            "P&L analysis tool returned an invalid result."
        )
    if result.tool_name != tool_name:
        raise PnLAnalysisAgentError(
            "P&L analysis tool returned an unexpected tool identity."
        )
    return result
