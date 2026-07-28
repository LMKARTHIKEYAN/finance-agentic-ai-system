"""Service-facing callable for the compiled autonomous runtime."""

from __future__ import annotations

from typing import Any, Mapping

from src.autonomous.context_adapter import (
    build_autonomous_execution_context,
)
from src.autonomous.runtime import AutonomousRuntime
from src.autonomous.schemas import AutonomousExecutionResult


class AutonomousServiceExecutor:
    """Invoke the autonomous graph using request-local internal context."""

    def __init__(self, runtime: AutonomousRuntime) -> None:
        if not isinstance(runtime, AutonomousRuntime):
            raise TypeError("runtime must be AutonomousRuntime.")
        self._runtime = runtime

    def __call__(
        self,
        question: str,
        deterministic_result: Any,
        internal_context: Mapping[str, Any] | None = None,
    ) -> AutonomousExecutionResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        if not isinstance(internal_context, Mapping):
            return _fallback("Internal deterministic context is unavailable.")
        graph_state = internal_context.get("graph_state")
        if not isinstance(graph_state, Mapping):
            return _fallback("Internal deterministic context is unavailable.")
        answer = getattr(deterministic_result, "answer", "")
        try:
            context = build_autonomous_execution_context(
                graph_state=graph_state,
                deterministic_answer=str(answer),
            )
            state = self._runtime.graph.invoke(
                {
                    "request": question.strip(),
                    "reporting_scope": context.reporting_scope,
                    "datasets": context.datasets,
                    "available_inputs": set(context.available_inputs),
                    "execution_arguments": {
                        "context": context.finance_context,
                        "draft_answer": context.draft_answer,
                        "anomaly_result": context.anomaly_result,
                        "operations_result": context.operations_result,
                        "revenue_variance_result": (
                            context.revenue_variance_result
                        ),
                    },
                }
            )
        except Exception:
            return _fallback("Autonomous runtime execution failed.")
        result = state.get("result") if isinstance(state, Mapping) else None
        if not isinstance(result, AutonomousExecutionResult):
            return _fallback("Autonomous runtime returned no safe result.")
        return result


def _fallback(reason: str) -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        status="fallback",
        fallback_flow="deterministic_planner",
        fallback_reason=reason,
        errors=(reason,),
    )
