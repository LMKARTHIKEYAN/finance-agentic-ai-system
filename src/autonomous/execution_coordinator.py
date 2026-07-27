"""Bounded coordinator for an approved autonomous finance plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.autonomous.agents.gp_decomposition_analysis_agent import (
    GPDecompositionAnalysisAgent,
)
from src.autonomous.agents.kpi_analysis_agent import KPIAnalysisAgent
from src.autonomous.agents.pnl_analysis_agent import PnLAnalysisAgent
from src.autonomous.agents.revenue_variance_analysis_agent import (
    RevenueVarianceAnalysisAgent,
)
from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.agents.root_cause_recommendation_agent import (
    RootCauseRecommendationAgent,
)
from src.autonomous.evidence_registry import EvidenceRegistry
from src.autonomous.execution_limits import (
    AutonomousExecutionLimits,
    ExecutionLimitExceededError,
    ExecutionUsageTracker,
)
from src.autonomous.reconciliation import AutonomousReconciler
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ManagementResponse,
    PlanStep,
    SupervisorPlan,
    ToolResult,
)
from src.autonomous.tools.data_tools import FinanceDataContext


_RESULT_TYPES = {
    "calculate_validated_kpis": "kpi",
    "generate_validated_pnl_analysis": "pnl",
    "calculate_validated_revenue_variance": "revenue_variance",
    "calculate_validated_gp_decomposition": "gp_decomposition",
}


class AutonomousExecutionCoordinator:
    """Execute a validated plan within deterministic safety limits."""

    def __init__(
        self,
        *,
        reviewer: ReviewerAgent,
        diagnostics: RootCauseRecommendationAgent | None = None,
        specialist_agents: Mapping[str, Any] | None = None,
        reconciler: AutonomousReconciler | None = None,
        limits: AutonomousExecutionLimits | None = None,
    ) -> None:
        if not isinstance(reviewer, ReviewerAgent):
            raise TypeError("reviewer must be a ReviewerAgent.")
        self._reviewer = reviewer
        self._diagnostics = (
            diagnostics or RootCauseRecommendationAgent()
        )
        self._specialists = dict(
            specialist_agents
            or {
                "kpi_analysis": KPIAnalysisAgent(),
                "pnl_analysis": PnLAnalysisAgent(),
                "revenue_variance": RevenueVarianceAnalysisAgent(),
                "gp_decomposition": GPDecompositionAnalysisAgent(),
            }
        )
        self._reconciler = reconciler or AutonomousReconciler()
        self._limits = limits or AutonomousExecutionLimits()

    def execute(
        self,
        plan: SupervisorPlan,
        *,
        context: FinanceDataContext,
        draft_answer: str,
        anomaly_result: object,
        operations_result: object,
        revenue_variance_result: object | None = None,
        llm_usage: Sequence[tuple[int, int, float]] = (),
    ) -> AutonomousExecutionResult:
        """Execute specialists, controls, diagnostics, and review."""

        if not isinstance(plan, SupervisorPlan):
            raise TypeError("plan must be a SupervisorPlan.")
        if not isinstance(context, FinanceDataContext):
            raise TypeError("context must be FinanceDataContext.")

        tracker = ExecutionUsageTracker(self._limits)
        registry = EvidenceRegistry()
        tool_results: list[ToolResult] = []
        try:
            for input_tokens, output_tokens, cost in llm_usage:
                tracker.record_llm_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=cost,
                )

            ordered = _topological_steps(plan.steps)
            specialist_steps = [
                step for step in ordered
                if step.capability in self._specialists
            ]
            for step in specialist_steps:
                result = self._run_with_retry(
                    step.arguments.get("agent_name", step.capability),
                    lambda step=step: self._specialists[
                        step.capability
                    ].execute(step, context),
                    tracker,
                    tool_calls=1,
                )
                if not isinstance(result, ToolResult):
                    raise TypeError(
                        "Specialist agent returned an invalid result."
                    )
                tool_results.append(result)
                registry.register(
                    result,
                    result_type=_RESULT_TYPES[result.tool_name],
                )

            if not tool_results:
                raise ValueError(
                    "Plan contains no executable finance specialist steps."
                )
            reconciliation = self._reconciler.reconcile(registry)
            if not reconciliation.passed:
                return _fallback(
                    "Evidence reconciliation failed.",
                    tracker,
                    reconciliation=reconciliation,
                )
            for record in registry.list():
                registry.mark_verified(record.evidence_id)
            evidence = registry.list()

            diagnostic_step = _required_step(
                ordered,
                "root_cause_recommendation",
            )
            diagnostics = self._run_with_retry(
                diagnostic_step.arguments.get(
                    "agent_name",
                    "root_cause_recommendation_agent",
                ),
                lambda: self._diagnostics.execute(
                    diagnostic_step,
                    evidence=evidence,
                    anomaly_result=anomaly_result,
                    operations_result=operations_result,
                    revenue_variance_result=revenue_variance_result,
                ),
                tracker,
                tool_calls=2,
            )

            _required_step(ordered, "review")
            review = self._run_with_retry(
                "reviewer_agent",
                lambda: self._reviewer.review(
                    plan=plan,
                    reconciliation=reconciliation,
                    evidence=evidence,
                    diagnostics=diagnostics,
                    draft_answer=draft_answer,
                ),
                tracker,
                tool_calls=0,
            )
            if review.decision not in {
                "approved",
                "approved_with_caveats",
            }:
                return _fallback(
                    f"Reviewer decision: {review.decision}.",
                    tracker,
                    reconciliation=reconciliation,
                    review=review,
                )

            answer = review.approved_answer or draft_answer
            management_response = ManagementResponse(
                answer=answer,
                evidence_ids=tuple(
                    item.evidence_id for item in evidence
                ),
                caveats=review.required_caveats,
            )
            return AutonomousExecutionResult(
                status="completed",
                management_response=management_response,
                review_result=review,
                reconciliation_result=reconciliation,
                evidence=tuple(
                    _reference(item) for item in evidence
                ),
                usage=tracker.snapshot(),
            )
        except Exception as exc:
            return _fallback(
                _safe_failure_reason(exc),
                tracker,
            )

    def _run_with_retry(
        self,
        agent_name: str,
        operation: Any,
        tracker: ExecutionUsageTracker,
        *,
        tool_calls: int,
    ) -> Any:
        attempts = self._limits.max_retries_per_agent + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                tracker.record_agent_run()
                for _ in range(tool_calls):
                    tracker.record_tool_call()
                return operation()
            except ExecutionLimitExceededError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                tracker.record_agent_retry(agent_name)
        raise RuntimeError(
            f"Agent {agent_name!r} failed after bounded retries."
        ) from last_error


def _topological_steps(
    steps: Sequence[PlanStep],
) -> tuple[PlanStep, ...]:
    remaining = {step.step_id: step for step in steps}
    completed: set[str] = set()
    ordered: list[PlanStep] = []
    while remaining:
        ready = [
            step for step in remaining.values()
            if set(step.depends_on) <= completed
        ]
        if not ready:
            raise ValueError("Plan dependencies cannot be resolved.")
        for step in ready:
            ordered.append(step)
            completed.add(step.step_id)
            del remaining[step.step_id]
    return tuple(ordered)


def _required_step(
    steps: Sequence[PlanStep],
    capability: str,
) -> PlanStep:
    matches = [step for step in steps if step.capability == capability]
    if len(matches) != 1:
        raise ValueError(
            f"Plan must contain exactly one {capability!r} step."
        )
    return matches[0]


def _reference(record: Any) -> Any:
    from src.autonomous.schemas import EvidenceReference

    return EvidenceReference(
        evidence_id=record.evidence_id,
        source=record.source_tool,
        result_type=record.result_type,
        reconciled=record.reconciled,
        period=record.period,
    )


def _fallback(
    reason: str,
    tracker: ExecutionUsageTracker,
    *,
    reconciliation: Any = None,
    review: Any = None,
) -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        status="fallback",
        reconciliation_result=reconciliation,
        review_result=review,
        usage=tracker.snapshot(),
        fallback_flow="deterministic_planner",
        fallback_reason=reason,
        errors=(reason,),
    )


def _safe_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ExecutionLimitExceededError):
        return str(exc)
    return "Autonomous execution failed; use deterministic fallback."
