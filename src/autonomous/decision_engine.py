"""Observation-driven next-action selection for autonomous finance goals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from src.autonomous.schemas import AutonomousDecision
from src.autonomous.state import AutonomousState
from src.autonomous.tools.registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


DecisionPolicy = Callable[[AutonomousState, ToolRegistry], AutonomousDecision]


_CRITERION_TO_TOOL = {
    "pnl_analysis": "generate_validated_pnl_analysis",
    "pnl_reconciliation": "generate_validated_pnl_analysis",
    "revenue_variance": "calculate_validated_revenue_variance",
    "revenue_reconciliation": "calculate_validated_revenue_variance",
    "gp_decomposition": "calculate_validated_gp_decomposition",
    "gp_reconciliation": "calculate_validated_gp_decomposition",
    "root_cause": "identify_supported_root_causes",
    "finance_answer": "calculate_validated_kpis",
    "kpi_analysis": "calculate_validated_kpis",
    "daily_kpi_extreme": "calculate_daily_kpi_extreme",
    "daily_order_root_cause": "analyze_daily_order_drivers",
    "trend_analysis": "analyze_trends",
    "period_comparison": "compare_periods",
    "drilldown_analysis": "analyze_drilldown",
    "anomaly_detection": "detect_anomalies",
    "category_profitability": "analyze_category_profitability",
    "customer_route_analysis": "analyze_customers_and_routes",
    "forecast_accuracy": "analyze_forecast_accuracy",
    "scenario_analysis": "analyze_scenario",
    "driver_forecast": "forecast_from_drivers",
    "profitability_alerts": "detect_profitability_alerts",
    "management_action": "prepare_management_action",
}


class DecisionEngine:
    """Choose one next action from current state and approved tools."""

    def __init__(
        self,
        *,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
        policy: DecisionPolicy | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry.")
        if policy is not None and not callable(policy):
            raise TypeError("policy must be callable.")
        self.registry = registry
        self._policy = policy

    def decide(self, state: AutonomousState) -> AutonomousDecision:
        if not isinstance(state, AutonomousState):
            raise TypeError("state must be an AutonomousState.")
        if self._policy is not None:
            decision = self._policy(state, self.registry)
            if not isinstance(decision, AutonomousDecision):
                raise TypeError("decision policy must return AutonomousDecision.")
            self._validate_tool(decision)
            return decision

        unresolved = tuple(
            item for item in state.goal.ambiguities
            if item not in state.clarification_answers
        )
        if unresolved:
            return AutonomousDecision(
                decision_id=_decision_id(),
                action="ask_user",
                rationale="The goal is ambiguous and cannot be executed safely.",
                question=state.goal.clarification_question,
            )

        missing = state.missing_criteria
        if not missing:
            return AutonomousDecision(
                decision_id=_decision_id(),
                action="finalize",
                rationale="All required goal criteria are satisfied.",
                final_answer=_compose_phase1_answer(state),
            )

        if missing == {"final_validation"}:
            return AutonomousDecision(
                decision_id=_decision_id(),
                action="validate",
                rationale="Financial evidence is complete and requires final validation.",
                target_criteria=("final_validation",),
            )

        criterion = next(
            (
                item.key for item in state.goal.criteria
                if item.key in missing and item.key != "final_validation"
            ),
            None,
        )
        tool_name = _CRITERION_TO_TOOL.get(criterion or "")
        if tool_name is None or tool_name not in self.registry.names:
            return AutonomousDecision(
                decision_id=_decision_id(),
                action="stop",
                rationale=(
                    f"No approved Phase 1 tool is registered for criterion "
                    f"{criterion!r}."
                ),
            )

        related = tuple(
            key for key, mapped_tool in _CRITERION_TO_TOOL.items()
            if mapped_tool == tool_name and key in missing
        )
        definition = self.registry.get(tool_name)
        arguments = {
            name: state.context[name]
            for name in definition.required_inputs
            if name in state.context
        }
        absent = set(definition.required_inputs) - set(arguments)
        if absent:
            return AutonomousDecision(
                decision_id=_decision_id(),
                action="ask_user",
                rationale="Required tool inputs are unavailable in shared state.",
                question=(
                    "The analysis is missing required inputs: "
                    + ", ".join(sorted(absent))
                    + ". Please provide or load them."
                ),
            )
        return AutonomousDecision(
            decision_id=_decision_id(),
            action="call_tool",
            tool_name=tool_name,
            arguments=arguments,
            rationale=(
                f"Criterion {criterion!r} is incomplete; use the approved "
                f"tool {tool_name!r}."
            ),
            target_criteria=related or ((criterion,) if criterion else ()),
        )

    def _validate_tool(self, decision: AutonomousDecision) -> None:
        if (
            decision.action == "call_tool"
            and decision.tool_name not in self.registry.names
        ):
            raise ValueError(
                f"Decision selected an unapproved tool: {decision.tool_name!r}."
            )


def _compose_phase1_answer(state: AutonomousState) -> str:
    summaries = [
        observation.summary
        for observation in state.observations
        if observation.status == "completed"
        and observation.source != "final_validation"
    ]
    if summaries:
        return " ".join(summaries)
    return f"Goal completed: {state.goal.objective}"


def _decision_id() -> str:
    return f"decision-{uuid4().hex}"
