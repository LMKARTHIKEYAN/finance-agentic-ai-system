"""Isolated supervised autonomous workflow graph."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.execution_limits import (
    AutonomousExecutionLimits,
)
from src.autonomous.plan_validator import AutonomousPlanValidator
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    DatasetAvailability,
    PlanValidationIssue,
    ReportingScope,
)
from src.autonomous.agents.supervisor_agent import FinanceSupervisorAgent


class _WorkflowState(TypedDict, total=False):
    request: str
    reporting_scope: ReportingScope
    datasets: tuple[DatasetAvailability, ...]
    available_inputs: set[str]
    execution_arguments: dict[str, Any]
    validation_issues: tuple[PlanValidationIssue, ...]
    plan: Any
    result: AutonomousExecutionResult
    replan_count: int
    route: str


def build_autonomous_graph(
    *,
    supervisor: FinanceSupervisorAgent,
    validator: AutonomousPlanValidator,
    coordinator: AutonomousExecutionCoordinator,
    limits: AutonomousExecutionLimits | None = None,
) -> Any:
    """Build an isolated LangGraph; it is not connected to the app router."""

    if not isinstance(supervisor, FinanceSupervisorAgent):
        raise TypeError("supervisor must be a FinanceSupervisorAgent.")
    if not isinstance(validator, AutonomousPlanValidator):
        raise TypeError("validator must be an AutonomousPlanValidator.")
    if not isinstance(coordinator, AutonomousExecutionCoordinator):
        raise TypeError(
            "coordinator must be an AutonomousExecutionCoordinator."
        )
    resolved_limits = limits or AutonomousExecutionLimits()

    def plan_node(state: _WorkflowState) -> dict[str, Any]:
        try:
            plan = supervisor.propose_plan(
                state["request"],
                reporting_scope=state.get("reporting_scope"),
                datasets=state.get("datasets", ()),
                validation_issues=state.get("validation_issues", ()),
            )
            return {"plan": plan}
        except Exception:
            return {
                "route": "fallback",
                "result": _graph_fallback(
                    "Supervisor planning failed."
                ),
            }

    def validate_node(state: _WorkflowState) -> dict[str, Any]:
        if state.get("route") == "fallback":
            return {}
        validation = validator.validate(
            state["plan"],
            available_inputs=state.get("available_inputs", set()),
        )
        if validation.valid:
            return {"route": "execute"}
        replan_count = state.get("replan_count", 0)
        if replan_count >= resolved_limits.max_replans:
            return {
                "route": "fallback",
                "result": _graph_fallback(
                    "Plan validation failed after replan limit."
                ),
            }
        return {
            "route": "replan",
            "validation_issues": validation.issues,
            "replan_count": replan_count + 1,
        }

    def execute_node(state: _WorkflowState) -> dict[str, Any]:
        arguments = dict(state.get("execution_arguments", {}))
        return {
            "result": coordinator.execute(state["plan"], **arguments)
        }

    builder = StateGraph(_WorkflowState)
    builder.add_node("supervisor", plan_node)
    builder.add_node("validator", validate_node)
    builder.add_node("coordinator", execute_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "validator")
    builder.add_conditional_edges(
        "validator",
        lambda state: state.get("route", "fallback"),
        {
            "execute": "coordinator",
            "replan": "supervisor",
            "fallback": END,
        },
    )
    builder.add_edge("coordinator", END)
    return builder.compile()


def _graph_fallback(reason: str) -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        status="fallback",
        fallback_flow="deterministic_planner",
        fallback_reason=reason,
        errors=(reason,),
    )
