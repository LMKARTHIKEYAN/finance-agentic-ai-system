"""Deterministically compile LLM selections into executable plan details."""

from __future__ import annotations

from src.agents.finance.kpi_agent import KPIAgent
from src.autonomous.schemas import (
    PlanStep,
    ReportingScope,
    SupervisorPlan,
)


_DEFAULT_PERFORMANCE_KPIS = (
    "total_orders",
    "completed_orders",
    "fulfillment_percentage",
    "cancellation_percentage",
    "actual_revenue",
    "actual_aov",
)


_TARGETS = {
    "kpi_analysis": (
        "kpi_agent",
        "calculate_validated_kpis",
        None,
    ),
    "pnl_analysis": (
        "pnl_agent",
        "generate_validated_pnl_analysis",
        "pnl_structure",
    ),
    "revenue_variance": (
        "revenue_variance_agent",
        "calculate_validated_revenue_variance",
        "revenue_variance",
    ),
    "gp_decomposition": (
        "gp_decomposition_agent",
        "calculate_validated_gp_decomposition",
        "gp_decomposition",
    ),
}


def compile_supervisor_plan(
    plan: SupervisorPlan,
    *,
    request: str | None = None,
) -> SupervisorPlan:
    """Canonicalize approved execution details without adding analysis."""

    if not isinstance(plan, SupervisorPlan):
        raise TypeError("plan must be a SupervisorPlan.")

    finance_ids = tuple(
        step.step_id for step in plan.steps if step.capability in _TARGETS
    )
    diagnostic_ids = tuple(
        step.step_id
        for step in plan.steps
        if step.capability == "root_cause_recommendation"
    )
    compiled_steps: list[PlanStep] = []
    reconciliations = list(plan.required_reconciliations)

    for step in plan.steps:
        arguments = step.arguments.to_execution_dict()
        depends_on = step.depends_on

        target = _TARGETS.get(step.capability)
        if target is not None:
            agent_name, tool_name, reconciliation = target
            arguments["agent_name"] = agent_name
            arguments["tool_name"] = tool_name
            if step.capability == "kpi_analysis":
                arguments = {
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "requested_kpis": list(
                        _compiled_kpi_selection(
                            arguments.get("requested_kpis"),
                            request=request,
                        )
                    ),
                }
            elif step.capability in {
                "pnl_analysis",
                "gp_decomposition",
            }:
                arguments = {
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    **_trusted_month_arguments(
                        plan.reporting_scope
                    ),
                }
            if (
                reconciliation is not None
                and reconciliation not in reconciliations
            ):
                reconciliations.append(reconciliation)

        elif step.capability == "root_cause_recommendation":
            arguments = {
                "agent_name": "root_cause_recommendation_agent",
                "tool_name": "identify_supported_root_causes",
                "recommendation_tool_name": (
                    "generate_supported_recommendations"
                ),
            }
            depends_on = finance_ids

        elif step.capability == "review":
            arguments = {"agent_name": "reviewer_agent"}
            depends_on = diagnostic_ids or finance_ids

        compiled_steps.append(
            PlanStep(
                step_id=step.step_id,
                capability=step.capability,
                depends_on=depends_on,
                required_evidence=step.required_evidence,
                arguments=arguments,
            )
        )

    return SupervisorPlan(
        objective=plan.objective,
        reporting_scope=plan.reporting_scope,
        steps=tuple(compiled_steps),
        required_reconciliations=tuple(reconciliations),
        expected_outputs=plan.expected_outputs,
    )


def _compiled_kpi_selection(
    value: object,
    *,
    request: str | None,
) -> tuple[str, ...]:
    """Return a supported, non-empty deterministic KPI selection."""

    if _is_broad_performance_request(request):
        return _DEFAULT_PERFORMANCE_KPIS

    requested = value if isinstance(value, list) else []
    supported = set(KPIAgent.KPI_METADATA)
    selected: list[str] = []
    for item in requested:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.strip().lower().split())
        canonical = KPIAgent.KPI_ALIASES.get(
            normalized,
            normalized.replace(" ", "_"),
        )
        if canonical in supported and canonical not in selected:
            selected.append(canonical)
    return tuple(selected) or _DEFAULT_PERFORMANCE_KPIS


def _is_broad_performance_request(request: str | None) -> bool:
    if not isinstance(request, str):
        return False
    normalized = " ".join(request.lower().split())
    return (
        "performance" in normalized
        and any(
            term in normalized
            for term in (
                "risk",
                "recommend",
                "management action",
            )
        )
    )


def _trusted_month_arguments(
    scope: ReportingScope,
) -> dict[str, str]:
    """Derive an executable month range from trusted parsed dates."""

    primary_dates = tuple(
        value
        for value in (
            scope.start_date,
            scope.end_date,
        )
        if value is not None
    )
    if not primary_dates:
        return {}

    comparison_dates = tuple(
        value
        for value in (
            scope.comparison_start_date,
            scope.comparison_end_date,
        )
        if value is not None
    )
    all_dates = (*primary_dates, *comparison_dates)
    return {
        "start_month": min(all_dates).strftime("%Y-%m"),
        "end_month": max(all_dates).strftime("%Y-%m"),
    }
