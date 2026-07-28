"""Deterministically compile LLM selections into executable plan details."""

from __future__ import annotations

from src.autonomous.schemas import (
    PlanStep,
    ReportingScope,
    SupervisorPlan,
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


def compile_supervisor_plan(plan: SupervisorPlan) -> SupervisorPlan:
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
            if step.capability == "pnl_analysis":
                arguments = {
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    **_trusted_pnl_month_arguments(
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


def _trusted_pnl_month_arguments(
    scope: ReportingScope,
) -> dict[str, str]:
    """Derive the executable P&L range from trusted parsed dates."""

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
