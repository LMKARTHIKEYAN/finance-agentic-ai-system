"""Tests for deterministic compilation of Supervisor selections."""

from datetime import date

from src.autonomous.plan_compiler import compile_supervisor_plan
from src.autonomous.plan_validator import AutonomousPlanValidator
from src.autonomous.schemas import (
    PlanStep,
    ReportingScope,
    SupervisorPlan,
)


def test_compiler_canonicalizes_pnl_diagnostic_plan() -> None:
    plan = SupervisorPlan(
        objective="Explain profit improvement",
        steps=(
            PlanStep(
                step_id="pnl",
                capability="pnl_analysis",
                arguments={"agent_name": "analysis"},
            ),
            PlanStep(
                step_id="diagnose",
                capability="root_cause_recommendation",
            ),
            PlanStep(step_id="review", capability="review"),
        ),
        expected_outputs=("answer",),
    )

    compiled = compile_supervisor_plan(plan)

    assert compiled.steps[0].arguments.get("tool_name") == (
        "generate_validated_pnl_analysis"
    )
    assert compiled.steps[1].depends_on == ("pnl",)
    assert compiled.steps[2].depends_on == ("diagnose",)
    assert "pnl_structure" in compiled.required_reconciliations
    validation = AutonomousPlanValidator().validate(
        compiled,
        available_inputs={
            "operations_data",
            "budget_data",
            "corporate_expenses_data",
            "budget_corporate_expenses_data",
        },
    )
    assert validation.valid is True


def test_compiler_preserves_kpi_selection_arguments() -> None:
    plan = SupervisorPlan(
        objective="Analyze KPIs",
        steps=(
            PlanStep(
                step_id="kpi",
                capability="kpi_analysis",
                arguments={"requested_kpis": ["actual_revenue"]},
            ),
            PlanStep(step_id="review", capability="review"),
        ),
        expected_outputs=("answer",),
    )

    compiled = compile_supervisor_plan(plan)

    assert compiled.steps[0].arguments.requested_kpis == [
        "actual_revenue"
    ]


def test_compiler_uses_trusted_primary_and_comparison_pnl_months() -> None:
    plan = SupervisorPlan(
        objective="Explain May profit versus April",
        reporting_scope=ReportingScope(
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 31),
            comparison_start_date=date(2026, 4, 1),
            comparison_end_date=date(2026, 4, 30),
        ),
        steps=(
            PlanStep(
                step_id="pnl",
                capability="pnl_analysis",
                arguments={
                    "agent_name": "invented_agent",
                    "tool_name": "invented_tool",
                    "start_date": date(2025, 1, 1),
                    "end_date": date(2025, 1, 31),
                    "start_month": "2027-12",
                    "end_month": "2027-01",
                },
            ),
        ),
        expected_outputs=("management_answer",),
    )

    compiled = compile_supervisor_plan(plan)
    arguments = compiled.steps[0].arguments.to_execution_dict()

    assert arguments == {
        "agent_name": "pnl_agent",
        "tool_name": "generate_validated_pnl_analysis",
        "start_month": "2026-04",
        "end_month": "2026-05",
    }


def test_compiler_uses_primary_pnl_month_without_comparison() -> None:
    plan = SupervisorPlan(
        objective="Generate May P&L",
        reporting_scope=ReportingScope(
            start_date=date(2026, 5, 1),
            end_date=date(2026, 5, 31),
        ),
        steps=(
            PlanStep(
                step_id="pnl",
                capability="pnl_analysis",
            ),
        ),
        expected_outputs=("management_answer",),
    )

    compiled = compile_supervisor_plan(plan)

    assert compiled.steps[0].arguments.start_month == "2026-05"
    assert compiled.steps[0].arguments.end_month == "2026-05"
