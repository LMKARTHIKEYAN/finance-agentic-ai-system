"""Tests for deterministic autonomous plan validation."""

from datetime import date

from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.plan_validator import AutonomousPlanValidator
from src.autonomous.schemas import PlanStep, ReportingScope, SupervisorPlan


def _plan(
    *steps: PlanStep,
    reconciliations: tuple[str, ...] = (),
    scope: ReportingScope | None = None,
) -> SupervisorPlan:
    return SupervisorPlan(
        objective="Analyze April performance",
        reporting_scope=scope or ReportingScope(),
        steps=steps,
        required_reconciliations=reconciliations,
        expected_outputs=("management_answer",),
    )


def test_valid_tool_plan_is_approved() -> None:
    plan = _plan(
        PlanStep(
            step_id="pnl",
            capability="pnl_analysis",
            arguments={
                "agent_name": "pnl_agent",
                "tool_name": "generate_validated_pnl_analysis",
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("pnl",),
            arguments={"agent_name": "reviewer_agent"},
        ),
        reconciliations=("pnl_structure",),
    )

    result = AutonomousPlanValidator().validate(
        plan,
        available_inputs={
            "operations_data",
            "budget_data",
            "corporate_expenses_data",
            "budget_corporate_expenses_data",
        },
    )

    assert result.valid is True
    assert result.approved_plan == plan
    assert result.selected_agents == ("pnl_agent", "reviewer_agent")
    assert result.selected_tools == (
        "generate_validated_pnl_analysis",
    )


def test_unknown_tool_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="unsafe",
            capability="pnl_analysis",
            arguments={
                "agent_name": "pnl_agent",
                "tool_name": "execute_python",
            },
        )
    )

    result = AutonomousPlanValidator().validate(plan)

    assert result.valid is False
    assert "unknown_tool" in {item.code for item in result.issues}


def test_missing_tool_inputs_are_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="variance",
            capability="revenue_variance",
            arguments={
                "agent_name": "variance_agent",
                "tool_name": "calculate_validated_revenue_variance",
            },
        ),
        reconciliations=("revenue_variance",),
    )

    result = AutonomousPlanValidator().validate(
        plan,
        available_inputs={"operations_result"},
    )

    assert result.valid is False
    assert "missing_tool_inputs" in {
        item.code for item in result.issues
    }


def test_missing_reconciliation_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="gp",
            capability="gp_decomposition",
            arguments={
                "agent_name": "gp_agent",
                "tool_name": "calculate_validated_gp_decomposition",
            },
        )
    )

    result = AutonomousPlanValidator().validate(
        plan,
        available_inputs={"operations_data", "budget_data"},
    )

    assert "missing_reconciliation" in {
        item.code for item in result.issues
    }


def test_finance_agent_without_tool_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="pnl",
            capability="pnl_analysis",
            arguments={"agent_name": "pnl_agent"},
        )
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "llm_calculation_forbidden" in {
        item.code for item in result.issues
    }


def test_circular_dependency_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="a",
            capability="root_cause",
            depends_on=("b",),
        ),
        PlanStep(
            step_id="b",
            capability="review",
            depends_on=("a",),
        ),
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "circular_dependency" in {
        item.code for item in result.issues
    }


def test_more_than_six_agents_is_rejected() -> None:
    plan = _plan(
        *(
            PlanStep(
                step_id=f"agent-{index}",
                capability="review",
                arguments={"agent_name": f"agent-{index}"},
            )
            for index in range(7)
        )
    )

    result = AutonomousPlanValidator(
        limits=AutonomousExecutionLimits(max_agents=6)
    ).validate(plan)

    assert "agent_limit_exceeded" in {
        item.code for item in result.issues
    }


def test_deterministic_tools_do_not_count_as_agents() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="calculate_validated_kpis",
            arguments={
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": ["actual_revenue"],
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("kpi",),
            arguments={"agent_name": "reviewer_agent"},
        ),
    )

    result = AutonomousPlanValidator().validate(plan)

    assert result.valid is True
    assert result.selected_agents == ("reviewer_agent",)


def test_plan_without_reviewer_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="calculate_validated_kpis",
            arguments={
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": ["actual_revenue"],
            },
        )
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "reviewer_required" in {
        item.code for item in result.issues
    }


def test_inconsistent_scope_override_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="calculate_validated_kpis",
            arguments={
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": ["actual_revenue"],
                "start_date": "2026-05-01",
            },
        ),
        scope=ReportingScope(
            start_date=date(2026, 4, 1),
            end_date=date(2026, 4, 30),
        ),
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "inconsistent_reporting_scope" in {
        item.code for item in result.issues
    }


def test_validator_does_not_mutate_plan() -> None:
    plan = _plan(
        PlanStep(
            step_id="review",
            capability="review",
            arguments={"agent_name": "reviewer"},
        )
    )
    before = plan.model_dump()

    AutonomousPlanValidator().validate(plan)

    assert plan.model_dump() == before


def test_empty_kpi_selection_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="kpi_analysis",
            arguments={
                "agent_name": "kpi_agent",
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": [],
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("kpi",),
            arguments={"agent_name": "reviewer_agent"},
        ),
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "missing_kpi_selection" in {
        item.code for item in result.issues
    }


def test_unsupported_kpi_selection_is_rejected() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="kpi_analysis",
            arguments={
                "agent_name": "kpi_agent",
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": ["invented_kpi"],
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("kpi",),
            arguments={"agent_name": "reviewer_agent"},
        ),
    )

    result = AutonomousPlanValidator().validate(plan)

    assert "unsupported_kpi_selection" in {
        item.code for item in result.issues
    }


def test_margin_request_requires_gp_capability() -> None:
    plan = _plan(
        PlanStep(
            step_id="pnl",
            capability="pnl_analysis",
            arguments={
                "agent_name": "pnl_agent",
                "tool_name": "generate_validated_pnl_analysis",
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("pnl",),
            arguments={"agent_name": "reviewer_agent"},
        ),
        reconciliations=("pnl_structure",),
    )

    result = AutonomousPlanValidator().validate(
        plan,
        request=(
            "Explain May margin changes versus April and recommend actions."
        ),
        available_inputs={
            "operations_data",
            "budget_data",
            "corporate_expenses_data",
            "budget_corporate_expenses_data",
        },
    )

    assert "missing_required_capability" in {
        item.code for item in result.issues
    }


def test_broad_performance_requires_all_finance_capabilities() -> None:
    plan = _plan(
        PlanStep(
            step_id="kpi",
            capability="kpi_analysis",
            arguments={
                "agent_name": "kpi_agent",
                "tool_name": "calculate_validated_kpis",
                "requested_kpis": ["actual_revenue"],
            },
        ),
        PlanStep(
            step_id="review",
            capability="review",
            depends_on=("kpi",),
            arguments={"agent_name": "reviewer_agent"},
        ),
    )

    result = AutonomousPlanValidator().validate(
        plan,
        request=(
            "Analyze May performance, identify financial risks, "
            "and recommend actions."
        ),
    )

    missing_capability_issues = [
        item
        for item in result.issues
        if item.code == "missing_required_capability"
    ]
    assert len(missing_capability_issues) == 3
