"""Unit tests for the deterministic finance execution planner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.orchestrator.planner import (
    ExecutionPlan,
    PlanStep,
    create_execution_plan,
    get_supported_flows,
    plan_user_request,
    validate_plan_against_state,
)


EXPECTED_FLOW_STEPS: dict[str, tuple[str, ...]] = {
    "kpi": (
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "kpi",
        "commentary",
        "complete",
    ),
    "budget": (
        "validate_budget",
        "clean_budget",
        "profile_budget",
        "budget",
        "kpi",
        "commentary",
        "complete",
    ),
    "forecast": (
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "forecast",
        "kpi",
        "commentary",
        "complete",
    ),
    "variance": (
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
        "operations_analysis",
        "budget",
        "variance",
        "anomaly",
        "root_cause",
        "recommendation",
        "kpi",
        "commentary",
        "complete",
    ),
    "scenario": (
        "validate_operations",
        "clean_operations",
        "operations_analysis",
        "forecast",
        "scenario",
        "kpi",
        "commentary",
        "complete",
    ),
    "full": (
        "validate_operations",
        "validate_budget",
        "clean_operations",
        "clean_budget",
        "profile_operations",
        "profile_budget",
        "operations_analysis",
        "budget",
        "forecast",
        "scenario",
        "variance",
        "finance_rules",
        "anomaly",
        "root_cause",
        "recommendation",
        "kpi",
        "commentary",
        "report",
    ),
}


EXPECTED_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "kpi": (
        "user_request",
        "operations_data",
    ),
    "budget": (
        "user_request",
        "budget_data",
    ),
    "forecast": (
        "user_request",
        "operations_data",
        "frequency",
        "rolling_window",
        "forecast_periods",
    ),
    "variance": (
        "user_request",
        "operations_data",
        "budget_data",
    ),
    "scenario": (
        "user_request",
        "operations_data",
        "business_assumptions",
        "scenario_name",
    ),
    "full": (
        "user_request",
        "operations_data",
        "budget_data",
        "business_assumptions",
        "frequency",
        "rolling_window",
        "forecast_periods",
        "scenario_name",
    ),
}


@pytest.mark.parametrize(
    ("flow", "expected_steps"),
    EXPECTED_FLOW_STEPS.items(),
)
def test_create_execution_plan_returns_correct_step_order(
    flow: str,
    expected_steps: tuple[str, ...],
) -> None:
    """Every supported workflow should return the approved step order."""

    plan = create_execution_plan(flow)

    assert isinstance(plan, ExecutionPlan)
    assert plan.selected_flow == flow
    assert plan.step_names == expected_steps
    assert plan.total_steps == len(expected_steps)


@pytest.mark.parametrize(
    ("flow", "expected_required_fields"),
    EXPECTED_REQUIRED_FIELDS.items(),
)
def test_create_execution_plan_returns_correct_required_state_fields(
    flow: str,
    expected_required_fields: tuple[str, ...],
) -> None:
    """Each workflow should expose the correct initial state requirements."""

    plan = create_execution_plan(flow)

    assert plan.required_state_fields == expected_required_fields


@pytest.mark.parametrize(
    ("flow", "expected_result_fields"),
    [
        (
            "kpi",
            (
                "operations_result",
                "kpi_result",
                "commentary_result",
            ),
        ),
        (
            "budget",
            (
                "budget_result",
                "kpi_result",
                "commentary_result",
            ),
        ),
        (
            "forecast",
            (
                "operations_result",
                "forecast_result",
                "kpi_result",
                "commentary_result",
            ),
        ),
        (
            "variance",
            (
                "operations_result",
                "budget_result",
                "variance_result",
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "kpi_result",
                "commentary_result",
            ),
        ),
        (
            "scenario",
            (
                "operations_result",
                "forecast_result",
                "scenario_result",
                "kpi_result",
                "commentary_result",
            ),
        ),
        (
            "full",
            (
                "operations_result",
                "budget_result",
                "forecast_result",
                "scenario_result",
                "variance_result",
                "finance_rules_result",
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "kpi_result",
                "commentary_result",
                "report_result",
            ),
        ),
    ],
)
def test_create_execution_plan_returns_expected_result_fields(
    flow: str,
    expected_result_fields: tuple[str, ...],
) -> None:
    """Expected result fields should match outputs produced by each workflow."""

    plan = create_execution_plan(flow)

    assert plan.expected_result_fields == expected_result_fields


@pytest.mark.parametrize(
    ("user_request", "expected_flow"),
    [
        ("Show KPI performance", "kpi"),
        ("Analyze the budget", "budget"),
        ("Create a revenue forecast", "forecast"),
        ("What is the revenue variance?", "variance"),
        ("Run a management scenario", "scenario"),
        ("Run full analysis", "full"),
    ],
)
def test_plan_user_request_routes_natural_language(
    user_request: str,
    expected_flow: str,
) -> None:
    """Natural-language requests should route to the correct workflow."""

    plan = plan_user_request(user_request)

    assert plan.selected_flow == expected_flow


def test_create_execution_plan_normalizes_case_and_whitespace() -> None:
    """Planner should normalize supported workflow names."""

    plan = create_execution_plan("  VARIANCE  ")

    assert plan.selected_flow == "variance"


def test_get_supported_flows_returns_all_approved_flows() -> None:
    """Supported-flow helper should return all six approved workflows."""

    assert get_supported_flows() == (
        "kpi",
        "budget",
        "forecast",
        "variance",
        "scenario",
        "full",
    )


def test_plan_contains_step_returns_true_and_false() -> None:
    """ExecutionPlan.contains_step should identify configured steps."""

    plan = create_execution_plan("variance")

    assert plan.contains_step("variance") is True
    assert plan.contains_step("forecast") is False


@pytest.mark.parametrize(
    "invalid_step_name",
    [None, 123, [], {}],
)
def test_contains_step_rejects_non_string_value(
    invalid_step_name: object,
) -> None:
    """ExecutionPlan.contains_step should reject non-string values."""

    plan = create_execution_plan("variance")

    with pytest.raises(
        TypeError,
        match="step_name must be a string",
    ):
        plan.contains_step(
            invalid_step_name  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("empty_step_name", ["", " ", "\t\n"])
def test_contains_step_rejects_empty_value(
    empty_step_name: str,
) -> None:
    """ExecutionPlan.contains_step should reject blank step names."""

    plan = create_execution_plan("variance")

    with pytest.raises(
        ValueError,
        match="step_name cannot be empty",
    ):
        plan.contains_step(empty_step_name)


def test_execution_plan_to_dict_is_json_compatible() -> None:
    """ExecutionPlan.to_dict should expose ordered, serializable metadata."""

    plan = create_execution_plan("variance")
    result = plan.to_dict()

    assert result["selected_flow"] == "variance"
    assert result["total_steps"] == 13
    assert result["required_state_fields"] == [
        "user_request",
        "operations_data",
        "budget_data",
    ]
    assert result["expected_result_fields"] == [
        "operations_result",
        "budget_result",
        "variance_result",
        "anomaly_result",
        "root_cause_result",
        "recommendation_result",
        "kpi_result",
        "commentary_result",
    ]

    steps = result["steps"]

    assert isinstance(steps, list)
    assert steps[0]["sequence"] == 1
    assert steps[0]["name"] == "validate_operations"
    assert steps[-1]["sequence"] == 13
    assert steps[-1]["name"] == "complete"


def test_validate_plan_against_state_returns_missing_fields() -> None:
    """State validation should report required fields that are missing."""

    plan = create_execution_plan("variance")

    missing = validate_plan_against_state(
        plan,
        {
            "user_request": "What is the revenue variance?",
            "operations_data": object(),
        },
    )

    assert missing == ("budget_data",)


def test_validate_plan_against_state_treats_none_as_missing() -> None:
    """A required state field with a None value should be considered missing."""

    plan = create_execution_plan("kpi")

    missing = validate_plan_against_state(
        plan,
        {
            "user_request": "Show KPI performance",
            "operations_data": None,
        },
    )

    assert missing == ("operations_data",)


def test_validate_plan_against_state_returns_empty_tuple_when_complete() -> None:
    """State validation should return no missing fields for valid input."""

    plan = create_execution_plan("forecast")

    missing = validate_plan_against_state(
        plan,
        {
            "user_request": "Create a forecast",
            "operations_data": object(),
            "frequency": "month",
            "rolling_window": 3,
            "forecast_periods": 6,
        },
    )

    assert missing == ()


@pytest.mark.parametrize(
    "invalid_plan",
    [None, "variance", {}, []],
)
def test_validate_plan_against_state_rejects_invalid_plan(
    invalid_plan: object,
) -> None:
    """State validation should require an ExecutionPlan instance."""

    with pytest.raises(
        TypeError,
        match="plan must be an ExecutionPlan",
    ):
        validate_plan_against_state(
            invalid_plan,  # type: ignore[arg-type]
            {},
        )


@pytest.mark.parametrize(
    "invalid_state",
    [None, [], "state", 123],
)
def test_validate_plan_against_state_rejects_non_mapping(
    invalid_state: object,
) -> None:
    """State validation should reject values that are not mappings."""

    plan = create_execution_plan("variance")

    with pytest.raises(
        TypeError,
        match="state must be a mapping",
    ):
        validate_plan_against_state(
            plan,
            invalid_state,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_flow",
    [None, 123, [], {}],
)
def test_create_execution_plan_rejects_non_string_flow(
    invalid_flow: object,
) -> None:
    """Planner should reject non-string workflow values."""

    with pytest.raises(
        TypeError,
        match="selected_flow must be a string",
    ):
        create_execution_plan(
            invalid_flow  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("empty_flow", ["", " ", "\t\n"])
def test_create_execution_plan_rejects_empty_flow(
    empty_flow: str,
) -> None:
    """Planner should reject blank workflow names."""

    with pytest.raises(
        ValueError,
        match="selected_flow cannot be empty",
    ):
        create_execution_plan(empty_flow)


def test_create_execution_plan_rejects_unknown_flow() -> None:
    """Planner should reject unsupported workflow names."""

    with pytest.raises(
        ValueError,
        match="Unsupported finance workflow",
    ):
        create_execution_plan("cash_flow")


@pytest.mark.parametrize(
    "invalid_request",
    [None, 123, [], {}],
)
def test_plan_user_request_rejects_non_string_request(
    invalid_request: object,
) -> None:
    """Natural-language planner should reject non-string requests."""

    with pytest.raises(
        TypeError,
        match="user_request must be a string",
    ):
        plan_user_request(
            invalid_request  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("empty_request", ["", " ", "\t\n"])
def test_plan_user_request_rejects_empty_request(
    empty_request: str,
) -> None:
    """Natural-language planner should reject blank requests."""

    with pytest.raises(
        ValueError,
        match="user_request cannot be empty",
    ):
        plan_user_request(empty_request)


def test_plan_user_request_rejects_unroutable_request() -> None:
    """Planner should reject requests that the router cannot classify."""

    with pytest.raises(
        ValueError,
        match="Unable to identify a supported finance workflow",
    ):
        plan_user_request("Tell me a joke about finance")


def test_plan_step_rejects_empty_name() -> None:
    """PlanStep should reject an empty logical node name."""

    with pytest.raises(
        ValueError,
        match="PlanStep.name cannot be empty",
    ):
        PlanStep(
            name="",
            purpose="Test purpose",
        )


def test_plan_step_rejects_name_with_outer_whitespace() -> None:
    """PlanStep should reject names with leading or trailing spaces."""

    with pytest.raises(
        ValueError,
        match="must not contain leading or trailing spaces",
    ):
        PlanStep(
            name=" variance ",
            purpose="Test purpose",
        )


def test_plan_step_rejects_empty_purpose() -> None:
    """PlanStep should require a meaningful purpose."""

    with pytest.raises(
        ValueError,
        match="PlanStep.purpose cannot be empty",
    ):
        PlanStep(
            name="variance",
            purpose=" ",
        )


def test_plan_step_rejects_empty_required_input() -> None:
    """PlanStep should reject blank required-input names."""

    with pytest.raises(
        ValueError,
        match="required_inputs cannot contain empty values",
    ):
        PlanStep(
            name="variance",
            purpose="Calculate variance.",
            required_inputs=("operations_result", ""),
        )


def test_plan_step_rejects_empty_output_field() -> None:
    """PlanStep should reject an empty output-field name."""

    with pytest.raises(
        ValueError,
        match="output_field cannot be an empty string",
    ):
        PlanStep(
            name="variance",
            purpose="Calculate variance.",
            output_field="",
        )


def test_execution_plan_rejects_duplicate_steps() -> None:
    """ExecutionPlan should reject duplicate logical step names."""

    duplicate_step = PlanStep(
        name="variance",
        purpose="Calculate variance.",
        output_field="variance_result",
    )

    with pytest.raises(
        ValueError,
        match="cannot contain duplicate step names",
    ):
        ExecutionPlan(
            selected_flow="variance",
            steps=(duplicate_step, duplicate_step),
            required_state_fields=(
                "user_request",
                "operations_data",
                "budget_data",
            ),
            expected_result_fields=("variance_result",),
            description="Variance workflow.",
        )


def test_execution_plan_rejects_empty_steps() -> None:
    """ExecutionPlan should require at least one configured step."""

    with pytest.raises(
        ValueError,
        match="ExecutionPlan.steps cannot be empty",
    ):
        ExecutionPlan(
            selected_flow="variance",
            steps=(),
            required_state_fields=(
                "user_request",
                "operations_data",
                "budget_data",
            ),
            expected_result_fields=(),
            description="Variance workflow.",
        )


def test_execution_plan_rejects_empty_description() -> None:
    """ExecutionPlan should require a human-readable description."""

    step = PlanStep(
        name="variance",
        purpose="Calculate variance.",
        output_field="variance_result",
    )

    with pytest.raises(
        ValueError,
        match="ExecutionPlan.description cannot be empty",
    ):
        ExecutionPlan(
            selected_flow="variance",
            steps=(step,),
            required_state_fields=(
                "user_request",
                "operations_data",
                "budget_data",
            ),
            expected_result_fields=("variance_result",),
            description=" ",
        )


def test_plan_objects_are_immutable() -> None:
    """PlanStep and ExecutionPlan should be immutable after construction."""

    plan = create_execution_plan("variance")

    with pytest.raises(FrozenInstanceError):
        plan.selected_flow = "kpi"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        plan.steps[0].name = "changed"  # type: ignore[misc]