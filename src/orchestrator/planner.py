"""
Deterministic execution planner for the Finance Agentic AI workflow.

This module converts a supported finance workflow into an immutable,
validated execution plan.

The planner does not:

- Execute agents
- Perform finance calculations
- Build the LangGraph
- Call OpenAI
- Modify input datasets

The execution sequences intentionally mirror the approved routes defined in
``src.orchestrator.graph``. Keeping planning deterministic makes finance
workflows predictable, auditable, inexpensive, and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.orchestrator.router import identify_flow
from src.orchestrator.state import FlowType


SUPPORTED_PLANNING_FLOWS: tuple[FlowType, ...] = (
    "kpi",
    "budget",
    "forecast",
    "variance",
    "scenario",
    "pnl",
    "full",
)


@dataclass(frozen=True, slots=True)
class PlanStep:
    """
    Describe one approved logical step in a finance execution plan.

    Attributes:
        name:
            Logical node name used by the orchestrator.

        purpose:
            Human-readable explanation of why the step is required.

        required_inputs:
            State fields that should be available before this step executes.

        output_field:
            State field normally produced by the step. Control and data
            preparation steps may use ``None``.
    """

    name: str
    purpose: str
    required_inputs: tuple[str, ...] = ()
    output_field: str | None = None

    def __post_init__(self) -> None:
        """Validate the immutable plan-step definition."""

        cleaned_name = self.name.strip()

        if not cleaned_name:
            raise ValueError("PlanStep.name cannot be empty.")

        if cleaned_name != self.name:
            raise ValueError(
                "PlanStep.name must not contain leading or trailing spaces."
            )

        if not self.purpose.strip():
            raise ValueError("PlanStep.purpose cannot be empty.")

        if any(not item.strip() for item in self.required_inputs):
            raise ValueError(
                "PlanStep.required_inputs cannot contain empty values."
            )

        if self.output_field is not None and not self.output_field.strip():
            raise ValueError(
                "PlanStep.output_field cannot be an empty string."
            )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """
    Immutable execution plan for one supported finance workflow.

    Attributes:
        selected_flow:
            Workflow selected by the deterministic router.

        steps:
            Ordered sequence of approved logical execution steps.

        required_state_fields:
            Initial state fields required before graph execution.

        expected_result_fields:
            Finance-result fields expected after successful execution.

        description:
            Human-readable description of the workflow.
    """

    selected_flow: FlowType
    steps: tuple[PlanStep, ...]
    required_state_fields: tuple[str, ...]
    expected_result_fields: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        """Validate execution-plan consistency."""

        if self.selected_flow not in SUPPORTED_PLANNING_FLOWS:
            raise ValueError(
                f"Unsupported planning flow: {self.selected_flow!r}."
            )

        if not self.steps:
            raise ValueError("ExecutionPlan.steps cannot be empty.")

        step_names = tuple(step.name for step in self.steps)

        if len(step_names) != len(set(step_names)):
            raise ValueError(
                "ExecutionPlan.steps cannot contain duplicate step names."
            )

        if not self.description.strip():
            raise ValueError("ExecutionPlan.description cannot be empty.")

        if any(not item.strip() for item in self.required_state_fields):
            raise ValueError(
                "required_state_fields cannot contain empty values."
            )

        if any(not item.strip() for item in self.expected_result_fields):
            raise ValueError(
                "expected_result_fields cannot contain empty values."
            )

    @property
    def step_names(self) -> tuple[str, ...]:
        """Return the ordered logical node names in the plan."""

        return tuple(step.name for step in self.steps)

    @property
    def total_steps(self) -> int:
        """Return the total number of execution steps."""

        return len(self.steps)

    def contains_step(self, step_name: str) -> bool:
        """
        Check whether a logical execution step is present.

        Args:
            step_name:
                Logical node name to search for.

        Returns:
            ``True`` when the plan contains the requested step.
        """

        if not isinstance(step_name, str):
            raise TypeError("step_name must be a string.")

        normalized_name = step_name.strip()

        if not normalized_name:
            raise ValueError("step_name cannot be empty.")

        return normalized_name in self.step_names

    def to_dict(self) -> dict[str, object]:
        """
        Convert the plan into a JSON-compatible dictionary.

        Returns:
            Dictionary suitable for logs, API debugging, or documentation.
        """

        return {
            "selected_flow": self.selected_flow,
            "description": self.description,
            "total_steps": self.total_steps,
            "required_state_fields": list(self.required_state_fields),
            "expected_result_fields": list(self.expected_result_fields),
            "steps": [
                {
                    "sequence": index,
                    "name": step.name,
                    "purpose": step.purpose,
                    "required_inputs": list(step.required_inputs),
                    "output_field": step.output_field,
                }
                for index, step in enumerate(self.steps, start=1)
            ],
        }


STEP_CATALOG: Mapping[str, PlanStep] = MappingProxyType(
    {
        "validate_operations": PlanStep(
            name="validate_operations",
            purpose="Validate operational data before finance analysis.",
            required_inputs=("operations_data",),
            output_field="operations_validation",
        ),
        "validate_budget": PlanStep(
            name="validate_budget",
            purpose="Validate budget data before budget-based analysis.",
            required_inputs=("budget_data",),
            output_field="budget_validation",
        ),
        "clean_operations": PlanStep(
            name="clean_operations",
            purpose="Clean validated operational data.",
            required_inputs=("operations_data", "operations_validation"),
            output_field="cleaned_operations_data",
        ),
        "clean_budget": PlanStep(
            name="clean_budget",
            purpose="Clean validated budget data.",
            required_inputs=("budget_data", "budget_validation"),
            output_field="cleaned_budget_data",
        ),
        "profile_operations": PlanStep(
            name="profile_operations",
            purpose="Profile cleaned operational data for completeness and shape.",
            required_inputs=("cleaned_operations_data",),
            output_field="operations_profile",
        ),
        "profile_budget": PlanStep(
            name="profile_budget",
            purpose="Profile cleaned budget data for completeness and shape.",
            required_inputs=("cleaned_budget_data",),
            output_field="budget_profile",
        ),
        "operations_analysis": PlanStep(
            name="operations_analysis",
            purpose="Produce operational metrics required by finance agents.",
            required_inputs=("cleaned_operations_data",),
            output_field="operations_result",
        ),
        "budget": PlanStep(
            name="budget",
            purpose="Analyze the supplied budget data.",
            required_inputs=("cleaned_budget_data",),
            output_field="budget_result",
        ),
        "forecast": PlanStep(
            name="forecast",
            purpose="Generate the approved finance forecast.",
            required_inputs=("operations_result",),
            output_field="forecast_result",
        ),
        "scenario": PlanStep(
            name="scenario",
            purpose="Evaluate the requested business scenario.",
            required_inputs=(
                "forecast_result",
                "business_assumptions",
            ),
            output_field="scenario_result",
        ),
        "variance": PlanStep(
            name="variance",
            purpose="Calculate and reconcile actual-versus-budget variance.",
            required_inputs=("operations_result", "budget_result"),
            output_field="variance_result",
        ),
        "pnl": PlanStep(
            name="pnl",
            purpose="Generate the Profit and Loss statement.",
            required_inputs=(
                "cleaned_operations_data",
                "cleaned_budget_data",
            ),
            output_field="pnl_result",
        ),
        "finance_rules": PlanStep(
            name="finance_rules",
            purpose="Apply deterministic finance controls and validation rules.",
            required_inputs=("variance_result",),
            output_field="finance_rules_result",
        ),
        "anomaly": PlanStep(
            name="anomaly",
            purpose="Identify material exceptions in supplied finance results.",
            required_inputs=("variance_result",),
            output_field="anomaly_result",
        ),
        "root_cause": PlanStep(
            name="root_cause",
            purpose="Identify evidence-supported drivers of finance exceptions.",
            required_inputs=("variance_result", "anomaly_result"),
            output_field="root_cause_result",
        ),
        "recommendation": PlanStep(
            name="recommendation",
            purpose="Produce management actions supported by analysis results.",
            required_inputs=("root_cause_result",),
            output_field="recommendation_result",
        ),
        "kpi": PlanStep(
            name="kpi",
            purpose="Produce the KPI scorecard for the selected workflow.",
            required_inputs=(),
            output_field="kpi_result",
        ),
        "commentary": PlanStep(
            name="commentary",
            purpose="Create management commentary from completed analysis.",
            required_inputs=(),
            output_field="commentary_result",
        ),
        "report": PlanStep(
            name="report",
            purpose="Assemble the complete management report.",
            required_inputs=("commentary_result",),
            output_field="report_result",
        ),
        "complete": PlanStep(
            name="complete",
            purpose="Mark the selected workflow as successfully completed.",
            required_inputs=(),
            output_field=None,
        ),
    }
)


FLOW_STEP_NAMES: Mapping[FlowType, tuple[str, ...]] = MappingProxyType(
    {
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
        "pnl": (
            "validate_operations",
            "validate_budget",
            "clean_operations",
            "clean_budget",
            "pnl",
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
)


FLOW_REQUIRED_STATE_FIELDS: Mapping[FlowType, tuple[str, ...]] = (
    MappingProxyType(
        {
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
            "pnl": (
                "user_request",
                "operations_data",
                "budget_data",
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
    )
)


FLOW_DESCRIPTIONS: Mapping[FlowType, str] = MappingProxyType(
    {
        "kpi": "Analyze operational performance and produce KPI commentary.",
        "budget": "Validate, profile, and analyze budget performance.",
        "forecast": "Analyze operations and produce a finance forecast.",
        "variance": (
            "Analyze actual-versus-budget variance, exceptions, root causes, "
            "recommendations, KPIs, and management commentary."
        ),
        "scenario": (
            "Create a forecast, apply business assumptions, and evaluate a "
            "management scenario."
        ),
        "pnl": (
            "Generate a Profit and Loss statement from operational and "
            "budget data."
        ),
        "full": (
            "Run the complete end-to-end FP&A management-analysis workflow."
        ),
    }
)


def _build_steps(step_names: tuple[str, ...]) -> tuple[PlanStep, ...]:
    """
    Resolve approved logical names into immutable plan steps.

    Args:
        step_names:
            Ordered logical names configured for a workflow.

    Returns:
        Ordered immutable plan-step tuple.

    Raises:
        RuntimeError:
            If the planner configuration references an unknown step.
    """

    steps: list[PlanStep] = []

    for step_name in step_names:
        try:
            step = STEP_CATALOG[step_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Planner configuration references unknown step "
                f"{step_name!r}."
            ) from exc

        steps.append(step)

    return tuple(steps)


def create_execution_plan(selected_flow: FlowType | str) -> ExecutionPlan:
    """
    Create the approved execution plan for a selected workflow.

    Args:
        selected_flow:
            Workflow name returned by the deterministic router.

    Returns:
        Immutable validated execution plan.

    Raises:
        TypeError:
            If ``selected_flow`` is not a string.

        ValueError:
            If the workflow is empty, unknown, or unsupported.
    """

    if not isinstance(selected_flow, str):
        raise TypeError("selected_flow must be a string.")

    normalized_flow = selected_flow.strip().lower()

    if not normalized_flow:
        raise ValueError("selected_flow cannot be empty.")

    if normalized_flow not in SUPPORTED_PLANNING_FLOWS:
        raise ValueError(
            f"Unsupported finance workflow: {normalized_flow!r}. "
            f"Supported workflows: "
            f"{', '.join(SUPPORTED_PLANNING_FLOWS)}."
        )

    flow = normalized_flow

    step_names = FLOW_STEP_NAMES[flow]
    steps = _build_steps(step_names)

    expected_result_fields = tuple(
        step.output_field
        for step in steps
        if step.output_field is not None
        and step.output_field.endswith("_result")
    )

    return ExecutionPlan(
        selected_flow=flow,  # type: ignore[arg-type]
        steps=steps,
        required_state_fields=FLOW_REQUIRED_STATE_FIELDS[flow],
        expected_result_fields=expected_result_fields,
        description=FLOW_DESCRIPTIONS[flow],
    )


def plan_user_request(user_request: str) -> ExecutionPlan:
    """
    Route a natural-language request and create its execution plan.

    Args:
        user_request:
            Natural-language finance request.

    Returns:
        Approved execution plan for the identified workflow.

    Raises:
        TypeError:
            If the request is not a string.

        ValueError:
            If the request is empty or cannot be routed.
    """

    if not isinstance(user_request, str):
        raise TypeError("user_request must be a string.")

    cleaned_request = user_request.strip()

    if not cleaned_request:
        raise ValueError("user_request cannot be empty.")

    selected_flow = identify_flow(cleaned_request)

    if selected_flow == "unknown":
        raise ValueError(
            "Unable to identify a supported finance workflow from the "
            "user request."
        )

    return create_execution_plan(selected_flow)


def validate_plan_against_state(
    plan: ExecutionPlan,
    state: Mapping[str, object],
) -> tuple[str, ...]:
    """
    Return required initial state fields that are missing or ``None``.

    The function performs orchestration validation only. It does not inspect
    DataFrame contents or perform finance data-quality checks.

    Args:
        plan:
            Approved execution plan.

        state:
            Initial graph-state mapping.

    Returns:
        Tuple of missing required state-field names.
    """

    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan.")

    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")

    return tuple(
        field_name
        for field_name in plan.required_state_fields
        if field_name not in state or state[field_name] is None
    )


def get_supported_flows() -> tuple[FlowType, ...]:
    """Return the immutable collection of supported planner workflows."""

    return SUPPORTED_PLANNING_FLOWS