"""Tests for bounded autonomous execution coordination."""

from typing import Any

from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.agents.root_cause_recommendation_agent import (
    RootCauseRecommendationAgent,
)
from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.schemas import (
    PlanStep,
    ReviewResult,
    SupervisorPlan,
    ToolResult,
)
from src.autonomous.tools.data_tools import FinanceDataContext
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class ReviewClient(StructuredLLMClient):
    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake"

    def generate_structured(self, **kwargs: Any) -> Any:
        return StructuredLLMResponse[ReviewResult](
            output=ReviewResult(
                decision="approved",
                approved_answer="Reviewed answer.",
            ),
            usage=LLMUsage(),
            metadata=LLMRequestMetadata(
                provider="fake",
                model="fake",
                elapsed_seconds=0,
            ),
        )


class VarianceSpecialist:
    def __init__(self, calls: list[str], *, fail_once: bool = False) -> None:
        self.calls = calls
        self.fail_once = fail_once

    def execute(self, step: PlanStep, context: FinanceDataContext) -> ToolResult:
        self.calls.append(step.step_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary")
        return ToolResult(
            call_id="variance",
            tool_name="calculate_validated_revenue_variance",
            status="completed",
            payload={"variance_check": 0.0},
        )


def _plan() -> SupervisorPlan:
    return SupervisorPlan(
        objective="Explain variance",
        steps=(
            PlanStep(
                step_id="variance",
                capability="revenue_variance",
                arguments={
                    "agent_name": "variance_agent",
                    "tool_name": "calculate_validated_revenue_variance",
                },
            ),
            PlanStep(
                step_id="diagnostics",
                capability="root_cause_recommendation",
                depends_on=("variance",),
                arguments={
                    "agent_name": "diagnostic_agent",
                    "tool_name": "identify_supported_root_causes",
                    "recommendation_tool_name": (
                        "generate_supported_recommendations"
                    ),
                },
            ),
            PlanStep(
                step_id="review",
                capability="review",
                depends_on=("diagnostics",),
                arguments={"agent_name": "reviewer_agent"},
            ),
        ),
        required_reconciliations=("revenue_variance",),
        expected_outputs=("management_answer",),
    )


def _coordinator(
    specialist: Any,
) -> AutonomousExecutionCoordinator:
    diagnostics = RootCauseRecommendationAgent(
        root_cause_tool=lambda **kwargs: ToolResult(
            call_id="root",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload={"causes": ["volume"]},
        ),
        recommendation_tool=lambda **kwargs: ToolResult(
            call_id="recommend",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload={"recommendations": ["monitor"]},
        ),
    )
    return AutonomousExecutionCoordinator(
        reviewer=ReviewerAgent(ReviewClient()),
        diagnostics=diagnostics,
        specialist_agents={"revenue_variance": specialist},
    )


def _execute(coordinator: AutonomousExecutionCoordinator) -> Any:
    return coordinator.execute(
        _plan(),
        context=FinanceDataContext(),
        draft_answer="Draft.",
        anomaly_result={},
        operations_result={},
    )


def test_coordinator_completes_controlled_workflow() -> None:
    calls: list[str] = []

    result = _execute(_coordinator(VarianceSpecialist(calls)))

    assert result.status == "completed"
    assert result.management_response.answer == "Reviewed answer."
    assert result.evidence[0].result_type == "revenue_variance"
    assert result.evidence[0].reconciled is True
    assert calls == ["variance"]


def test_coordinator_retries_one_failed_agent_attempt() -> None:
    calls: list[str] = []

    result = _execute(
        _coordinator(VarianceSpecialist(calls, fail_once=True))
    )

    assert result.status == "completed"
    assert calls == ["variance", "variance"]
    assert result.usage.agent_runs == 4


def test_coordinator_falls_back_for_unknown_specialist() -> None:
    result = _execute(_coordinator(VarianceSpecialist([])))
    invalid = _plan().model_copy(
        update={
            "steps": (
                _plan().steps[0].model_copy(
                    update={"capability": "unknown_finance"}
                ),
                *_plan().steps[1:],
            )
        }
    )

    fallback = _coordinator(VarianceSpecialist([])).execute(
        invalid,
        context=FinanceDataContext(),
        draft_answer="Draft.",
        anomaly_result={},
        operations_result={},
    )

    assert result.status == "completed"
    assert fallback.status == "fallback"
    assert fallback.fallback_flow == "deterministic_planner"


def test_coordinator_preserves_dependency_order() -> None:
    calls: list[str] = []
    first = _plan().steps[0].model_copy(
        update={"step_id": "first"}
    )
    second = _plan().steps[0].model_copy(
        update={"step_id": "second", "depends_on": ("first",)}
    )
    diagnostics = _plan().steps[1].model_copy(
        update={"depends_on": ("second",)}
    )
    plan = _plan().model_copy(
        update={"steps": (second, first, diagnostics, _plan().steps[2])}
    )

    result = _coordinator(VarianceSpecialist(calls)).execute(
        plan,
        context=FinanceDataContext(),
        draft_answer="Draft.",
        anomaly_result={},
        operations_result={},
    )

    assert result.status == "completed"
    assert calls == ["first", "second"]
