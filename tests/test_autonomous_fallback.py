"""Fallback-path tests for autonomous coordination."""

from typing import Any

from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.agents.root_cause_recommendation_agent import (
    RootCauseRecommendationAgent,
)
from src.autonomous.schemas import PlanStep, ReviewResult, SupervisorPlan
from src.autonomous.tools.data_tools import FinanceDataContext
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class Client(StructuredLLMClient):
    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake"

    def generate_structured(self, **kwargs: Any) -> Any:
        return StructuredLLMResponse[ReviewResult](
            output=ReviewResult(decision="failed"),
            usage=LLMUsage(),
            metadata=LLMRequestMetadata(
                provider="fake", model="fake", elapsed_seconds=0
            ),
        )


def _minimal_plan() -> SupervisorPlan:
    return SupervisorPlan(
        objective="Analyze",
        steps=(
            PlanStep(
                step_id="review",
                capability="review",
                arguments={"agent_name": "reviewer_agent"},
            ),
        ),
        expected_outputs=("answer",),
    )


def test_missing_specialist_steps_returns_fallback() -> None:
    coordinator = AutonomousExecutionCoordinator(
        reviewer=ReviewerAgent(Client()),
        diagnostics=RootCauseRecommendationAgent(),
    )

    result = coordinator.execute(
        _minimal_plan(),
        context=FinanceDataContext(),
        draft_answer="Draft",
        anomaly_result={},
        operations_result={},
    )

    assert result.status == "fallback"
    assert result.fallback_flow == "deterministic_planner"
    assert result.management_response is None


def test_fallback_hides_internal_exception_details() -> None:
    class Broken:
        def execute(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("secret dataframe row")

    plan = _minimal_plan().model_copy(
        update={
            "steps": (
                PlanStep(
                    step_id="kpi",
                    capability="kpi_analysis",
                    arguments={
                        "agent_name": "kpi_agent",
                        "tool_name": "calculate_validated_kpis",
                    },
                ),
                *_minimal_plan().steps,
            )
        }
    )
    coordinator = AutonomousExecutionCoordinator(
        reviewer=ReviewerAgent(Client()),
        specialist_agents={"kpi_analysis": Broken()},
    )

    result = coordinator.execute(
        plan,
        context=FinanceDataContext(),
        draft_answer="Draft",
        anomaly_result={},
        operations_result={},
    )

    assert result.status == "fallback"
    assert "secret dataframe row" not in result.fallback_reason
