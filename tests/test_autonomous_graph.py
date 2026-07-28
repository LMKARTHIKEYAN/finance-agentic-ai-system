"""Tests for the isolated supervised autonomous LangGraph."""

from typing import Any

from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.agents.supervisor_agent import FinanceSupervisorAgent
from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.graph import build_autonomous_graph
from src.autonomous.plan_validator import AutonomousPlanValidator
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    PlanStep,
    ReviewResult,
    SupervisorPlan,
)
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMStructuredOutputError,
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class QueueClient(StructuredLLMClient):
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, Any]] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake"

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return StructuredLLMResponse(
            output=output,
            usage=LLMUsage(),
            metadata=LLMRequestMetadata(
                provider="fake", model="fake", elapsed_seconds=0
            ),
        )


def _valid_plan() -> SupervisorPlan:
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


def _invalid_plan() -> SupervisorPlan:
    return SupervisorPlan(
        objective="Analyze",
        steps=(
            PlanStep(
                step_id="unsafe",
                capability="analysis",
                arguments={
                    "agent_name": "unsafe_agent",
                    "tool_name": "execute_python",
                },
            ),
        ),
        expected_outputs=("answer",),
    )


def _coordinator() -> AutonomousExecutionCoordinator:
    reviewer_client = QueueClient([ReviewResult(decision="failed")])
    return AutonomousExecutionCoordinator(
        reviewer=ReviewerAgent(reviewer_client)
    )


def test_graph_executes_validated_plan() -> None:
    supervisor_client = QueueClient([_valid_plan()])
    coordinator = _coordinator()
    completed = AutonomousExecutionResult(status="completed")
    calls: list[SupervisorPlan] = []

    def execute(plan: SupervisorPlan, **kwargs: Any) -> Any:
        calls.append(plan)
        return completed

    coordinator.execute = execute  # type: ignore[method-assign]
    graph = build_autonomous_graph(
        supervisor=FinanceSupervisorAgent(supervisor_client),
        validator=AutonomousPlanValidator(),
        coordinator=coordinator,
    )

    state = graph.invoke(
        {
            "request": "Why did profit improve?",
            "available_inputs": set(),
            "execution_arguments": {},
        }
    )

    assert state["result"] == completed
    assert calls == [_valid_plan()]


def test_graph_replans_invalid_plan_then_executes() -> None:
    supervisor_client = QueueClient([_invalid_plan(), _valid_plan()])
    coordinator = _coordinator()
    coordinator.execute = lambda plan, **kwargs: (  # type: ignore[method-assign]
        AutonomousExecutionResult(status="completed")
    )
    graph = build_autonomous_graph(
        supervisor=FinanceSupervisorAgent(supervisor_client),
        validator=AutonomousPlanValidator(),
        coordinator=coordinator,
    )

    state = graph.invoke(
        {
            "request": "Analyze",
            "available_inputs": set(),
            "execution_arguments": {},
        }
    )

    assert state["result"].status == "completed"
    assert state["replan_count"] == 1
    assert len(supervisor_client.calls) == 2
    second_context = supervisor_client.calls[1]["messages"][1]["content"]
    assert "unknown_tool" in second_context


def test_graph_falls_back_after_replan_limit() -> None:
    supervisor_client = QueueClient(
        [_invalid_plan(), _invalid_plan()]
    )
    graph = build_autonomous_graph(
        supervisor=FinanceSupervisorAgent(supervisor_client),
        validator=AutonomousPlanValidator(),
        coordinator=_coordinator(),
        limits=AutonomousExecutionLimits(max_replans=1),
    )

    state = graph.invoke(
        {
            "request": "Analyze",
            "available_inputs": set(),
            "execution_arguments": {},
        }
    )

    assert state["result"].status == "fallback"
    assert state["result"].fallback_flow == "deterministic_planner"
    assert state["replan_count"] == 1


def test_graph_supervisor_failure_returns_fallback() -> None:
    class FailingClient(QueueClient):
        def generate_structured(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider failure")

    graph = build_autonomous_graph(
        supervisor=FinanceSupervisorAgent(FailingClient([])),
        validator=AutonomousPlanValidator(),
        coordinator=_coordinator(),
    )

    state = graph.invoke(
        {
            "request": "Analyze",
            "available_inputs": set(),
            "execution_arguments": {},
        }
    )

    assert state["result"].status == "fallback"
    assert state["result"].fallback_reason == (
        "Supervisor planning failed: unexpected_error."
    )


def test_graph_reports_safe_supervisor_failure_category() -> None:
    secret = "secret raw finance data"

    class InvalidStructuredOutputClient(QueueClient):
        def generate_structured(self, **kwargs: Any) -> Any:
            raise LLMStructuredOutputError(secret)

    graph = build_autonomous_graph(
        supervisor=FinanceSupervisorAgent(
            InvalidStructuredOutputClient([])
        ),
        validator=AutonomousPlanValidator(),
        coordinator=_coordinator(),
    )

    state = graph.invoke(
        {
            "request": "Analyze",
            "available_inputs": set(),
            "execution_arguments": {},
        }
    )

    reason = state["result"].fallback_reason
    assert reason == (
        "Supervisor planning failed: structured_output_invalid."
    )
    assert secret not in reason
