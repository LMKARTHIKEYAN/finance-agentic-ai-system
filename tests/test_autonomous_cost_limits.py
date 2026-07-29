"""Execution-limit tests at the autonomous coordinator boundary."""

from typing import Any

import pytest

from src.autonomous.agents.reviewer_agent import ReviewerAgent
from src.autonomous.execution_coordinator import (
    AutonomousExecutionCoordinator,
)
from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.execution_limits import (
    ExecutionLimitExceededError,
    ExecutionUsageTracker,
)
from src.autonomous.schemas import PlanStep, ReviewResult, SupervisorPlan
from src.autonomous.tools.data_tools import FinanceDataContext
from src.llm.client import StructuredLLMClient


class UnusedClient(StructuredLLMClient):
    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake"

    def generate_structured(self, **kwargs: Any) -> Any:
        raise AssertionError("LLM should not be called")


def _plan() -> SupervisorPlan:
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


def _run(
    limits: AutonomousExecutionLimits,
    usage: tuple[tuple[int, int, float], ...],
) -> Any:
    coordinator = AutonomousExecutionCoordinator(
        reviewer=ReviewerAgent(UnusedClient(), limits=limits),
        limits=limits,
    )
    return coordinator.execute(
        _plan(),
        context=FinanceDataContext(),
        draft_answer="Draft",
        anomaly_result={},
        operations_result={},
        llm_usage=usage,
    )


def test_total_token_limit_returns_deterministic_fallback() -> None:
    limits = AutonomousExecutionLimits(
        max_input_tokens=100,
        max_output_tokens=100,
        max_total_tokens=10,
    )

    result = _run(limits, ((6, 5, 0.0),))

    assert result.status == "fallback"
    assert "total-token limit" in result.fallback_reason


def test_cost_limit_returns_deterministic_fallback() -> None:
    limits = AutonomousExecutionLimits(max_cost_usd=0.01)

    result = _run(limits, ((1, 1, 0.02),))

    assert result.status == "fallback"
    assert "cost limit" in result.fallback_reason


def test_per_request_output_limit_returns_fallback() -> None:
    limits = AutonomousExecutionLimits(max_output_tokens=5)

    result = _run(limits, ((1, 6, 0.0),))

    assert result.status == "fallback"
    assert "per-request token limit" in result.fallback_reason


def test_six_agent_run_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_agents=6)
    )
    for _ in range(6):
        tracker.record_agent_run()

    with pytest.raises(
        ExecutionLimitExceededError,
        match="distinct autonomous agents",
    ):
        tracker.record_agent_run()


def test_one_retry_per_agent_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_retries_per_agent=1)
    )
    tracker.record_agent_retry("pnl_agent")

    with pytest.raises(
        ExecutionLimitExceededError,
        match="retries",
    ):
        tracker.record_agent_retry("pnl_agent")


def test_tool_call_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_tool_calls=1)
    )
    tracker.record_tool_call()

    with pytest.raises(
        ExecutionLimitExceededError,
        match="tool calls",
    ):
        tracker.record_tool_call()


def test_execution_time_limit_is_enforced() -> None:
    ticks = iter((0.0, 2.0))
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_execution_seconds=1.0),
        clock=lambda: next(ticks),
    )

    with pytest.raises(
        ExecutionLimitExceededError,
        match="execution-time",
    ):
        tracker.check_time()
