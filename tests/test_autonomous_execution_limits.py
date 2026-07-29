"""Tests for deterministic autonomous execution limits."""

import pytest
from pydantic import ValidationError

from src.autonomous.execution_limits import (
    AutonomousExecutionLimits,
    ExecutionLimitExceededError,
    ExecutionUsageTracker,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_default_limits_match_architecture() -> None:
    limits = AutonomousExecutionLimits()

    assert limits.max_agents == 6
    assert limits.max_replans == 2
    assert limits.max_retries_per_agent == 1


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("max_agents", 0),
        ("max_replans", -1),
        ("max_retries_per_agent", -1),
        ("max_tool_calls", 0),
        ("max_total_tokens", 0),
        ("max_cost_usd", 0),
        ("max_execution_seconds", 0),
    ],
)
def test_limits_reject_invalid_configuration(
    field_name: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        AutonomousExecutionLimits(**{field_name: value})


def test_agent_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_agents=2)
    )
    tracker.record_agent_run()
    tracker.record_agent_run()

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_agent_run()


def test_allowed_retry_does_not_consume_distinct_agent_limit() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(
            max_agents=2,
            max_retries_per_agent=1,
        )
    )

    tracker.record_agent_run("pnl")
    tracker.record_agent_retry("pnl")
    tracker.record_agent_run("pnl")
    tracker.record_agent_run("reviewer")
    tracker.record_agent_retry("reviewer")
    tracker.record_agent_run("reviewer")

    assert tracker.snapshot().agent_runs == 4

    with pytest.raises(
        ExecutionLimitExceededError,
        match="execution attempts",
    ):
        tracker.record_agent_run("reviewer")


def test_distinct_agent_limit_is_enforced_separately() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(
            max_agents=2,
            max_retries_per_agent=1,
        )
    )
    tracker.record_agent_run("pnl")
    tracker.record_agent_run("reviewer")

    with pytest.raises(
        ExecutionLimitExceededError,
        match="distinct",
    ):
        tracker.record_agent_run("third_agent")


def test_replan_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_replans=2)
    )
    tracker.record_replan()
    tracker.record_replan()

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_replan()


def test_retry_limit_is_tracked_per_agent() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_retries_per_agent=1)
    )
    tracker.record_agent_retry("pnl")
    tracker.record_agent_retry("reviewer")

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_agent_retry("pnl")


def test_tool_call_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_tool_calls=1)
    )
    tracker.record_tool_call()

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_tool_call()


def test_token_and_cost_usage_is_recorded() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(
            max_input_tokens=100,
            max_output_tokens=50,
            max_total_tokens=200,
            max_cost_usd=1,
        )
    )

    tracker.record_llm_usage(
        input_tokens=80,
        output_tokens=20,
        estimated_cost_usd=0.1,
    )

    usage = tracker.snapshot()
    assert usage.input_tokens == 80
    assert usage.output_tokens == 20
    assert usage.total_tokens == 100
    assert usage.estimated_cost_usd == pytest.approx(0.1)


def test_total_token_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(
            max_input_tokens=100,
            max_output_tokens=100,
            max_total_tokens=100,
        )
    )
    tracker.record_llm_usage(input_tokens=40, output_tokens=10)

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_llm_usage(input_tokens=40, output_tokens=20)


def test_cost_limit_is_enforced() -> None:
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_cost_usd=0.1)
    )

    with pytest.raises(ExecutionLimitExceededError):
        tracker.record_llm_usage(
            input_tokens=1,
            output_tokens=1,
            estimated_cost_usd=0.11,
        )


def test_time_limit_is_enforced() -> None:
    clock = FakeClock()
    tracker = ExecutionUsageTracker(
        AutonomousExecutionLimits(max_execution_seconds=5),
        clock=clock,
    )
    clock.value = 6

    with pytest.raises(ExecutionLimitExceededError):
        tracker.check_time()
