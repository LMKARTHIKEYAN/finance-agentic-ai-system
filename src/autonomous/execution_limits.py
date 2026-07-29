"""Deterministic execution limits for the future autonomous workflow."""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous.schemas import AutonomousExecutionUsage


class ExecutionLimitExceededError(RuntimeError):
    """Raised when an autonomous execution exceeds an approved limit."""


class AutonomousExecutionLimits(BaseModel):
    """Immutable autonomous execution limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_agents: int = Field(default=6, gt=0)
    max_replans: int = Field(default=2, ge=0)
    max_retries_per_agent: int = Field(default=1, ge=0)
    max_tool_calls: int = Field(default=10, gt=0)
    max_input_tokens: int = Field(default=12_000, gt=0)
    max_output_tokens: int = Field(default=1_200, gt=0)
    max_total_tokens: int = Field(default=20_000, gt=0)
    max_cost_usd: float = Field(default=0.03, gt=0.0)
    max_execution_seconds: float = Field(default=120.0, gt=0.0)


class ExecutionUsageTracker:
    """Track usage and reject work that exceeds configured limits."""

    def __init__(
        self,
        limits: AutonomousExecutionLimits | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")

        self._limits = limits or AutonomousExecutionLimits()
        self._clock = clock
        self._started_at = clock()
        self._agent_runs = 0
        self._agent_names: set[str] = set()
        self._replans = 0
        self._tool_calls = 0
        self._agent_retries: dict[str, int] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._estimated_cost_usd = 0.0

    @property
    def limits(self) -> AutonomousExecutionLimits:
        return self._limits

    def record_agent_run(self, agent_name: str | None = None) -> None:
        self.check_time()
        cleaned_name = (
            _required_text(agent_name, "agent_name")
            if agent_name is not None
            else f"anonymous_agent_{self._agent_runs + 1}"
        )
        next_names = {*self._agent_names, cleaned_name}
        if len(next_names) > self._limits.max_agents:
            raise ExecutionLimitExceededError(
                "Maximum distinct autonomous agents exceeded."
            )
        maximum_runs = self._limits.max_agents * (
            self._limits.max_retries_per_agent + 1
        )
        if self._agent_runs >= maximum_runs:
            raise ExecutionLimitExceededError(
                "Maximum autonomous agent execution attempts exceeded."
            )
        self._agent_names = next_names
        self._agent_runs += 1

    def record_replan(self) -> None:
        self.check_time()
        if self._replans >= self._limits.max_replans:
            raise ExecutionLimitExceededError(
                "Maximum supervisor replans exceeded."
            )
        self._replans += 1

    def record_agent_retry(self, agent_name: str) -> None:
        self.check_time()
        cleaned_name = _required_text(agent_name, "agent_name")
        current = self._agent_retries.get(cleaned_name, 0)
        if current >= self._limits.max_retries_per_agent:
            raise ExecutionLimitExceededError(
                f"Maximum retries exceeded for agent {cleaned_name!r}."
            )
        self._agent_retries[cleaned_name] = current + 1

    def record_tool_call(self) -> None:
        self.check_time()
        if self._tool_calls >= self._limits.max_tool_calls:
            raise ExecutionLimitExceededError(
                "Maximum autonomous tool calls exceeded."
            )
        self._tool_calls += 1

    def record_llm_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        self.check_time()
        validated_input = _non_negative_integer(
            input_tokens,
            "input_tokens",
        )
        validated_output = _non_negative_integer(
            output_tokens,
            "output_tokens",
        )
        validated_cost = _non_negative_number(
            estimated_cost_usd,
            "estimated_cost_usd",
        )

        if validated_input > self._limits.max_input_tokens:
            raise ExecutionLimitExceededError(
                "LLM input exceeded the per-request token limit."
            )
        if validated_output > self._limits.max_output_tokens:
            raise ExecutionLimitExceededError(
                "LLM output exceeded the per-request token limit."
            )

        next_total = (
            self._total_tokens
            + validated_input
            + validated_output
        )
        next_cost = self._estimated_cost_usd + validated_cost

        if next_total > self._limits.max_total_tokens:
            raise ExecutionLimitExceededError(
                "Autonomous workflow exceeded the total-token limit."
            )
        if next_cost > self._limits.max_cost_usd:
            raise ExecutionLimitExceededError(
                "Autonomous workflow exceeded the cost limit."
            )

        self._input_tokens += validated_input
        self._output_tokens += validated_output
        self._total_tokens = next_total
        self._estimated_cost_usd = next_cost

    def check_time(self) -> None:
        if self.elapsed_seconds > self._limits.max_execution_seconds:
            raise ExecutionLimitExceededError(
                "Autonomous workflow exceeded the execution-time limit."
            )

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    def snapshot(self) -> AutonomousExecutionUsage:
        return AutonomousExecutionUsage(
            agent_runs=self._agent_runs,
            replans=self._replans,
            tool_calls=self._tool_calls,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._total_tokens,
            estimated_cost_usd=self._estimated_cost_usd,
            elapsed_seconds=self.elapsed_seconds,
        )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return value


def _non_negative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    numeric_value = float(value)
    if numeric_value < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return numeric_value
