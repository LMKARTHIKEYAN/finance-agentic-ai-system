"""Deterministic completion and safety-stop rules for the agent loop."""

from __future__ import annotations

from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.schemas import StopDecision
from src.autonomous.state import AutonomousState


class StopConditions:
    """Evaluate success, clarification, approval, failure, and hard limits."""

    def __init__(
        self,
        *,
        limits: AutonomousExecutionLimits | None = None,
        max_iterations: int = 20,
        max_consecutive_failures: int = 2,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive.")
        self.limits = limits or AutonomousExecutionLimits()
        self.max_iterations = max_iterations
        self.max_consecutive_failures = max_consecutive_failures

    def evaluate(self, state: AutonomousState) -> StopDecision:
        if not isinstance(state, AutonomousState):
            raise TypeError("state must be an AutonomousState.")
        if state.pending_approval:
            return StopDecision(
                outcome="request_approval",
                reason=state.pending_approval,
                missing_criteria=tuple(sorted(state.missing_criteria)),
            )
        if state.pending_question:
            return StopDecision(
                outcome="ask_user",
                reason=state.pending_question,
                missing_criteria=tuple(sorted(state.missing_criteria)),
            )
        if not state.missing_criteria:
            return StopDecision(
                outcome="complete",
                reason="All required goal criteria are satisfied.",
            )
        if state.iteration >= self.max_iterations:
            return StopDecision(
                outcome="limit_reached",
                reason="Maximum autonomous loop iterations reached.",
                missing_criteria=tuple(sorted(state.missing_criteria)),
            )
        if state.tool_call_count >= self.limits.max_tool_calls:
            return StopDecision(
                outcome="limit_reached",
                reason="Maximum autonomous tool calls reached.",
                missing_criteria=tuple(sorted(state.missing_criteria)),
            )
        failures = 0
        for observation in reversed(state.observations):
            if observation.status != "failed":
                break
            failures += 1
        if failures >= self.max_consecutive_failures:
            return StopDecision(
                outcome="failed",
                reason="Repeated autonomous actions failed.",
                missing_criteria=tuple(sorted(state.missing_criteria)),
            )
        return StopDecision(
            outcome="continue",
            reason="The goal remains incomplete and execution may continue.",
            missing_criteria=tuple(sorted(state.missing_criteria)),
        )
