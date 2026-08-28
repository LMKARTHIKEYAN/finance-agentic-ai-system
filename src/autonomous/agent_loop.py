"""Bounded Decide -> Act -> Observe loop for autonomous finance goals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.autonomous.decision_engine import DecisionEngine
from src.autonomous.execution_limits import (
    AutonomousExecutionLimits,
    ExecutionLimitExceededError,
    ExecutionUsageTracker,
)
from src.autonomous.observation_manager import ObservationManager
from src.autonomous.schemas import AgentLoopResult, StopDecision
from src.autonomous.state import AutonomousState
from src.autonomous.stop_conditions import StopConditions
from src.autonomous.tool_executor import ToolExecutionError, ToolExecutor


ValidationCallback = Callable[
    [AutonomousState],
    tuple[bool, str, dict[str, Any]],
]


class AgentLoop:
    """Run one autonomous goal until completion, pause, or safe stop."""

    def __init__(
        self,
        *,
        decision_engine: DecisionEngine,
        tool_executor: ToolExecutor,
        observation_manager: ObservationManager | None = None,
        stop_conditions: StopConditions | None = None,
        limits: AutonomousExecutionLimits | None = None,
        validator: ValidationCallback | None = None,
    ) -> None:
        if not isinstance(decision_engine, DecisionEngine):
            raise TypeError("decision_engine must be a DecisionEngine.")
        if not isinstance(tool_executor, ToolExecutor):
            raise TypeError("tool_executor must be a ToolExecutor.")
        self.decision_engine = decision_engine
        self.tool_executor = tool_executor
        self.observation_manager = observation_manager or ObservationManager()
        self.limits = limits or AutonomousExecutionLimits()
        self.stop_conditions = stop_conditions or StopConditions(
            limits=self.limits
        )
        self.validator = validator or _default_validator

    def run(self, state: AutonomousState) -> AgentLoopResult:
        if not isinstance(state, AutonomousState):
            raise TypeError("state must be an AutonomousState.")
        tracker = ExecutionUsageTracker(self.limits)
        state.status = "running"
        if state.pending_question and all(
            key in state.clarification_answers
            for key in state.goal.ambiguities
        ):
            state.pending_question = None

        while True:
            try:
                tracker.check_time()
            except ExecutionLimitExceededError as exc:
                state.status = "failed"
                state.stop_reason = str(exc)
                return _result(
                    state,
                    StopDecision(
                        outcome="limit_reached",
                        reason=str(exc),
                        missing_criteria=tuple(sorted(state.missing_criteria)),
                    ),
                )

            stop = self.stop_conditions.evaluate(state)
            if stop.outcome != "continue":
                return _finish_from_stop(state, stop)

            state.iteration += 1
            decision = self.decision_engine.decide(state)
            state.record_decision(decision)

            if decision.action == "ask_user":
                state.status = "waiting_for_user"
                state.pending_question = decision.question
                return _result(
                    state,
                    StopDecision(
                        outcome="ask_user",
                        reason=decision.question or decision.rationale,
                        missing_criteria=tuple(sorted(state.missing_criteria)),
                    ),
                    question=decision.question,
                )
            if decision.action == "request_approval":
                state.status = "waiting_for_approval"
                state.pending_approval = decision.question
                return _result(
                    state,
                    StopDecision(
                        outcome="request_approval",
                        reason=decision.question or decision.rationale,
                        missing_criteria=tuple(sorted(state.missing_criteria)),
                    ),
                    question=decision.question,
                )
            if decision.action == "stop":
                state.status = "failed"
                state.stop_reason = decision.rationale
                return _result(
                    state,
                    StopDecision(
                        outcome="failed",
                        reason=decision.rationale,
                        missing_criteria=tuple(sorted(state.missing_criteria)),
                    ),
                )
            if decision.action == "finalize":
                state.final_answer = decision.final_answer
                state.status = "completed"
                return _result(
                    state,
                    StopDecision(
                        outcome="complete",
                        reason="The supervisor finalized the completed goal.",
                    ),
                    answer=decision.final_answer,
                )
            if decision.action == "validate":
                passed, details, payload = self.validator(state)
                observation = self.observation_manager.validation_observation(
                    decision,
                    passed=passed,
                    details=details,
                    payload=payload,
                )
                state.record_observation(observation)
                continue
            if decision.action != "call_tool":
                state.status = "failed"
                return _result(
                    state,
                    StopDecision(
                        outcome="failed",
                        reason=f"Unsupported decision action: {decision.action}.",
                    ),
                )

            try:
                tracker.record_tool_call()
                state.tool_call_count += 1
                tool_result = self.tool_executor.execute(decision)
                observation = self.observation_manager.from_tool_result(
                    decision,
                    tool_result,
                )
            except (ToolExecutionError, ExecutionLimitExceededError) as exc:
                state.status = "failed"
                state.errors.append(str(exc))
                return _result(
                    state,
                    StopDecision(
                        outcome=(
                            "limit_reached"
                            if isinstance(exc, ExecutionLimitExceededError)
                            else "failed"
                        ),
                        reason=str(exc),
                        missing_criteria=tuple(sorted(state.missing_criteria)),
                    ),
                )
            state.record_observation(observation)


def _default_validator(
    state: AutonomousState,
) -> tuple[bool, str, dict[str, Any]]:
    failed = [
        observation.source for observation in state.observations
        if observation.status == "failed"
    ]
    if failed:
        return False, "Final validation failed because tool errors remain.", {
            "failed_sources": failed
        }
    evidence_ids = [
        evidence_id
        for observation in state.observations
        for evidence_id in observation.evidence_ids
    ]
    if not evidence_ids:
        return False, "Final validation failed because no evidence exists.", {}
    return True, "Final evidence validation passed.", {
        "evidence_ids": evidence_ids
    }


def _finish_from_stop(
    state: AutonomousState,
    stop: StopDecision,
) -> AgentLoopResult:
    if stop.outcome == "complete":
        state.status = "completed"
        state.final_answer = state.final_answer or _answer_from_observations(state)
        state.stop_reason = stop.reason
        return _result(state, stop, answer=state.final_answer)
    if stop.outcome == "ask_user":
        state.status = "waiting_for_user"
        return _result(state, stop, question=state.pending_question)
    if stop.outcome == "request_approval":
        state.status = "waiting_for_approval"
        return _result(state, stop, question=state.pending_approval)
    state.status = "failed"
    state.stop_reason = stop.reason
    return _result(state, stop)


def _answer_from_observations(state: AutonomousState) -> str:
    summaries = [
        observation.summary
        for observation in state.observations
        if observation.status == "completed"
        and observation.source != "final_validation"
    ]
    return " ".join(summaries) or f"Goal completed: {state.goal.objective}"


def _result(
    state: AutonomousState,
    stop: StopDecision,
    *,
    answer: str | None = None,
    question: str | None = None,
) -> AgentLoopResult:
    return AgentLoopResult(
        goal_id=state.goal.goal_id,
        status=state.status,  # type: ignore[arg-type]
        stop=stop,
        answer=answer,
        question=question,
        iterations=state.iteration,
        tool_calls=state.tool_call_count,
        observations=tuple(state.observations),
    )
