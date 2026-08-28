"""Shared mutable state for the bounded autonomous agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from src.autonomous.schemas import (
    AutonomousExecutionUsage,
    AutonomousDecision,
    ComplexityDecision,
    EvidenceReference,
    FinanceGoal,
    ManagementResponse,
    Observation,
    RecommendationFinding,
    ReconciliationResult,
    ReviewResult,
    RootCauseFinding,
    SupervisorPlan,
    ToolResult,
)


class AutonomousGraphState(TypedDict, total=False):
    """Compatibility shape for graph or API serialization."""

    user_request: str
    resolved_request: str
    parsed_intent: Any
    complexity_decision: ComplexityDecision
    execution_mode: str
    supervisor_plan: SupervisorPlan
    validated_plan: SupervisorPlan
    replan_count: int
    agent_retry_counts: dict[str, int]
    selected_agents: list[str]
    tool_results: list[ToolResult]
    evidence: list[EvidenceReference]
    root_causes: list[RootCauseFinding]
    recommendations: list[RecommendationFinding]
    reconciliation_result: ReconciliationResult
    review_result: ReviewResult
    management_response: ManagementResponse
    execution_usage: AutonomousExecutionUsage
    execution_status: str
    fallback_flow: str
    fallback_reason: str
    errors: list[str]


@dataclass(slots=True)
class AutonomousState:
    """Mutable runtime state shared by every Phase 1 loop component."""

    goal: FinanceGoal
    status: str = "pending"
    iteration: int = 0
    tool_call_count: int = 0
    completed_criteria: set[str] = field(default_factory=set)
    decisions: list[AutonomousDecision] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    clarification_answers: dict[str, str] = field(default_factory=dict)
    pending_question: str | None = None
    pending_approval: str | None = None
    final_answer: str | None = None
    stop_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.goal, FinanceGoal):
            raise TypeError("goal must be a FinanceGoal.")

    @property
    def required_criteria(self) -> set[str]:
        return {
            item.key for item in self.goal.criteria if item.required
        }

    @property
    def missing_criteria(self) -> set[str]:
        return self.required_criteria - self.completed_criteria

    @property
    def latest_observation(self) -> Observation | None:
        return self.observations[-1] if self.observations else None

    def record_decision(self, decision: AutonomousDecision) -> None:
        if not isinstance(decision, AutonomousDecision):
            raise TypeError("decision must be an AutonomousDecision.")
        self.decisions.append(decision)

    def record_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation.")
        self.observations.append(observation)
        if observation.status == "completed":
            self.completed_criteria.update(observation.satisfied_criteria)
            self.context.update(observation.payload)
        elif observation.status == "failed":
            self.errors.append(observation.summary)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return compact LLM-safe state without raw credentials or frames."""

        return {
            "goal": self.goal.model_dump(mode="json"),
            "status": self.status,
            "iteration": self.iteration,
            "tool_call_count": self.tool_call_count,
            "completed_criteria": sorted(self.completed_criteria),
            "missing_criteria": sorted(self.missing_criteria),
            "latest_observation": (
                self.latest_observation.model_dump(mode="json")
                if self.latest_observation is not None
                else None
            ),
            "pending_question": self.pending_question,
            "pending_approval": self.pending_approval,
            "errors": tuple(self.errors),
        }
