"""State definition for the isolated autonomous workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from src.autonomous.schemas import (
    AutonomousExecutionUsage,
    ComplexityDecision,
    EvidenceReference,
    ManagementResponse,
    RecommendationFinding,
    ReconciliationResult,
    ReviewResult,
    RootCauseFinding,
    SupervisorPlan,
    ToolResult,
)


class AutonomousGraphState(TypedDict, total=False):
    """Shared state used only by the future autonomous LangGraph."""

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
