"""Structured contracts for the supervised autonomous workflow."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


ExecutionMode = Literal["deterministic", "autonomous"]
RequestType = Literal[
    "simple_report",
    "simple_calculation",
    "diagnostic",
    "decision_support",
    "multi_analysis",
    "unknown",
]
ToolStatus = Literal["completed", "failed", "skipped"]
ReviewDecision = Literal[
    "approved",
    "approved_with_caveats",
    "replan_required",
    "failed",
]
AutonomousStatus = Literal[
    "pending",
    "running",
    "completed",
    "fallback",
    "failed",
]
IssueSeverity = Literal["error", "warning"]


class FrozenModel(BaseModel):
    """Base class for immutable autonomous workflow contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ComplexityDecision(FrozenModel):
    """Validated decision selecting deterministic or autonomous execution."""

    execution_mode: ExecutionMode
    request_type: RequestType
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(min_length=1)
    fallback_flow: str = Field(default="unknown", min_length=1)


class ReportingScope(FrozenModel):
    """Reporting period and optional business dimension."""

    start_date: date | None = None
    end_date: date | None = None
    comparison_start_date: date | None = None
    comparison_end_date: date | None = None
    category: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "ReportingScope":
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date cannot be after end_date.")
        if (
            self.comparison_start_date is not None
            and self.comparison_end_date is not None
            and self.comparison_start_date > self.comparison_end_date
        ):
            raise ValueError(
                "comparison_start_date cannot be after "
                "comparison_end_date."
            )
        return self


class DatasetAvailability(FrozenModel):
    """Metadata describing one internally available dataset."""

    dataset_name: str = Field(min_length=1)
    available: bool
    row_count: int | None = Field(default=None, ge=0)
    period_start: date | None = None
    period_end: date | None = None


class PlanStepArguments(FrozenModel):
    """Allow-listed arguments an LLM may assign to a plan step."""

    agent_name: str | None = None
    tool_name: str | None = None
    recommendation_tool_name: str | None = None
    requested_kpis: list[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    category: str | None = None
    start_month: str | None = None
    end_month: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    forecast_period: str | None = None
    scenario_period: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_arguments(cls, value: Any) -> Any:
        """Reject unapproved fields with the established safety wording."""

        if isinstance(value, dict):
            allowed = {
                "agent_name",
                "tool_name",
                "recommendation_tool_name",
                "requested_kpis",
                "start_date",
                "end_date",
                "category",
                "start_month",
                "end_month",
                "dimension",
                "dimension_value",
                "forecast_period",
                "scenario_period",
            }
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    "unsupported step arguments: "
                    f"{sorted(unknown)}."
                )
        return value

    def to_execution_dict(self) -> dict[str, Any]:
        """Return only explicitly populated execution arguments."""

        return self.model_dump(exclude_none=True, mode="python")

    def get(self, key: str, default: Any = None) -> Any:
        """Provide read-only dictionary compatibility to existing agents."""

        return self.to_execution_dict().get(key, default)

    def keys(self) -> Any:
        """Return populated argument names."""

        return self.to_execution_dict().keys()

    def __getitem__(self, key: str) -> Any:
        return self.to_execution_dict()[key]

    def __iter__(self) -> Any:
        return iter(self.to_execution_dict())

    def __len__(self) -> int:
        return len(self.to_execution_dict())


class PlanStep(FrozenModel):
    """One proposed specialist capability in a supervisor plan."""

    step_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    arguments: PlanStepArguments = Field(
        default_factory=PlanStepArguments
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> "PlanStep":
        if self.step_id in self.depends_on:
            raise ValueError("a plan step cannot depend on itself.")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("depends_on cannot contain duplicates.")
        return self


class SupervisorPlan(FrozenModel):
    """Structured execution plan proposed by the LLM supervisor."""

    objective: str = Field(min_length=1)
    reporting_scope: ReportingScope = Field(default_factory=ReportingScope)
    steps: tuple[PlanStep, ...] = Field(min_length=1)
    required_reconciliations: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan_graph(self) -> "SupervisorPlan":
        step_ids = [step.step_id for step in self.steps]

        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique.")

        known_ids = set(step_ids)
        for step in self.steps:
            unknown = set(step.depends_on) - known_ids
            if unknown:
                raise ValueError(
                    f"step {step.step_id!r} has unknown dependencies: "
                    f"{sorted(unknown)}."
                )
        return self


class ToolCall(FrozenModel):
    """One validated call to an allow-listed deterministic tool."""

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(FrozenModel):
    """Compact result returned by a deterministic finance tool."""

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    status: ToolStatus
    evidence_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None


class EvidenceReference(FrozenModel):
    """Reference to a result held in the internal evidence registry."""

    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    reconciled: bool
    period: str | None = None


class RootCauseFinding(FrozenModel):
    """Evidence-supported root-cause conclusion."""

    cause: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    unresolved: bool = False


class RecommendationFinding(FrozenModel):
    """Management recommendation linked to causes or risks."""

    recommendation: str = Field(min_length=1)
    priority: Literal["high", "medium", "low"]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    requires_human_approval: bool = False


class ReconciliationCheck(FrozenModel):
    """Result of one deterministic reconciliation check."""

    name: str = Field(min_length=1)
    passed: bool
    details: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()


class ReconciliationResult(FrozenModel):
    """Combined deterministic reconciliation status."""

    passed: bool
    checks: tuple[ReconciliationCheck, ...] = Field(min_length=1)
    warnings: tuple[str, ...] = ()


class ReviewResult(FrozenModel):
    """Structured reviewer decision over evidence and conclusions."""

    decision: ReviewDecision
    unsupported_claims: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    reconciliation_issues: tuple[str, ...] = ()
    required_caveats: tuple[str, ...] = ()
    approved_answer: str | None = None


class ManagementResponse(FrozenModel):
    """Reviewed management response returned by the autonomous path."""

    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    caveats: tuple[str, ...] = ()


class AutonomousExecutionUsage(FrozenModel):
    """Aggregated usage recorded during autonomous execution."""

    agent_runs: int = Field(default=0, ge=0)
    replans: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class AutonomousExecutionResult(FrozenModel):
    """Final structured result of an autonomous workflow attempt."""

    status: AutonomousStatus
    management_response: ManagementResponse | None = None
    review_result: ReviewResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    usage: AutonomousExecutionUsage = Field(
        default_factory=AutonomousExecutionUsage
    )
    fallback_flow: str | None = None
    fallback_reason: str | None = None
    errors: tuple[str, ...] = ()


class PlanValidationIssue(FrozenModel):
    """One deterministic plan-validation issue."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: IssueSeverity
    step_id: str | None = None


class PlanValidationResult(FrozenModel):
    """Result returned by the deterministic autonomous plan validator."""

    valid: bool
    approved_plan: SupervisorPlan | None = None
    issues: tuple[PlanValidationIssue, ...] = ()
    selected_agents: tuple[str, ...] = ()
    selected_tools: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_approval(self) -> "PlanValidationResult":
        has_errors = any(
            issue.severity == "error"
            for issue in self.issues
        )
        if self.valid and (self.approved_plan is None or has_errors):
            raise ValueError(
                "a valid result requires an approved plan and no errors."
            )
        if not self.valid and self.approved_plan is not None:
            raise ValueError(
                "an invalid result cannot contain an approved plan."
            )
        return self


class EvidenceRecord(FrozenModel):
    """LLM-safe metadata and compact payload for registered evidence."""

    evidence_id: str = Field(min_length=1)
    source_tool: str = Field(min_length=1)
    result_type: str = Field(min_length=1)
    tool_status: ToolStatus
    compact_payload: dict[str, Any] = Field(default_factory=dict)
    period: str | None = None
    category: str | None = None
    reconciled: bool = False
    verified: bool = False
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_evidence_status(self) -> "EvidenceRecord":
        if self.verified and (
            self.tool_status != "completed" or not self.reconciled
        ):
            raise ValueError(
                "verified evidence must be completed and reconciled."
            )
        return self
