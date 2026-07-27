"""Controlled deterministic root-cause and recommendation specialist."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous.schemas import EvidenceRecord, PlanStep, ToolResult
from src.autonomous.tools.diagnostic_tools import (
    generate_supported_recommendations,
    identify_supported_root_causes,
)


class RootCauseRecommendationAgentError(RuntimeError):
    """Raised when controlled diagnostic execution cannot complete."""


class DiagnosticAnalysisResult(BaseModel):
    """Deterministic diagnostic results linked to verified evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_cause_result: ToolResult
    recommendation_result: ToolResult
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class RootCauseRecommendationAgent:
    """Run approved diagnostic tools without performing LLM calculations."""

    name = "Root-Cause and Recommendation Agent"
    capability = "root_cause_recommendation"
    root_cause_tool_name = "identify_supported_root_causes"
    recommendation_tool_name = "generate_supported_recommendations"

    def __init__(
        self,
        *,
        root_cause_tool: Callable[..., ToolResult] = (
            identify_supported_root_causes
        ),
        recommendation_tool: Callable[..., ToolResult] = (
            generate_supported_recommendations
        ),
    ) -> None:
        if not callable(root_cause_tool):
            raise TypeError("root_cause_tool must be callable.")
        if not callable(recommendation_tool):
            raise TypeError("recommendation_tool must be callable.")
        self._root_cause_tool = root_cause_tool
        self._recommendation_tool = recommendation_tool

    def execute(
        self,
        step: PlanStep,
        *,
        evidence: Sequence[EvidenceRecord],
        anomaly_result: object,
        operations_result: object,
        revenue_variance_result: object | None = None,
    ) -> DiagnosticAnalysisResult:
        """Execute both deterministic diagnostic stages in order."""

        _validate_step(step)
        verified_evidence = _validated_evidence(evidence)
        if anomaly_result is None:
            raise ValueError("anomaly_result is required.")
        if operations_result is None:
            raise ValueError("operations_result is required.")

        try:
            root_cause_result = self._root_cause_tool(
                anomaly_result=anomaly_result,
                operations_result=operations_result,
                revenue_variance_result=revenue_variance_result,
            )
            _validate_tool_result(
                root_cause_result,
                self.root_cause_tool_name,
            )
            recommendation_result = self._recommendation_tool(
                root_cause_result=root_cause_result.payload,
            )
            _validate_tool_result(
                recommendation_result,
                self.recommendation_tool_name,
            )
        except RootCauseRecommendationAgentError:
            raise
        except Exception as exc:
            raise RootCauseRecommendationAgentError(
                "Diagnostic tool execution failed."
            ) from exc

        return DiagnosticAnalysisResult(
            root_cause_result=root_cause_result,
            recommendation_result=recommendation_result,
            evidence_ids=tuple(
                item.evidence_id for item in verified_evidence
            ),
        )


def _validate_step(step: object) -> None:
    if not isinstance(step, PlanStep):
        raise TypeError("step must be a PlanStep.")
    if step.capability != RootCauseRecommendationAgent.capability:
        raise ValueError(
            "step capability must be "
            f"{RootCauseRecommendationAgent.capability!r}."
        )
    if (
        step.arguments.get("tool_name")
        != RootCauseRecommendationAgent.root_cause_tool_name
    ):
        raise ValueError("step contains an unapproved root-cause tool.")
    if (
        step.arguments.get("recommendation_tool_name")
        != RootCauseRecommendationAgent.recommendation_tool_name
    ):
        raise ValueError("step contains an unapproved recommendation tool.")
    allowed = {
        "agent_name",
        "tool_name",
        "recommendation_tool_name",
    }
    unknown = set(step.arguments) - allowed
    if unknown:
        raise ValueError(f"unsupported step arguments: {sorted(unknown)}.")


def _validated_evidence(
    evidence: object,
) -> tuple[EvidenceRecord, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(
        evidence,
        Sequence,
    ):
        raise TypeError("evidence must be a sequence.")
    records = tuple(evidence)
    if not records:
        raise ValueError("At least one evidence record is required.")
    if any(not isinstance(item, EvidenceRecord) for item in records):
        raise TypeError("every evidence item must be an EvidenceRecord.")
    if any(
        item.tool_status != "completed"
        or not item.reconciled
        or not item.verified
        for item in records
    ):
        raise ValueError(
            "Diagnostic analysis requires completed, reconciled, "
            "verified evidence."
        )
    return records


def _validate_tool_result(
    result: object,
    expected_tool_name: str,
) -> None:
    if not isinstance(result, ToolResult):
        raise RootCauseRecommendationAgentError(
            "Diagnostic tool returned an invalid result."
        )
    if result.tool_name != expected_tool_name:
        raise RootCauseRecommendationAgentError(
            "Diagnostic tool returned an unexpected tool identity."
        )
    if result.status != "completed":
        raise RootCauseRecommendationAgentError(
            "Diagnostic tool did not complete successfully."
        )
