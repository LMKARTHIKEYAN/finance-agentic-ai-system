"""Evidence-controlled LLM reviewer for autonomous finance analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.autonomous.agents.root_cause_recommendation_agent import (
    DiagnosticAnalysisResult,
)
from src.autonomous.execution_limits import AutonomousExecutionLimits
from src.autonomous.schemas import (
    EvidenceRecord,
    ReconciliationResult,
    ReviewResult,
    SupervisorPlan,
)
from src.llm.client import LLMMessage, StructuredLLMClient


class ReviewerAgentError(RuntimeError):
    """Raised when the reviewer cannot return a safe structured result."""


class ReviewerAgent:
    """Review compact evidence and conclusions without recalculating values."""

    name = "Finance Reviewer Agent"
    description = (
        "Checks evidence support, reconciliation, and management claims."
    )
    instructions = """
You are the independent reviewer for a controlled FP&A workflow.
Return only a structured ReviewResult matching the supplied schema.

Review rules:
- Never calculate, change, estimate, or invent financial values.
- Approve only claims supported by supplied verified evidence.
- Check that diagnostic conclusions reference supplied evidence IDs.
- Treat reconciliation issues, missing evidence, and unsupported claims as
  reasons to require replanning or fail the review.
- Recommendations must be supported by evidence and must not imply that an
  external action has already occurred.
- External writes and communications always require human approval.
- If approved, approved_answer must contain the reviewed management answer.
- Use required_caveats for limitations that do not invalidate the analysis.
""".strip()

    def __init__(
        self,
        llm_client: StructuredLLMClient,
        *,
        limits: AutonomousExecutionLimits | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not isinstance(llm_client, StructuredLLMClient):
            raise TypeError(
                "llm_client must implement StructuredLLMClient."
            )
        resolved_limits = limits or AutonomousExecutionLimits()
        resolved_max_tokens = (
            resolved_limits.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        if (
            isinstance(resolved_max_tokens, bool)
            or not isinstance(resolved_max_tokens, int)
        ):
            raise TypeError("max_output_tokens must be an integer.")
        if resolved_max_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")
        if resolved_max_tokens > resolved_limits.max_output_tokens:
            raise ValueError(
                "max_output_tokens cannot exceed the autonomous limit."
            )
        self._llm_client = llm_client
        self._limits = resolved_limits
        self._max_output_tokens = resolved_max_tokens

    def review(
        self,
        *,
        plan: SupervisorPlan,
        reconciliation: ReconciliationResult,
        evidence: Sequence[EvidenceRecord],
        diagnostics: DiagnosticAnalysisResult,
        draft_answer: str,
    ) -> ReviewResult:
        """Review one draft using reconciled, verified compact evidence."""

        if not isinstance(plan, SupervisorPlan):
            raise TypeError("plan must be a SupervisorPlan.")
        if not isinstance(reconciliation, ReconciliationResult):
            raise TypeError(
                "reconciliation must be a ReconciliationResult."
            )
        if not isinstance(diagnostics, DiagnosticAnalysisResult):
            raise TypeError(
                "diagnostics must be a DiagnosticAnalysisResult."
            )
        cleaned_answer = _required_text(draft_answer, "draft_answer")
        records = _validated_evidence_sequence(evidence)

        precheck = _precheck(reconciliation, records, diagnostics)
        if precheck is not None:
            return precheck

        messages = self._build_messages(
            plan=plan,
            reconciliation=reconciliation,
            evidence=records,
            diagnostics=diagnostics,
            draft_answer=cleaned_answer,
        )
        try:
            response = self._llm_client.generate_structured(
                messages=messages,
                response_model=ReviewResult,
                max_output_tokens=self._max_output_tokens,
                timeout_seconds=self._limits.max_execution_seconds,
            )
        except Exception as exc:
            raise ReviewerAgentError(
                "Reviewer structured generation failed."
            ) from exc

        result = response.output
        if not isinstance(result, ReviewResult):
            raise ReviewerAgentError(
                "Reviewer returned an invalid structured result."
            )
        _validate_review_decision(result)
        return result

    def _build_messages(
        self,
        *,
        plan: SupervisorPlan,
        reconciliation: ReconciliationResult,
        evidence: tuple[EvidenceRecord, ...],
        diagnostics: DiagnosticAnalysisResult,
        draft_answer: str,
    ) -> tuple[LLMMessage, ...]:
        context: dict[str, Any] = {
            "plan": plan.model_dump(mode="json"),
            "reconciliation": reconciliation.model_dump(mode="json"),
            "evidence": [
                item.model_dump(mode="json") for item in evidence
            ],
            "diagnostics": diagnostics.model_dump(mode="json"),
            "draft_answer": draft_answer,
        }
        _reject_dataframes(context)
        return (
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": json.dumps(
                    context,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )


def _validated_evidence_sequence(
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
    return records


def _precheck(
    reconciliation: ReconciliationResult,
    evidence: tuple[EvidenceRecord, ...],
    diagnostics: DiagnosticAnalysisResult,
) -> ReviewResult | None:
    reconciliation_issues = tuple(
        check.details
        for check in reconciliation.checks
        if not check.passed
    )
    if not reconciliation.passed or reconciliation_issues:
        return ReviewResult(
            decision="replan_required",
            reconciliation_issues=(
                reconciliation_issues
                or ("Deterministic reconciliation failed.",)
            ),
        )

    unusable = tuple(
        item.evidence_id
        for item in evidence
        if item.tool_status != "completed"
        or not item.reconciled
        or not item.verified
    )
    available_ids = {item.evidence_id for item in evidence}
    missing = tuple(
        item
        for item in diagnostics.evidence_ids
        if item not in available_ids
    )
    if unusable or missing:
        return ReviewResult(
            decision="replan_required",
            missing_evidence=tuple(dict.fromkeys((*unusable, *missing))),
        )
    return None


def _validate_review_decision(result: ReviewResult) -> None:
    if result.decision in {"approved", "approved_with_caveats"}:
        if not result.approved_answer:
            raise ReviewerAgentError(
                "An approved review requires an approved answer."
            )
        if (
            result.unsupported_claims
            or result.missing_evidence
            or result.reconciliation_issues
        ):
            raise ReviewerAgentError(
                "Reviewer approval contains blocking issues."
            )


def _reject_dataframes(value: Any) -> None:
    if isinstance(value, pd.DataFrame):
        raise TypeError("Reviewer context cannot contain DataFrames.")
    if isinstance(value, dict):
        for item in value.values():
            _reject_dataframes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_dataframes(item)


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned
