"""Tests for autonomous workflow contracts."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ComplexityDecision,
    EvidenceReference,
    ManagementResponse,
    PlanStep,
    ReportingScope,
    ReviewResult,
    SupervisorPlan,
)


def test_complexity_decision_accepts_valid_values() -> None:
    decision = ComplexityDecision(
        execution_mode="autonomous",
        request_type="diagnostic",
        confidence=0.9,
        reasons=("The request asks why profit changed.",),
        fallback_flow="pnl",
    )

    assert decision.execution_mode == "autonomous"
    assert decision.confidence == pytest.approx(0.9)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_complexity_decision_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=confidence,
            reasons=("reason",),
        )


def test_reporting_scope_rejects_reversed_dates() -> None:
    with pytest.raises(ValidationError):
        ReportingScope(
            start_date=date(2026, 5, 1),
            end_date=date(2026, 4, 1),
        )


def test_plan_step_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError):
        PlanStep(
            step_id="pnl",
            capability="pnl_analysis",
            depends_on=("pnl",),
        )


def test_supervisor_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SupervisorPlan(
            objective="Analyze performance",
            steps=(
                PlanStep(step_id="kpi", capability="kpi"),
                PlanStep(step_id="kpi", capability="pnl"),
            ),
            expected_outputs=("management_answer",),
        )


def test_supervisor_plan_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown dependencies"):
        SupervisorPlan(
            objective="Analyze performance",
            steps=(
                PlanStep(
                    step_id="review",
                    capability="review",
                    depends_on=("missing",),
                ),
            ),
            expected_outputs=("management_answer",),
        )


def test_execution_result_serializes_nested_contracts() -> None:
    result = AutonomousExecutionResult(
        status="completed",
        management_response=ManagementResponse(
            answer="April performance improved.",
            evidence_ids=("pnl-1",),
        ),
        review_result=ReviewResult(
            decision="approved",
            approved_answer="April performance improved.",
        ),
        evidence=(
            EvidenceReference(
                evidence_id="pnl-1",
                source="PnlAgent",
                result_type="pnl",
                reconciled=True,
            ),
        ),
    )

    payload = result.model_dump(mode="json")

    assert payload["status"] == "completed"
    assert payload["evidence"][0]["evidence_id"] == "pnl-1"


def test_evidence_reference_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(
            evidence_id="",
            source="PnlAgent",
            result_type="pnl",
            reconciled=True,
        )
