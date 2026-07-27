"""Tests for the evidence-controlled structured reviewer."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.autonomous.agents.reviewer_agent import (
    ReviewerAgent,
    ReviewerAgentError,
)
from src.autonomous.agents.root_cause_recommendation_agent import (
    DiagnosticAnalysisResult,
)
from src.autonomous.schemas import (
    EvidenceRecord,
    PlanStep,
    ReconciliationCheck,
    ReconciliationResult,
    ReviewResult,
    SupervisorPlan,
    ToolResult,
)
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class FakeReviewerClient(StructuredLLMClient):
    def __init__(
        self,
        output: object,
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-reviewer"

    def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if isinstance(self.output, ReviewResult):
            return StructuredLLMResponse[ReviewResult](
                output=self.output,
                usage=LLMUsage(),
                metadata=LLMRequestMetadata(
                    provider=self.provider,
                    model=self.model,
                    elapsed_seconds=0.01,
                ),
            )
        return type("InvalidResponse", (), {"output": self.output})()


def _plan() -> SupervisorPlan:
    return SupervisorPlan(
        objective="Explain profit improvement",
        steps=(
            PlanStep(
                step_id="review",
                capability="review",
                arguments={"agent_name": "reviewer_agent"},
            ),
        ),
        expected_outputs=("management_answer",),
    )


def _evidence(
    evidence_id: str = "pnl-001",
    *,
    verified: bool = True,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_tool="generate_validated_pnl_analysis",
        result_type="pnl",
        tool_status="completed",
        compact_payload={"net_profit_variance": 25},
        reconciled=True,
        verified=verified,
    )


def _reconciliation(*, passed: bool = True) -> ReconciliationResult:
    return ReconciliationResult(
        passed=passed,
        checks=(
            ReconciliationCheck(
                name="pnl",
                passed=passed,
                details="P&L reconciled." if passed else "P&L failed.",
                evidence_ids=("pnl-001",),
            ),
        ),
    )


def _diagnostics(
    evidence_ids: tuple[str, ...] = ("pnl-001",),
) -> DiagnosticAnalysisResult:
    return DiagnosticAnalysisResult(
        root_cause_result=ToolResult(
            call_id="root",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload={"causes": ["volume"]},
        ),
        recommendation_result=ToolResult(
            call_id="recommend",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload={"recommendations": ["monitor volume"]},
        ),
        evidence_ids=evidence_ids,
    )


def _review(
    client: FakeReviewerClient,
    **overrides: Any,
) -> ReviewResult:
    arguments = {
        "plan": _plan(),
        "reconciliation": _reconciliation(),
        "evidence": (_evidence(),),
        "diagnostics": _diagnostics(),
        "draft_answer": "Profit improved because volume increased.",
        **overrides,
    }
    return ReviewerAgent(client).review(**arguments)


def test_reviewer_returns_structured_approval() -> None:
    expected = ReviewResult(
        decision="approved",
        approved_answer="Reviewed management answer.",
    )
    client = FakeReviewerClient(expected)

    result = _review(client)

    assert result == expected
    assert client.calls[0]["response_model"] is ReviewResult


def test_reviewer_sends_compact_verified_evidence() -> None:
    client = FakeReviewerClient(
        ReviewResult(
            decision="approved",
            approved_answer="Approved.",
        )
    )

    _review(client)
    context = json.loads(
        client.calls[0]["messages"][1]["content"]
    )

    assert context["evidence"][0]["evidence_id"] == "pnl-001"
    assert context["evidence"][0]["verified"] is True
    assert context["evidence"][0]["compact_payload"] == {
        "net_profit_variance": 25
    }


def test_failed_reconciliation_short_circuits_without_llm() -> None:
    client = FakeReviewerClient(
        ReviewResult(decision="approved", approved_answer="unsafe")
    )

    result = _review(
        client,
        reconciliation=_reconciliation(passed=False),
    )

    assert result.decision == "replan_required"
    assert result.reconciliation_issues == ("P&L failed.",)
    assert client.calls == []


def test_unverified_evidence_short_circuits_without_llm() -> None:
    client = FakeReviewerClient(
        ReviewResult(decision="approved", approved_answer="unsafe")
    )

    result = _review(client, evidence=(_evidence(verified=False),))

    assert result.decision == "replan_required"
    assert result.missing_evidence == ("pnl-001",)
    assert client.calls == []


def test_missing_diagnostic_evidence_short_circuits() -> None:
    client = FakeReviewerClient(
        ReviewResult(decision="approved", approved_answer="unsafe")
    )

    result = _review(
        client,
        diagnostics=_diagnostics(("unknown-999",)),
    )

    assert result.decision == "replan_required"
    assert result.missing_evidence == ("unknown-999",)
    assert client.calls == []


def test_reviewer_rejects_approval_without_answer() -> None:
    client = FakeReviewerClient(ReviewResult(decision="approved"))

    with pytest.raises(ReviewerAgentError, match="approved answer"):
        _review(client)


def test_reviewer_rejects_approval_with_blocking_issues() -> None:
    client = FakeReviewerClient(
        ReviewResult(
            decision="approved",
            unsupported_claims=("Unsupported margin claim.",),
            approved_answer="Answer.",
        )
    )

    with pytest.raises(ReviewerAgentError, match="blocking issues"):
        _review(client)


def test_reviewer_wraps_provider_error_without_details() -> None:
    client = FakeReviewerClient(
        ReviewResult(decision="failed"),
        error=RuntimeError("secret provider output"),
    )

    with pytest.raises(
        ReviewerAgentError,
        match="structured generation failed",
    ) as error:
        _review(client)

    assert "secret provider output" not in str(error.value)


def test_reviewer_rejects_non_structured_output() -> None:
    client = FakeReviewerClient({"decision": "approved"})

    with pytest.raises(ReviewerAgentError, match="invalid structured"):
        _review(client)


def test_reviewer_instructions_prohibit_finance_calculation_and_writes() -> None:
    instructions = ReviewerAgent.instructions.lower()

    assert "never calculate" in instructions
    assert "verified evidence" in instructions
    assert "external writes" in instructions
    assert "human approval" in instructions
