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
    LLMOutputLimitError,
    LLMStructuredOutputError,
    LLMTimeoutError,
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
        "original_request": "Explain profit improvement.",
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
    assert context["original_request"] == "Explain profit improvement."


def test_reviewer_compacts_multi_period_pnl_evidence() -> None:
    client = FakeReviewerClient(
        ReviewResult(
            decision="approved",
            approved_answer="Approved.",
        )
    )
    pnl_evidence = _evidence().model_copy(
        update={
            "compact_payload": {
                "actual_pnl": [
                    {
                        "month": "2026-04",
                        "revenue": 100,
                        "net_profit": 20,
                        "internal_detail": "exclude",
                    },
                    {
                        "month": "2026-05",
                        "revenue": 120,
                        "net_profit": 25,
                        "internal_detail": "exclude",
                    },
                ],
                "budget_pnl": [{"large": "exclude"}],
                "variance_pnl": [{"large": "exclude"}],
            }
        }
    )

    _review(client, evidence=(pnl_evidence,))
    context = json.loads(
        client.calls[0]["messages"][1]["content"]
    )
    compact_payload = context["evidence"][0]["compact_payload"]

    assert compact_payload == {
        "actual_pnl": [
            {
                "month": "2026-04",
                "revenue": 100,
                "net_profit": 20,
            },
            {
                "month": "2026-05",
                "revenue": 120,
                "net_profit": 25,
            },
        ]
    }
    assert "budget_pnl" not in compact_payload
    assert "variance_pnl" not in compact_payload


def test_reviewer_compacts_gp_product_and_portfolio_evidence() -> None:
    client = FakeReviewerClient(
        ReviewResult(
            decision="approved",
            approved_answer="Approved.",
        )
    )
    gp_evidence = _evidence().model_copy(
        update={
            "result_type": "gp_decomposition",
            "compact_payload": {
                "budget_gp_percentage": 30.0,
                "actual_gp_percentage": 32.0,
                "total_variance_percentage_points": 2.0,
                "reconciliation_status": "PASS",
                "reconciliation_difference": 0.0,
                "product_level": [
                    {
                        "category": "2W",
                        "actual_gp_percentage": 32.0,
                        "budget_gp_percentage": 30.0,
                        "large_internal_detail": "exclude",
                    }
                ],
                "category_analysis": [{"large": "exclude"}],
            },
        }
    )

    _review(
        client,
        evidence=(gp_evidence,),
        diagnostics=_diagnostics(),
        original_request="Explain margin changes.",
    )
    context = json.loads(
        client.calls[0]["messages"][1]["content"]
    )
    compact_payload = context["evidence"][0]["compact_payload"]

    assert compact_payload["actual_gp_percentage"] == 32.0
    assert compact_payload["product_level"] == [
        {
            "category": "2W",
            "actual_gp_percentage": 32.0,
            "budget_gp_percentage": 30.0,
        }
    ]
    assert "category_analysis" not in compact_payload


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


def test_margin_request_requires_gp_decomposition_evidence() -> None:
    client = FakeReviewerClient(
        ReviewResult(
            decision="approved",
            approved_answer="unsafe",
        )
    )

    result = _review(
        client,
        original_request=(
            "Explain May margin changes and recommend actions."
        ),
    )

    assert result.decision == "replan_required"
    assert result.missing_evidence == (
        "required:gp_decomposition",
    )
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
    assert error.value.failure_code == "unexpected_error"


def test_reviewer_rejects_non_structured_output() -> None:
    client = FakeReviewerClient({"decision": "approved"})

    with pytest.raises(
        ReviewerAgentError,
        match="invalid structured",
    ) as error:
        _review(client)

    assert error.value.failure_code == "structured_output_invalid"


@pytest.mark.parametrize(
    ("provider_error", "expected_code"),
    [
        (LLMTimeoutError("private timeout"), "timeout"),
        (
            LLMOutputLimitError("private output"),
            "output_token_limit",
        ),
        (
            LLMStructuredOutputError("private schema"),
            "structured_output_invalid",
        ),
    ],
)
def test_reviewer_classifies_safe_llm_failure_codes(
    provider_error: Exception,
    expected_code: str,
) -> None:
    client = FakeReviewerClient(
        ReviewResult(decision="failed"),
        error=provider_error,
    )

    with pytest.raises(ReviewerAgentError) as error:
        _review(client)

    assert error.value.failure_code == expected_code
    assert "private" not in str(error.value)


def test_reviewer_instructions_prohibit_finance_calculation_and_writes() -> None:
    instructions = ReviewerAgent.instructions.lower()

    assert "never calculate" in instructions
    assert "verified evidence" in instructions
    assert "external writes" in instructions
    assert "human approval" in instructions
