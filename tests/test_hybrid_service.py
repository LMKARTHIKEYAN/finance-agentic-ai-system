"""Tests for enabled hybrid service routing."""

from typing import Any

from src.api.finance_response_context import (
    build_finance_response_context,
)
from src.api.schemas import AskResponse
from src.api.service import AskServiceResult, FinanceAskService
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ComplexityDecision,
    EvidenceReference,
    ManagementResponse,
    ReviewResult,
)


class Classifier:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def classify(self, request: str) -> ComplexityDecision:
        return ComplexityDecision(
            execution_mode=self.mode,
            request_type=(
                "diagnostic"
                if self.mode == "autonomous"
                else "simple_report"
            ),
            confidence=1,
            reasons=("test",),
            fallback_flow="pnl",
        )


def deterministic_result() -> AskServiceResult:
    return AskServiceResult(
        answer="Deterministic answer.",
        sources=[],
        selected_flow="pnl",
        execution_status="completed",
        used_fallback=False,
        session_id="session-1",
        memory_status="completed",
    )


def service(
    *,
    mode: str,
    executor: Any = None,
    enabled: bool = True,
    shadow: bool = False,
) -> FinanceAskService:
    instance = object.__new__(FinanceAskService)
    instance._complexity_classifier = Classifier(mode)
    instance._autonomous_executor = executor
    instance._autonomous_enabled = enabled
    instance._autonomous_shadow_mode = shadow
    instance._ask_deterministic = (  # type: ignore[method-assign]
        lambda question, **kwargs: deterministic_result()
    )
    return instance


def completed_result() -> AutonomousExecutionResult:
    return AutonomousExecutionResult(
        status="completed",
        management_response=ManagementResponse(
            answer="Reviewed autonomous answer.",
            evidence_ids=("pnl-001",),
        ),
        review_result=ReviewResult(
            decision="approved",
            approved_answer="Reviewed autonomous answer.",
        ),
        evidence=(
            EvidenceReference(
                evidence_id="pnl-001",
                source="pnl_tool",
                result_type="pnl",
                reconciled=True,
            ),
        ),
    )


def test_simple_request_uses_deterministic_path() -> None:
    autonomous_calls: list[str] = []
    result = service(
        mode="deterministic",
        executor=lambda question, deterministic: autonomous_calls.append(
            question
        ),
    ).ask("Show April P&L")

    assert result.answer == "Deterministic answer."
    assert result.hybrid_metadata["execution_mode"] == "deterministic"
    assert result.hybrid_metadata["autonomous_status"] == "not_selected"
    assert autonomous_calls == []


def test_complex_request_returns_reviewed_autonomous_answer() -> None:
    received: list[tuple[str, AskServiceResult]] = []

    def execute(
        question: str,
        deterministic: AskServiceResult,
    ) -> AutonomousExecutionResult:
        received.append((question, deterministic))
        return completed_result()

    result = service(mode="autonomous", executor=execute).ask(
        "Why did April profit improve?",
        session_id="session-1",
    )

    assert result.answer == "Reviewed autonomous answer."
    assert result.session_id == "session-1"
    assert result.memory_status == "completed"
    assert result.hybrid_metadata["execution_mode"] == "autonomous"
    assert result.hybrid_metadata["review_decision"] == "approved"
    assert result.hybrid_metadata["evidence_ids"] == ["pnl-001"]
    assert received[0][1].answer == "Deterministic answer."


def test_api_schema_remains_backward_compatible() -> None:
    response = AskResponse(
        answer="Existing answer",
        execution_status="completed",
    )

    assert response.hybrid_metadata is None


def test_response_context_includes_only_safe_hybrid_metadata() -> None:
    context = build_finance_response_context(
        "pnl",
        {"pnl_result": {"net_profit": 10}},
        {
            "execution_mode": "autonomous",
            "evidence_ids": ["pnl-001"],
            "internal_prompt": "must not be copied",
        },
    )

    assert context["hybrid_execution"] == {
        "execution_mode": "autonomous",
        "evidence_ids": ["pnl-001"],
    }
