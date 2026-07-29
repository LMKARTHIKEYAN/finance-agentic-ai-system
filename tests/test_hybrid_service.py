"""Tests for enabled hybrid service routing."""

from typing import Any

from src.api.finance_response_context import (
    build_finance_response_context,
)
from src.api.schemas import AskResponse
from src.api.service import (
    AskServiceResult,
    FinanceAskService,
    _autonomous_metadata,
)
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


def test_complex_margin_request_uses_gp_baseline_and_original_request() -> None:
    baseline_questions: list[str] = []
    autonomous_questions: list[str] = []
    instance = service(
        mode="autonomous",
        executor=(
            lambda question, deterministic: (
                autonomous_questions.append(question)
                or completed_result()
            )
        ),
    )
    instance._complexity_classifier = type(
        "MarginClassifier",
        (),
        {
            "classify": lambda self, request: ComplexityDecision(
                execution_mode="autonomous",
                request_type="decision_support",
                confidence=1,
                reasons=("test",),
                fallback_flow="gp_variance",
            )
        },
    )()
    instance._ask_deterministic = (  # type: ignore[method-assign]
        lambda question, **kwargs: (
            baseline_questions.append(question)
            or deterministic_result()
        )
    )
    question = (
        "Explain May 2026 margin changes versus April 2026 "
        "and recommend management actions."
    )

    result = instance.ask(question)

    assert result.hybrid_metadata["execution_mode"] == "autonomous"
    assert baseline_questions == [
        "Show GP% decomposition for May 2026 versus April 2026"
    ]
    assert autonomous_questions == [question]


def test_complex_performance_request_uses_full_management_baseline() -> None:
    baseline_questions: list[str] = []
    instance = service(
        mode="autonomous",
        executor=lambda question, deterministic: completed_result(),
    )
    instance._complexity_classifier = type(
        "PerformanceClassifier",
        (),
        {
            "classify": lambda self, request: ComplexityDecision(
                execution_mode="autonomous",
                request_type="decision_support",
                confidence=1,
                reasons=("test",),
                fallback_flow="kpi",
            )
        },
    )()
    instance._ask_deterministic = (  # type: ignore[method-assign]
        lambda question, **kwargs: (
            baseline_questions.append(question)
            or deterministic_result()
        )
    )

    instance.ask(
        "Analyze May 2026 performance, identify financial risks, "
        "and recommend actions."
    )

    assert baseline_questions == [
        "Generate management report for May 2026"
    ]


def test_simple_request_keeps_original_deterministic_question() -> None:
    deterministic_questions: list[str] = []
    instance = service(mode="deterministic")
    instance._ask_deterministic = (  # type: ignore[method-assign]
        lambda question, **kwargs: (
            deterministic_questions.append(question)
            or deterministic_result()
        )
    )

    instance.ask("Show May 2026 KPI")

    assert deterministic_questions == ["Show May 2026 KPI"]


def test_api_schema_remains_backward_compatible() -> None:
    response = AskResponse(
        answer="Existing answer",
        execution_status="completed",
    )

    assert response.hybrid_metadata is None


def test_autonomous_metadata_exposes_only_review_issue_counts() -> None:
    metadata = _autonomous_metadata(
        AutonomousExecutionResult(
            status="fallback",
            review_result=ReviewResult(
                decision="failed",
                unsupported_claims=("Private claim text.",),
                missing_evidence=("private-id",),
            ),
            fallback_flow="deterministic_planner",
            fallback_reason="Reviewer decision failed.",
        )
    )

    assert metadata["review_issue_counts"] == {
        "unsupported_claims": 1,
        "missing_evidence": 1,
    }
    assert "Private claim text" not in str(metadata)
    assert "private-id" not in str(metadata)


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
