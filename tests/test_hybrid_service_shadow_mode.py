"""Tests for non-user-facing autonomous shadow execution."""

from typing import Any

from src.api.service import AskServiceResult, FinanceAskService
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ComplexityDecision,
    ManagementResponse,
)


class ComplexClassifier:
    def classify(self, request: str) -> ComplexityDecision:
        return ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=1,
            reasons=("diagnostic",),
            fallback_flow="pnl",
        )


def _service(executor: Any) -> FinanceAskService:
    instance = object.__new__(FinanceAskService)
    instance._complexity_classifier = ComplexClassifier()
    instance._autonomous_executor = executor
    instance._autonomous_enabled = False
    instance._autonomous_shadow_mode = True
    instance._ask_deterministic = lambda *args, **kwargs: AskServiceResult(  # type: ignore[method-assign]
        answer="Visible deterministic answer.",
        sources=[],
        selected_flow="pnl",
        execution_status="completed",
        used_fallback=False,
    )
    return instance


def test_shadow_mode_never_replaces_deterministic_answer() -> None:
    autonomous = AutonomousExecutionResult(
        status="completed",
        management_response=ManagementResponse(
            answer="Hidden autonomous answer.",
            evidence_ids=("pnl-001",),
        ),
    )

    result = _service(
        lambda question, deterministic: autonomous
    ).ask("Why did profit improve?")

    assert result.answer == "Visible deterministic answer."
    assert result.hybrid_metadata["execution_mode"] == "deterministic"
    assert result.hybrid_metadata["autonomous_status"] == (
        "shadow_completed"
    )


def test_shadow_failure_does_not_affect_response() -> None:
    def fail(*args: Any) -> Any:
        raise RuntimeError("sensitive provider detail")

    result = _service(fail).ask("Why did profit improve?")

    assert result.answer == "Visible deterministic answer."
    assert result.execution_status == "completed"
    assert result.hybrid_metadata["autonomous_status"] == "failed"
    assert "sensitive provider detail" not in str(result.hybrid_metadata)


def test_shadow_fallback_is_recorded_but_not_returned_as_answer() -> None:
    autonomous = AutonomousExecutionResult(
        status="fallback",
        fallback_flow="deterministic_planner",
        fallback_reason="Reviewer requested replan.",
    )

    result = _service(
        lambda question, deterministic: autonomous
    ).ask("Why did profit improve?")

    assert result.answer == "Visible deterministic answer."
    assert result.hybrid_metadata["autonomous_status"] == "shadow_fallback"
    assert result.hybrid_metadata["fallback_used"] is True
