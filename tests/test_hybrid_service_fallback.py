"""Tests for safe deterministic hybrid fallback."""

from typing import Any

from src.api.service import AskServiceResult, FinanceAskService
from src.autonomous.schemas import (
    AutonomousExecutionResult,
    ComplexityDecision,
)


class Classifier:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def classify(self, request: str) -> ComplexityDecision:
        if self.fail:
            raise RuntimeError("classifier secret")
        return ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=1,
            reasons=("diagnostic",),
            fallback_flow="variance",
        )


def _service(
    *,
    classifier: Any,
    executor: Any,
) -> FinanceAskService:
    instance = object.__new__(FinanceAskService)
    instance._complexity_classifier = classifier
    instance._autonomous_executor = executor
    instance._autonomous_enabled = True
    instance._autonomous_shadow_mode = False
    instance._ask_deterministic = lambda *args, **kwargs: AskServiceResult(  # type: ignore[method-assign]
        answer="Safe deterministic answer.",
        sources=[],
        selected_flow="variance",
        execution_status="completed",
        used_fallback=False,
        session_id="preserved-session",
        memory_status="completed",
    )
    return instance


def test_classifier_failure_uses_deterministic_result() -> None:
    result = _service(
        classifier=Classifier(fail=True),
        executor=lambda *args: None,
    ).ask("Analyze performance")

    assert result.answer == "Safe deterministic answer."
    assert result.hybrid_metadata["autonomous_status"] == (
        "classifier_fallback"
    )
    assert "classifier secret" not in str(result.hybrid_metadata)


def test_autonomous_exception_uses_deterministic_result() -> None:
    def fail(*args: Any) -> Any:
        raise RuntimeError("raw dataframe details")

    result = _service(
        classifier=Classifier(),
        executor=fail,
    ).ask("Why did revenue change?")

    assert result.answer == "Safe deterministic answer."
    assert result.session_id == "preserved-session"
    assert result.memory_status == "completed"
    assert "raw dataframe details" not in str(result.hybrid_metadata)


def test_autonomous_fallback_preserves_safe_reason() -> None:
    autonomous = AutonomousExecutionResult(
        status="fallback",
        fallback_flow="deterministic_planner",
        fallback_reason="Execution cost limit exceeded.",
    )

    result = _service(
        classifier=Classifier(),
        executor=lambda *args: autonomous,
    ).ask("Why did revenue change?")

    assert result.answer == "Safe deterministic answer."
    assert result.hybrid_metadata["fallback_used"] is True
    assert result.hybrid_metadata["fallback_reason"] == (
        "Execution cost limit exceeded."
    )


def test_missing_autonomous_executor_falls_back() -> None:
    result = _service(
        classifier=Classifier(),
        executor=None,
    ).ask("Why did revenue change?")

    assert result.answer == "Safe deterministic answer."
    assert result.hybrid_metadata["autonomous_status"] == "unavailable"
