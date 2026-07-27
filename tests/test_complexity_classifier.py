"""Tests for deterministic-first complexity classification."""

from typing import Any

import pytest

from src.autonomous.complexity_classifier import ComplexityClassifier
from src.autonomous.schemas import ComplexityDecision
from src.llm.client import StructuredLLMClient
from src.llm.schemas import (
    LLMRequestMetadata,
    LLMUsage,
    StructuredLLMResponse,
)


class FakeStructuredClient(StructuredLLMClient):
    def __init__(
        self,
        decision: ComplexityDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def generate_structured(
        self,
        **kwargs: Any,
    ) -> StructuredLLMResponse[ComplexityDecision]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return StructuredLLMResponse[ComplexityDecision](
            output=self.decision,
            usage=LLMUsage(),
            metadata=LLMRequestMetadata(
                provider="fake",
                model="fake-model",
                elapsed_seconds=0,
            ),
        )


@pytest.mark.parametrize(
    "user_request, fallback_flow",
    [
        ("Show April KPI", "kpi"),
        ("Generate April P&L", "pnl"),
        ("Compare Actual vs Budget revenue", "variance"),
        ("Show April GP% decomposition", "gp_variance"),
    ],
)
def test_clear_simple_requests_use_deterministic_path(
    user_request: str,
    fallback_flow: str,
) -> None:
    decision = ComplexityClassifier().classify(user_request)

    assert decision.execution_mode == "deterministic"
    assert decision.fallback_flow == fallback_flow


@pytest.mark.parametrize(
    "user_request, request_type",
    [
        ("Why did April profit improve?", "diagnostic"),
        (
            "Identify risks and recommend actions.",
            "decision_support",
        ),
        (
            "What caused the margin change and what should management do?",
            "multi_analysis",
        ),
    ],
)
def test_clear_complex_requests_use_autonomous_path(
    user_request: str,
    request_type: str,
) -> None:
    decision = ComplexityClassifier().classify(user_request)

    assert decision.execution_mode == "autonomous"
    assert decision.request_type == request_type


def test_clear_request_does_not_call_llm() -> None:
    fake = FakeStructuredClient(
        decision=ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=1,
            reasons=("fake",),
        )
    )

    ComplexityClassifier(
        llm_client=fake,
        llm_enabled=True,
    ).classify("Show April KPI")

    assert fake.calls == []


def test_ambiguous_request_uses_injected_llm() -> None:
    fake = FakeStructuredClient(
        decision=ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=0.95,
            reasons=("Requires diagnostic reasoning.",),
            fallback_flow="kpi",
        )
    )

    decision = ComplexityClassifier(
        llm_client=fake,
        llm_enabled=True,
    ).classify("Assess April performance")

    assert decision.execution_mode == "autonomous"
    assert len(fake.calls) == 1


def test_low_confidence_llm_decision_falls_back() -> None:
    fake = FakeStructuredClient(
        decision=ComplexityDecision(
            execution_mode="autonomous",
            request_type="diagnostic",
            confidence=0.6,
            reasons=("Uncertain.",),
            fallback_flow="kpi",
        )
    )

    decision = ComplexityClassifier(
        llm_client=fake,
        llm_enabled=True,
        confidence_threshold=0.8,
    ).classify("Assess April performance")

    assert decision.execution_mode == "deterministic"


def test_llm_failure_falls_back() -> None:
    fake = FakeStructuredClient(error=RuntimeError("failure"))

    decision = ComplexityClassifier(
        llm_client=fake,
        llm_enabled=True,
    ).classify("Assess April performance")

    assert decision.execution_mode == "deterministic"


@pytest.mark.parametrize("user_request", ["", "   "])
def test_classifier_rejects_empty_request(user_request: str) -> None:
    with pytest.raises(ValueError):
        ComplexityClassifier().classify(user_request)
