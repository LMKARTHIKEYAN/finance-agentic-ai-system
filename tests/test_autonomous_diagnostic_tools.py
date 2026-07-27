"""Tests for deterministic diagnostic wrappers."""

from dataclasses import dataclass, field
from typing import Any

from src.autonomous.tools.diagnostic_tools import (
    generate_supported_recommendations,
    identify_supported_root_causes,
)


@dataclass
class FakeRootResult:
    overall_status: str = "completed"
    findings: list[dict[str, Any]] = field(
        default_factory=lambda: [{"root_cause_code": "volume"}]
    )


@dataclass
class FakeRecommendationResult:
    overall_status: str = "completed"
    recommendations: list[dict[str, Any]] = field(
        default_factory=lambda: [{"recommended_action": "Recover volume"}]
    )


class FakeRootAgent:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def analyze(self, **kwargs: Any) -> FakeRootResult:
        self.kwargs = kwargs
        return FakeRootResult()


class FakeRecommendationAgent:
    def __init__(self) -> None:
        self.value: Any = None

    def analyze(self, value: Any) -> FakeRecommendationResult:
        self.value = value
        return FakeRecommendationResult()


def test_root_cause_wrapper_preserves_arguments_and_findings() -> None:
    agent = FakeRootAgent()
    anomaly = object()
    operations = object()
    variance = object()

    result = identify_supported_root_causes(
        anomaly_result=anomaly,
        operations_result=operations,
        revenue_variance_result=variance,
        agent=agent,
    )

    assert result.payload["findings"][0]["root_cause_code"] == "volume"
    assert agent.kwargs["anomaly_result"] is anomaly
    assert agent.kwargs["variance_result"] is variance


def test_recommendation_wrapper_uses_root_cause_result() -> None:
    root_result = object()
    agent = FakeRecommendationAgent()

    result = generate_supported_recommendations(
        root_cause_result=root_result,
        agent=agent,
    )

    assert agent.value is root_result
    assert (
        result.payload["recommendations"][0]["recommended_action"]
        == "Recover volume"
    )
