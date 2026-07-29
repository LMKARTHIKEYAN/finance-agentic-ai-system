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


def test_pnl_diagnostics_compare_monthly_calculation_evidence() -> None:
    result = identify_supported_root_causes(
        pnl_result={
            "actual_pnl": [
                {
                    "month": "2026-03",
                    "revenue": 100.0,
                    "direct_cost": 60.0,
                    "sales_marketing": 5.0,
                    "other_opex": 4.0,
                    "depreciation": 2.0,
                    "interest": 1.0,
                    "income_tax": 7.0,
                    "net_profit": 21.0,
                },
                {
                    "month": "2026-04",
                    "revenue": 120.0,
                    "direct_cost": 65.0,
                    "sales_marketing": 6.0,
                    "other_opex": 4.0,
                    "depreciation": 2.0,
                    "interest": 1.0,
                    "income_tax": 10.0,
                    "net_profit": 32.0,
                },
            ]
        }
    )

    assert result.payload["base_month"] == "2026-03"
    assert result.payload["current_month"] == "2026-04"
    assert result.payload["net_profit_change"] == 11.0
    assert result.payload["drivers"][0]["metric"] == "revenue"


def test_pnl_recommendations_use_supported_driver_payload() -> None:
    result = generate_supported_recommendations(
        root_cause_result={
            "analysis_type": "pnl_period_change",
            "drivers": [
                {
                    "metric": "revenue",
                    "impact": "favorable",
                }
            ],
        }
    )

    assert result.payload["recommendations"][0]["metric"] == "revenue"


def test_multi_evidence_diagnostics_extract_supported_risks() -> None:
    result = identify_supported_root_causes(
        evidence_payloads=[
            {
                "evidence_id": "variance-001",
                "result_type": "revenue_variance",
                "payload": {
                    "actual_revenue": 90.0,
                    "budget_revenue": 100.0,
                    "revenue_variance": -10.0,
                    "price_effect": -6.0,
                    "volume_effect": -4.0,
                },
            },
            {
                "evidence_id": "gp-001",
                "result_type": "gp_decomposition",
                "payload": {
                    "budget_gp_percentage": 30.0,
                    "actual_gp_percentage": 28.0,
                    "total_variance_percentage_points": -2.0,
                },
            },
        ]
    )

    assert result.payload["analysis_type"] == (
        "multi_evidence_performance"
    )
    assert {
        item["metric"]
        for item in result.payload["risk_findings"]
    } == {
        "revenue_variance",
        "total_variance_percentage_points",
    }
    assert result.payload["evidence_ids"] == [
        "variance-001",
        "gp-001",
    ]


def test_multi_evidence_recommendations_link_risk_evidence() -> None:
    result = generate_supported_recommendations(
        root_cause_result={
            "analysis_type": "multi_evidence_performance",
            "risk_findings": [
                {
                    "metric": "revenue_variance",
                    "evidence_id": "variance-001",
                }
            ],
        }
    )

    recommendation = result.payload["recommendations"][0]
    assert recommendation["metric"] == "revenue_variance"
    assert recommendation["evidence_id"] == "variance-001"
