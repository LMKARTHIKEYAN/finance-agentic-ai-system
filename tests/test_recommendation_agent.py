"""Tests for RecommendationAgent."""

from types import SimpleNamespace

import pytest

from src.agents.analytics.recommendation_agent import RecommendationAgent


def make_root_cause(
    *,
    root_cause_code: str = "RC-ANOM-001-COMPLETED_ORDERS",
    anomaly_code: str = "ANOM-001",
    level: str = "period",
    dimension_value: str = "2026-06",
    target_metric: str = "total_revenue",
    anomaly_direction: str = "decrease",
    anomaly_severity: str = "high",
    cause_metric: str = "completed_orders",
    cause_description: str = "Completed orders contributed to the revenue decline.",
    confidence: str = "high",
    impact: str = "unfavorable",
    evidence_score: object = 80.0,
    evidence: object = None,
    recommended_next_check: str = "Review completed orders by cluster.",
) -> SimpleNamespace:
    """Create a root-cause object with the attributes required by the agent."""

    if evidence is None:
        evidence = {
            "completed_orders": 8000,
            "total_orders": 10000,
        }

    return SimpleNamespace(
        root_cause_code=root_cause_code,
        anomaly_code=anomaly_code,
        level=level,
        dimension_value=dimension_value,
        target_metric=target_metric,
        anomaly_direction=anomaly_direction,
        anomaly_severity=anomaly_severity,
        cause_metric=cause_metric,
        cause_description=cause_description,
        confidence=confidence,
        impact=impact,
        evidence_score=evidence_score,
        evidence=evidence,
        recommended_next_check=recommended_next_check,
    )


def make_result(*findings: SimpleNamespace) -> SimpleNamespace:
    """Create a minimal RootCauseResult-like object."""

    return SimpleNamespace(findings=list(findings))


def test_constructor_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="minimum_evidence_score"):
        RecommendationAgent(minimum_evidence_score=-1)

    with pytest.raises(ValueError, match="minimum_evidence_score"):
        RecommendationAgent(minimum_evidence_score=101)

    with pytest.raises(ValueError, match="high_evidence_score"):
        RecommendationAgent(high_evidence_score=101)

    with pytest.raises(ValueError, match="critical_evidence_score"):
        RecommendationAgent(critical_evidence_score=101)

    with pytest.raises(ValueError, match="must be greater"):
        RecommendationAgent(
            high_evidence_score=80,
            critical_evidence_score=80,
        )

    with pytest.raises(ValueError, match="cannot be below"):
        RecommendationAgent(
            minimum_evidence_score=75,
            high_evidence_score=70,
        )

    with pytest.raises(ValueError, match="must be positive"):
        RecommendationAgent(max_recommendations_per_root_cause=0)

    with pytest.raises(TypeError, match="must be a bool"):
        RecommendationAgent(include_favorable_monitoring="yes")  # type: ignore[arg-type]


def test_empty_result_returns_no_recommendation_required() -> None:
    result = RecommendationAgent().analyze(make_result())

    assert result.overall_status == "NO_RECOMMENDATION_REQUIRED"
    assert result.root_causes_received == 0
    assert result.root_causes_processed == 0
    assert result.recommendation_count == 0
    assert result.recommendations == []
    assert result.unresolved_root_causes == []


def test_critical_unfavorable_root_cause_creates_critical_action() -> None:
    root_cause = make_root_cause(
        anomaly_severity="critical",
        confidence="high",
        evidence_score=90.0,
        cause_metric="completed_orders",
    )

    result = RecommendationAgent().analyze(make_result(root_cause))
    recommendation = result.recommendations[0]

    assert result.overall_status == "CRITICAL_ACTION_REQUIRED"
    assert result.critical_priority_count == 1
    assert recommendation.priority == "critical"
    assert recommendation.status == "open"
    assert recommendation.owner == "Operations Manager"
    assert recommendation.success_metric == "completed_orders"
    assert recommendation.target_direction == "increase"
    assert recommendation.implementation_steps


def test_high_priority_action_uses_matching_action_template() -> None:
    root_cause = make_root_cause(
        cause_metric="cancellation_percentage",
        anomaly_severity="high",
        confidence="high",
        evidence_score=75.0,
        evidence={"cancellation_percentage": 18.0},
    )

    result = RecommendationAgent().analyze(make_result(root_cause))
    recommendation = result.recommendations[0]

    assert result.overall_status == "ACTION_REQUIRED"
    assert result.high_priority_count == 1
    assert recommendation.priority == "high"
    assert recommendation.owner == "Operations Quality Manager"
    assert recommendation.success_metric == "cancellation_percentage"
    assert recommendation.target_direction == "decrease"
    assert "Reduce cancellation" in recommendation.recommended_action


def test_medium_priority_is_created_for_medium_confidence() -> None:
    root_cause = make_root_cause(
        anomaly_severity="medium",
        confidence="medium",
        evidence_score=55.0,
        cause_metric="fulfillment_percentage",
    )

    result = RecommendationAgent().analyze(make_result(root_cause))

    assert result.overall_status == "REVIEW"
    assert result.recommendations[0].priority == "medium"
    assert result.recommendations[0].status == "open"


def test_favorable_root_cause_creates_monitoring_recommendation() -> None:
    root_cause = make_root_cause(
        impact="favorable",
        anomaly_direction="increase",
        evidence_score=88.0,
        cause_metric="price_effect",
    )

    result = RecommendationAgent().analyze(make_result(root_cause))
    recommendation = result.recommendations[0]

    assert result.overall_status == "REVIEW"
    assert result.monitor_count == 1
    assert recommendation.priority == "low"
    assert recommendation.status == "monitor"
    assert recommendation.recommended_action.startswith(
        "Preserve the favorable driver"
    )


def test_favorable_monitoring_can_be_excluded() -> None:
    root_cause = make_root_cause(
        impact="favorable",
        cause_metric="volume_effect",
        evidence_score=80.0,
    )

    agent = RecommendationAgent(include_favorable_monitoring=False)
    result = agent.analyze(make_result(root_cause))

    assert result.root_causes_received == 1
    assert result.root_causes_processed == 1
    assert result.recommendation_count == 0
    assert result.overall_status == "NO_RECOMMENDATION_REQUIRED"


def test_below_threshold_and_invalid_scores_become_unresolved() -> None:
    weak = make_root_cause(
        root_cause_code="RC-WEAK",
        evidence_score=10.0,
    )
    invalid = make_root_cause(
        root_cause_code="RC-INVALID",
        anomaly_code="ANOM-002",
        evidence_score="not-a-number",
    )

    result = RecommendationAgent(minimum_evidence_score=20).analyze(
        make_result(weak, invalid)
    )

    assert result.recommendation_count == 0
    assert len(result.unresolved_root_causes) == 2
    assert result.overall_status == "REVIEW"
    assert "below the minimum" in result.unresolved_root_causes[0]["message"]
    assert "invalid evidence score" in result.unresolved_root_causes[1]["message"]


def test_unknown_cause_metric_uses_default_action_and_target_metric() -> None:
    root_cause = make_root_cause(
        cause_metric="unknown_business_driver",
        target_metric="gross_profit",
        evidence_score=72.0,
    )

    result = RecommendationAgent().analyze(make_result(root_cause))
    recommendation = result.recommendations[0]

    assert recommendation.owner == "Business Performance Manager"
    assert recommendation.success_metric == "gross_profit"
    assert recommendation.target_direction == "improve"
    assert "validate the driver" in recommendation.recommended_action.lower()


def test_supporting_evidence_contains_root_cause_context() -> None:
    root_cause = make_root_cause(
        cause_metric="average_order_value",
        evidence={"average_order_value": 925.5},
        confidence="medium",
        impact="neutral",
        evidence_score=60.0,
    )

    result = RecommendationAgent().analyze(make_result(root_cause))
    recommendation = result.recommendations[0]

    assert recommendation.status == "monitor"
    assert recommendation.priority == "low"
    assert recommendation.supporting_evidence["average_order_value"] == 925.5
    assert recommendation.supporting_evidence["root_cause_confidence"] == "medium"
    assert recommendation.supporting_evidence["root_cause_impact"] == "neutral"
    assert "root_cause_description" in recommendation.supporting_evidence
    assert "root_cause_next_check" in recommendation.supporting_evidence


def test_recommendations_are_sorted_by_priority_and_score() -> None:
    medium = make_root_cause(
        root_cause_code="RC-MEDIUM",
        anomaly_code="ANOM-MEDIUM",
        confidence="medium",
        anomaly_severity="medium",
        evidence_score=60.0,
        cause_metric="total_orders",
    )
    critical = make_root_cause(
        root_cause_code="RC-CRITICAL",
        anomaly_code="ANOM-CRITICAL",
        confidence="high",
        anomaly_severity="critical",
        evidence_score=95.0,
        cause_metric="volume_effect",
    )
    high = make_root_cause(
        root_cause_code="RC-HIGH",
        anomaly_code="ANOM-HIGH",
        confidence="high",
        anomaly_severity="high",
        evidence_score=78.0,
        cause_metric="price_effect",
    )

    result = RecommendationAgent().analyze(
        make_result(medium, high, critical)
    )

    assert [item.priority for item in result.recommendations] == [
        "critical",
        "high",
        "medium",
    ]
    assert result.critical_priority_count == 1
    assert result.high_priority_count == 1


def test_input_validation_errors() -> None:
    agent = RecommendationAgent()

    with pytest.raises(ValueError, match="root_cause_result is required"):
        agent.analyze(None)

    with pytest.raises(ValueError, match="must contain a findings"):
        agent.analyze(SimpleNamespace())

    with pytest.raises(TypeError, match="findings must be a list"):
        agent.analyze(SimpleNamespace(findings="invalid"))

    incomplete = SimpleNamespace(root_cause_code="RC-001")
    with pytest.raises(ValueError, match="missing required attributes"):
        agent.analyze(make_result(incomplete))


def test_invalid_confidence_impact_and_evidence_type_fail() -> None:
    agent = RecommendationAgent()

    with pytest.raises(ValueError, match="Unsupported root-cause confidence"):
        agent.analyze(make_result(make_root_cause(confidence="very_high")))

    with pytest.raises(ValueError, match="Unsupported root-cause impact"):
        agent.analyze(make_result(make_root_cause(impact="bad")))

    with pytest.raises(TypeError, match="evidence must be a dictionary"):
        agent.analyze(make_result(make_root_cause(evidence=["invalid"])))