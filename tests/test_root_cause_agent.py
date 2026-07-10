"""Unit tests for RootCauseAgent."""

from types import SimpleNamespace

import pytest

from src.agents.analytics.root_cause_agent import (
    RootCauseAgent,
    RootCauseFinding,
    RootCauseResult,
)


def make_anomaly(
    *,
    anomaly_code: str = "ANOM-001",
    level: str = "vehicle_category",
    dimension_value: str = "Tata Ace",
    metric: str = "total_revenue",
    actual_value: float = 80_000.0,
    comparison_value: float = 100_000.0,
    percentage_change: float = -20.0,
    z_score: float = -2.5,
    direction: str = "decrease",
    severity: str = "high",
) -> SimpleNamespace:
    """Create an anomaly object containing every required attribute."""

    return SimpleNamespace(
        anomaly_code=anomaly_code,
        level=level,
        dimension_value=dimension_value,
        metric=metric,
        actual_value=actual_value,
        comparison_value=comparison_value,
        percentage_change=percentage_change,
        z_score=z_score,
        direction=direction,
        severity=severity,
    )


def make_anomaly_result(*anomalies: SimpleNamespace) -> SimpleNamespace:
    """Create an AnomalyResult-like object."""

    return SimpleNamespace(findings=list(anomalies))


def make_operations_result() -> SimpleNamespace:
    """Create an OperationsAnalysisResult-like object with all summaries."""

    return SimpleNamespace(
        period_summary=[
            {
                "period": "2026-06",
                "total_orders": 1_000,
                "completed_orders": 620,
                "cancelled_orders": 250,
                "total_revenue": 620_000.0,
                "average_order_value": 1_000.0,
                "fulfillment_percentage": 62.0,
                "cancellation_percentage": 25.0,
            }
        ],
        vehicle_summary=[
            {
                "vehicle_category": "Tata Ace",
                "total_orders": 500,
                "completed_orders": 300,
                "cancelled_orders": 150,
                "total_revenue": 300_000.0,
                "average_order_value": 1_000.0,
                "fulfillment_percentage": 60.0,
                "cancellation_percentage": 30.0,
            },
            {
                "vehicle_category": "2W",
                "total_orders": 800,
                "completed_orders": 720,
                "cancelled_orders": 40,
                "total_revenue": 360_000.0,
                "average_order_value": 500.0,
                "fulfillment_percentage": 90.0,
                "cancellation_percentage": 5.0,
            },
        ],
        cluster_summary=[
            {
                "pickup_cluster": "Chennai Central",
                "total_orders": 400,
                "completed_orders": 260,
                "cancelled_orders": 100,
                "total_revenue": 260_000.0,
                "average_order_value": 1_000.0,
                "fulfillment_percentage": 65.0,
                "cancellation_percentage": 25.0,
            }
        ],
    )


def test_constructor_rejects_invalid_thresholds() -> None:
    """Invalid configuration values must fail immediately."""

    with pytest.raises(ValueError, match="cannot be negative"):
        RootCauseAgent(minimum_evidence_score=-1)

    with pytest.raises(ValueError, match="between 0 and 100"):
        RootCauseAgent(medium_confidence_score=101)

    with pytest.raises(ValueError, match="must be greater"):
        RootCauseAgent(
            medium_confidence_score=70,
            high_confidence_score=70,
        )

    with pytest.raises(ValueError, match="must be positive"):
        RootCauseAgent(max_causes_per_anomaly=0)


def test_empty_anomaly_result_returns_no_root_cause_required() -> None:
    """No anomalies should produce a clean empty result."""

    agent = RootCauseAgent()

    result = agent.analyze(
        anomaly_result=make_anomaly_result(),
        operations_result=make_operations_result(),
    )

    assert isinstance(result, RootCauseResult)
    assert result.overall_status == "NO_ROOT_CAUSE_REQUIRED"
    assert result.anomalies_received == 0
    assert result.anomalies_analyzed == 0
    assert result.root_cause_count == 0
    assert result.high_confidence_count == 0
    assert result.unresolved_count == 0
    assert result.findings == []
    assert result.unresolved_anomalies == []


def test_revenue_anomaly_identifies_operational_drivers() -> None:
    """A revenue decrease should identify direct and leakage drivers."""

    anomaly = make_anomaly()
    agent = RootCauseAgent(max_causes_per_anomaly=5)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    cause_metrics = {finding.cause_metric for finding in result.findings}

    assert result.overall_status == "ACTION_REQUIRED"
    assert result.anomalies_received == 1
    assert result.anomalies_analyzed == 1
    assert result.unresolved_count == 0
    assert result.root_cause_count >= 4
    assert "completed_orders" in cause_metrics
    assert "average_order_value" in cause_metrics
    assert "fulfillment_percentage" in cause_metrics
    assert "cancellation_percentage" in cause_metrics

    for finding in result.findings:
        assert isinstance(finding, RootCauseFinding)
        assert finding.anomaly_code == "ANOM-001"
        assert finding.level == "vehicle_category"
        assert finding.dimension_value == "Tata Ace"
        assert 0 <= finding.evidence_score <= 100
        assert finding.recommended_next_check


def test_vehicle_revenue_anomaly_uses_matching_variance_row() -> None:
    """Vehicle-level analysis must use that vehicle's variance effects."""

    anomaly = make_anomaly()
    variance_result = SimpleNamespace(
        price_effect=-10_000.0,
        volume_effect=-20_000.0,
        new_discontinued_effect=0.0,
        vehicle_variance_summary=[
            {
                "vehicle_category": "Tata Ace",
                "price_effect": -30_000.0,
                "volume_effect": -70_000.0,
                "new_discontinued_effect": 0.0,
            },
            {
                "vehicle_category": "2W",
                "price_effect": 5_000.0,
                "volume_effect": 10_000.0,
                "new_discontinued_effect": 0.0,
            },
        ],
    )
    agent = RootCauseAgent(max_causes_per_anomaly=10)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
        variance_result=variance_result,
    )

    findings_by_metric = {
        finding.cause_metric: finding for finding in result.findings
    }

    assert findings_by_metric["price_effect"].evidence["price_effect"] == -30_000.0
    assert findings_by_metric["volume_effect"].evidence["volume_effect"] == -70_000.0
    assert (
        findings_by_metric["volume_effect"]
        .evidence["absolute_contribution_percentage"]
        == 70.0
    )
    assert findings_by_metric["price_effect"].impact == "unfavorable"
    assert findings_by_metric["volume_effect"].impact == "unfavorable"


def test_aov_anomaly_includes_only_price_variance_effect() -> None:
    """AOV analysis must not add volume effect from variance decomposition."""

    anomaly = make_anomaly(
        metric="average_order_value",
        actual_value=900.0,
        comparison_value=1_000.0,
        percentage_change=-10.0,
    )
    variance_result = SimpleNamespace(
        price_effect=-25_000.0,
        volume_effect=-75_000.0,
        new_discontinued_effect=0.0,
        vehicle_variance_summary=[],
    )
    agent = RootCauseAgent(max_causes_per_anomaly=10)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
        variance_result=variance_result,
    )

    cause_metrics = {finding.cause_metric for finding in result.findings}

    assert "price_or_trip_mix" in cause_metrics
    assert "price_effect" in cause_metrics
    assert "volume_effect" not in cause_metrics


def test_completed_orders_anomaly_is_limited_to_configured_causes() -> None:
    """The agent must keep only the configured number of top causes."""

    anomaly = make_anomaly(
        metric="completed_orders",
        actual_value=300,
        comparison_value=450,
        percentage_change=-33.33,
        severity="critical",
    )
    agent = RootCauseAgent(max_causes_per_anomaly=2)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    assert result.root_cause_count == 2
    assert len(result.findings) == 2
    assert result.findings[0].evidence_score >= result.findings[1].evidence_score


def test_unknown_metric_becomes_unresolved() -> None:
    """A supported level with an unsupported metric should be unresolved."""

    anomaly = make_anomaly(metric="partner_productivity")
    agent = RootCauseAgent()

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    assert result.overall_status == "REVIEW"
    assert result.root_cause_count == 0
    assert result.unresolved_count == 1
    assert result.unresolved_anomalies[0]["anomaly_code"] == "ANOM-001"
    assert result.unresolved_anomalies[0]["metric"] == "partner_productivity"


def test_missing_summary_match_can_leave_revenue_anomaly_unresolved() -> None:
    """A revenue anomaly without a matching row or variance evidence is unresolved."""

    anomaly = make_anomaly(dimension_value="Unknown Vehicle")
    agent = RootCauseAgent()

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    assert result.overall_status == "REVIEW"
    assert result.root_cause_count == 0
    assert result.unresolved_count == 1


def test_dimension_matching_is_case_and_whitespace_insensitive() -> None:
    """Dimension values should match after normalization."""

    anomaly = make_anomaly(dimension_value="  TATA ACE  ")
    agent = RootCauseAgent(max_causes_per_anomaly=5)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    assert result.root_cause_count > 0
    assert result.unresolved_count == 0


def test_minimum_evidence_score_filters_findings() -> None:
    """Findings below a strict minimum score should be removed."""

    anomaly = make_anomaly(
        metric="total_orders",
        severity="low",
        percentage_change=-1.0,
        z_score=-0.1,
    )
    agent = RootCauseAgent(minimum_evidence_score=99.0)

    result = agent.analyze(
        anomaly_result=make_anomaly_result(anomaly),
        operations_result=make_operations_result(),
    )

    assert result.root_cause_count == 0
    assert result.unresolved_count == 1
    assert result.overall_status == "REVIEW"


def test_input_validation_errors() -> None:
    """Required result structures and anomaly fields must be validated."""

    agent = RootCauseAgent()
    operations_result = make_operations_result()

    with pytest.raises(ValueError, match="anomaly_result is required"):
        agent.analyze(None, operations_result)

    with pytest.raises(ValueError, match="findings attribute"):
        agent.analyze(SimpleNamespace(), operations_result)

    with pytest.raises(TypeError, match="findings must be a list"):
        agent.analyze(
            SimpleNamespace(findings="not-a-list"),
            operations_result,
        )

    with pytest.raises(ValueError, match="operations_result is required"):
        agent.analyze(make_anomaly_result(), None)

    incomplete_operations_result = SimpleNamespace(
        period_summary=[],
        vehicle_summary=[],
    )
    with pytest.raises(ValueError, match="missing required summaries"):
        agent.analyze(
            make_anomaly_result(),
            incomplete_operations_result,
        )


def test_invalid_anomaly_level_and_missing_attribute_fail() -> None:
    """Unsupported levels and incomplete anomaly objects must fail."""

    agent = RootCauseAgent()
    operations_result = make_operations_result()

    unsupported_level = make_anomaly(level="customer")
    with pytest.raises(ValueError, match="Unsupported anomaly level"):
        agent.analyze(
            make_anomaly_result(unsupported_level),
            operations_result,
        )

    incomplete_anomaly = SimpleNamespace(
        anomaly_code="ANOM-X",
        level="period",
    )
    with pytest.raises(ValueError, match="missing required attributes"):
        agent.analyze(
            make_anomaly_result(incomplete_anomaly),
            operations_result,
        )