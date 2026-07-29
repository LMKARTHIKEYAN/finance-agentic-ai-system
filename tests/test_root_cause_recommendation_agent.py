"""Tests for controlled root-cause and recommendation execution."""

from typing import Any

import pytest

from src.autonomous.agents.root_cause_recommendation_agent import (
    RootCauseRecommendationAgent,
    RootCauseRecommendationAgentError,
)
from src.autonomous.schemas import EvidenceRecord, PlanStep, ToolResult


def _step(**arguments: Any) -> PlanStep:
    return PlanStep(
        step_id="diagnostics",
        capability="root_cause_recommendation",
        arguments={
            "agent_name": "root_cause_recommendation_agent",
            "tool_name": "identify_supported_root_causes",
            "recommendation_tool_name": (
                "generate_supported_recommendations"
            ),
            **arguments,
        },
    )


def _evidence(
    evidence_id: str = "pnl-001",
    *,
    verified: bool = True,
    compact_payload: dict[str, Any] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_tool="generate_validated_pnl_analysis",
        result_type="pnl",
        tool_status="completed",
        compact_payload=compact_payload or {"summary": "compact"},
        reconciled=True,
        verified=verified,
    )


def test_diagnostic_agent_calls_tools_in_order_and_links_evidence() -> None:
    calls: list[tuple[str, Any]] = []

    def root_tool(**kwargs: Any) -> ToolResult:
        calls.append(("root", kwargs))
        return ToolResult(
            call_id="root",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload={"causes": ["volume"]},
        )

    def recommendation_tool(**kwargs: Any) -> ToolResult:
        calls.append(("recommendation", kwargs))
        return ToolResult(
            call_id="recommend",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload={"recommendations": ["monitor volume"]},
        )

    result = RootCauseRecommendationAgent(
        root_cause_tool=root_tool,
        recommendation_tool=recommendation_tool,
    ).execute(
        _step(),
        evidence=(_evidence(),),
        anomaly_result={"anomaly": "volume"},
        operations_result={"revenue": 100},
        revenue_variance_result={"variance": 10},
    )

    assert result.evidence_ids == ("pnl-001",)
    assert [item[0] for item in calls] == ["root", "recommendation"]
    assert calls[1][1]["root_cause_result"] == {"causes": ["volume"]}


def test_diagnostic_agent_requires_verified_evidence() -> None:
    with pytest.raises(ValueError, match="verified evidence"):
        RootCauseRecommendationAgent().execute(
            _step(),
            evidence=(_evidence(verified=False),),
            anomaly_result={},
            operations_result={},
        )


def test_diagnostic_agent_rejects_unapproved_tools() -> None:
    with pytest.raises(ValueError, match="unapproved root-cause"):
        RootCauseRecommendationAgent().execute(
            _step(tool_name="execute_python"),
            evidence=(_evidence(),),
            anomaly_result={},
            operations_result={},
        )


def test_diagnostic_agent_rejects_missing_internal_results() -> None:
    with pytest.raises(
        RootCauseRecommendationAgentError,
        match="Root-cause tool execution failed",
    ) as error:
        RootCauseRecommendationAgent().execute(
            _step(),
            evidence=(_evidence(),),
            anomaly_result=None,
            operations_result={},
        )

    assert error.value.failure_code == "root_cause_tool"


def test_diagnostic_agent_uses_multi_period_pnl_evidence() -> None:
    result = RootCauseRecommendationAgent().execute(
        _step(),
        evidence=(
            _evidence(
                compact_payload={
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
            ),
        ),
        anomaly_result=None,
        operations_result=None,
    )

    assert result.root_cause_result.payload["net_profit_change"] == 11.0
    assert result.recommendation_result.payload["recommendations"]


def test_diagnostic_agent_uses_gp_evidence_without_legacy_inputs() -> None:
    result = RootCauseRecommendationAgent().execute(
        _step(),
        evidence=(
            EvidenceRecord(
                evidence_id="gp-001",
                source_tool="calculate_validated_gp_decomposition",
                result_type="gp_decomposition",
                tool_status="completed",
                compact_payload={
                    "budget_gp_percentage": 30.0,
                    "actual_gp_percentage": 28.0,
                    "mix_effect_percentage_points": -0.5,
                    "price_effect_percentage_points": -0.5,
                    "cost_effect_percentage_points": -1.0,
                    "total_variance_percentage_points": -2.0,
                },
                reconciled=True,
                verified=True,
            ),
        ),
        anomaly_result=None,
        operations_result=None,
    )

    assert result.root_cause_result.payload["analysis_type"] == (
        "multi_evidence_performance"
    )
    assert result.root_cause_result.payload["risk_findings"][0][
        "metric"
    ] == "total_variance_percentage_points"
    assert result.recommendation_result.payload["recommendations"]


def test_diagnostic_agent_stops_when_root_cause_tool_fails() -> None:
    recommendation_called = False

    def root_tool(**kwargs: Any) -> ToolResult:
        raise RuntimeError("private tool detail")

    def recommendation_tool(**kwargs: Any) -> ToolResult:
        nonlocal recommendation_called
        recommendation_called = True
        raise AssertionError

    with pytest.raises(
        RootCauseRecommendationAgentError,
        match="execution failed",
    ) as error:
        RootCauseRecommendationAgent(
            root_cause_tool=root_tool,
            recommendation_tool=recommendation_tool,
        ).execute(
            _step(),
            evidence=(_evidence(),),
            anomaly_result={},
            operations_result={},
        )

    assert recommendation_called is False
    assert "private tool detail" not in str(error.value)
    assert error.value.failure_code == "root_cause_tool"


def test_diagnostic_agent_rejects_wrong_tool_result_identity() -> None:
    def root_tool(**kwargs: Any) -> ToolResult:
        return ToolResult(
            call_id="wrong",
            tool_name="calculate_validated_kpis",
            status="completed",
        )

    with pytest.raises(
        RootCauseRecommendationAgentError,
        match="tool identity",
    ):
        RootCauseRecommendationAgent(root_cause_tool=root_tool).execute(
            _step(),
            evidence=(_evidence(),),
            anomaly_result={},
            operations_result={},
        )
