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
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_tool="generate_validated_pnl_analysis",
        result_type="pnl",
        tool_status="completed",
        compact_payload={"summary": "compact"},
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
    with pytest.raises(ValueError, match="anomaly_result"):
        RootCauseRecommendationAgent().execute(
            _step(),
            evidence=(_evidence(),),
            anomaly_result=None,
            operations_result={},
        )


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
