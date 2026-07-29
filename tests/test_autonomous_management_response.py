"""Tests for deterministic grounded management-response composition."""

from src.autonomous.agents.root_cause_recommendation_agent import (
    DiagnosticAnalysisResult,
)
from src.autonomous.management_response import ManagementResponseComposer
from src.autonomous.schemas import ToolResult


def _pnl_diagnostics() -> DiagnosticAnalysisResult:
    return DiagnosticAnalysisResult(
        root_cause_result=ToolResult(
            call_id="root",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload={
                "analysis_type": "pnl_period_change",
                "base_month": "2026-04",
                "current_month": "2026-05",
                "base_net_profit": 100.0,
                "current_net_profit": 125.0,
                "net_profit_change": 25.0,
                "drivers": [
                    {
                        "metric": "revenue",
                        "change": 40.0,
                        "profit_effect": 40.0,
                        "impact": "favorable",
                    },
                    {
                        "metric": "direct_cost",
                        "change": 15.0,
                        "profit_effect": -15.0,
                        "impact": "unfavorable",
                    },
                ],
            },
        ),
        recommendation_result=ToolResult(
            call_id="recommend",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload={
                "recommendations": [
                    {
                        "action": (
                            "Monitor and sustain the favorable revenue movement"
                        )
                    }
                ]
            },
        ),
        evidence_ids=("pnl-001",),
    )


def test_composer_uses_only_supplied_pnl_diagnostics() -> None:
    result = ManagementResponseComposer().compose(
        _pnl_diagnostics(),
        fallback_answer="Unsupported old draft.",
    )

    assert "2026-04" in result
    assert "2026-05" in result
    assert "₹25.00" in result
    assert "₹40.00 profit effect (favorable)" in result
    assert "-₹15.00 profit effect (unfavorable)" in result
    assert "Evidence: pnl-001." in result
    assert "Unsupported old draft" not in result


def test_composer_keeps_fallback_for_non_pnl_diagnostics() -> None:
    diagnostics = _pnl_diagnostics().model_copy(
        update={
            "root_cause_result": ToolResult(
                call_id="root",
                tool_name="identify_supported_root_causes",
                status="completed",
                payload={"analysis_type": "operational"},
            )
        }
    )

    result = ManagementResponseComposer().compose(
        diagnostics,
        fallback_answer="Existing grounded operational answer.",
    )

    assert result == "Existing grounded operational answer."


def test_composer_builds_grounded_multi_evidence_response() -> None:
    diagnostics = DiagnosticAnalysisResult(
        root_cause_result=ToolResult(
            call_id="root",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload={
                "analysis_type": "multi_evidence_performance",
                "findings": [
                    {
                        "evidence_id": "variance-001",
                        "result_type": "revenue_variance",
                        "metric": "revenue_variance",
                        "value": -10.0,
                        "unit": "currency",
                        "risk": True,
                    }
                ],
                "risk_findings": [
                    {
                        "evidence_id": "variance-001",
                        "result_type": "revenue_variance",
                        "metric": "revenue_variance",
                        "value": -10.0,
                        "unit": "currency",
                        "risk": True,
                    }
                ],
            },
        ),
        recommendation_result=ToolResult(
            call_id="recommend",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload={
                "recommendations": [
                    {
                        "action": (
                            "Investigate the adverse revenue variance"
                        )
                    }
                ]
            },
        ),
        evidence_ids=("variance-001",),
    )

    result = ManagementResponseComposer().compose(
        diagnostics,
        fallback_answer="Old broad response.",
    )

    assert "Revenue Variance: -₹10.00 [variance-001]." in result
    assert "Financial risk indicators" in result
    assert "Investigate the adverse revenue variance." in result
    assert "Old broad response" not in result
