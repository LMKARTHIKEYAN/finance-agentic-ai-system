from datetime import date
from pathlib import Path

import pandas as pd

from src.autonomous.goal_builder import GoalBuilder
from src.autonomous.permission_policy import PermissionPolicy
from src.autonomous.recovery_policy import RecoveryPolicy
from src.autonomous.release_controller import ReleaseController
from src.autonomous.scheduled_goal_builder import ScheduledGoalBuilder
from src.autonomous.tools.snowflake_tools import load_finance_data
from src.evaluation.autonomous_metrics import AutonomousMetrics
from src.autonomous.schemas import ToolResult
from src.services.autonomous_finance_service import AutonomousFinanceService


class RepositoryStub:
    def get_orders(self, *_):
        return pd.DataFrame({"order_date": ["2026-08-01"], "commission_amount": [100.0]})

    def get_budget(self, *_):
        return pd.DataFrame({"month": ["2026-08"], "vehicle_category": ["2W"]})

    def get_corporate_expenses(self, *_):
        return pd.DataFrame({"month": ["2026-08"]})

    def get_budget_corporate_expenses(self, *_):
        return pd.DataFrame({"month": ["2026-08"]})


def test_snowflake_tool_keeps_frames_internal_and_commission_revenue():
    result = load_finance_data(
        repository=RepositoryStub(), start_date=date(2026, 8, 1), end_date=date(2026, 8, 31)
    )
    assert result["revenue_definition"] == "commission_amount"
    assert result["finance_context"].operations_data.shape == (1, 2)
    assert result["dataset_summaries"]["operations_data"].row_count == 1


def test_permission_and_recovery_are_bounded():
    policy = PermissionPolicy(allowed_tools={"read", "email"}, external_write_tools={"email"})
    assert policy.evaluate("read").allowed
    assert policy.evaluate("email").requires_approval
    assert not policy.evaluate("delete").allowed
    assert RecoveryPolicy(max_retries=1).decide(attempt=1, recoverable=True).action == "retry"
    assert RecoveryPolicy(max_retries=1).decide(attempt=2, recoverable=True).action == "stop"


def test_release_controller_blocks_unapproved_recipient(tmp_path: Path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    controller = ReleaseController(["cfo@example.com"])
    assert controller.evaluate(
        recipients=["other@example.com"], validated=True, pdf_path=pdf, human_approved=True
    ).approved is False


def test_scheduled_monthly_goal_requests_full_management_pack():
    goal = ScheduledGoalBuilder().build("monthly", run_date=date(2026, 8, 1))
    keys = {item.key for item in goal.criteria}
    assert {"pnl_analysis", "revenue_variance", "gp_decomposition", "forecast"} <= keys


def test_ambiguous_pnl_requests_clarification():
    goal = GoalBuilder().build("P&L 2026")
    assert goal.ambiguities == ("annual_period_basis",)


def test_goal_builder_recognizes_vehicle_category_and_tata_ace_family():
    assert GoalBuilder().build("P&L for August 2026 for 2w").reporting_scope.category == "2W"
    assert GoalBuilder().build("P&L for August 2026 tata ace").reporting_scope.category == "Tata Ace"


def test_autonomous_metrics_rates():
    metrics = AutonomousMetrics(goals=2, completed=1, validated_answers=1)
    assert metrics.completion_rate() == 0.5
    assert metrics.evidence_validation_rate() == 1.0


def test_revenue_variance_answer_explains_financial_drivers():
    from src.services.autonomous_finance_service import _finance_summary

    answer = _finance_summary(
        "revenue variance",
        {
            "actual_revenue": 120.0,
            "budget_revenue": 100.0,
            "revenue_variance": 20.0,
            "actual_orders": 12,
            "budget_orders": 10,
            "actual_aov": 10.0,
            "budget_aov": 10.0,
            "volume_effect": 20.0,
            "price_effect": 0.0,
            "vehicle_variance_summary": [
                {"vehicle_category": "2W", "revenue_variance": 20.0}
            ],
        },
        "revenue-001",
    )
    assert "20.00%" in answer
    assert "volume effect" in answer
    assert "Major category drivers: 2W" in answer


def test_end_to_end_goal_data_tool_reconciliation_and_cited_answer(monkeypatch):
    def fake_pnl(*, finance_context):
        assert finance_context.operations_data is not None
        return ToolResult(
            call_id="pnl-call",
            tool_name="calculate_pnl",
            status="completed",
            payload={
                "actual_pnl": [{"revenue": 120.0}],
                "budget_pnl": [{"revenue": 100.0}],
                "variance_pnl": [{"revenue_variance": 20.0}],
                "pnl_summary": {
                    "actual": {"revenue": 120.0, "gross_profit": 40.0, "net_profit": 20.0},
                    "budget": {"revenue": 100.0, "net_profit": 15.0},
                    "variance": {"net_profit_variance": 5.0},
                },
                "available_months": ["2026-08"],
                "excluded_actual_months": [],
                "excluded_budget_months": [],
                "revenue_definition": "commission_amount",
            },
        )

    monkeypatch.setattr("src.services.autonomous_finance_service.calculate_pnl", fake_pnl)
    response = AutonomousFinanceService(repository=RepositoryStub()).ask(
        "Show P&L for August 2026"
    )
    assert response.status == "completed"
    assert "actual revenue INR 120.00" in response.answer
    assert "commission_amount" in response.answer
    assert "[pnl-001]" in response.answer
    assert response.evidence[0]["verified"] is True
    assert response.management_commentary is not None
    assert "Revenue was INR 120.00" in response.management_commentary["executive_summary"]
    assert "Net profit was INR 20.00" in response.management_commentary["executive_summary"]
