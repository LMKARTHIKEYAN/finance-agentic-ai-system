"""Tests for the shared LangGraph finance state."""

from typing import get_type_hints

import pandas as pd

from src.orchestrator.state import FinanceGraphState


def test_finance_graph_state_can_be_created() -> None:
    """FinanceGraphState should accept the main workflow fields."""

    operations_data = pd.DataFrame(
        {
            "revenue": [1000, 1200],
            "cost": [700, 800],
        }
    )

    state: FinanceGraphState = {
        "user_request": "Show KPI performance",
        "selected_flow": "kpi",
        "operations_data": operations_data,
        "errors": [],
        "execution_status": "pending",
    }

    assert state["user_request"] == "Show KPI performance"
    assert state["selected_flow"] == "kpi"
    assert state["execution_status"] == "pending"
    assert state["errors"] == []
    assert state["operations_data"].equals(operations_data)


def test_finance_graph_state_allows_partial_state() -> None:
    """
    FinanceGraphState uses total=False.

    This means the graph can begin with only the fields required at the
    current stage and later nodes can add additional results.
    """

    state: FinanceGraphState = {
        "user_request": "Create forecast",
    }

    assert state["user_request"] == "Create forecast"
    assert "forecast_result" not in state
    assert "execution_status" not in state


def test_finance_graph_state_contains_required_field_definitions() -> None:
    """The state schema should define all required workflow fields."""

    type_hints = get_type_hints(FinanceGraphState)

    expected_fields = {
        "user_request",
        "selected_flow",
        "operations_data",
        "budget_data",
        "forecast_data",
        "scenario_data",
        "validation_result",
        "cleaned_data",
        "profile_result",
        "operations_result",
        "budget_result",
        "forecast_result",
        "scenario_result",
        "variance_result",
        "finance_rules_result",
        "anomaly_result",
        "root_cause_result",
        "recommendation_result",
        "kpi_result",
        "commentary_result",
        "report_result",
        "errors",
        "error_message",
        "execution_status",
    }

    assert expected_fields.issubset(type_hints.keys())


def test_state_can_store_agent_results() -> None:
    """Nodes should be able to add agent outputs to the shared state."""

    state: FinanceGraphState = {
        "user_request": "Run complete management analysis",
        "selected_flow": "full",
        "execution_status": "running",
        "errors": [],
    }

    state["kpi_result"] = {"revenue_growth": 10.5}
    state["anomaly_result"] = {"anomalies_found": 2}
    state["commentary_result"] = "Revenue increased by 10.5%."
    state["execution_status"] = "completed"

    assert state["kpi_result"]["revenue_growth"] == 10.5
    assert state["anomaly_result"]["anomalies_found"] == 2
    assert state["commentary_result"] == "Revenue increased by 10.5%."
    assert state["execution_status"] == "completed"


def test_state_can_store_errors() -> None:
    """The state should support safe error recording."""

    state: FinanceGraphState = {
        "user_request": "Run forecast",
        "selected_flow": "forecast",
        "errors": [],
        "execution_status": "running",
    }

    state["errors"].append("Forecast output is empty")
    state["error_message"] = "Forecast output is empty"
    state["execution_status"] = "failed"

    assert state["errors"] == ["Forecast output is empty"]
    assert state["error_message"] == "Forecast output is empty"
    assert state["execution_status"] == "failed"