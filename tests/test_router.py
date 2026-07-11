"""Tests for the deterministic LangGraph request router."""

import pytest

from src.orchestrator.router import identify_flow, route_request
from src.orchestrator.state import FinanceGraphState


@pytest.mark.parametrize(
    ("user_request", "expected_flow"),
    [
        ("Show KPI performance", "kpi"),
        ("Display key performance indicators", "kpi"),
        ("Prepare budget analysis", "budget"),
        ("Create a budget plan", "budget"),
        ("Forecast next 3 months", "forecast"),
        ("Predict future revenue", "forecast"),
        ("Explain revenue variance", "variance"),
        ("Why is revenue below budget?", "variance"),
        ("Run scenario analysis", "scenario"),
        ("Show best case and worst case", "scenario"),
        ("Run complete management report", "full"),
        ("Run full analysis", "full"),
        ("Hello", "unknown"),
        ("", "unknown"),
        ("   ", "unknown"),
    ],
)
def test_identify_flow(user_request: str, expected_flow: str) -> None:
    """The router should identify the correct flow from the request."""

    assert identify_flow(user_request) == expected_flow


def test_route_request_updates_state_for_known_flow() -> None:
    """A valid request should update the state and start execution."""

    state: FinanceGraphState = {
        "user_request": "Show KPI performance",
        "execution_status": "pending",
        "errors": [],
    }

    result = route_request(state)

    assert result["selected_flow"] == "kpi"
    assert result["execution_status"] == "running"
    assert result["error_message"] == ""
    assert result["errors"] == []


def test_route_request_preserves_existing_state() -> None:
    """The router should preserve existing input data and state values."""

    state: FinanceGraphState = {
        "user_request": "Prepare budget analysis",
        "operations_result": {"total_revenue": 100000},
        "errors": [],
        "execution_status": "pending",
    }

    result = route_request(state)

    assert result["selected_flow"] == "budget"
    assert result["operations_result"] == {"total_revenue": 100000}
    assert result["execution_status"] == "running"


def test_route_request_handles_unknown_request() -> None:
    """An unsupported request should safely mark the state as failed."""

    state: FinanceGraphState = {
        "user_request": "Tell me a joke",
        "errors": [],
        "execution_status": "pending",
    }

    result = route_request(state)

    assert result["selected_flow"] == "unknown"
    assert result["execution_status"] == "failed"
    assert result["error_message"] == "Unable to identify a supported workflow."
    assert result["errors"] == [
        "Unable to identify a supported workflow."
    ]


def test_route_request_appends_error_without_removing_existing_errors() -> None:
    """Unknown routing should preserve and append to existing errors."""

    state: FinanceGraphState = {
        "user_request": "",
        "errors": ["Previous warning"],
        "execution_status": "pending",
    }

    result = route_request(state)

    assert result["selected_flow"] == "unknown"
    assert result["execution_status"] == "failed"
    assert result["errors"] == [
        "Previous warning",
        "Unable to identify a supported workflow.",
    ]


def test_route_request_handles_missing_user_request() -> None:
    """The router should safely handle a missing user_request field."""

    state: FinanceGraphState = {
        "errors": [],
        "execution_status": "pending",
    }

    result = route_request(state)

    assert result["selected_flow"] == "unknown"
    assert result["execution_status"] == "failed"