"""Tests for Finance Agentic AI LangGraph node wrappers."""

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.orchestrator.nodes as nodes
from src.orchestrator.state import FinanceGraphState


# ----------------------------------------------------------------------
# Reusable test data
# ----------------------------------------------------------------------


@pytest.fixture
def valid_operations_data() -> pd.DataFrame:
    """Return a minimal valid operations DataFrame."""

    return pd.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-2"],
            "order_date": ["2026-01-01", "2026-01-02"],
            "pickup_cluster": ["Chennai Central", "Guindy"],
            "drop_cluster": ["T Nagar", "Velachery"],
            "vehicle_category": ["2W", "3W"],
            "order_status": ["Completed", "Completed"],
            "fare": [200.0, 350.0],
            "commission_amount": [20.0, 35.0],
            "partner_payout": [150.0, 260.0],
            "incentive": [10.0, 15.0],
            "goodwill": [0.0, 0.0],
            "dry_run": [0.0, 0.0],
            "surge": [5.0, 10.0],
            "trip_distance_km": [5.0, 8.0],
            "delivery_time_minutes": [25.0, 40.0],
        }
    )


@pytest.fixture
def valid_budget_data() -> pd.DataFrame:
    """Return a minimal valid budget DataFrame."""

    return pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "vehicle_category": ["2W", "3W"],
            "budget_orders": [1000, 800],
            "budget_revenue": [200000.0, 280000.0],
        }
    )


@pytest.fixture
def base_state() -> FinanceGraphState:
    """Return the common starting graph state."""

    return {
        "user_request": "Run complete management analysis",
        "selected_flow": "full",
        "execution_status": "running",
        "errors": [],
        "executed_nodes": [],
    }


# ----------------------------------------------------------------------
# Validation-node tests
# ----------------------------------------------------------------------


def test_validate_operations_node_success(
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """Valid operations data should be stored in state."""

    state: FinanceGraphState = {
        **base_state,
        "operations_data": valid_operations_data,
    }

    result = nodes.validate_operations_node(state)

    assert result["execution_status"] == "running"
    assert result["operations_validation"].is_valid is True
    assert result["validation_result"].is_valid is True
    assert result["error_message"] == ""
    assert result["failed_node"] == ""
    assert "validate_operations" in result["executed_nodes"]


def test_validate_budget_node_success(
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """Valid budget data should be stored in state."""

    state: FinanceGraphState = {
        **base_state,
        "budget_data": valid_budget_data,
    }

    result = nodes.validate_budget_node(state)

    assert result["execution_status"] == "running"
    assert result["budget_validation"].is_valid is True
    assert result["validation_result"].is_valid is True
    assert result["errors"] == []
    assert "validate_budget" in result["executed_nodes"]


def test_validate_operations_node_handles_missing_dataframe(
    base_state: FinanceGraphState,
) -> None:
    """Missing operations data should produce a failed state."""

    result = nodes.validate_operations_node(base_state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "validate_operations"
    assert "operations_data" in result["error_message"]
    assert len(result["errors"]) == 1


def test_validate_operations_node_handles_empty_dataframe(
    base_state: FinanceGraphState,
) -> None:
    """An empty operations DataFrame should produce a failed state."""

    state: FinanceGraphState = {
        **base_state,
        "operations_data": pd.DataFrame(),
    }

    result = nodes.validate_operations_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "validate_operations"
    assert "empty DataFrame" in result["error_message"]


def test_validate_operations_node_handles_invalid_columns(
    base_state: FinanceGraphState,
) -> None:
    """Invalid operations columns should fail validation safely."""

    invalid_data = pd.DataFrame(
        {
            "order_id": ["ORD-1"],
        }
    )

    state: FinanceGraphState = {
        **base_state,
        "operations_data": invalid_data,
    }

    result = nodes.validate_operations_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "validate_operations"
    assert "Operations validation failed" in result["error_message"]
    assert result["errors"]


# ----------------------------------------------------------------------
# Cleaning and profiling tests
# ----------------------------------------------------------------------


def test_clean_operations_node_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """The cleaning node should store cleaned operations data."""

    expected_cleaned_data = valid_operations_data.copy()
    expected_cleaned_data["cleaned"] = True

    monkeypatch.setattr(
        nodes.CleaningAgent,
        "clean_operations_data",
        lambda dataframe: expected_cleaned_data,
    )

    state: FinanceGraphState = {
        **base_state,
        "operations_data": valid_operations_data,
        "operations_validation": SimpleNamespace(is_valid=True),
    }

    result = nodes.clean_operations_node(state)

    assert result["execution_status"] == "running"
    assert result["cleaned_operations_data"].equals(
        expected_cleaned_data
    )
    assert result["cleaned_data"].equals(expected_cleaned_data)
    assert "clean_operations" in result["executed_nodes"]


def test_clean_operations_node_requires_validation(
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """Cleaning should not run before validation."""

    state: FinanceGraphState = {
        **base_state,
        "operations_data": valid_operations_data,
    }

    result = nodes.clean_operations_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "clean_operations"
    assert "operations_validation" in result["error_message"]


def test_profile_operations_node_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """The profile node should save the returned profile object."""

    expected_profile = SimpleNamespace(
        total_rows=2,
        total_columns=len(valid_operations_data.columns),
    )

    monkeypatch.setattr(
        nodes.ProfilingAgent,
        "profile_operations_data",
        lambda dataframe: expected_profile,
    )

    state: FinanceGraphState = {
        **base_state,
        "cleaned_operations_data": valid_operations_data,
    }

    result = nodes.profile_operations_node(state)

    assert result["operations_profile"] is expected_profile
    assert result["profile_result"] is expected_profile
    assert result["execution_status"] == "running"
    assert "profile_operations" in result["executed_nodes"]


# ----------------------------------------------------------------------
# Analysis-node tests
# ----------------------------------------------------------------------


def test_operations_analysis_node_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """The operations node should call the existing agent."""

    expected_result = SimpleNamespace(
        total_orders=2,
        period_summary=[
            {
                "period": "2026-01",
                "total_orders": 2,
                "total_revenue": 550.0,
            }
        ],
    )

    def fake_analyze(
        self: Any,
        data: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        vehicle_category: str | None = None,
        pickup_cluster: str | None = None,
        group_by: str = "month",
    ) -> Any:
        assert data.equals(valid_operations_data)
        assert group_by == "month"
        return expected_result

    monkeypatch.setattr(
        nodes.OperationsAnalysisAgent,
        "analyze",
        fake_analyze,
    )

    state: FinanceGraphState = {
        **base_state,
        "cleaned_operations_data": valid_operations_data,
        "frequency": "month",
        "filters": {},
    }

    result = nodes.operations_analysis_node(state)

    assert result["operations_result"] is expected_result
    assert result["execution_status"] == "running"
    assert "operations_analysis" in result["executed_nodes"]


def test_forecast_node_success(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """The forecast node should store a non-empty forecast result."""

    operations_result = SimpleNamespace(
        period_summary=[
            {
                "period": "2026-01",
                "total_orders": 100,
                "total_revenue": 20000.0,
            },
            {
                "period": "2026-02",
                "total_orders": 110,
                "total_revenue": 23000.0,
            },
        ]
    )

    expected_forecast = SimpleNamespace(
        forecast_summary=[
            {
                "forecast_period": "2026-03",
                "forecast_orders": 120,
                "forecast_revenue": 25000.0,
                "forecast_aov": 208.33,
            }
        ]
    )

    def fake_analyze(
        self: Any,
        period_summary: list[dict[str, Any]],
        frequency: str,
        rolling_window: int,
        forecast_periods: int,
    ) -> Any:
        assert period_summary == operations_result.period_summary
        assert frequency == "month"
        assert rolling_window == 3
        assert forecast_periods == 2
        return expected_forecast

    monkeypatch.setattr(
        nodes.ForecastAgent,
        "analyze",
        fake_analyze,
    )

    state: FinanceGraphState = {
        **base_state,
        "operations_result": operations_result,
        "frequency": "month",
        "rolling_window": 3,
        "forecast_periods": 2,
    }

    result = nodes.forecast_node(state)

    assert result["forecast_result"] is expected_forecast
    assert result["execution_status"] == "running"
    assert "forecast" in result["executed_nodes"]


def test_forecast_node_rejects_empty_output(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """An empty forecast output should safely fail the node."""

    operations_result = SimpleNamespace(
        period_summary=[
            {
                "period": "2026-01",
                "total_orders": 100,
                "total_revenue": 20000.0,
            }
        ]
    )

    monkeypatch.setattr(
        nodes.ForecastAgent,
        "analyze",
        lambda self, **kwargs: SimpleNamespace(
            forecast_summary=[]
        ),
    )

    state: FinanceGraphState = {
        **base_state,
        "operations_result": operations_result,
    }

    result = nodes.forecast_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "forecast"
    assert "empty forecast output" in result["error_message"]


def test_variance_node_success(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """The variance node should pass actual and budget results."""

    operations_result = SimpleNamespace(total_revenue=100000.0)
    budget_result = SimpleNamespace(total_budget_revenue=110000.0)
    expected_variance = SimpleNamespace(
        total_revenue_variance=-10000.0
    )

    def fake_analyze(
        self: Any,
        actual_result: Any,
        budget_result: Any,
    ) -> Any:
        assert actual_result is operations_result
        assert budget_result is not None
        return expected_variance

    monkeypatch.setattr(
        nodes.RevenueVarianceAgent,
        "analyze",
        fake_analyze,
    )

    state: FinanceGraphState = {
        **base_state,
        "operations_result": operations_result,
        "budget_result": budget_result,
    }

    result = nodes.variance_node(state)

    assert result["variance_result"] is expected_variance
    assert result["execution_status"] == "running"
    assert "variance" in result["executed_nodes"]


def test_pnl_node_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """The P&L node should store all structured P&L outputs."""

    corporate_expenses_data = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "expense_category": ["Salary", "Rent"],
            "actual_amount": [50000.0, 20000.0],
        }
    )
    budget_corporate_expenses_data = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "expense_category": ["Salary", "Rent"],
            "budget_amount": [48000.0, 22000.0],
        }
    )

    actual_pnl = {
        "revenue": 550.0,
        "gross_profit": 135.0,
        "operating_profit": 65000.0,
    }
    budget_pnl = {
        "revenue": 480000.0,
        "gross_profit": 120000.0,
        "operating_profit": 50000.0,
    }
    variance_pnl = {
        "revenue_variance": -479450.0,
        "gross_profit_variance": -119865.0,
        "operating_profit_variance": 15000.0,
    }
    pnl_summary = {
        "actual_revenue": 550.0,
        "budget_revenue": 480000.0,
        "revenue_variance": -479450.0,
    }

    expected_result = SimpleNamespace(
        actual_pnl=actual_pnl,
        budget_pnl=budget_pnl,
        variance_pnl=variance_pnl,
        pnl_summary=pnl_summary,
        available_months=["2026-01", "2026-02"],
        excluded_actual_months=[],
        excluded_budget_months=[],
    )

    def fake_analyze(
        self: Any,
        operations_data: pd.DataFrame,
        budget_data: pd.DataFrame,
        corporate_expenses_data: pd.DataFrame,
        budget_corporate_expenses_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> Any:
        assert operations_data.equals(valid_operations_data)
        assert budget_data.equals(valid_budget_data)
        assert corporate_expenses_data.equals(
            state["corporate_expenses_data"]
        )
        assert budget_corporate_expenses_data.equals(
            state["budget_corporate_expenses_data"]
        )
        assert start_month == "2026-01"
        assert end_month == "2026-02"
        return expected_result

    monkeypatch.setattr(
        nodes.PnlAgent,
        "analyze",
        fake_analyze,
    )

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "cleaned_budget_data": valid_budget_data,
        "corporate_expenses_data": corporate_expenses_data,
        "budget_corporate_expenses_data": (
            budget_corporate_expenses_data
        ),
        "start_month": "2026-01",
        "end_month": "2026-02",
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "running"
    assert result["pnl_result"] is expected_result
    assert result["actual_pnl"] == actual_pnl
    assert result["budget_pnl"] == budget_pnl
    assert result["variance_pnl"] == variance_pnl
    assert result["pnl_summary"] == pnl_summary
    assert result["pnl_available_months"] == [
        "2026-01",
        "2026-02",
    ]
    assert result["pnl_excluded_actual_months"] == []
    assert result["pnl_excluded_budget_months"] == []
    assert result["error_message"] == ""
    assert result["failed_node"] == ""
    assert "pnl" in result["executed_nodes"]


def test_pnl_node_requires_cleaned_operations_data(
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """P&L execution should fail without cleaned operations data."""

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_budget_data": valid_budget_data,
        "corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "actual_amount": [50000.0],
            }
        ),
        "budget_corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "budget_amount": [48000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert "cleaned_operations_data" in result["error_message"]
    assert result["errors"]


def test_pnl_node_requires_cleaned_budget_data(
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """P&L execution should fail without cleaned budget data."""

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "actual_amount": [50000.0],
            }
        ),
        "budget_corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "budget_amount": [48000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert "cleaned_budget_data" in result["error_message"]


def test_pnl_node_requires_corporate_expenses_data(
    valid_operations_data: pd.DataFrame,
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """P&L execution should fail without actual corporate expenses."""

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "cleaned_budget_data": valid_budget_data,
        "budget_corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "budget_amount": [48000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert "corporate_expenses_data" in result["error_message"]


def test_pnl_node_requires_budget_corporate_expenses_data(
    valid_operations_data: pd.DataFrame,
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """P&L execution should fail without budget corporate expenses."""

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "cleaned_budget_data": valid_budget_data,
        "corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "actual_amount": [50000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert (
        "budget_corporate_expenses_data"
        in result["error_message"]
    )


def test_pnl_node_rejects_missing_result_output(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """P&L node should reject an incomplete P&L agent result."""

    monkeypatch.setattr(
        nodes.PnlAgent,
        "analyze",
        lambda self, **kwargs: SimpleNamespace(
            actual_pnl={},
            budget_pnl={},
            variance_pnl=None,
            pnl_summary={},
            available_months=[],
            excluded_actual_months=[],
            excluded_budget_months=[],
        ),
    )

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "cleaned_budget_data": valid_budget_data,
        "corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "actual_amount": [50000.0],
            }
        ),
        "budget_corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "budget_amount": [48000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert "variance_pnl" in result["error_message"]


def test_pnl_node_records_agent_exception(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    valid_budget_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """Unexpected P&L agent exceptions should be recorded in state."""

    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Unexpected P&L agent failure")

    monkeypatch.setattr(
        nodes.PnlAgent,
        "analyze",
        raise_error,
    )

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "pnl",
        "cleaned_operations_data": valid_operations_data,
        "cleaned_budget_data": valid_budget_data,
        "corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "actual_amount": [50000.0],
            }
        ),
        "budget_corporate_expenses_data": pd.DataFrame(
            {
                "month": ["2026-01"],
                "expense_category": ["Salary"],
                "budget_amount": [48000.0],
            }
        ),
    }

    result = nodes.pnl_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "pnl"
    assert result["error_message"] == "Unexpected P&L agent failure"
    assert any(
        "pnl: Unexpected P&L agent failure" in error
        for error in result["errors"]
    )




def test_recommendation_node_success(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """Recommendation node should store agent output."""

    root_cause_result = SimpleNamespace(root_causes=["Low volume"])
    expected_recommendation = SimpleNamespace(
        recommendations=["Improve demand generation"]
    )

    monkeypatch.setattr(
        nodes.RecommendationAgent,
        "analyze",
        lambda self, result: expected_recommendation,
    )

    state: FinanceGraphState = {
        **base_state,
        "root_cause_result": root_cause_result,
    }

    result = nodes.recommendation_node(state)

    assert result["recommendation_result"] is expected_recommendation
    assert result["execution_status"] == "running"
    assert "recommendation" in result["executed_nodes"]


# ----------------------------------------------------------------------
# KPI, commentary and report tests
# ----------------------------------------------------------------------


def test_kpi_node_success(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """The KPI node should choose default KPI fields for the KPI route."""

    expected_result = SimpleNamespace(
        selected_kpis={"total_orders": 100}
    )

    def fake_analyze(
        self: Any,
        requested_kpis: list[str],
        **kwargs: Any,
    ) -> Any:
        assert "total_orders" in requested_kpis
        assert "actual_revenue" in requested_kpis
        return expected_result

    monkeypatch.setattr(
        nodes.KPIAgent,
        "analyze",
        fake_analyze,
    )

    state: FinanceGraphState = {
        **base_state,
        "selected_flow": "kpi",
        "operations_result": SimpleNamespace(total_orders=100),
        "filters": {},
    }

    result = nodes.kpi_node(state)

    assert result["kpi_result"] is expected_result
    assert result["execution_status"] == "running"
    assert "total_orders" in result["requested_kpis"]
    assert "kpi" in result["executed_nodes"]


def test_commentary_node_requires_kpi_result(
    base_state: FinanceGraphState,
) -> None:
    """Commentary cannot run without KPI results."""

    result = nodes.commentary_node(base_state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "commentary"
    assert "kpi_result" in result["error_message"]


def test_report_node_completes_execution(
    monkeypatch: pytest.MonkeyPatch,
    base_state: FinanceGraphState,
) -> None:
    """A successful report node should complete graph execution."""

    commentary_result = SimpleNamespace(
        executive_summary="Revenue performance summary"
    )
    expected_report = SimpleNamespace(
        markdown_report="# Management Report"
    )

    monkeypatch.setattr(
        nodes.ReportAgent,
        "analyze",
        lambda self, **kwargs: expected_report,
    )

    state: FinanceGraphState = {
        **base_state,
        "commentary_result": commentary_result,
    }

    result = nodes.report_node(state)

    assert result["report_result"] is expected_report
    assert result["execution_status"] == "completed"
    assert result["error_message"] == ""
    assert "report" in result["executed_nodes"]


# ----------------------------------------------------------------------
# Error and graph-control tests
# ----------------------------------------------------------------------


def test_agent_exception_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
    valid_operations_data: pd.DataFrame,
    base_state: FinanceGraphState,
) -> None:
    """Unexpected agent exceptions should be written into state."""

    def raise_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Unexpected operations agent failure")

    monkeypatch.setattr(
        nodes.OperationsAnalysisAgent,
        "analyze",
        raise_error,
    )

    state: FinanceGraphState = {
        **base_state,
        "cleaned_operations_data": valid_operations_data,
        "filters": {},
    }

    result = nodes.operations_analysis_node(state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "operations_analysis"
    assert result["error_message"] == (
        "Unexpected operations agent failure"
    )
    assert any(
        "operations_analysis" in error
        for error in result["errors"]
    )


def test_complete_node_marks_execution_completed(
    base_state: FinanceGraphState,
) -> None:
    """The completion node should finish a successful route."""

    result = nodes.complete_node(base_state)

    assert result["execution_status"] == "completed"
    assert result["error_message"] == ""
    assert "complete" in result["executed_nodes"]


def test_complete_node_preserves_failed_state(
    base_state: FinanceGraphState,
) -> None:
    """A failed route must not be changed to completed."""

    failed_state: FinanceGraphState = {
        **base_state,
        "execution_status": "failed",
        "failed_node": "forecast",
        "error_message": "Forecast failed",
        "errors": ["forecast: Forecast failed"],
    }

    result = nodes.complete_node(failed_state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "forecast"
    assert result["error_message"] == "Forecast failed"


def test_error_node_preserves_failure_details(
    base_state: FinanceGraphState,
) -> None:
    """The error node should keep the original failure information."""

    failed_state: FinanceGraphState = {
        **base_state,
        "execution_status": "failed",
        "failed_node": "scenario",
        "error_message": "Scenario output is empty",
        "errors": ["scenario: Scenario output is empty"],
    }

    result = nodes.error_node(failed_state)

    assert result["execution_status"] == "failed"
    assert result["failed_node"] == "scenario"
    assert result["error_message"] == "Scenario output is empty"
    assert result["errors"] == [
        "scenario: Scenario output is empty"
    ]


@pytest.mark.parametrize(
    ("execution_status", "expected_route"),
    [
        ("pending", "continue"),
        ("running", "continue"),
        ("completed", "continue"),
        ("failed", "error"),
    ],
)
def test_should_continue(
    execution_status: str,
    expected_route: str,
) -> None:
    """Conditional graph routing should detect failed states."""

    state: FinanceGraphState = {
        "execution_status": execution_status,  # type: ignore[typeddict-item]
        "errors": [],
    }

    assert nodes.should_continue(state) == expected_route