"""Integration tests for the direct pre-LangGraph pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from main import (
    FinancePipeline,
    build_graph_state,
    build_parser,
    load_csv,
    print_graph_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_PATH = PROJECT_ROOT / "data" / "operations" / "sample_orders.csv"
BUDGET_PATH = PROJECT_ROOT / "data" / "planning" / "sample_budget.csv"
ASSUMPTIONS_PATH = PROJECT_ROOT / "data" / "assumptions" / "business_assumptions.csv"
CORPORATE_EXPENSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "operations"
    / "sample_corporate_expenses.csv"
)
BUDGET_CORPORATE_EXPENSES_PATH = (
    PROJECT_ROOT
    / "data"
    / "planning"
    / "sample_budget_corporate_expenses.csv"
)


@pytest.fixture(scope="module")
def operations_data() -> pd.DataFrame:
    return pd.read_csv(OPERATIONS_PATH)


@pytest.fixture(scope="module")
def budget_data() -> pd.DataFrame:
    return pd.read_csv(BUDGET_PATH)


@pytest.fixture(scope="module")
def assumptions_data() -> pd.DataFrame:
    return pd.read_csv(ASSUMPTIONS_PATH)


def test_kpi_only_flow(operations_data: pd.DataFrame) -> None:
    result = FinancePipeline().run_kpi_flow(operations_data)

    assert result.flow == "kpi"
    assert result.operations_validation is not None
    assert result.operations_validation.is_valid is True
    assert result.operations_result is not None
    assert result.kpi_result is not None
    assert result.kpi_result.selected_kpis
    assert result.commentary_result is not None
    assert result.report_result is None


def test_budget_only_flow(budget_data: pd.DataFrame) -> None:
    result = FinancePipeline().run_budget_flow(budget_data)

    assert result.flow == "budget"
    assert result.budget_validation is not None
    assert result.budget_validation.is_valid is True
    assert result.budget_result.total_budget_orders > 0
    assert result.budget_result.total_budget_revenue > 0
    assert result.operations_result is None


def test_forecast_only_flow(operations_data: pd.DataFrame) -> None:
    result = FinancePipeline().run_forecast_flow(
        operations_data,
        rolling_window=3,
        forecast_periods=2,
    )

    assert result.flow == "forecast"
    assert len(result.forecast_result.forecast_summary) == 2
    assert result.kpi_result.selected_kpis
    assert result.commentary_result is not None


def test_full_analysis_flow(
    operations_data: pd.DataFrame,
    budget_data: pd.DataFrame,
    assumptions_data: pd.DataFrame,
) -> None:
    result = FinancePipeline().run_full_analysis(
        operations_data=operations_data,
        budget_data=budget_data,
        assumptions=assumptions_data,
        rolling_window=3,
        forecast_periods=6,
    )

    assert result.flow == "full"
    assert result.operations_profile is not None
    assert result.budget_profile is not None
    assert result.operations_result is not None
    assert result.budget_result is not None
    assert result.forecast_result is not None
    assert result.scenario_result is not None
    assert result.variance_result is not None
    assert result.finance_rules_result is not None
    assert result.anomaly_result is not None
    assert result.root_cause_result is not None
    assert result.recommendation_result is not None
    assert result.kpi_result is not None
    assert result.commentary_result is not None
    assert result.report_result is not None
    assert result.report_result.markdown_report
    assert result.report_result.source_availability["recommendation"] is True


def test_invalid_operations_data_stops_pipeline() -> None:
    invalid_data = pd.DataFrame({"order_id": ["ORD-1"]})

    with pytest.raises(ValueError, match="Operations data validation failed"):
        FinancePipeline().run_kpi_flow(invalid_data)


def test_load_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        load_csv(PROJECT_ROOT / "data" / "missing.csv")


def test_build_graph_state_loads_all_pnl_datasets() -> None:
    """CLI graph state should supply every raw dataset required by P&L."""

    args = build_parser().parse_args(
        [
            "--mode",
            "graph",
            "--request",
            "Generate P&L for April 2026",
            "--operations",
            str(OPERATIONS_PATH),
            "--budget",
            str(BUDGET_PATH),
            "--corporate-expenses",
            str(CORPORATE_EXPENSES_PATH),
            "--budget-corporate-expenses",
            str(BUDGET_CORPORATE_EXPENSES_PATH),
        ]
    )

    state = build_graph_state(args)

    assert state["selected_flow"] == "pnl"
    assert state["start_date"] == "2026-04-01"
    assert state["end_date"] == "2026-04-30"
    assert state["start_month"] == "2026-04"
    assert state["end_month"] == "2026-04"
    assert isinstance(state["operations_data"], pd.DataFrame)
    assert isinstance(state["budget_data"], pd.DataFrame)
    assert isinstance(state["corporate_expenses_data"], pd.DataFrame)
    assert isinstance(
        state["budget_corporate_expenses_data"],
        pd.DataFrame,
    )
    assert not state["operations_data"].empty
    assert not state["budget_data"].empty
    assert not state["corporate_expenses_data"].empty
    assert not state["budget_corporate_expenses_data"].empty


def test_print_graph_result_prints_pnl_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Successful P&L CLI runs should expose their calculated result."""

    print_graph_result(
        {
            "selected_flow": "pnl",
            "execution_status": "completed",
            "executed_nodes": ["pnl", "complete"],
            "pnl_result": {"reporting_period": "April 2026"},
        }
    )

    output = capsys.readouterr().out

    assert "Selected flow: pnl" in output
    assert "Execution status: completed" in output
    assert "'reporting_period': 'April 2026'" in output
