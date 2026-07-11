"""Integration tests for the direct pre-LangGraph pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from main import FinancePipeline, load_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_PATH = PROJECT_ROOT / "data" / "operations" / "sample_orders.csv"
BUDGET_PATH = PROJECT_ROOT / "data" / "planning" / "sample_budget.csv"
ASSUMPTIONS_PATH = PROJECT_ROOT / "data" / "assumptions" / "business_assumptions.csv"


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