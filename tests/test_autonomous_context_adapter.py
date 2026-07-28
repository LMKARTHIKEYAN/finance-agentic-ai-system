"""Tests for deterministic-to-autonomous internal context adaptation."""

import pandas as pd

from src.autonomous.context_adapter import (
    build_autonomous_execution_context,
)


def test_adapter_preserves_dataframes_and_calculated_results() -> None:
    operations = pd.DataFrame({"commission_amount": [100.0]})
    budget = pd.DataFrame({"budget_revenue": [90.0]})
    operations_result = {"total_revenue": 100.0}
    state = {
        "operations_data": operations,
        "budget_data": budget,
        "operations_result": operations_result,
        "budget_result": {"total_budget_revenue": 90.0},
        "variance_result": {"revenue_variance": 10.0},
        "anomaly_result": {"anomalies": []},
        "filters": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-30",
            "category": "A",
        },
    }

    context = build_autonomous_execution_context(
        graph_state=state,
        deterministic_answer="Existing reviewed draft.",
    )

    assert context.finance_context.operations_data is operations
    assert context.finance_context.budget_data is budget
    assert context.operations_result is operations_result
    assert context.reporting_scope.category == "A"
    assert "operations_data" in context.available_inputs
    assert context.datasets[0].row_count == 1


def test_adapter_dataset_metadata_contains_no_rows() -> None:
    context = build_autonomous_execution_context(
        graph_state={
            "operations_data": pd.DataFrame(
                {"commission_amount": [100.0, 200.0]}
            )
        },
        deterministic_answer="Draft.",
    )

    serialized = context.datasets[0].model_dump()
    assert serialized["row_count"] == 2
    assert "commission_amount" not in str(serialized)
    assert 100.0 not in serialized.values()


def test_adapter_ignores_invalid_optional_dates() -> None:
    context = build_autonomous_execution_context(
        graph_state={"filters": {"start_date": "April"}},
        deterministic_answer="Draft.",
    )

    assert context.reporting_scope.start_date is None


def test_adapter_preserves_comparison_period_scope() -> None:
    context = build_autonomous_execution_context(
        graph_state={
            "filters": {
                "start_date": "2026-04-01",
                "end_date": "2026-04-30",
                "comparison_start_date": "2026-03-01",
                "comparison_end_date": "2026-03-31",
            }
        },
        deterministic_answer="Draft.",
    )

    assert str(context.reporting_scope.comparison_start_date) == (
        "2026-03-01"
    )
    assert str(context.reporting_scope.comparison_end_date) == "2026-03-31"
