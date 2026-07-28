"""Adapt trusted deterministic graph state for autonomous execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import pandas as pd

from src.autonomous.schemas import DatasetAvailability, ReportingScope
from src.autonomous.tools.data_tools import FinanceDataContext


@dataclass(frozen=True)
class AutonomousExecutionContext:
    """Internal-only inputs for one autonomous workflow invocation."""

    finance_context: FinanceDataContext
    reporting_scope: ReportingScope
    datasets: tuple[DatasetAvailability, ...]
    available_inputs: frozenset[str]
    anomaly_result: Any
    operations_result: Any
    revenue_variance_result: Any
    draft_answer: str


def build_autonomous_execution_context(
    *,
    graph_state: Mapping[str, Any],
    deterministic_answer: str,
) -> AutonomousExecutionContext:
    """Reuse deterministic inputs/results without recalculating values."""

    if not isinstance(graph_state, Mapping):
        raise TypeError("graph_state must be a mapping.")
    if not isinstance(deterministic_answer, str):
        raise TypeError("deterministic_answer must be a string.")

    finance_context = FinanceDataContext(
        operations_data=_dataframe_or_none(
            graph_state.get("operations_data")
        ),
        budget_data=_dataframe_or_none(graph_state.get("budget_data")),
        corporate_expenses_data=_dataframe_or_none(
            graph_state.get("corporate_expenses_data")
        ),
        budget_corporate_expenses_data=_dataframe_or_none(
            graph_state.get("budget_corporate_expenses_data")
        ),
        business_assumptions=graph_state.get("business_assumptions"),
        operations_result=graph_state.get("operations_result"),
        budget_result=graph_state.get("budget_result"),
        revenue_variance_result=graph_state.get("variance_result"),
        gp_portfolio_result=graph_state.get("gp_variance_result"),
        forecast_result=graph_state.get("forecast_result"),
        scenario_result=graph_state.get("scenario_result"),
        finance_rules_result=graph_state.get("finance_rules_result"),
    )
    dataset_fields = (
        "operations_data",
        "budget_data",
        "corporate_expenses_data",
        "budget_corporate_expenses_data",
    )
    datasets = tuple(
        _dataset_availability(name, getattr(finance_context, name))
        for name in dataset_fields
    )
    available = {
        name
        for name in dataset_fields
        if isinstance(getattr(finance_context, name), pd.DataFrame)
        and not getattr(finance_context, name).empty
    }
    result_fields = {
        "operations_result": finance_context.operations_result,
        "budget_result": finance_context.budget_result,
        "revenue_variance_result": (
            finance_context.revenue_variance_result
        ),
        "gp_portfolio_result": finance_context.gp_portfolio_result,
        "forecast_result": finance_context.forecast_result,
        "scenario_result": finance_context.scenario_result,
        "finance_rules_result": finance_context.finance_rules_result,
        "anomaly_result": (
            graph_state.get("anomaly_result")
            if graph_state.get("anomaly_result") is not None
            else {}
        ),
        "root_cause_result": graph_state.get("root_cause_result"),
    }
    available.update(
        name for name, value in result_fields.items() if value is not None
    )
    filters = graph_state.get("filters")
    scope_values = filters if isinstance(filters, Mapping) else {}
    return AutonomousExecutionContext(
        finance_context=finance_context,
        reporting_scope=ReportingScope(
            start_date=_optional_date(scope_values.get("start_date")),
            end_date=_optional_date(scope_values.get("end_date")),
            comparison_start_date=_optional_date(
                scope_values.get("comparison_start_date")
            ),
            comparison_end_date=_optional_date(
                scope_values.get("comparison_end_date")
            ),
            category=_optional_text(scope_values.get("category")),
        ),
        datasets=datasets,
        available_inputs=frozenset(available),
        anomaly_result=(
            graph_state.get("anomaly_result")
            if graph_state.get("anomaly_result") is not None
            else {}
        ),
        operations_result=graph_state.get("operations_result"),
        revenue_variance_result=graph_state.get("variance_result"),
        draft_answer=deterministic_answer.strip(),
    )


def _dataframe_or_none(value: Any) -> pd.DataFrame | None:
    return value if isinstance(value, pd.DataFrame) else None


def _dataset_availability(
    name: str,
    value: pd.DataFrame | None,
) -> DatasetAvailability:
    return DatasetAvailability(
        dataset_name=name,
        available=isinstance(value, pd.DataFrame) and not value.empty,
        row_count=len(value) if isinstance(value, pd.DataFrame) else None,
    )


def _optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
