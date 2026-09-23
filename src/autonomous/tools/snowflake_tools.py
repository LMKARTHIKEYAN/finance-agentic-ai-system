"""Read-only Snowflake tools for the autonomous finance agent."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.autonomous.tools.data_tools import FinanceDataContext


def load_finance_data(
    *,
    repository: Any,
    start_date: date,
    end_date: date,
    category: str | None = None,
) -> dict[str, Any]:
    """Load the approved reporting datasets for one bounded period."""

    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("start_date and end_date must be date values.")
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date.")
    start_month = start_date.strftime("%Y-%m")
    end_month = end_date.strftime("%Y-%m")
    all_operations = repository.get_orders(start_date, end_date)
    all_budget = repository.get_budget(start_month, end_month)
    expenses = repository.get_corporate_expenses(start_month, end_month)
    budget_expenses = repository.get_budget_corporate_expenses(start_month, end_month)
    allocation_basis = None
    if category:
        operations = _filter_category(all_operations, category)
        budget = _filter_category(all_budget, category)
        actual_share = _revenue_share(all_operations, operations, "commission_amount")
        budget_share = _revenue_share(all_budget, budget, "budget_revenue")
        expenses = _scale_expenses(expenses, actual_share)
        budget_expenses = _scale_expenses(budget_expenses, budget_share)
        allocation_basis = (
            "Corporate expenses allocated by the selected category's share "
            "of actual and budget revenue."
        )
    else:
        operations, budget = all_operations, all_budget
    context = FinanceDataContext(
        operations_data=operations,
        budget_data=budget,
        corporate_expenses_data=expenses,
        budget_corporate_expenses_data=budget_expenses,
    )
    summaries = {
        field: context.summarize_dataset(field)
        for field in (
            "operations_data",
            "budget_data",
            "corporate_expenses_data",
            "budget_corporate_expenses_data",
        )
    }
    if not summaries["operations_data"].available:
        raise ValueError("No order data exists for the requested period.")
    return {
        "finance_context": context,
        "dataset_summaries": summaries,
        "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "revenue_definition": "commission_amount",
        "category": category,
        "allocation_basis": allocation_basis,
        "summary": "PostgreSQL finance data loaded for the requested period.",
    }


def _filter_category(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    if "vehicle_category" not in frame.columns:
        raise ValueError("The dataset has no vehicle_category column.")
    values = frame["vehicle_category"].astype(str).str.strip().str.casefold()
    target = category.casefold()
    mask = values.str.startswith("tata ace") if target == "tata ace" else values.eq(target)
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No data exists for vehicle category {category!r}.")
    return selected


def _revenue_share(all_rows: pd.DataFrame, selected: pd.DataFrame, column: str) -> float:
    if column not in all_rows.columns:
        raise ValueError(f"The dataset has no {column} column.")
    if column == "commission_amount" and "order_status" in all_rows.columns:
        all_rows = all_rows.loc[all_rows["order_status"].astype(str).str.casefold().eq("completed")]
        selected = selected.loc[selected["order_status"].astype(str).str.casefold().eq("completed")]
    total = float(pd.to_numeric(all_rows[column], errors="coerce").fillna(0).sum())
    part = float(pd.to_numeric(selected[column], errors="coerce").fillna(0).sum())
    if total <= 0:
        raise ValueError(f"Cannot allocate expenses because total {column} is zero.")
    return part / total


def _scale_expenses(frame: pd.DataFrame, share: float) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if str(column).lower() == "month":
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.notna().all():
            result[column] = numeric * share
    return result


def check_data_freshness(*, repository: Any, expected_through: date) -> dict[str, Any]:
    """Compare the latest available order date with the required cutoff."""

    latest = repository.latest_order_date()
    current = latest is not None and latest >= expected_through
    return {
        "latest_order_date": latest,
        "expected_through": expected_through,
        "is_current": current,
        "summary": (
            "PostgreSQL data is current."
            if current
            else "PostgreSQL data is not current for the requested cutoff."
        ),
    }
