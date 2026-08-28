"""Vehicle-category profitability and unit economics."""

from __future__ import annotations

import pandas as pd

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, numeric, prepared_orders, safe_records
from src.autonomous.tools.data_tools import FinanceDataContext


def analyze_category_profitability(*, finance_context: FinanceDataContext) -> ToolResult:
    frame = prepared_orders(finance_context)
    if "vehicle_category" not in frame.columns:
        raise ValueError("vehicle_category is required for profitability analysis.")
    categories = aggregate(frame, ["vehicle_category"])
    expenses = finance_context.corporate_expenses_data
    corporate_total = 0.0
    if isinstance(expenses, pd.DataFrame) and not expenses.empty:
        excluded = {"month", "date", "period", "vehicle_category"}
        for column in expenses.columns:
            if str(column).lower() not in excluded:
                corporate_total += float(numeric(expenses, column).sum())
    total_revenue = float(categories["revenue"].sum())
    categories["corporate_expense_allocation"] = categories["revenue"].apply(
        lambda value: round(corporate_total * float(value) / total_revenue, 2) if total_revenue else 0.0
    )
    categories["net_profit"] = (categories["gross_profit"] - categories["corporate_expense_allocation"]).round(2)
    categories["profit_per_completed_order"] = categories["net_profit"].div(categories["completed_orders"].replace(0, pd.NA)).fillna(0).round(2)
    categories = categories.sort_values("net_profit", ascending=False)
    return ToolResult(call_id="category-profitability", tool_name="analyze_category_profitability", status="completed", payload={"rows": safe_records(categories), "corporate_expense_allocation_method": "commission revenue share", "direct_cost_definition": "incentive + goodwill + dry_run + surge", "revenue_definition": "commission_amount"})
