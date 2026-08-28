"""Current-versus-comparison period KPI analysis."""

from __future__ import annotations

import pandas as pd

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders
from src.autonomous.tools.data_tools import FinanceDataContext


def compare_periods(
    *, finance_context: FinanceDataContext, current_start: str, current_end: str,
    comparison_start: str, comparison_end: str, comparison_label: str = "previous period",
) -> ToolResult:
    frame = prepared_orders(finance_context)
    current = frame[frame["order_date"].between(pd.Timestamp(current_start), pd.Timestamp(current_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
    prior = frame[frame["order_date"].between(pd.Timestamp(comparison_start), pd.Timestamp(comparison_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))]
    current_row = aggregate(current, []).iloc[0].to_dict() if not current.empty else {}
    prior_row = aggregate(prior, []).iloc[0].to_dict() if not prior.empty else {}
    metrics = ("total_orders", "completed_orders", "revenue", "aov", "fulfillment_percentage", "cancellation_percentage", "direct_cost", "gross_profit", "gp_percentage")
    comparison = []
    for metric in metrics:
        actual = float(current_row.get(metric, 0) or 0)
        baseline = float(prior_row.get(metric, 0) or 0)
        comparison.append({
            "metric": metric, "current": round(actual, 2), "comparison": round(baseline, 2),
            "variance": round(actual - baseline, 2),
            "variance_percentage": round((actual - baseline) / abs(baseline) * 100, 2) if baseline else None,
        })
    return ToolResult(
        call_id="period-comparison", tool_name="compare_periods", status="completed",
        payload={"comparison_label": comparison_label, "current_period": f"{current_start} to {current_end}", "comparison_period": f"{comparison_start} to {comparison_end}", "metrics": comparison, "revenue_definition": "commission_amount"},
    )
