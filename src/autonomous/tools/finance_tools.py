"""Unified adapters over deterministic FP&A calculation tools."""

from __future__ import annotations

import pandas as pd

from src.autonomous.schemas import ToolResult
from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent
from src.agents.finance.budget_agent import BudgetAgent
from src.agents.finance.forecast_agent import ForecastAgent
from src.agents.finance.variance_agent import RevenueVarianceAgent
from src.autonomous.tools.data_tools import serialize_tool_payload
from src.autonomous.tools.data_tools import FinanceDataContext
from src.autonomous.tools.gp_decomposition_tools import calculate_validated_gp_decomposition
from src.autonomous.tools.kpi_tools import calculate_validated_kpis
from src.autonomous.tools.pnl_tools import generate_validated_pnl_analysis
from src.autonomous.tools.revenue_variance_tools import calculate_validated_revenue_variance


def calculate_pnl(*, finance_context: FinanceDataContext) -> ToolResult:
    return generate_validated_pnl_analysis(finance_context)


def calculate_revenue_variance(*, finance_context: FinanceDataContext) -> ToolResult:
    if finance_context.operations_result is None:
        operations = finance_context.require_dataframe("operations_data")
        finance_context.operations_result = OperationsAnalysisAgent().analyze(
            operations, group_by="month"
        )
    if finance_context.budget_result is None:
        budget = finance_context.require_dataframe("budget_data")
        finance_context.budget_result = BudgetAgent().analyze(budget)
    return calculate_validated_revenue_variance(finance_context)


def calculate_gp_decomposition(*, finance_context: FinanceDataContext) -> ToolResult:
    return calculate_validated_gp_decomposition(finance_context)


def calculate_kpis(
    *, finance_context: FinanceDataContext, requested_kpis: list[str]
) -> ToolResult:
    """Prepare deterministic source results before selecting requested KPIs."""

    if finance_context.operations_result is None:
        operations = finance_context.require_dataframe("operations_data")
        finance_context.operations_result = OperationsAnalysisAgent().analyze(
            operations, group_by="month"
        )
    normalized_kpis = {str(item).strip().lower() for item in requested_kpis}
    budget_kpis = {"budget orders", "budget revenue", "budget aov", "budget average order value"}
    variance_kpis = {
        "order variance", "revenue variance", "aov variance", "price effect",
        "volume effect", "new discontinued effect", "new launch effect", "variance check",
    }
    if normalized_kpis & (budget_kpis | variance_kpis):
        if finance_context.budget_result is None:
            budget = finance_context.require_dataframe("budget_data")
            finance_context.budget_result = BudgetAgent().analyze(budget)
    if normalized_kpis & variance_kpis and finance_context.revenue_variance_result is None:
        finance_context.revenue_variance_result = RevenueVarianceAgent().analyze(
            actual_result=finance_context.operations_result,
            budget_result=finance_context.budget_result,
        )
    result = calculate_validated_kpis(finance_context, requested_kpis)
    actual_by_category = {
        row["vehicle_category"]: row
        for row in finance_context.operations_result.vehicle_summary
    }
    result.payload["category_kpis"] = [
        {
            "vehicle_category": category,
            **actual_by_category[category],
        }
        for category in sorted(actual_by_category)
    ]
    return result


def calculate_daily_kpi_extreme(
    *, finance_context: FinanceDataContext, extreme: str = "lowest",
    analysis_mode: str = "orders",
) -> ToolResult:
    """Compare daily operational KPIs and identify the lowest/highest order day."""

    operations = finance_context.require_dataframe("operations_data").copy()
    operations["order_date"] = pd.to_datetime(operations["order_date"], errors="coerce")
    operations = operations.dropna(subset=["order_date"])
    if operations.empty:
        return ToolResult(
            call_id="daily-kpi-extreme",
            tool_name="calculate_daily_kpi_extreme",
            status="failed",
            payload={"error": "No valid daily order data is available."},
        )
    operations["date"] = operations["order_date"].dt.date
    status = operations.get("order_status", pd.Series("", index=operations.index)).astype(str).str.lower()
    completed = status.eq("completed")
    cancelled = status.eq("cancelled")
    revenue = pd.to_numeric(operations.get("commission_amount", 0), errors="coerce").fillna(0)
    daily = (
        operations.assign(
            completed_order=completed.astype(int),
            cancelled_order=cancelled.astype(int),
            completed_revenue=revenue.where(completed, 0.0),
        )
        .groupby("date", as_index=False)
        .agg(
            total_orders=("date", "size"),
            completed_orders=("completed_order", "sum"),
            cancelled_orders=("cancelled_order", "sum"),
            revenue=("completed_revenue", "sum"),
        )
        .sort_values("date")
    )
    daily["aov"] = daily["revenue"].div(daily["completed_orders"].replace(0, pd.NA)).fillna(0.0)
    daily["fulfillment_percentage"] = daily["completed_orders"] / daily["total_orders"] * 100
    daily["cancellation_percentage"] = daily["cancelled_orders"] / daily["total_orders"] * 100
    requested = "highest" if extreme == "highest" else "lowest"
    index = daily["total_orders"].idxmax() if requested == "highest" else daily["total_orders"].idxmin()
    selected = daily.loc[index]
    highest_revenue = daily.loc[daily["revenue"].idxmax()]
    lowest_orders = daily.loc[daily["total_orders"].idxmin()]
    highest_aov = daily.loc[daily["aov"].idxmax()]
    average_orders = float(daily["total_orders"].mean())
    selected_orders = int(selected["total_orders"])
    difference = selected_orders - average_orders
    rows = daily.copy()
    rows["date"] = rows["date"].astype(str)
    return ToolResult(
        call_id="daily-kpi-extreme",
        tool_name="calculate_daily_kpi_extreme",
        status="completed",
        payload={
            "requested_extreme": requested,
            "selected_day": str(selected["date"]),
            "selected_total_orders": selected_orders,
            "daily_average_orders": round(average_orders, 2),
            "difference_from_daily_average": round(difference, 2),
            "difference_from_daily_average_percentage": round(
                difference / average_orders * 100 if average_orders else 0.0, 2
            ),
            "selected_day_kpis": {
                "completed_orders": int(selected["completed_orders"]),
                "cancelled_orders": int(selected["cancelled_orders"]),
                "revenue": round(float(selected["revenue"]), 2),
                "aov": round(float(selected["aov"]), 2),
                "fulfillment_percentage": round(float(selected["fulfillment_percentage"]), 2),
                "cancellation_percentage": round(float(selected["cancellation_percentage"]), 2),
            },
            "analysis_mode": analysis_mode,
            "highest_revenue_day": {
                "date": str(highest_revenue["date"]),
                "revenue": round(float(highest_revenue["revenue"]), 2),
                "total_orders": int(highest_revenue["total_orders"]),
                "aov": round(float(highest_revenue["aov"]), 2),
            },
            "lowest_order_day": {
                "date": str(lowest_orders["date"]),
                "revenue": round(float(lowest_orders["revenue"]), 2),
                "total_orders": int(lowest_orders["total_orders"]),
                "aov": round(float(lowest_orders["aov"]), 2),
                "fulfillment_percentage": round(float(lowest_orders["fulfillment_percentage"]), 2),
                "cancellation_percentage": round(float(lowest_orders["cancellation_percentage"]), 2),
            },
            "highest_aov_day": {
                "date": str(highest_aov["date"]),
                "revenue": round(float(highest_aov["revenue"]), 2),
                "total_orders": int(highest_aov["total_orders"]),
                "aov": round(float(highest_aov["aov"]), 2),
            },
            "daily_kpis": rows.to_dict(orient="records"),
            "revenue_definition": "commission_amount",
        },
    )


def analyze_daily_order_drivers(
    *, finance_context: FinanceDataContext, target_date: str
) -> ToolResult:
    """Explain observable operational drivers for one low-order date."""

    operations = finance_context.require_dataframe("operations_data").copy()
    operations["order_date"] = pd.to_datetime(operations["order_date"], errors="coerce")
    operations = operations.dropna(subset=["order_date"])
    target = pd.Timestamp(target_date).normalize()
    operations["date"] = operations["order_date"].dt.normalize()
    status = operations.get("order_status", pd.Series("", index=operations.index)).astype(str).str.lower()
    operations["completed_order"] = status.eq("completed").astype(int)
    operations["cancelled_order"] = status.eq("cancelled").astype(int)
    operations["completed_revenue"] = pd.to_numeric(
        operations.get("commission_amount", 0), errors="coerce"
    ).fillna(0).where(status.eq("completed"), 0.0)
    daily = operations.groupby("date", as_index=False).agg(
        total_orders=("date", "size"),
        completed_orders=("completed_order", "sum"),
        cancelled_orders=("cancelled_order", "sum"),
        revenue=("completed_revenue", "sum"),
    )
    match = daily[daily["date"] == target]
    if match.empty:
        return ToolResult(
            call_id="daily-order-root-cause",
            tool_name="analyze_daily_order_drivers",
            status="failed",
            payload={"error": f"No orders are available for {target_date}."},
        )
    row = match.iloc[0]
    daily["weekday"] = daily["date"].dt.day_name()
    peers = daily[(daily["weekday"] == target.day_name()) & (daily["date"] != target)]
    prior = daily[daily["date"] < target].sort_values("date").tail(1)
    monthly_average = float(daily["total_orders"].mean())
    weekday_average = float(peers["total_orders"].mean()) if not peers.empty else monthly_average
    prior_orders = int(prior.iloc[0]["total_orders"]) if not prior.empty else None
    total = int(row["total_orders"])
    completed = int(row["completed_orders"])
    cancelled = int(row["cancelled_orders"])
    revenue = float(row["revenue"])
    cancellation_pct = cancelled / total * 100 if total else 0.0
    fulfillment_pct = completed / total * 100 if total else 0.0
    signals = []
    if total < weekday_average:
        signals.append(
            f"Order volume was {(weekday_average - total) / weekday_average * 100:.2f}% below the other "
            f"{target.day_name()} average."
        )
    monthly_cancel_pct = float(daily["cancelled_orders"].sum() / daily["total_orders"].sum() * 100)
    if cancellation_pct > monthly_cancel_pct:
        signals.append(
            f"Cancellation was {cancellation_pct:.2f}%, above the monthly rate of {monthly_cancel_pct:.2f}%."
        )
    else:
        signals.append(
            f"Cancellation was {cancellation_pct:.2f}%, not above the monthly rate of {monthly_cancel_pct:.2f}%."
        )
    target_rows = operations[operations["date"] == target]
    cluster_column = "pickup_cluster" if "pickup_cluster" in target_rows.columns else None
    top_clusters = []
    if cluster_column:
        top_clusters = [
            {"pickup_cluster": str(name), "orders": int(count)}
            for name, count in target_rows[cluster_column].fillna("Unknown").value_counts().head(3).items()
        ]
    return ToolResult(
        call_id="daily-order-root-cause",
        tool_name="analyze_daily_order_drivers",
        status="completed",
        payload={
            "target_date": target_date,
            "weekday": target.day_name(),
            "total_orders": total,
            "completed_orders": completed,
            "cancelled_orders": cancelled,
            "revenue": round(revenue, 2),
            "aov": round(revenue / completed if completed else 0.0, 2),
            "fulfillment_percentage": round(fulfillment_pct, 2),
            "cancellation_percentage": round(cancellation_pct, 2),
            "monthly_daily_average_orders": round(monthly_average, 2),
            "same_weekday_average_orders": round(weekday_average, 2),
            "previous_day_orders": prior_orders,
            "monthly_cancellation_percentage": round(monthly_cancel_pct, 2),
            "operational_signals": signals,
            "top_pickup_clusters": top_clusters,
            "cause_status": "No approved management document proves a causal explanation; these are structured-data indicators.",
            "revenue_definition": "commission_amount",
        },
    )
def calculate_rolling_forecast(
    *, finance_context: FinanceDataContext, rolling_window: int = 3, forecast_periods: int = 3
) -> ToolResult:
    operations = finance_context.require_dataframe("operations_data")
    historical = OperationsAnalysisAgent().analyze(operations, group_by="month")
    result = ForecastAgent().analyze(
        historical.period_summary,
        frequency="month",
        rolling_window=rolling_window,
        forecast_periods=forecast_periods,
    )
    finance_context.forecast_result = result
    category_forecasts = []
    for category in sorted(operations["vehicle_category"].dropna().astype(str).str.strip().unique()):
        category_history = OperationsAnalysisAgent().analyze(
            operations,
            vehicle_category=category,
            group_by="month",
        )
        if len(category_history.period_summary) < rolling_window:
            continue
        category_result = ForecastAgent().analyze(
            category_history.period_summary,
            frequency="month",
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
        )
        for row in category_result.forecast_summary:
            category_forecasts.append({"vehicle_category": category, **row})
    return ToolResult(
        call_id="finance-forecast",
        tool_name="calculate_rolling_forecast",
        status="completed",
        payload={
            **serialize_tool_payload(result),
            "historical_summary": serialize_tool_payload(historical.period_summary),
            "category_forecast_summary": category_forecasts,
            "revenue_definition": "commission_amount",
        },
    )


FINANCE_TOOL_NAMES = (
    "calculate_pnl",
    "calculate_revenue_variance",
    "calculate_gp_decomposition",
    "calculate_kpis",
    "calculate_daily_kpi_extreme",
    "analyze_daily_order_drivers",
    "calculate_rolling_forecast",
)
