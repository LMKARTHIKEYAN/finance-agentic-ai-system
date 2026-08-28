"""Threshold-based profitability and operating alerts."""

from __future__ import annotations

from src.autonomous.schemas import ToolResult
from src.autonomous.tools.analysis_common import aggregate, prepared_orders
from src.autonomous.tools.data_tools import FinanceDataContext


def detect_profitability_alerts(
    *, finance_context: FinanceDataContext, minimum_gp_percentage: float = 30.0,
    maximum_cancellation_percentage: float = 8.0,
) -> ToolResult:
    frame = prepared_orders(finance_context)
    dimensions = ["vehicle_category"] if "vehicle_category" in frame.columns else []
    rows = aggregate(frame, dimensions)
    alerts = []
    for row in rows.to_dict(orient="records"):
        scope = row.get("vehicle_category", "overall")
        if row["gross_profit"] < 0:
            alerts.append({"severity": "critical", "scope": scope, "metric": "gross_profit", "value": row["gross_profit"], "message": "Negative gross profit."})
        if row["gp_percentage"] < minimum_gp_percentage:
            alerts.append({"severity": "high", "scope": scope, "metric": "gp_percentage", "value": row["gp_percentage"], "message": f"GP% is below {minimum_gp_percentage:.2f}%."})
        if row["cancellation_percentage"] > maximum_cancellation_percentage:
            alerts.append({"severity": "high", "scope": scope, "metric": "cancellation_percentage", "value": row["cancellation_percentage"], "message": f"Cancellation exceeds {maximum_cancellation_percentage:.2f}%."})
    lowest_gp = rows.loc[rows["gp_percentage"].idxmin()].to_dict()
    highest_cancellation = rows.loc[rows["cancellation_percentage"].idxmax()].to_dict()
    lowest_profit = rows.loc[rows["gross_profit"].idxmin()].to_dict()
    monitoring_summary = {
        "categories_evaluated": len(rows),
        "lowest_gp_category": lowest_gp.get("vehicle_category", "overall"),
        "lowest_gp_percentage": lowest_gp["gp_percentage"],
        "gp_headroom_percentage_points": round(float(lowest_gp["gp_percentage"]) - minimum_gp_percentage, 2),
        "highest_cancellation_category": highest_cancellation.get("vehicle_category", "overall"),
        "highest_cancellation_percentage": highest_cancellation["cancellation_percentage"],
        "cancellation_headroom_percentage_points": round(maximum_cancellation_percentage - float(highest_cancellation["cancellation_percentage"]), 2),
        "lowest_gross_profit_category": lowest_profit.get("vehicle_category", "overall"),
        "lowest_gross_profit": lowest_profit["gross_profit"],
    }
    return ToolResult(call_id="profitability-alert", tool_name="detect_profitability_alerts", status="completed", payload={"alert_count": len(alerts), "alerts": alerts, "monitoring_summary": monitoring_summary, "thresholds": {"minimum_gp_percentage": minimum_gp_percentage, "maximum_cancellation_percentage": maximum_cancellation_percentage}, "revenue_definition": "commission_amount", "direct_cost_definition": "incentive + goodwill + dry_run + surge"})
