"""Forecast error, bias, MAPE, and category/month accuracy."""

from __future__ import annotations

import pandas as pd

from src.autonomous.schemas import ToolResult


def analyze_forecast_accuracy(*, actuals: list[dict] | None = None, forecasts: list[dict] | None = None, finance_context=None) -> ToolResult:
    actuals = actuals or []
    forecasts = forecasts or []
    actual = pd.DataFrame(actuals)
    forecast = pd.DataFrame(forecasts)
    keys = [name for name in ("period", "vehicle_category") if name in actual.columns and name in forecast.columns]
    if "period" not in keys or "actual_revenue" not in actual.columns or "forecast_revenue" not in forecast.columns:
        return ToolResult(call_id="forecast-accuracy", tool_name="analyze_forecast_accuracy", status="completed", payload={"available": False, "reason": "Saved forecast snapshots and matching actual_revenue by period are required.", "rows": []})
    merged = actual.merge(forecast, on=keys, how="inner")
    if merged.empty:
        return ToolResult(call_id="forecast-accuracy", tool_name="analyze_forecast_accuracy", status="completed", payload={"available": False, "reason": "No matching actual and forecast periods were found.", "rows": []})
    merged["error"] = merged["actual_revenue"] - merged["forecast_revenue"]
    merged["absolute_percentage_error"] = merged["error"].abs().div(merged["actual_revenue"].abs().replace(0, pd.NA)).mul(100)
    summary = {"mean_error_bias": round(float(merged["error"].mean()), 2), "mape": round(float(merged["absolute_percentage_error"].mean()), 2), "forecast_accuracy_percentage": round(max(0.0, 100 - float(merged["absolute_percentage_error"].mean())), 2)}
    return ToolResult(call_id="forecast-accuracy", tool_name="analyze_forecast_accuracy", status="completed", payload={"available": True, "summary": summary, "rows": merged.where(pd.notna(merged), None).to_dict(orient="records")})
