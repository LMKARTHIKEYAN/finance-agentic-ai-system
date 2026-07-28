"""Deterministic root-cause and recommendation tool wrappers."""

from __future__ import annotations

from typing import Any

from src.agents.analytics.recommendation_agent import RecommendationAgent
from src.agents.analytics.root_cause_agent import RootCauseAgent
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import serialize_tool_payload


def identify_supported_root_causes(
    *,
    anomaly_result: Any | None = None,
    operations_result: Any | None = None,
    revenue_variance_result: Any | None = None,
    pnl_result: dict[str, Any] | None = None,
    agent: Any | None = None,
) -> ToolResult:
    """Run the existing deterministic root-cause analysis."""

    if pnl_result is not None:
        return ToolResult(
            call_id="identify_supported_root_causes",
            tool_name="identify_supported_root_causes",
            status="completed",
            payload=_identify_pnl_change_drivers(pnl_result),
        )
    if anomaly_result is None or operations_result is None:
        raise ValueError(
            "Operational diagnostics require anomaly_result and "
            "operations_result."
        )
    root_cause_agent = agent if agent is not None else RootCauseAgent()
    result = root_cause_agent.analyze(
        anomaly_result=anomaly_result,
        operations_result=operations_result,
        variance_result=revenue_variance_result,
    )
    return ToolResult(
        call_id="identify_supported_root_causes",
        tool_name="identify_supported_root_causes",
        status="completed",
        payload=serialize_tool_payload(result),
    )


def generate_supported_recommendations(
    *,
    root_cause_result: Any,
    agent: Any | None = None,
) -> ToolResult:
    """Run the existing deterministic recommendation analysis."""

    if (
        isinstance(root_cause_result, dict)
        and root_cause_result.get("analysis_type")
        == "pnl_period_change"
    ):
        return ToolResult(
            call_id="generate_supported_recommendations",
            tool_name="generate_supported_recommendations",
            status="completed",
            payload=_pnl_driver_recommendations(root_cause_result),
        )
    recommendation_agent = (
        agent if agent is not None else RecommendationAgent()
    )
    result = recommendation_agent.analyze(root_cause_result)
    return ToolResult(
        call_id="generate_supported_recommendations",
        tool_name="generate_supported_recommendations",
        status="completed",
        payload=serialize_tool_payload(result),
    )


def _identify_pnl_change_drivers(
    pnl_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare trusted monthly P&L outputs without changing formulas."""

    rows = pnl_result.get("actual_pnl")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(
            "P&L period-change diagnostics require at least two months."
        )
    ordered = sorted(rows, key=lambda item: str(item.get("month", "")))
    base = ordered[0]
    current = ordered[-1]
    base_month = str(base.get("month", ""))
    current_month = str(current.get("month", ""))
    if not base_month or not current_month:
        raise ValueError("P&L evidence requires valid month values.")

    net_profit_change = _numeric(current, "net_profit") - _numeric(
        base,
        "net_profit",
    )
    driver_specs = (
        ("revenue", 1.0),
        ("direct_cost", -1.0),
        ("sales_marketing", -1.0),
        ("other_opex", -1.0),
        ("depreciation", -1.0),
        ("interest", -1.0),
        ("income_tax", -1.0),
    )
    drivers = []
    for metric, direction in driver_specs:
        change = _numeric(current, metric) - _numeric(base, metric)
        contribution = change * direction
        if change == 0:
            continue
        drivers.append(
            {
                "metric": metric,
                "base_value": round(_numeric(base, metric), 2),
                "current_value": round(_numeric(current, metric), 2),
                "change": round(change, 2),
                "profit_effect": round(contribution, 2),
                "impact": (
                    "favorable" if contribution > 0 else "unfavorable"
                ),
            }
        )
    drivers.sort(
        key=lambda item: abs(float(item["profit_effect"])),
        reverse=True,
    )
    return {
        "analysis_type": "pnl_period_change",
        "base_month": base_month,
        "current_month": current_month,
        "base_net_profit": round(_numeric(base, "net_profit"), 2),
        "current_net_profit": round(_numeric(current, "net_profit"), 2),
        "net_profit_change": round(net_profit_change, 2),
        "drivers": drivers,
    }


def _pnl_driver_recommendations(
    root_cause_result: dict[str, Any],
) -> dict[str, Any]:
    """Create controlled actions linked to deterministic P&L drivers."""

    recommendations = []
    for driver in root_cause_result.get("drivers", [])[:5]:
        metric = str(driver.get("metric", "unknown"))
        favorable = driver.get("impact") == "favorable"
        recommendations.append(
            {
                "metric": metric,
                "priority": "medium" if favorable else "high",
                "action": (
                    f"Monitor and sustain the favorable {metric} movement."
                    if favorable
                    else f"Investigate and address the adverse {metric} movement."
                ),
                "requires_human_approval": False,
            }
        )
    return {
        "analysis_type": "pnl_period_change_recommendations",
        "recommendations": recommendations,
    }


def _numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"P&L evidence field {key!r} must be numeric.")
    return float(value)
