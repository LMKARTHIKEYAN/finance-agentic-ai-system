"""Application service for a bounded autonomous FP&A execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re
from typing import Any

from src.autonomous.citations import append_evidence_citations
from src.autonomous.evidence_registry import EvidenceRegistry
from src.autonomous.goal_builder import GoalBuilder
from src.autonomous.observability import ObservabilityRecorder
from src.autonomous.reconciliation import AutonomousReconciler
from src.autonomous.schemas import FinanceGoal, ToolResult
from src.autonomous.tools.finance_tools import (
    calculate_gp_decomposition,
    calculate_kpis,
    calculate_daily_kpi_extreme,
    analyze_daily_order_drivers,
    calculate_pnl,
    calculate_revenue_variance,
    calculate_rolling_forecast,
)
from src.autonomous.tools.snowflake_tools import load_finance_data
from src.autonomous.tools.retrieval_tools import retrieve_company_context
from src.autonomous.tools.trend_analysis_tool import analyze_trends
from src.autonomous.tools.period_comparison_tool import compare_periods
from src.autonomous.tools.drilldown_tool import analyze_drilldown
from src.autonomous.tools.root_cause_tool import identify_operational_drivers
from src.autonomous.tools.anomaly_detection_tool import detect_anomalies
from src.autonomous.tools.category_profitability_tool import analyze_category_profitability
from src.autonomous.tools.customer_route_tool import analyze_customers_and_routes
from src.autonomous.tools.forecast_accuracy_tool import analyze_forecast_accuracy
from src.autonomous.tools.scenario_analysis_tool import analyze_scenario
from src.autonomous.tools.driver_forecast_tool import forecast_from_drivers
from src.autonomous.tools.profitability_alert_tool import detect_profitability_alerts


@dataclass(frozen=True)
class AutonomousFinanceResponse:
    status: str
    answer: str | None
    question: str | None
    goal_id: str
    evidence: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    management_commentary: dict[str, Any] | None = None


class AutonomousFinanceService:
    """Dynamically select tools from the goal and observed evidence."""

    def __init__(
        self,
        *,
        repository: Any,
        goal_builder: GoalBuilder | None = None,
        retriever: Any | None = None,
    ) -> None:
        self.repository = repository
        self.goal_builder = goal_builder or GoalBuilder()
        self.retriever = retriever
        self.observability = ObservabilityRecorder()

    def ask(self, question: str) -> AutonomousFinanceResponse:
        return self.execute_goal(self.goal_builder.build(question))

    def execute_goal(self, goal: FinanceGoal) -> AutonomousFinanceResponse:
        # Metrics are scoped to one goal instead of accumulating across chats.
        self.observability = ObservabilityRecorder()
        self.observability.record("goal_started", goal.goal_id)
        if goal.ambiguities:
            self.observability.record("clarification_requested", goal.goal_id)
            return AutonomousFinanceResponse(
                "waiting_for_user", None, goal.clarification_question,
                goal.goal_id, (), self.observability.metrics(),
            )
        scope = goal.reporting_scope
        if scope.start_date is None or scope.end_date is None:
            return AutonomousFinanceResponse(
                "waiting_for_user", None,
                "Please provide the reporting period.", goal.goal_id, (),
                self.observability.metrics(),
            )

        loaded = load_finance_data(
            repository=self.repository,
            start_date=scope.start_date,
            end_date=scope.end_date,
            category=scope.category,
        )
        context = loaded["finance_context"]
        forecast_context = None
        requested = {item.key for item in goal.criteria}
        if "forecast" in requested:
            forecast_history_start = scope.start_date - timedelta(days=190)
            forecast_context = load_finance_data(
                repository=self.repository,
                start_date=forecast_history_start,
                end_date=scope.end_date,
                category=scope.category,
            )["finance_context"]
        registry = EvidenceRegistry()
        selected: list[tuple[str, Any, str, dict[str, Any]]] = []
        if "management_action" in requested:
            return AutonomousFinanceResponse(
                "waiting_for_user", None,
                "Please provide the issue, recommended action, owner, and due date. The action will remain a draft until approved.",
                goal.goal_id, (), self.observability.metrics(),
            )
        if "daily_order_root_cause" in requested:
            month_start = scope.start_date.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            context = load_finance_data(
                repository=self.repository,
                start_date=month_start,
                end_date=month_end,
                category=scope.category,
            )["finance_context"]
            selected.append(
                (
                    "daily order drivers",
                    analyze_daily_order_drivers,
                    "daily_order_root_cause",
                    {"target_date": scope.start_date.isoformat()},
                )
            )
        if "daily_kpi_extreme" in requested:
            normalized_request = goal.original_request.lower()
            combined_revenue_orders = (
                "revenue" in normalized_request
                and "order" in normalized_request
                and any(word in normalized_request for word in ("high", "highest", "maximum"))
                and any(word in normalized_request for word in ("low", "lowest", "minimum", "least"))
            )
            extreme = (
                "highest"
                if any(word in normalized_request for word in ("highest", "high", "maximum", "most"))
                else "lowest"
            )
            selected.append(
                (
                    "daily KPI extreme",
                    calculate_daily_kpi_extreme,
                    "daily_kpi_extreme",
                    {
                        "extreme": extreme,
                        "analysis_mode": "high_revenue_low_orders" if combined_revenue_orders else "orders",
                    },
                )
            )
        if "kpi_analysis" in requested:
            selected.append(
                (
                    "KPI",
                    calculate_kpis,
                    "kpi",
                    {
                        "requested_kpis": [
                            "total orders", "completed orders", "cancelled orders",
                            "actual revenue", "actual aov", "fulfillment",
                            "cancellation",
                        ]
                    },
                )
            )
        if "trend_analysis" in requested:
            normalized = goal.original_request.lower()
            frequency = "weekly" if "week" in normalized else "monthly" if "month" in normalized and "daily" not in normalized else "daily"
            selected.append(("trend analysis", analyze_trends, "trend", {"frequency": frequency}))
        if "period_comparison" in requested:
            normalized = goal.original_request.lower()
            if "year-over-year" in normalized or "yoy" in normalized:
                comparison_start = scope.start_date.replace(year=scope.start_date.year - 1)
                comparison_end = scope.end_date.replace(year=scope.end_date.year - 1)
            elif scope.start_date.day == 1 and (scope.end_date + timedelta(days=1)).day == 1:
                comparison_end = scope.start_date - timedelta(days=1)
                comparison_start = comparison_end.replace(day=1)
            else:
                day_count = (scope.end_date - scope.start_date).days + 1
                comparison_end = scope.start_date - timedelta(days=1)
                comparison_start = comparison_end - timedelta(days=day_count - 1)
            context = load_finance_data(
                repository=self.repository, start_date=comparison_start,
                end_date=scope.end_date, category=scope.category,
            )["finance_context"]
            selected.append(("period comparison", compare_periods, "period_comparison", {
                "current_start": scope.start_date.isoformat(), "current_end": scope.end_date.isoformat(),
                "comparison_start": comparison_start.isoformat(), "comparison_end": comparison_end.isoformat(),
            }))
        if "drilldown_analysis" in requested:
            normalized = goal.original_request.lower()
            dimension = "pickup_cluster" if "pickup" in normalized else "drop_cluster" if "drop" in normalized else "order_status" if "status" in normalized else "weekday" if "weekday" in normalized or "day of week" in normalized else "vehicle_category"
            selected.append(("drill-down", analyze_drilldown, "drilldown", {"dimension": dimension}))
        if "anomaly_detection" in requested:
            selected.append(("anomaly detection", detect_anomalies, "anomaly", {}))
        if "category_profitability" in requested:
            selected.append(("category profitability", analyze_category_profitability, "category_profitability", {}))
        if "customer_route_analysis" in requested:
            normalized = goal.original_request.lower()
            analysis_scope = "both" if "customer" in normalized and "route" in normalized else "customer" if "customer" in normalized else "route"
            selected.append(("customer and route analysis", analyze_customers_and_routes, "customer_route", {"analysis_scope": analysis_scope}))
        if "forecast_accuracy" in requested:
            selected.append(("forecast accuracy", analyze_forecast_accuracy, "forecast_accuracy", {"actuals": [], "forecasts": []}))
        if "scenario_analysis" in requested:
            normalized = goal.original_request.lower()
            order_match = re.search(r"orders?\s+(?:fall|falls|decline|increase|rise|rises)\s+(?:by\s+)?(-?\d+(?:\.\d+)?)%", normalized)
            order_change = float(order_match.group(1)) if order_match else 0.0
            if order_match and any(word in order_match.group(0) for word in ("fall", "decline")):
                order_change = -abs(order_change)
            aov_match = re.search(r"aov\s+(?:increase|rise|decrease|fall)\s+(?:by\s+)?(?:₹|inr)?\s*(-?\d+(?:\.\d+)?)", normalized)
            aov_change = float(aov_match.group(1)) if aov_match else 0.0
            if aov_match and any(word in aov_match.group(0) for word in ("decrease", "fall")):
                aov_change = -abs(aov_change)
            target_match = re.search(r"(?:gp|gross margin)[^\d]*(\d+(?:\.\d+)?)%", normalized)
            selected.append(("scenario analysis", analyze_scenario, "scenario", {"order_change_percentage": order_change, "aov_change_amount": aov_change, "target_gp_percentage": float(target_match.group(1)) if target_match else None}))
        if "driver_forecast" in requested:
            history_start = scope.start_date - timedelta(days=370)
            context = load_finance_data(repository=self.repository, start_date=history_start, end_date=scope.end_date, category=scope.category)["finance_context"]
            selected.append(("driver-based forecast", forecast_from_drivers, "driver_forecast", {}))
        if "profitability_alerts" in requested:
            selected.append(("profitability alerts", detect_profitability_alerts, "profitability_alert", {}))
        if "root_cause" in requested and "daily_order_root_cause" not in requested:
            selected.append(("operational drivers", identify_operational_drivers, "operational_drivers", {"metric": "revenue" if "revenue" in goal.original_request.lower() else "orders"}))
        if requested & {"pnl_analysis", "pnl_reconciliation"}:
            selected.append(("pnl", calculate_pnl, "pnl", {}))
        if requested & {"revenue_variance", "revenue_reconciliation"}:
            selected.append(("revenue variance", calculate_revenue_variance, "revenue_variance", {}))
        if requested & {"gp_decomposition", "gp_reconciliation"}:
            selected.append(("GP decomposition", calculate_gp_decomposition, "gp_decomposition", {}))
        if "forecast" in requested:
            selected.append(("rolling forecast", calculate_rolling_forecast, "forecast", {}))
        if not selected:
            selected.append(("P&L", calculate_pnl, "pnl", {}))

        summaries: list[str] = []
        period = f"{scope.start_date.isoformat()} to {scope.end_date.isoformat()}"
        for label, tool, result_type, tool_arguments in selected:
            self.observability.record("tool_called", goal.goal_id, tool=tool.__name__)
            active_context = forecast_context if result_type == "forecast" else context
            result: ToolResult = tool(finance_context=active_context, **tool_arguments)
            if result.status != "completed":
                return AutonomousFinanceResponse(
                    "failed", None, None, goal.goal_id, (), self.observability.metrics()
                )
            record = registry.register(
                result, result_type=result_type, period=period, category=scope.category
            )
            summaries.append(
                _finance_summary(label, result.payload, record.evidence_id, category=scope.category)
            )
            self.observability.record("observation_recorded", goal.goal_id, evidence_id=record.evidence_id)

        if "root_cause" in requested and self.retriever is not None:
            self.observability.record("tool_called", goal.goal_id, tool="retrieve_company_context")
            rag_result = retrieve_company_context(self.retriever, goal.original_request, top_k=3)
            rag_record = registry.register(
                rag_result,
                result_type="rag",
                period=period,
                category=scope.category,
            )
            documents = rag_result.payload.get("documents") or []
            summaries.append(
                f"Retrieved {len(documents)} approved document source(s) for management context "
                f"[{rag_record.evidence_id}]."
            )
            self.observability.record(
                "observation_recorded", goal.goal_id, evidence_id=rag_record.evidence_id
            )

        reconciliation = AutonomousReconciler().reconcile(registry)
        if not reconciliation.passed:
            return AutonomousFinanceResponse(
                "failed", "Validation or reconciliation failed; no answer was released.",
                None, goal.goal_id,
                tuple(item.model_dump(mode="json") for item in registry.list()),
                self.observability.metrics(),
            )
        for record in registry.list():
            registry.mark_verified(record.evidence_id)
        answer = append_evidence_citations(
            " ".join(summaries) + " Revenue is based only on commission_amount, never fare.",
            registry.list(),
        )
        self.observability.record("goal_completed", goal.goal_id)
        commentary = _management_commentary(registry.list())
        return AutonomousFinanceResponse(
            "completed", answer, None, goal.goal_id,
            tuple(item.model_dump(mode="json") for item in registry.list()),
            self.observability.metrics(), commentary,
        )


def _finance_summary(
    label: str, payload: dict[str, Any], evidence_id: str, *, category: str | None = None
) -> str:
    """Create a compact numeric answer directly from validated tool output."""

    if label == "revenue variance":
        actual = payload.get("actual_revenue")
        budget = payload.get("budget_revenue")
        variance = payload.get("revenue_variance")
        budget_value = float(budget) if isinstance(budget, (int, float)) and budget else 0.0
        variance_pct = float(variance) / budget_value * 100 if budget_value else None
        direction = "favourable" if isinstance(variance, (int, float)) and variance >= 0 else "unfavourable"
        scope = f" for {category}" if category else ""
        drivers = _top_revenue_drivers(payload.get("vehicle_variance_summary", []))
        return (
            f"Revenue variance{scope}: actual INR {_number(actual)} versus budget INR "
            f"{_number(budget)}, producing a {direction} variance of INR {_number(variance)} "
            f"({_percentage(variance_pct)}) [{evidence_id}]. "
            f"Completed orders were {_integer(payload.get('actual_orders'))} versus "
            f"{_integer(payload.get('budget_orders'))} budget, while AOV was INR "
            f"{_number(payload.get('actual_aov'))} versus INR {_number(payload.get('budget_aov'))}. "
            f"The volume effect was INR {_number(payload.get('volume_effect'))}; "
            f"the price/AOV effect was INR {_number(payload.get('price_effect'))}."
            + (f" Major category drivers: {drivers}." if drivers else "")
        )
    if label == "GP decomposition":
        portfolio = payload.get("portfolio_level", {})
        return (
            f"GP%: budget {_number(portfolio.get('budget_gp_percentage'))}%, "
            f"actual {_number(portfolio.get('actual_gp_percentage'))}%; "
            f"mix {_number(portfolio.get('mix_effect_percentage_points'))} pp, "
            f"price {_number(portfolio.get('price_effect_percentage_points'))} pp, "
            f"cost {_number(portfolio.get('cost_effect_percentage_points'))} pp "
            f"[{evidence_id}]."
        )
    if label == "KPI":
        selected_kpis = payload.get("selected_kpis") or []
        values = [
            f"{item.get('display_name')}: {_kpi_value(item.get('value'), item.get('unit'))}"
            for item in selected_kpis
            if isinstance(item, dict)
        ]
        return f"KPI summary: {'; '.join(values)} [{evidence_id}]."
    if label == "daily KPI extreme":
        selected = payload.get("selected_day_kpis") or {}
        if payload.get("analysis_mode") == "high_revenue_low_orders":
            high_revenue = payload.get("highest_revenue_day") or {}
            low_orders = payload.get("lowest_order_day") or {}
            high_aov = payload.get("highest_aov_day") or {}
            return (
                f"Highest-revenue day: {high_revenue.get('date')} with commission revenue INR "
                f"{_number(high_revenue.get('revenue'))}, {_integer(high_revenue.get('total_orders'))} "
                f"orders and AOV INR {_number(high_revenue.get('aov'))}. "
                f"Lowest-order day: {low_orders.get('date')} with "
                f"{_integer(low_orders.get('total_orders'))} orders, commission revenue INR "
                f"{_number(low_orders.get('revenue'))} and AOV INR {_number(low_orders.get('aov'))}. "
                f"The strongest high-revenue-per-order day was {high_aov.get('date')} with AOV INR "
                f"{_number(high_aov.get('aov'))} across {_integer(high_aov.get('total_orders'))} orders "
                f"[{evidence_id}]."
            )
        direction = payload.get("requested_extreme", "lowest")
        difference_pct = payload.get("difference_from_daily_average_percentage")
        comparison = (
            f"{abs(float(difference_pct)):.2f}% {'below' if float(difference_pct) < 0 else 'above'}"
            if isinstance(difference_pct, (int, float)) else "not comparable with"
        )
        return (
            f"The {direction}-order day was {payload.get('selected_day')} with "
            f"{_integer(payload.get('selected_total_orders'))} total orders, {comparison} the "
            f"daily average of {_number(payload.get('daily_average_orders'))}. "
            f"Completed orders were {_integer(selected.get('completed_orders'))}, cancellations "
            f"were {_integer(selected.get('cancelled_orders'))}, fulfillment was "
            f"{_percentage(selected.get('fulfillment_percentage'))}, and commission revenue was "
            f"INR {_number(selected.get('revenue'))} [{evidence_id}]."
        )
    if label == "daily order drivers":
        signals = " ".join(payload.get("operational_signals") or [])
        prior = payload.get("previous_day_orders")
        prior_text = f", versus {_integer(prior)} on the previous available day" if prior is not None else ""
        return (
            f"On {payload.get('target_date')} ({payload.get('weekday')}), the selected category had "
            f"{_integer(payload.get('total_orders'))} orders{prior_text}. The monthly daily average was "
            f"{_number(payload.get('monthly_daily_average_orders'))}, and the same-weekday average was "
            f"{_number(payload.get('same_weekday_average_orders'))}. {signals} "
            f"Fulfillment was {_percentage(payload.get('fulfillment_percentage'))}; commission revenue "
            f"was INR {_number(payload.get('revenue'))}. {payload.get('cause_status')} [{evidence_id}]."
        )
    if label == "pnl":
        summary = payload.get("pnl_summary", {})
        actual = summary.get("actual", {})
        budget = summary.get("budget", {})
        variance = summary.get("variance", {})
        # Accept the earlier flat contract as a compatibility fallback.
        actual_revenue = actual.get("revenue", summary.get("actual_revenue"))
        actual_gp = actual.get("gross_profit", summary.get("actual_gross_profit"))
        actual_np = actual.get("net_profit", summary.get("actual_net_profit"))
        budget_np = budget.get("net_profit", summary.get("budget_net_profit"))
        net_profit_variance = variance.get("net_profit_variance")
        actual_direct_cost = actual.get("direct_cost")
        budget_direct_cost = budget.get("direct_cost")
        prefix = f"P&L for {category}" if category else "P&L for all vehicle categories"
        allocation = (
            " Corporate expenses were allocated by revenue share."
            if category else ""
        )
        return (
            f"{prefix}: actual revenue INR {_number(actual_revenue)}, "
            f"actual direct cost INR {_number(actual_direct_cost)} versus budget INR "
            f"{_number(budget_direct_cost)}, "
            f"actual gross profit INR {_number(actual_gp)}, "
            f"actual net profit INR {_number(actual_np)}, "
            f"budget net profit INR {_number(budget_np)}, "
            f"net profit variance INR {_number(net_profit_variance)} "
            f"[{evidence_id}].{allocation}"
        )
    if label == "rolling forecast":
        rows = payload.get("forecast_summary", [])
        if rows:
            first = rows[0]
            return (
                f"Rolling forecast {first.get('forecast_period')}: revenue INR "
                f"{_number(first.get('forecast_revenue'))} [{evidence_id}]."
            )
    if label == "trend analysis":
        rows = payload.get("trend") or []
        summary = payload.get("summary") or {}
        return (
            f"{payload.get('frequency', 'daily').title()} trend across {len(rows)} periods: average orders "
            f"{_number(summary.get('average_orders'))} and average commission revenue INR "
            f"{_number(summary.get('average_revenue'))}. Highest orders were "
            f"{_integer(summary.get('highest_orders'))} on {str(summary.get('highest_order_period'))[:10]}; "
            f"lowest orders were {_integer(summary.get('lowest_orders'))} on "
            f"{str(summary.get('lowest_order_period'))[:10]}. Highest revenue was INR "
            f"{_number(summary.get('highest_revenue'))} on {str(summary.get('highest_revenue_period'))[:10]}; "
            f"first-to-last revenue movement was {_percentage(summary.get('first_to_last_revenue_change_percentage'))} "
            f"[{evidence_id}]."
        )
    if label == "period comparison":
        metrics = payload.get("metrics") or []
        revenue = next((item for item in metrics if item.get("metric") == "revenue"), {})
        orders = next((item for item in metrics if item.get("metric") == "total_orders"), {})
        return f"Period comparison: revenue variance INR {_number(revenue.get('variance'))}; order variance {_integer(orders.get('variance'))} [{evidence_id}]."
    if label == "drill-down":
        rows = payload.get("rows") or []
        return f"Drill-down by {payload.get('dimension')} produced {len(rows)} segment(s) [{evidence_id}]."
    if label == "anomaly detection":
        return f"Detected {_integer(payload.get('anomaly_count'))} robust daily KPI anomaly/anomalies [{evidence_id}]."
    if label == "category profitability":
        rows = payload.get("rows") or []
        top = rows[0] if rows else {}
        return f"Category profitability calculated for {len(rows)} categories; highest net profit was {top.get('vehicle_category')} at INR {_number(top.get('net_profit'))} [{evidence_id}]."
    if label == "customer and route analysis":
        if payload.get("analysis_scope") == "route":
            summary = payload.get("route_summary") or {}
            top = summary.get("highest_gross_profit_route") or {}
            bottom = summary.get("lowest_gross_profit_route") or {}
            return (
                f"Route profitability analysed {summary.get('route_count', 0)} routes on a gross-profit basis. "
                f"Highest contribution: {top.get('pickup_cluster')} to {top.get('drop_cluster')} with gross "
                f"profit INR {_number(top.get('gross_profit'))} and GP% {_percentage(top.get('gp_percentage'))}. "
                f"Lowest contribution: {bottom.get('pickup_cluster')} to {bottom.get('drop_cluster')} with gross "
                f"profit INR {_number(bottom.get('gross_profit'))} and GP% {_percentage(bottom.get('gp_percentage'))}. "
                f"Loss-making routes: {_integer(summary.get('loss_making_route_count'))}. Corporate expenses are "
                f"not allocated to routes [{evidence_id}]."
            )
        return f"Customer analysis available: {payload.get('customer_analysis_available')}; route analysis returned {len(payload.get('routes') or [])} route(s) [{evidence_id}]."
    if label == "forecast accuracy":
        if not payload.get("available"):
            return f"Forecast accuracy is unavailable: {payload.get('reason')} [{evidence_id}]."
        summary = payload.get("summary") or {}
        return f"Forecast MAPE was {_percentage(summary.get('mape'))}; bias was INR {_number(summary.get('mean_error_bias'))} [{evidence_id}]."
    if label == "scenario analysis":
        scenario = payload.get("scenario") or {}
        return f"Scenario revenue is INR {_number(scenario.get('revenue'))}, gross profit INR {_number(scenario.get('gross_profit'))}, and GP% {_percentage(scenario.get('gp_percentage'))} [{evidence_id}]."
    if label == "driver-based forecast":
        rows = payload.get("forecast") or []
        first = rows[0] if rows else {}
        return f"Driver-based forecast {first.get('period', 'unavailable')}: revenue INR {_number(first.get('revenue'))} and GP% {_percentage(first.get('gp_percentage'))} [{evidence_id}]."
    if label == "profitability alerts":
        summary = payload.get("monitoring_summary") or {}
        thresholds = payload.get("thresholds") or {}
        return (
            f"Profitability monitoring evaluated {_integer(summary.get('categories_evaluated'))} categories "
            f"and identified {_integer(payload.get('alert_count'))} threshold breach(es). Lowest GP% was "
            f"{_percentage(summary.get('lowest_gp_percentage'))} for {summary.get('lowest_gp_category')}, "
            f"against a minimum threshold of {_percentage(thresholds.get('minimum_gp_percentage'))}. "
            f"Highest cancellation was {_percentage(summary.get('highest_cancellation_percentage'))} for "
            f"{summary.get('highest_cancellation_category')}, against a maximum threshold of "
            f"{_percentage(thresholds.get('maximum_cancellation_percentage'))} [{evidence_id}]."
        )
    if label == "operational drivers":
        return f"Operational driver analysis evaluated {len(payload.get('drivers') or [])} supported dimensions; associations are not claimed as proven causes [{evidence_id}]."
    return f"{label.title()} completed [{evidence_id}]."


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.2f}"
    return "not available"


def _integer(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{int(value):,}"
    return "not available"


def _percentage(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):,.2f}%"
    return "percentage not available"


def _kpi_value(value: Any, unit: Any) -> str:
    if value is None:
        return "not available"
    if unit == "count":
        return _integer(value)
    if unit in {"percentage", "decimal_percentage", "percentage_points"}:
        return f"{_number(value)}%"
    if unit in {"currency", "currency_per_order"}:
        return f"INR {_number(value)}"
    return str(value)


def _top_revenue_drivers(rows: Any) -> str:
    if not isinstance(rows, list):
        return ""
    valid = [
        row for row in rows
        if isinstance(row, dict) and isinstance(row.get("revenue_variance"), (int, float))
    ]
    valid.sort(key=lambda row: abs(float(row["revenue_variance"])), reverse=True)
    return "; ".join(
        f"{row.get('vehicle_category')} INR {_number(row.get('revenue_variance'))}"
        for row in valid[:3]
    )


def _management_commentary(records: Any) -> dict[str, Any]:
    findings: list[str] = []
    risks: list[str] = []
    recommendations: list[str] = []
    for record in records:
        payload = record.compact_payload
        if record.result_type == "pnl":
            summary = payload.get("pnl_summary") or payload.get("summary") or {}
            actual = summary.get("actual") or {}
            budget = summary.get("budget") or {}
            variance = summary.get("variance") or {}
            actual_revenue = actual.get("revenue")
            budget_revenue = budget.get("revenue")
            actual_net_profit = actual.get("net_profit")
            budget_net_profit = budget.get("net_profit")
            actual_margin = actual.get("gross_margin_percentage")
            budget_margin = budget.get("gross_margin_percentage")
            if all(isinstance(value, (int, float)) for value in (actual_revenue, budget_revenue)):
                revenue_variance = float(actual_revenue) - float(budget_revenue)
                direction = "above" if revenue_variance >= 0 else "below"
                findings.append(
                    f"Revenue was INR {_number(actual_revenue)}, INR {_number(abs(revenue_variance))} "
                    f"{direction} budget."
                )
            if all(isinstance(value, (int, float)) for value in (actual_net_profit, budget_net_profit)):
                profit_variance = float(actual_net_profit) - float(budget_net_profit)
                direction = "above" if profit_variance >= 0 else "below"
                findings.append(
                    f"Net profit was INR {_number(actual_net_profit)}, INR {_number(abs(profit_variance))} "
                    f"{direction} budget."
                )
                if profit_variance < 0:
                    risks.append(f"Net profit missed budget by INR {_number(abs(profit_variance))}.")
                    recommendations.append("Review operating expenses and category contribution to recover the profit gap.")
            if all(isinstance(value, (int, float)) for value in (actual_margin, budget_margin)):
                margin_variance = float(actual_margin) - float(budget_margin)
                findings.append(
                    f"Gross margin was {float(actual_margin):.2f}%, "
                    f"{abs(margin_variance):.2f} percentage points "
                    f"{'above' if margin_variance >= 0 else 'below'} budget."
                )
                if margin_variance < 0:
                    risks.append(f"Gross margin is {abs(margin_variance):.2f} pp below budget.")
                    recommendations.append("Investigate direct cost per order and pricing by vehicle category.")
            if variance.get("net_profit_variance") is not None and not findings:
                findings.append(
                    f"Net profit variance was INR {_number(variance.get('net_profit_variance'))}."
                )
        elif record.result_type == "daily_kpi_extreme":
            selected = payload.get("selected_day_kpis") or {}
            day = payload.get("selected_day")
            difference_pct = float(payload.get("difference_from_daily_average_percentage") or 0)
            if payload.get("analysis_mode") == "high_revenue_low_orders":
                high_revenue = payload.get("highest_revenue_day") or {}
                low_orders = payload.get("lowest_order_day") or {}
                day = low_orders.get("date")
                selected = low_orders
                findings.append(
                    f"Highest revenue occurred on {high_revenue.get('date')}; lowest order volume "
                    f"occurred on {low_orders.get('date')}."
                )
            else:
                findings.append(
                    f"{day} recorded the lowest order volume at {_integer(payload.get('selected_total_orders'))} "
                    f"orders, {abs(difference_pct):.2f}% below the month's daily average."
                )
            risks.append(
                f"Low-volume day {day} had fulfillment of "
                f"{_percentage(selected.get('fulfillment_percentage'))} and cancellation of "
                f"{_percentage(selected.get('cancellation_percentage'))}."
            )
            recommendations.append(
                f"Review weekday, vehicle-category, location, and cancellation drivers for {day}."
            )
        elif record.result_type == "daily_order_root_cause":
            findings.extend(payload.get("operational_signals") or [])
            risks.append(payload.get("cause_status"))
            recommendations.append(
                "Check dispatch capacity, customer demand, promotions, outages, holidays, and cluster-level availability for the date."
            )
        elif record.result_type == "trend":
            summary = payload.get("summary") or {}
            revenue_change = float(summary.get("first_to_last_revenue_change_percentage") or 0)
            order_change = float(summary.get("first_to_last_order_change_percentage") or 0)
            findings.append(
                f"From the first to last period, revenue moved {revenue_change:+.2f}% and orders moved {order_change:+.2f}%."
            )
            risks.append(
                f"The lowest order period was {str(summary.get('lowest_order_period'))[:10]} with "
                f"{_integer(summary.get('lowest_orders'))} orders."
            )
            recommendations.append(
                "Review the lowest-volume date by vehicle category, pickup cluster, cancellations, and service levels."
            )
        elif record.result_type == "customer_route" and payload.get("route_summary"):
            summary = payload.get("route_summary") or {}
            top = summary.get("highest_gross_profit_route") or {}
            bottom = summary.get("lowest_gross_profit_route") or {}
            findings.append(
                f"The highest gross-profit route was {top.get('pickup_cluster')} to {top.get('drop_cluster')} "
                f"at INR {_number(top.get('gross_profit'))}."
            )
            risks.append(
                f"The lowest gross-profit route was {bottom.get('pickup_cluster')} to {bottom.get('drop_cluster')} "
                f"at INR {_number(bottom.get('gross_profit'))}; route results exclude corporate-expense allocation."
            )
            recommendations.append(
                "Review low-contribution routes for order volume, AOV, cancellation, fulfilment, and direct cost per completed order."
            )
        elif record.result_type == "profitability_alert":
            summary = payload.get("monitoring_summary") or {}
            if int(payload.get("alert_count") or 0) == 0:
                findings.append(
                    f"All {_integer(summary.get('categories_evaluated'))} categories remained within configured GP% and cancellation thresholds."
                )
                risks.append(
                    f"The highest cancellation rate was {_percentage(summary.get('highest_cancellation_percentage'))} "
                    f"for {summary.get('highest_cancellation_category')}."
                )
                recommendations.append(
                    f"Continue monitoring {summary.get('highest_cancellation_category')} as the category closest to the cancellation limit."
                )
            else:
                findings.append(f"{_integer(payload.get('alert_count'))} profitability threshold breach(es) require review.")
                risks.extend(item.get("message") for item in (payload.get("alerts") or [])[:3])
                recommendations.append("Assign owners to investigate and resolve the highest-severity profitability alerts.")
        elif record.result_type == "revenue_variance":
            rows = payload.get("vehicle_variance_summary") or []
            ranked = sorted(rows, key=lambda row: float(row.get("revenue_variance", 0)))
            if ranked:
                findings.append(
                    f"Largest favourable revenue category: {ranked[-1].get('vehicle_category')} "
                    f"(INR {_number(ranked[-1].get('revenue_variance'))})."
                )
                if float(ranked[0].get("revenue_variance", 0)) < 0:
                    risks.append(
                        f"{ranked[0].get('vehicle_category')} has the largest adverse revenue variance "
                        f"of INR {_number(ranked[0].get('revenue_variance'))}."
                    )
                    recommendations.append(
                        f"Review volume and AOV drivers for {ranked[0].get('vehicle_category')}."
                    )
        elif record.result_type == "gp_decomposition":
            portfolio = payload.get("portfolio_level") or payload
            effects = {
                "mix": float(portfolio.get("mix_effect_percentage_points") or 0),
                "price": float(portfolio.get("price_effect_percentage_points") or 0),
                "cost": float(portfolio.get("cost_effect_percentage_points") or 0),
            }
            driver = max(effects, key=lambda key: abs(effects[key]))
            findings.append(f"The largest GP% bridge driver is {driver}: {effects[driver]:+.2f} pp.")
            if effects[driver] < 0:
                risks.append(f"The {driver} effect reduced GP% by {abs(effects[driver]):.2f} pp.")
                recommendations.append(f"Prioritize corrective action on the {driver} driver.")
        elif record.result_type == "forecast":
            forecast = payload.get("forecast_summary") or []
            if forecast:
                findings.append(
                    f"The next forecast period is {forecast[0].get('forecast_period')} with revenue "
                    f"INR {_number(forecast[0].get('forecast_revenue'))}."
                )
    if not risks:
        risks.append("No material adverse exception was identified in the requested evidence.")
    if not recommendations:
        recommendations.append("Continue monitoring category revenue, service levels, and GP% drivers.")
    return {
        "executive_summary": " ".join(findings) or "Validated finance analysis completed.",
        "risks": risks[:3],
        "recommendations": recommendations[:3],
    }
