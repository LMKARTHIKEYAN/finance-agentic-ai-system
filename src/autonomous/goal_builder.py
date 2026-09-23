"""Convert natural-language FP&A requests into measurable finance goals."""

from __future__ import annotations

import re
from datetime import date
from uuid import uuid4

from src.autonomous.schemas import (
    FinanceGoal,
    GoalCompletionCriterion,
    ReportingScope,
)


_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november",
            "december",
        ),
        start=1,
    )
}


class GoalBuilder:
    """Build explicit objectives, scope, ambiguity, and completion criteria."""

    def build(
        self,
        request: str,
        *,
        reference_date: date | None = None,
        goal_id: str | None = None,
    ) -> FinanceGoal:
        cleaned = _required_text(request, "request")
        normalized = " ".join(cleaned.lower().split())
        today = reference_date or date.today()
        scope, ambiguities, question = _reporting_scope(normalized, today)
        category = _category_from_request(normalized)
        if category is not None:
            scope = scope.model_copy(update={"category": category})
        criteria = _completion_criteria(normalized)
        return FinanceGoal(
            goal_id=goal_id or f"goal-{uuid4().hex}",
            original_request=cleaned,
            objective=_objective(cleaned),
            reporting_scope=scope,
            criteria=criteria,
            ambiguities=ambiguities,
            clarification_question=question,
        )


def _objective(request: str) -> str:
    stripped = request.strip().rstrip(".?!")
    return stripped[0].upper() + stripped[1:] if stripped else request


def _completion_criteria(
    normalized: str,
) -> tuple[GoalCompletionCriterion, ...]:
    criteria: list[GoalCompletionCriterion] = []

    exact_date_reason = (
        bool(re.search(r"\b20\d{2}-\d{2}-\d{2}\b", normalized))
        and "order" in normalized
        and any(word in normalized for word in ("why", "reason", "cause"))
    )
    if exact_date_reason:
        criteria.append(
            GoalCompletionCriterion(
                key="daily_order_root_cause",
                description="The selected day's order volume is compared with relevant operational benchmarks and drivers.",
            )
        )

    if any(word in normalized for word in ("trend", "trending", "trajectory")):
        criteria.append(GoalCompletionCriterion(key="trend_analysis", description="KPI trends are calculated at the requested frequency."))
    if any(word in normalized for word in ("compare", "comparison", "versus", " vs ", "year-over-year", "yoy")) and not any(word in normalized for word in ("budget", "forecast")):
        criteria.append(GoalCompletionCriterion(key="period_comparison", description="Current and comparison-period KPIs are calculated."))
    if any(word in normalized for word in ("drill down", "drill-down", "breakdown")):
        criteria.append(GoalCompletionCriterion(key="drilldown_analysis", description="Performance is analysed by the requested business dimension."))
    if any(word in normalized for word in ("anomaly", "anomalies", "unusual", "outlier")):
        criteria.append(GoalCompletionCriterion(key="anomaly_detection", description="Unusual KPI observations are detected using robust statistics."))
    if "profitability" in normalized and any(word in normalized for word in ("category", "vehicle")):
        criteria.append(GoalCompletionCriterion(key="category_profitability", description="Category profitability and unit economics are calculated."))
    if any(word in normalized for word in ("customer", "route")) and any(word in normalized for word in ("analysis", "profit", "concentration", "retention", "churn", "top")):
        criteria.append(GoalCompletionCriterion(key="customer_route_analysis", description="Customer and route performance are analysed where source fields exist."))
    if "forecast" in normalized and any(word in normalized for word in ("accuracy", "mape", "bias", "error")):
        criteria.append(GoalCompletionCriterion(key="forecast_accuracy", description="Forecast accuracy, bias, and MAPE are calculated from saved forecast snapshots."))
    if any(phrase in normalized for phrase in ("what if", "scenario", "required to achieve")):
        criteria.append(GoalCompletionCriterion(key="scenario_analysis", description="The requested deterministic what-if scenario is calculated."))
    if "forecast" in normalized and any(word in normalized for word in ("driver-based", "driver based", "orders and aov", "drivers")):
        criteria.append(GoalCompletionCriterion(key="driver_forecast", description="A driver-based forecast is calculated from orders, AOV, and cost drivers."))
    if any(word in normalized for word in ("alert", "alerts", "threshold breach")):
        criteria.append(GoalCompletionCriterion(key="profitability_alerts", description="Profitability and operating thresholds are evaluated."))
    if any(phrase in normalized for phrase in (
        "action tracker", "track action", "create management action",
        "record management action", "assign management action",
    )):
        criteria.append(GoalCompletionCriterion(key="management_action", description="A management action draft is prepared for approval."))

    daily_extreme_request = (
        any(word in normalized for word in ("lowest", "low", "minimum", "least", "highest", "high", "maximum", "most"))
        and "day" in normalized
        and (re.search(r"\bkpis?\b", normalized) or "order" in normalized)
    )
    if daily_extreme_request:
        criteria.append(
            GoalCompletionCriterion(
                key="daily_kpi_extreme",
                description="Daily KPIs are compared and the requested lowest or highest day is identified.",
            )
        )
    elif re.search(r"\bkpis?\b", normalized):
        criteria.append(
            GoalCompletionCriterion(
                key="kpi_analysis",
                description="The standard actual, budget, and variance KPIs are calculated.",
            )
        )

    if "p&l" in normalized or "pnl" in normalized or "profit and loss" in normalized:
        criteria.extend(
            (
                GoalCompletionCriterion(
                    key="pnl_analysis",
                    description="Actual and budget P&L has been calculated.",
                ),
                GoalCompletionCriterion(
                    key="pnl_reconciliation",
                    description="P&L formulas and totals reconcile.",
                ),
            )
        )
    if "revenue" in normalized and any(
        word in normalized for word in ("variance", "decline", "driver", "why")
    ):
        criteria.extend(
            (
                GoalCompletionCriterion(
                    key="revenue_variance",
                    description="Actual-versus-budget revenue variance is calculated.",
                ),
                GoalCompletionCriterion(
                    key="revenue_reconciliation",
                    description="Revenue bridge effects reconcile to total variance.",
                ),
            )
        )
    if "gp%" in normalized or "gross margin" in normalized or "gp decomposition" in normalized:
        criteria.extend(
            (
                GoalCompletionCriterion(
                    key="gp_decomposition",
                    description="GP% mix, price, and cost effects are calculated.",
                ),
                GoalCompletionCriterion(
                    key="gp_reconciliation",
                    description="GP% effects reconcile to total GP% movement.",
                ),
            )
        )
    specialized_forecast = any(key in {item.key for item in criteria} for key in ("forecast_accuracy", "driver_forecast"))
    if any(word in normalized for word in ("forecast", "projection", "outlook")) and not specialized_forecast:
        criteria.append(
            GoalCompletionCriterion(
                key="forecast",
                description="The requested rolling forecast is calculated.",
            )
        )
    if any(word in normalized for word in ("why", "driver", "cause", "explain")):
        criteria.append(
            GoalCompletionCriterion(
                key="root_cause",
                description="Major drivers are supported by financial evidence.",
            )
        )
    if not criteria:
        criteria.append(
            GoalCompletionCriterion(
                key="finance_answer",
                description="The requested financial result is calculated.",
            )
        )
    criteria.append(
        GoalCompletionCriterion(
            key="final_validation",
            description="Required evidence is validated before finalization.",
        )
    )
    unique = {item.key: item for item in criteria}
    return tuple(unique.values())


def _reporting_scope(
    normalized: str,
    today: date,
) -> tuple[ReportingScope, tuple[str, ...], str | None]:
    exact_date_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", normalized)
    if exact_date_match:
        try:
            selected = date(*(int(value) for value in exact_date_match.groups()))
        except ValueError:
            return ReportingScope(), ("reporting_period",), "Please provide a valid reporting date."
        return ReportingScope(start_date=selected, end_date=selected), (), None

    if "this month" in normalized:
        start = today.replace(day=1)
        return ReportingScope(start_date=start, end_date=today), (), None

    month_match = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(20\d{2})\b",
        normalized,
    )
    if month_match:
        month = _MONTHS[month_match.group(1)]
        year = int(month_match.group(2))
        end = _month_end(year, month)
        return ReportingScope(
            start_date=date(year, month, 1),
            end_date=end,
        ), (), None

    year_month_match = re.search(
        r"\b(20\d{2})\s+(" + "|".join(_MONTHS) + r")\b",
        normalized,
    )
    if year_month_match:
        year = int(year_month_match.group(1))
        month = _MONTHS[year_month_match.group(2)]
        return ReportingScope(
            start_date=date(year, month, 1),
            end_date=_month_end(year, month),
        ), (), None

    fy_match = re.search(r"\bfy\s*(20\d{2})(?:\s*[-/]\s*(\d{2,4}))?\b", normalized)
    if fy_match:
        year = int(fy_match.group(1))
        return ReportingScope(
            start_date=date(year, 4, 1),
            end_date=date(year + 1, 3, 31),
        ), (), None

    year_match = re.search(r"\b(20\d{2})\b", normalized)
    if year_match:
        year = int(year_match.group(1))
        if any(term in normalized for term in ("p&l", "pnl", "profit and loss")):
            return ReportingScope(), (
                "annual_period_basis",
            ), (
                f"For {year} P&L, do you mean calendar year {year} "
                f"(January-December) or financial year FY {year}-{str(year + 1)[-2:]} "
                "(April-March)?"
            )
        return ReportingScope(
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
        ), (), None

    return ReportingScope(), ("reporting_period",), (
        "Which reporting period should I analyse?"
    )


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date.fromordinal(date(year, month + 1, 1).toordinal() - 1)


def _category_from_request(normalized: str) -> str | None:
    """Recognize the approved vehicle-category vocabulary."""

    aliases = (
        (r"\btata ace open\b", "Tata Ace Open"),
        (r"\btata ace closed\b", "Tata Ace Closed"),
        (r"\btata ace\b", "Tata Ace"),
        (r"\bpackers\s*(?:&|and)\s*movers\b", "Packers & Movers"),
        (r"\bcompact auto\b", "Compact Auto"),
        (r"\b17\s*ft\b", "17 FT"),
        (r"\b14\s*ft\b", "14 FT"),
        (r"\b10\s*ft\b", "10 FT"),
        (r"\b9\s*ft\b", "9 FT"),
        (r"\b8\s*ft\b", "8 FT"),
        (r"\b3w\b", "3W"),
        (r"\b2w\b", "2W"),
    )
    for pattern, category in aliases:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return category
    return None


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned
