"""
Dashboard response builder for the Finance Agentic AI API.

This module converts existing finance-agent outputs into a structured
DashboardPayload that can be consumed by Streamlit, Excel exports and
PDF management reports.

This module must not:

- Recalculate finance metrics
- Execute agents
- Call LangGraph
- Call OpenAI
- Perform RAG retrieval
- Contain Streamlit presentation logic
"""

from __future__ import annotations

from datetime import datetime, timezone
from numbers import Real
from typing import Any

from src.api.schemas import (
    DashboardCommentary,
    DashboardKpiCard,
    DashboardPayload,
    DashboardRecommendation,
    DashboardReportMetadata,
    DashboardRisk,
    DashboardTable,
    DashboardTrendPoint,
    DashboardWaterfallPoint,
)


def build_dashboard_response(
    *,
    selected_flow: str,
    finance_analysis: dict[str, Any],
    period: str | None = None,
    analysis_type: str | None = None,
    comparison: str | None = None,
    category: str | None = None,
) -> DashboardPayload:
    """
    Build a structured FP&A dashboard response.

    Args:
        selected_flow:
            Finance flow selected by the router.

        finance_analysis:
            Serialized outputs produced by existing finance agents.

        period:
            Optional reporting period selected by the user.

        analysis_type:
            Optional dashboard analysis type.

        comparison:
            Optional comparison such as Actual vs Budget.

        category:
            Optional selected vehicle or business category.

    Returns:
        Structured DashboardPayload.

    Raises:
        TypeError:
            If selected_flow or finance_analysis is invalid.
    """

    if not isinstance(selected_flow, str):
        raise TypeError("selected_flow must be a string.")

    if not isinstance(finance_analysis, dict):
        raise TypeError("finance_analysis must be a dictionary.")

    normalized_flow = selected_flow.strip().lower() or "unknown"

    operations_result = _get_mapping(
        finance_analysis,
        "operations_result",
    )
    budget_result = _get_mapping(
        finance_analysis,
        "budget_result",
    )
    forecast_result = _get_mapping(
        finance_analysis,
        "forecast_result",
    )
    scenario_result = _get_mapping(
        finance_analysis,
        "scenario_result",
    )
    variance_result = _get_mapping(
        finance_analysis,
        "variance_result",
    )
    recommendation_result = _get_mapping(
        finance_analysis,
        "recommendation_result",
    )
    commentary_result = _get_mapping(
        finance_analysis,
        "commentary_result",
    )
    report_result = _get_mapping(
        finance_analysis,
        "report_result",
    )
    kpi_result = _get_mapping(
        finance_analysis,
        "kpi_result",
    )

    kpi_cards = _build_kpi_cards(
        operations_result=operations_result,
        budget_result=budget_result,
        variance_result=variance_result,
        kpi_result=kpi_result,
    )

    kpi_table = _build_kpi_table(
        kpi_cards
    )

    variance_table = _build_variance_table(
        variance_result
    )

    category_table = _build_category_table(
        operations_result=operations_result,
        budget_result=budget_result,
        variance_result=variance_result,
    )

    trend_data = _build_trend_data(
        operations_result=operations_result,
        budget_result=budget_result,
        forecast_result=forecast_result,
        scenario_result=scenario_result,
    )

    waterfall_data = _build_waterfall_data(
        variance_result
    )

    recommendations = _build_recommendations(
        recommendation_result,
        report_result,
    )

    risks = _build_risks(
        commentary_result,
        report_result,
    )

    executive_summary = _get_executive_summary(
        report_result,
        commentary_result,
    )

    commentary = _build_commentary(
        commentary_result
    )

    data_limitations = _build_data_limitations(
        finance_analysis=finance_analysis,
        kpi_result=kpi_result,
    )

    available_sections = _identify_available_sections(
        kpi_cards=kpi_cards,
        kpi_table=kpi_table,
        variance_table=variance_table,
        category_table=category_table,
        trend_data=trend_data,
        waterfall_data=waterfall_data,
        recommendations=recommendations,
        risks=risks,
        executive_summary=executive_summary,
    )

    expected_sections = {
        "kpi_cards",
        "kpi_table",
        "variance_table",
        "category_table",
        "trend_data",
        "waterfall_data",
        "recommendations",
        "risks",
        "executive_summary",
    }

    unavailable_sections = sorted(
        expected_sections - set(available_sections)
    )

    return DashboardPayload(
        report_metadata=DashboardReportMetadata(
            title=_get_report_title(
                report_result
            ),
            period=period,
            analysis_type=(
                analysis_type
                or _get_report_type(
                    report_result
                )
                or normalized_flow
            ),
            comparison=comparison,
            category=category,
            generated_at=_get_generated_at(
                report_result
            ),
            overall_status=_as_optional_string(
                report_result.get(
                    "overall_status"
                )
            ),
        ),
        kpi_cards=kpi_cards,
        kpi_table=kpi_table,
        variance_table=variance_table,
        category_table=category_table,
        trend_data=trend_data,
        waterfall_data=waterfall_data,
        recommendations=recommendations,
        risks=risks,
        executive_summary=executive_summary,
        commentary=commentary,
        data_limitations=data_limitations,
        available_sections=available_sections,
        unavailable_sections=unavailable_sections,
    )


def _build_kpi_cards(
    *,
    operations_result: dict[str, Any],
    budget_result: dict[str, Any],
    variance_result: dict[str, Any],
    kpi_result: dict[str, Any],
) -> list[DashboardKpiCard]:
    """Create dashboard KPI cards from existing results."""

    selected_kpis = _get_list(
        kpi_result,
        "selected_kpis",
    )

    if selected_kpis:
        cards = [
            _kpi_item_to_card(item)
            for item in selected_kpis
            if isinstance(item, dict)
        ]

        return [
            card
            for card in cards
            if card is not None
        ]

    cards: list[DashboardKpiCard] = []

    actual_revenue = _as_number(
        operations_result.get("total_revenue")
    )
    budget_revenue = _as_number(
        budget_result.get("total_budget_revenue")
    )

    if actual_revenue is not None:
        cards.append(
            _create_card(
                key="total_revenue",
                label="Revenue",
                value=actual_revenue,
                unit="currency",
                comparison_label=(
                    "Budget"
                    if budget_revenue is not None
                    else None
                ),
                comparison_value=budget_revenue,
                favourable_when_positive=True,
            )
        )

    completed_orders = _as_number(
        operations_result.get(
            "completed_orders"
        )
    )
    budget_orders = _as_number(
        budget_result.get(
            "total_budget_orders"
        )
    )

    if completed_orders is not None:
        cards.append(
            _create_card(
                key="completed_orders",
                label="Completed Orders",
                value=completed_orders,
                unit="count",
                comparison_label=(
                    "Budget"
                    if budget_orders is not None
                    else None
                ),
                comparison_value=budget_orders,
                favourable_when_positive=True,
            )
        )

    average_order_value = _as_number(
        operations_result.get(
            "average_order_value"
        )
    )
    budget_aov = _as_number(
        budget_result.get(
            "budget_average_order_value"
        )
    )

    if average_order_value is not None:
        cards.append(
            _create_card(
                key="average_order_value",
                label="Average Order Value",
                value=average_order_value,
                unit="currency",
                comparison_label=(
                    "Budget"
                    if budget_aov is not None
                    else None
                ),
                comparison_value=budget_aov,
                favourable_when_positive=True,
            )
        )

    fulfillment = _as_number(
        operations_result.get(
            "fulfillment_percentage"
        )
    )

    if fulfillment is not None:
        cards.append(
            DashboardKpiCard(
                key="fulfillment_percentage",
                label="Fulfillment",
                value=fulfillment,
                formatted_value=_format_value(
                    fulfillment,
                    "percentage",
                ),
                unit="percentage",
                status=_percentage_status(
                    fulfillment
                ),
                favourability=(
                    "positive"
                    if fulfillment >= 90
                    else "negative"
                ),
            )
        )

    revenue_variance = _as_number(
        variance_result.get(
            "revenue_variance"
        )
    )

    if revenue_variance is not None:
        budget_value = _as_number(
            variance_result.get(
                "budget_revenue"
            )
        )

        cards.append(
            _create_card(
                key="revenue_variance",
                label="Revenue Variance",
                value=revenue_variance,
                unit="currency",
                comparison_label="Budget",
                comparison_value=budget_value,
                favourable_when_positive=True,
                value_is_delta=True,
            )
        )

    return cards


def _kpi_item_to_card(
    item: dict[str, Any],
) -> DashboardKpiCard | None:
    """Convert one serialized KPIValue into a dashboard card."""

    key = _as_optional_string(
        item.get("kpi")
    )
    label = _as_optional_string(
        item.get("display_name")
    )
    value = _as_number(
        item.get("value")
    )

    if not key or not label:
        return None

    unit = _as_optional_string(
        item.get("unit")
    )

    return DashboardKpiCard(
        key=key,
        label=label,
        value=value,
        formatted_value=(
            _format_value(value, unit)
            if value is not None
            else _as_optional_string(
                item.get("value")
            )
        ),
        unit=unit,
        status=(
            _as_optional_string(
                item.get("value")
            )
            if unit == "status"
            else None
        ),
    )


def _create_card(
    *,
    key: str,
    label: str,
    value: float | int,
    unit: str,
    comparison_label: str | None,
    comparison_value: float | int | None,
    favourable_when_positive: bool,
    value_is_delta: bool = False,
) -> DashboardKpiCard:
    """Create a KPI card with comparison values."""

    if value_is_delta:
        delta = value
    elif comparison_value is not None:
        delta = value - comparison_value
    else:
        delta = None

    delta_percentage = _calculate_percentage(
        delta,
        comparison_value,
    )

    favourability = _get_favourability(
        delta,
        favourable_when_positive,
    )

    return DashboardKpiCard(
        key=key,
        label=label,
        value=value,
        formatted_value=_format_value(
            value,
            unit,
        ),
        unit=unit,
        comparison_label=comparison_label,
        comparison_value=comparison_value,
        delta=delta,
        delta_percentage=delta_percentage,
        status=_variance_status(delta),
        favourability=favourability,
    )


def _build_kpi_table(
    kpi_cards: list[DashboardKpiCard],
) -> DashboardTable | None:
    """Create a detailed KPI table from dashboard cards."""

    if not kpi_cards:
        return None

    rows = [
        {
            "KPI": card.label,
            "Actual": card.value,
            "Unit": card.unit,
            "Comparison": card.comparison_value,
            "Variance": card.delta,
            "Variance %": card.delta_percentage,
            "Status": card.status,
        }
        for card in kpi_cards
    ]

    return DashboardTable(
        title="KPI Summary",
        columns=[
            "KPI",
            "Actual",
            "Unit",
            "Comparison",
            "Variance",
            "Variance %",
            "Status",
        ],
        rows=rows,
    )


def _build_variance_table(
    variance_result: dict[str, Any],
) -> DashboardTable | None:
    """Create the actual-versus-budget variance table."""

    if not variance_result:
        return None

    rows = [
        {
            "Metric": "Orders",
            "Actual": variance_result.get(
                "actual_orders"
            ),
            "Budget": variance_result.get(
                "budget_orders"
            ),
            "Variance": variance_result.get(
                "order_variance"
            ),
        },
        {
            "Metric": "Revenue",
            "Actual": variance_result.get(
                "actual_revenue"
            ),
            "Budget": variance_result.get(
                "budget_revenue"
            ),
            "Variance": variance_result.get(
                "revenue_variance"
            ),
        },
        {
            "Metric": "Average Order Value",
            "Actual": variance_result.get(
                "actual_aov"
            ),
            "Budget": variance_result.get(
                "budget_aov"
            ),
            "Variance": variance_result.get(
                "aov_variance"
            ),
        },
    ]

    return DashboardTable(
        title="Actual vs Budget Variance",
        columns=[
            "Metric",
            "Actual",
            "Budget",
            "Variance",
        ],
        rows=rows,
    )


def _build_category_table(
    *,
    operations_result: dict[str, Any],
    budget_result: dict[str, Any],
    variance_result: dict[str, Any],
) -> DashboardTable | None:
    """Create the vehicle-category performance table."""

    variance_rows = _get_list(
        variance_result,
        "vehicle_variance_summary",
    )

    if variance_rows:
        columns = _collect_columns(
            variance_rows
        )

        return DashboardTable(
            title="Category Variance Analysis",
            columns=columns,
            rows=[
                row
                for row in variance_rows
                if isinstance(row, dict)
            ],
        )

    actual_rows = _get_list(
        operations_result,
        "vehicle_summary",
    )
    budget_rows = _get_list(
        budget_result,
        "vehicle_summary",
    )

    if not actual_rows and not budget_rows:
        return None

    merged_rows = _merge_category_rows(
        actual_rows=actual_rows,
        budget_rows=budget_rows,
    )

    return DashboardTable(
        title="Category Performance",
        columns=_collect_columns(
            merged_rows
        ),
        rows=merged_rows,
    )


def _build_trend_data(
    *,
    operations_result: dict[str, Any],
    budget_result: dict[str, Any],
    forecast_result: dict[str, Any],
    scenario_result: dict[str, Any],
) -> list[DashboardTrendPoint]:
    """Create chart-ready actual, budget and forecast trend data."""

    trend_by_period: dict[str, dict[str, Any]] = {}

    _add_period_rows(
        trend_by_period,
        _get_list(
            operations_result,
            "period_summary",
        ),
        value_keys=(
            "total_revenue",
            "revenue",
        ),
        target_key="actual",
    )

    _add_period_rows(
        trend_by_period,
        _get_list(
            budget_result,
            "period_summary",
        ),
        value_keys=(
            "budget_revenue",
            "total_budget_revenue",
        ),
        target_key="budget",
    )

    _add_period_rows(
        trend_by_period,
        _get_list(
            forecast_result,
            "forecast_summary",
        ),
        value_keys=(
            "forecast_revenue",
            "total_revenue",
            "revenue",
        ),
        target_key="forecast",
    )

    _add_period_rows(
        trend_by_period,
        _get_list(
            scenario_result,
            "adjusted_forecast",
        ),
        value_keys=(
            "adjusted_revenue",
            "forecast_revenue",
            "revenue",
        ),
        target_key="forecast",
    )

    return [
        DashboardTrendPoint(
            period=period,
            actual=values.get("actual"),
            budget=values.get("budget"),
            forecast=values.get("forecast"),
            prior_year=values.get("prior_year"),
        )
        for period, values in sorted(
            trend_by_period.items()
        )
    ]


def _build_waterfall_data(
    variance_result: dict[str, Any],
) -> list[DashboardWaterfallPoint]:
    """Create revenue variance waterfall chart data."""

    budget_revenue = _as_number(
        variance_result.get("budget_revenue")
    )
    actual_revenue = _as_number(
        variance_result.get("actual_revenue")
    )

    if (
        budget_revenue is None
        or actual_revenue is None
    ):
        return []

    components = [
        (
            "Budget Revenue",
            budget_revenue,
            "absolute",
            "neutral",
        ),
        (
            "Price Effect",
            _as_number(
                variance_result.get(
                    "price_effect"
                )
            )
            or 0,
            "relative",
            None,
        ),
        (
            "Volume Effect",
            _as_number(
                variance_result.get(
                    "volume_effect"
                )
            )
            or 0,
            "relative",
            None,
        ),
        (
            "New / Discontinued",
            _as_number(
                variance_result.get(
                    "new_discontinued_effect"
                )
            )
            or 0,
            "relative",
            None,
        ),
        (
            "Actual Revenue",
            actual_revenue,
            "total",
            "neutral",
        ),
    ]

    return [
        DashboardWaterfallPoint(
            label=label,
            value=value,
            measure=measure,
            sequence=index,
            favourability=(
                favourability
                or _get_favourability(
                    value,
                    favourable_when_positive=True,
                )
            ),
            unit="currency",
        )
        for index, (
            label,
            value,
            measure,
            favourability,
        ) in enumerate(
            components,
            start=1,
        )
    ]


def _build_recommendations(
    recommendation_result: dict[str, Any],
    report_result: dict[str, Any],
) -> list[DashboardRecommendation]:
    """Create structured management recommendations."""

    recommendation_rows = _get_list(
        recommendation_result,
        "recommendations",
    )

    recommendations: list[
        DashboardRecommendation
    ] = []

    for row in recommendation_rows:
        if not isinstance(row, dict):
            continue

        action = _as_optional_string(
            row.get("recommended_action")
        )

        if not action:
            continue

        recommendations.append(
            DashboardRecommendation(
                priority=_as_optional_string(
                    row.get("priority")
                ),
                title=(
                    _as_optional_string(
                        row.get(
                            "recommendation_code"
                        )
                    )
                    or "Management Recommendation"
                ),
                action=action,
                owner=_as_optional_string(
                    row.get("owner")
                ),
                time_horizon=_as_optional_string(
                    row.get("time_horizon")
                ),
                expected_impact=_as_optional_string(
                    row.get("expected_impact")
                ),
                source="recommendation_result",
            )
        )

    if recommendations:
        return recommendations

    for action in _get_list(
        report_result,
        "management_actions",
    ):
        action_text = _as_optional_string(action)

        if action_text:
            recommendations.append(
                DashboardRecommendation(
                    title="Management Action",
                    action=action_text,
                    source="report_result",
                )
            )

    return recommendations


def _build_risks(
    commentary_result: dict[str, Any],
    report_result: dict[str, Any],
) -> list[DashboardRisk]:
    """Create structured dashboard risks."""

    risk_texts = _get_list(
        report_result,
        "key_risks",
    )

    source = "report_result"

    if not risk_texts:
        risk_texts = _get_list(
            commentary_result,
            "risks",
        )
        source = "commentary_result"

    risks: list[DashboardRisk] = []

    for risk in risk_texts:
        risk_text = _as_optional_string(risk)

        if not risk_text:
            continue

        risks.append(
            DashboardRisk(
                severity="review",
                title="Financial or Operational Risk",
                description=risk_text,
                source=source,
            )
        )

    return risks


def _build_commentary(
    commentary_result: dict[str, Any],
) -> DashboardCommentary:
    """Convert structured commentary into dashboard sections."""

    return DashboardCommentary(
        executive_summary=_as_optional_string(
            commentary_result.get(
                "executive_summary"
            )
        ),
        financial_performance=None,
        kpi_commentary=_join_text(
            commentary_result.get(
                "kpi_commentary"
            )
        ),
        variance_commentary=_join_text(
            commentary_result.get(
                "variance_commentary"
            )
        ),
        forecast_commentary=_join_text(
            commentary_result.get(
                "forecast_commentary"
            )
        ),
        scenario_commentary=_join_text(
            commentary_result.get(
                "scenario_commentary"
            )
        ),
        management_attention=[
            str(item)
            for item in _get_list(
                commentary_result,
                "management_attention",
            )
            if str(item).strip()
        ],
    )


def _get_executive_summary(
    report_result: dict[str, Any],
    commentary_result: dict[str, Any],
) -> str | None:
    """Return the strongest available executive summary."""

    return (
        _as_optional_string(
            report_result.get(
                "executive_summary"
            )
        )
        or _as_optional_string(
            commentary_result.get(
                "executive_summary"
            )
        )
    )


def _build_data_limitations(
    *,
    finance_analysis: dict[str, Any],
    kpi_result: dict[str, Any],
) -> list[str]:
    """Create transparent data-availability limitations."""

    limitations: list[str] = []

    unavailable_kpis = _get_list(
        kpi_result,
        "unavailable_kpis",
    )

    if unavailable_kpis:
        limitations.append(
            "Unavailable KPIs: "
            + ", ".join(
                str(item)
                for item in unavailable_kpis
            )
        )

    unknown_kpis = _get_list(
        kpi_result,
        "unknown_kpis",
    )

    if unknown_kpis:
        limitations.append(
            "Unknown KPI requests: "
            + ", ".join(
                str(item)
                for item in unknown_kpis
            )
        )

    expected_results = {
        "operations_result",
        "budget_result",
        "forecast_result",
        "variance_result",
        "recommendation_result",
        "commentary_result",
        "report_result",
    }

    missing_results = sorted(
        result_name
        for result_name in expected_results
        if not finance_analysis.get(result_name)
    )

    if missing_results:
        limitations.append(
            "Some dashboard sections are unavailable because "
            "the selected workflow did not produce: "
            + ", ".join(missing_results)
            + "."
        )

    return limitations


def _identify_available_sections(
    *,
    kpi_cards: list[DashboardKpiCard],
    kpi_table: DashboardTable | None,
    variance_table: DashboardTable | None,
    category_table: DashboardTable | None,
    trend_data: list[DashboardTrendPoint],
    waterfall_data: list[DashboardWaterfallPoint],
    recommendations: list[DashboardRecommendation],
    risks: list[DashboardRisk],
    executive_summary: str | None,
) -> list[str]:
    """Identify successfully populated dashboard sections."""

    sections: list[str] = []

    section_values = {
        "kpi_cards": kpi_cards,
        "kpi_table": kpi_table,
        "variance_table": variance_table,
        "category_table": category_table,
        "trend_data": trend_data,
        "waterfall_data": waterfall_data,
        "recommendations": recommendations,
        "risks": risks,
        "executive_summary": executive_summary,
    }

    for section_name, value in section_values.items():
        if value:
            sections.append(section_name)

    return sections


def _merge_category_rows(
    *,
    actual_rows: list[Any],
    budget_rows: list[Any],
) -> list[dict[str, Any]]:
    """Merge actual and budget category summaries."""

    merged: dict[str, dict[str, Any]] = {}

    for row in actual_rows:
        if not isinstance(row, dict):
            continue

        category = _category_name(row)

        if not category:
            continue

        merged[category] = {
            "vehicle_category": category,
            **row,
        }

    for row in budget_rows:
        if not isinstance(row, dict):
            continue

        category = _category_name(row)

        if not category:
            continue

        existing = merged.setdefault(
            category,
            {"vehicle_category": category},
        )

        for key, value in row.items():
            if key == "vehicle_category":
                continue

            existing[
                _budget_column_name(key)
            ] = value

    return list(merged.values())


def _add_period_rows(
    trend_by_period: dict[str, dict[str, Any]],
    rows: list[Any],
    *,
    value_keys: tuple[str, ...],
    target_key: str,
) -> None:
    """Add serialized period rows to a trend mapping."""

    for row in rows:
        if not isinstance(row, dict):
            continue

        period = _period_name(row)

        if not period:
            continue

        value = _first_number(
            row,
            value_keys,
        )

        if value is None:
            continue

        trend_by_period.setdefault(
            period,
            {},
        )[target_key] = value


def _period_name(
    row: dict[str, Any],
) -> str | None:
    """Return a period value from a serialized result row."""

    for key in (
        "period",
        "month",
        "week",
        "date",
        "forecast_period",
    ):
        value = _as_optional_string(
            row.get(key)
        )

        if value:
            return value

    return None


def _category_name(
    row: dict[str, Any],
) -> str | None:
    """Return a category value from a serialized result row."""

    for key in (
        "vehicle_category",
        "category",
        "dimension_value",
    ):
        value = _as_optional_string(
            row.get(key)
        )

        if value:
            return value

    return None


def _budget_column_name(
    key: str,
) -> str:
    """Return a clear budget column name."""

    if key.startswith("budget_"):
        return key

    return f"budget_{key}"


def _collect_columns(
    rows: list[Any],
) -> list[str]:
    """Collect table columns while preserving first-seen order."""

    columns: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        for key in row:
            if key not in columns:
                columns.append(key)

    return columns


def _get_mapping(
    source: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Safely return one nested mapping."""

    value = source.get(key)

    if isinstance(value, dict):
        return value

    return {}


def _get_list(
    source: dict[str, Any],
    key: str,
) -> list[Any]:
    """Safely return one nested list."""

    value = source.get(key)

    if isinstance(value, list):
        return value

    return []


def _first_number(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> float | int | None:
    """Return the first valid numeric value from candidate keys."""

    for key in keys:
        value = _as_number(
            row.get(key)
        )

        if value is not None:
            return value

    return None


def _as_number(
    value: Any,
) -> float | int | None:
    """Convert valid real numbers into JSON-safe numeric values."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, Real):
        return None

    numeric_value = float(value)

    if numeric_value.is_integer():
        return int(numeric_value)

    return numeric_value


def _as_optional_string(
    value: Any,
) -> str | None:
    """Convert a non-empty value into a stripped string."""

    if value is None:
        return None

    cleaned = str(value).strip()

    return cleaned or None


def _join_text(
    value: Any,
) -> str | None:
    """Convert commentary lists into display-ready text."""

    if isinstance(value, list):
        items = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

        return "\n".join(items) or None

    return _as_optional_string(value)


def _calculate_percentage(
    numerator: float | int | None,
    denominator: float | int | None,
) -> float | None:
    """Calculate a safe percentage."""

    if (
        numerator is None
        or denominator is None
        or denominator == 0
    ):
        return None

    return round(
        (float(numerator) / abs(float(denominator)))
        * 100,
        2,
    )


def _get_favourability(
    value: float | int | None,
    favourable_when_positive: bool,
) -> str | None:
    """Return financial favourability for a variance."""

    if value is None:
        return None

    if value == 0:
        return "neutral"

    is_positive = value > 0

    if is_positive == favourable_when_positive:
        return "positive"

    return "negative"


def _variance_status(
    variance: float | int | None,
) -> str | None:
    """Return a simple variance status."""

    if variance is None:
        return None

    if variance > 0:
        return "FAVOURABLE"

    if variance < 0:
        return "UNFAVOURABLE"

    return "ON TARGET"


def _percentage_status(
    value: float | int,
) -> str:
    """Return operational status for a percentage KPI."""

    if value >= 95:
        return "STRONG"

    if value >= 90:
        return "ACCEPTABLE"

    return "REVIEW"


def _format_value(
    value: float | int | None,
    unit: str | None,
) -> str | None:
    """Format a numeric KPI for dashboard display."""

    if value is None:
        return None

    normalized_unit = (
        str(unit).strip().lower()
        if unit
        else ""
    )

    if normalized_unit in {
        "currency",
        "inr",
        "rupees",
    }:
        return f"₹{value:,.2f}"

    if normalized_unit in {
        "percentage",
        "percent",
        "%",
    }:
        return f"{value:,.2f}%"

    if normalized_unit == "count":
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def _get_report_title(
    report_result: dict[str, Any],
) -> str:
    """Return the report title."""

    return (
        _as_optional_string(
            report_result.get(
                "report_title"
            )
        )
        or "Finance Performance Dashboard"
    )


def _get_report_type(
    report_result: dict[str, Any],
) -> str | None:
    """Return the report type."""

    return _as_optional_string(
        report_result.get("report_type")
    )


def _get_generated_at(
    report_result: dict[str, Any],
) -> str:
    """Return the report generation timestamp."""

    return (
        _as_optional_string(
            report_result.get(
                "generated_at"
            )
        )
        or datetime.now(
            timezone.utc
        ).isoformat()
    )