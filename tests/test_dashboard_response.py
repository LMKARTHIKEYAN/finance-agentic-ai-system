"""
Tests for the dashboard response builder.

These tests verify that existing finance-agent outputs are converted
into structured dashboard data without recalculating finance results.
"""

from __future__ import annotations

import pytest

from src.api.dashboard_response import build_dashboard_response
from src.api.schemas import DashboardPayload


@pytest.fixture
def sample_finance_analysis() -> dict:
    """Return representative serialized finance-agent results."""

    return {
        "operations_result": {
            "total_revenue": 720_000,
            "completed_orders": 2_500,
            "average_order_value": 288,
            "fulfillment_percentage": 94.5,
            "vehicle_summary": [
                {
                    "vehicle_category": "2W",
                    "completed_orders": 1_200,
                    "total_revenue": 180_000,
                },
                {
                    "vehicle_category": "3W",
                    "completed_orders": 800,
                    "total_revenue": 240_000,
                },
            ],
            "period_summary": [
                {
                    "period": "2026-05",
                    "total_revenue": 680_000,
                },
                {
                    "period": "2026-06",
                    "total_revenue": 720_000,
                },
            ],
        },
        "budget_result": {
            "total_budget_revenue": 690_000,
            "total_budget_orders": 2_400,
            "budget_average_order_value": 287.5,
            "vehicle_summary": [
                {
                    "vehicle_category": "2W",
                    "completed_orders": 1_100,
                    "total_revenue": 170_000,
                },
                {
                    "vehicle_category": "3W",
                    "completed_orders": 850,
                    "total_revenue": 250_000,
                },
            ],
            "period_summary": [
                {
                    "period": "2026-05",
                    "budget_revenue": 670_000,
                },
                {
                    "period": "2026-06",
                    "budget_revenue": 690_000,
                },
            ],
        },
        "forecast_result": {
            "forecast_summary": [
                {
                    "period": "2026-05",
                    "forecast_revenue": 675_000,
                },
                {
                    "period": "2026-06",
                    "forecast_revenue": 710_000,
                },
            ],
        },
        "variance_result": {
            "actual_orders": 2_500,
            "budget_orders": 2_400,
            "order_variance": 100,
            "actual_revenue": 720_000,
            "budget_revenue": 690_000,
            "revenue_variance": 30_000,
            "actual_aov": 288,
            "budget_aov": 287.5,
            "aov_variance": 0.5,
            "price_effect": 12_000,
            "volume_effect": 15_000,
            "new_discontinued_effect": 3_000,
            "vehicle_variance_summary": [
                {
                    "vehicle_category": "2W",
                    "actual_revenue": 180_000,
                    "budget_revenue": 170_000,
                    "revenue_variance": 10_000,
                },
                {
                    "vehicle_category": "3W",
                    "actual_revenue": 240_000,
                    "budget_revenue": 250_000,
                    "revenue_variance": -10_000,
                },
            ],
        },
        "kpi_result": {
            "selected_kpis": [],
            "unavailable_kpis": [],
            "unknown_kpis": [],
        },
        "recommendation_result": {
            "recommendations": [
                {
                    "recommendation_code": "REV_GROWTH",
                    "priority": "HIGH",
                    "recommended_action": (
                        "Protect the revenue growth from the 2W category."
                    ),
                    "owner": "Operations",
                    "time_horizon": "Immediate",
                    "expected_impact": "Maintain favourable revenue variance.",
                }
            ],
        },
        "commentary_result": {
            "executive_summary": (
                "Revenue exceeded budget, supported by order growth."
            ),
            "kpi_commentary": [
                "Revenue was above budget.",
                "Order volume improved.",
            ],
            "variance_commentary": [
                "Price and volume effects were favourable."
            ],
            "forecast_commentary": [
                "Forecast performance remains stable."
            ],
            "scenario_commentary": [
                "The base case remains achievable."
            ],
            "risks": [
                "The 3W category was below budget."
            ],
            "management_attention": [
                "Review the 3W category performance."
            ],
        },
        "report_result": {
            "report_title": "June 2026 Management Report",
            "report_type": "monthly",
            "generated_at": "2026-07-16T12:00:00+00:00",
            "overall_status": "FAVOURABLE",
            "executive_summary": (
                "June revenue exceeded budget by ₹30,000."
            ),
            "key_risks": [
                "The 3W category remained below budget."
            ],
            "management_actions": [
                "Investigate the 3W order decline."
            ],
        },
    }


def test_build_dashboard_response_returns_payload(
    sample_finance_analysis: dict,
) -> None:
    """Builder should return the declared dashboard schema."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
        period="June 2026",
        analysis_type="Management Report",
        comparison="Actual vs Budget",
        category="All Categories",
    )

    assert isinstance(result, DashboardPayload)


def test_build_dashboard_response_sets_metadata(
    sample_finance_analysis: dict,
) -> None:
    """Report metadata should preserve filters and report information."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
        period="June 2026",
        analysis_type="Management Report",
        comparison="Actual vs Budget",
        category="All Categories",
    )

    metadata = result.report_metadata

    assert metadata.title == "June 2026 Management Report"
    assert metadata.period == "June 2026"
    assert metadata.analysis_type == "Management Report"
    assert metadata.comparison == "Actual vs Budget"
    assert metadata.category == "All Categories"
    assert metadata.overall_status == "FAVOURABLE"


def test_build_dashboard_response_creates_kpi_cards(
    sample_finance_analysis: dict,
) -> None:
    """Operations and variance results should produce KPI cards."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    cards_by_key = {
        card.key: card
        for card in result.kpi_cards
    }

    assert cards_by_key["total_revenue"].value == 720_000
    assert cards_by_key["total_revenue"].comparison_value == 690_000
    assert cards_by_key["total_revenue"].delta == 30_000
    assert cards_by_key["total_revenue"].favourability == "positive"

    assert cards_by_key["completed_orders"].value == 2_500
    assert cards_by_key["average_order_value"].value == 288
    assert cards_by_key["fulfillment_percentage"].value == 94.5
    assert cards_by_key["revenue_variance"].value == 30_000


def test_build_dashboard_response_creates_kpi_table(
    sample_finance_analysis: dict,
) -> None:
    """KPI cards should also be represented in the KPI table."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert result.kpi_table is not None
    assert result.kpi_table.title == "KPI Summary"
    assert len(result.kpi_table.rows) == len(result.kpi_cards)
    assert "KPI" in result.kpi_table.columns
    assert "Variance" in result.kpi_table.columns


def test_build_dashboard_response_creates_variance_table(
    sample_finance_analysis: dict,
) -> None:
    """Variance agent output should create the variance table."""

    result = build_dashboard_response(
        selected_flow="variance",
        finance_analysis=sample_finance_analysis,
    )

    assert result.variance_table is not None
    assert result.variance_table.title == "Actual vs Budget Variance"
    assert len(result.variance_table.rows) == 3

    revenue_row = next(
        row
        for row in result.variance_table.rows
        if row["Metric"] == "Revenue"
    )

    assert revenue_row["Actual"] == 720_000
    assert revenue_row["Budget"] == 690_000
    assert revenue_row["Variance"] == 30_000


def test_build_dashboard_response_creates_category_table(
    sample_finance_analysis: dict,
) -> None:
    """Vehicle-level variance rows should populate the category table."""

    result = build_dashboard_response(
        selected_flow="variance",
        finance_analysis=sample_finance_analysis,
    )

    assert result.category_table is not None
    assert result.category_table.title == "Category Variance Analysis"
    assert len(result.category_table.rows) == 2

    first_row = result.category_table.rows[0]

    assert first_row["vehicle_category"] == "2W"
    assert first_row["revenue_variance"] == 10_000


def test_build_dashboard_response_creates_trend_data(
    sample_finance_analysis: dict,
) -> None:
    """Actual, budget and forecast period data should be combined."""

    result = build_dashboard_response(
        selected_flow="forecast",
        finance_analysis=sample_finance_analysis,
    )

    assert len(result.trend_data) == 2

    june = next(
        point
        for point in result.trend_data
        if point.period == "2026-06"
    )

    assert june.actual == 720_000
    assert june.budget == 690_000
    assert june.forecast == 710_000


def test_build_dashboard_response_creates_waterfall(
    sample_finance_analysis: dict,
) -> None:
    """Variance output should produce ordered waterfall points."""

    result = build_dashboard_response(
        selected_flow="variance",
        finance_analysis=sample_finance_analysis,
    )

    labels = [
        point.label
        for point in result.waterfall_data
    ]

    assert labels == [
        "Budget Revenue",
        "Price Effect",
        "Volume Effect",
        "New / Discontinued",
        "Actual Revenue",
    ]

    assert result.waterfall_data[0].measure == "absolute"
    assert result.waterfall_data[-1].measure == "total"
    assert result.waterfall_data[1].value == 12_000
    assert result.waterfall_data[2].value == 15_000


def test_build_dashboard_response_creates_recommendations(
    sample_finance_analysis: dict,
) -> None:
    """Recommendation-agent results should populate recommendations."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert len(result.recommendations) == 1

    recommendation = result.recommendations[0]

    assert recommendation.priority == "HIGH"
    assert recommendation.title == "REV_GROWTH"
    assert recommendation.owner == "Operations"
    assert recommendation.source == "recommendation_result"


def test_build_dashboard_response_creates_risks(
    sample_finance_analysis: dict,
) -> None:
    """Report risks should take priority over commentary risks."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert len(result.risks) == 1
    assert (
        result.risks[0].description
        == "The 3W category remained below budget."
    )
    assert result.risks[0].source == "report_result"


def test_build_dashboard_response_uses_report_executive_summary(
    sample_finance_analysis: dict,
) -> None:
    """Report summary should take priority over commentary summary."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert (
        result.executive_summary
        == "June revenue exceeded budget by ₹30,000."
    )


def test_build_dashboard_response_creates_commentary(
    sample_finance_analysis: dict,
) -> None:
    """Structured commentary should be preserved for the UI."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert result.commentary.executive_summary is not None
    assert "Revenue was above budget." in (
        result.commentary.kpi_commentary or ""
    )
    assert "Price and volume effects" in (
        result.commentary.variance_commentary or ""
    )
    assert result.commentary.management_attention == [
        "Review the 3W category performance."
    ]


def test_build_dashboard_response_identifies_available_sections(
    sample_finance_analysis: dict,
) -> None:
    """Populated dashboard sections should be marked available."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert "kpi_cards" in result.available_sections
    assert "variance_table" in result.available_sections
    assert "trend_data" in result.available_sections
    assert "waterfall_data" in result.available_sections
    assert "recommendations" in result.available_sections
    assert "risks" in result.available_sections
    assert "executive_summary" in result.available_sections


def test_build_dashboard_response_handles_empty_analysis() -> None:
    """Empty agent results should return a valid empty dashboard."""

    result = build_dashboard_response(
        selected_flow="full",
        finance_analysis={},
    )

    assert isinstance(result, DashboardPayload)
    assert result.kpi_cards == []
    assert result.kpi_table is None
    assert result.variance_table is None
    assert result.category_table is None
    assert result.trend_data == []
    assert result.waterfall_data == []
    assert result.recommendations == []
    assert result.risks == []
    assert result.executive_summary is None
    assert result.available_sections == []
    assert result.unavailable_sections


def test_build_dashboard_response_records_unavailable_kpis() -> None:
    """Unavailable and unknown KPI names should become limitations."""

    finance_analysis = {
        "kpi_result": {
            "selected_kpis": [],
            "unavailable_kpis": [
                "gross_profit",
                "gross_margin",
            ],
            "unknown_kpis": [
                "customer_happiness_index",
            ],
        }
    }

    result = build_dashboard_response(
        selected_flow="kpi",
        finance_analysis=finance_analysis,
    )

    limitations = " ".join(result.data_limitations)

    assert "gross_profit" in limitations
    assert "gross_margin" in limitations
    assert "customer_happiness_index" in limitations


def test_build_dashboard_response_uses_selected_kpis() -> None:
    """Serialized KPI-agent output should be converted directly."""

    finance_analysis = {
        "kpi_result": {
            "selected_kpis": [
                {
                    "kpi": "revenue",
                    "display_name": "Revenue",
                    "value": 500_000,
                    "unit": "currency",
                },
                {
                    "kpi": "fulfillment_rate",
                    "display_name": "Fulfillment Rate",
                    "value": 93.5,
                    "unit": "percentage",
                },
            ],
            "unavailable_kpis": [],
            "unknown_kpis": [],
        }
    }

    result = build_dashboard_response(
        selected_flow="kpi",
        finance_analysis=finance_analysis,
    )

    assert len(result.kpi_cards) == 2
    assert result.kpi_cards[0].key == "revenue"
    assert result.kpi_cards[0].formatted_value == "₹500,000.00"
    assert result.kpi_cards[1].formatted_value == "93.50%"


def test_build_dashboard_response_rejects_non_string_flow() -> None:
    """Selected flow must be a string."""

    with pytest.raises(
        TypeError,
        match="selected_flow must be a string",
    ):
        build_dashboard_response(
            selected_flow=123,  # type: ignore[arg-type]
            finance_analysis={},
        )


def test_build_dashboard_response_rejects_non_dictionary_analysis() -> None:
    """Finance analysis must be a dictionary."""

    with pytest.raises(
        TypeError,
        match="finance_analysis must be a dictionary",
    ):
        build_dashboard_response(
            selected_flow="full",
            finance_analysis=[],  # type: ignore[arg-type]
        )


def test_build_dashboard_response_does_not_mutate_input(
    sample_finance_analysis: dict,
) -> None:
    """Dashboard transformation should not change agent outputs."""

    original_revenue = sample_finance_analysis[
        "operations_result"
    ]["total_revenue"]

    build_dashboard_response(
        selected_flow="full",
        finance_analysis=sample_finance_analysis,
    )

    assert (
        sample_finance_analysis[
            "operations_result"
        ]["total_revenue"]
        == original_revenue
    )