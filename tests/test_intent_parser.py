"""
Tests for the deterministic finance intent parser.

These tests verify:

- Analysis-flow detection
- Comparison extraction
- Single-date extraction
- Month and year extraction
- Date-range extraction
- Relative-period extraction
- Category extraction
- KPI extraction
- Scenario extraction
- Missing-information detection
- Clarification questions
- Temporary slot filling
- Graph-compatible filter creation
"""

from __future__ import annotations

from datetime import date

import pytest

from src.orchestrator.intent_parser import (
    FinanceIntent,
    ParsedPeriod,
    merge_finance_intent,
    parse_finance_intent,
)


def test_parse_actual_vs_budget_month_request() -> None:
    """Complete variance request should extract month and comparison."""

    result = parse_finance_intent(
        "Show Actual vs Budget for January 2026"
    )

    assert result.selected_flow == "variance"
    assert result.comparison == "actual_vs_budget"
    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-01-31"
    assert result.period.display_value == "January 2026"
    assert result.period.granularity == "month"
    assert result.is_complete is True
    assert result.missing_fields == ()
    assert result.clarification_question is None


def test_parse_kpi_specific_date_request() -> None:
    """KPI request should extract a specific calendar date."""

    result = parse_finance_intent(
        "Show KPI for 20 October 2026"
    )

    assert result.selected_flow == "kpi"
    assert result.period.start_date == "2026-10-20"
    assert result.period.end_date == "2026-10-20"
    assert result.period.display_value == "20 October 2026"
    assert result.period.granularity == "day"
    assert result.is_complete is True


def test_parse_numeric_dot_date() -> None:
    """Dates using DD.MM.YYYY should be supported."""

    result = parse_finance_intent(
        "Show KPI for 20.10.2026"
    )

    assert result.period.start_date == "2026-10-20"
    assert result.period.end_date == "2026-10-20"
    assert result.period.granularity == "day"


def test_parse_numeric_slash_date() -> None:
    """Dates using DD/MM/YYYY should be supported."""

    result = parse_finance_intent(
        "Show revenue KPI for 20/10/2026"
    )

    assert result.period.start_date == "2026-10-20"
    assert result.period.display_value == "20 October 2026"


def test_parse_iso_date() -> None:
    """ISO dates should be extracted correctly."""

    result = parse_finance_intent(
        "Show KPI performance for 2026-10-20"
    )

    assert result.period.start_date == "2026-10-20"
    assert result.period.end_date == "2026-10-20"


def test_parse_named_month_abbreviation() -> None:
    """Abbreviated month names should be supported."""

    result = parse_finance_intent(
        "Show forecast for Jan 2026"
    )

    assert result.selected_flow == "forecast"
    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-01-31"
    assert result.period.display_value == "January 2026"


def test_parse_leap_year_february() -> None:
    """Month end should respect leap years."""

    result = parse_finance_intent(
        "Show KPI for February 2028"
    )

    assert result.period.start_date == "2028-02-01"
    assert result.period.end_date == "2028-02-29"


def test_parse_non_leap_year_february() -> None:
    """February should end on day 28 in a non-leap year."""

    result = parse_finance_intent(
        "Show KPI for February 2026"
    )

    assert result.period.start_date == "2026-02-01"
    assert result.period.end_date == "2026-02-28"


def test_parse_year_period() -> None:
    """A year-only request should create a full-year period."""

    result = parse_finance_intent(
        "Show KPI performance for 2026"
    )

    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-12-31"
    assert result.period.display_value == "2026"
    assert result.period.granularity == "year"


def test_parse_named_date_range() -> None:
    """Named start and end dates should create a date range."""

    result = parse_finance_intent(
        "Show Actual vs Budget from "
        "1 January 2026 to 31 March 2026"
    )

    assert result.selected_flow == "variance"
    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-03-31"
    assert result.period.granularity == "range"
    assert result.period.display_value == (
        "01 January 2026 to 31 March 2026"
    )
    assert result.is_complete is True


def test_parse_numeric_date_range() -> None:
    """Numeric start and end dates should create a date range."""

    result = parse_finance_intent(
        "Show KPI between 01/01/2026 and 31/01/2026"
    )

    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-01-31"
    assert result.period.granularity == "range"


def test_parse_iso_date_range() -> None:
    """ISO start and end dates should create a date range."""

    result = parse_finance_intent(
        "Show KPI from 2026-01-01 to 2026-03-31"
    )

    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-03-31"
    assert result.period.granularity == "range"


def test_parse_range_with_year_only_on_end_date() -> None:
    """Start date should inherit the year from the end date."""

    result = parse_finance_intent(
        "Show KPI from 1 Jan to 31 Mar 2026"
    )

    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-03-31"
    assert result.period.granularity == "range"


def test_invalid_reverse_date_range_is_unresolved() -> None:
    """End date before start date should require clarification."""

    result = parse_finance_intent(
        "Show KPI from 31 March 2026 to 1 January 2026"
    )

    assert result.period.start_date is None
    assert result.period.end_date is None
    assert result.is_complete is False
    assert "period" in result.missing_fields


def test_invalid_range_date_is_unresolved() -> None:
    """Invalid dates inside a range should not be accepted."""

    result = parse_finance_intent(
        "Show KPI from 31 February 2026 to 31 March 2026"
    )

    assert result.period.start_date is None
    assert result.is_complete is False
    assert "period" in result.missing_fields


def test_parse_today_relative_period() -> None:
    """Today should use the provided reference date."""

    result = parse_finance_intent(
        "Show KPI for today",
        reference_date=date(2026, 7, 16),
    )

    assert result.period.start_date == "2026-07-16"
    assert result.period.end_date == "2026-07-16"
    assert result.period.display_value == "16 July 2026"


def test_parse_yesterday_relative_period() -> None:
    """Yesterday should resolve relative to the reference date."""

    result = parse_finance_intent(
        "Show KPI for yesterday",
        reference_date=date(2026, 7, 16),
    )

    assert result.period.start_date == "2026-07-15"
    assert result.period.end_date == "2026-07-15"


def test_parse_this_month_relative_period() -> None:
    """This month should create the complete current month."""

    result = parse_finance_intent(
        "Show Actual vs Budget for this month",
        reference_date=date(2026, 7, 16),
    )

    assert result.period.start_date == "2026-07-01"
    assert result.period.end_date == "2026-07-31"
    assert result.period.display_value == "July 2026"


def test_parse_last_month_across_year_boundary() -> None:
    """Last month in January should resolve to December."""

    result = parse_finance_intent(
        "Show KPI for last month",
        reference_date=date(2026, 1, 10),
    )

    assert result.period.start_date == "2025-12-01"
    assert result.period.end_date == "2025-12-31"
    assert result.period.display_value == "December 2025"


def test_parse_this_year_relative_period() -> None:
    """This year should create a complete annual period."""

    result = parse_finance_intent(
        "Show budget for this year",
        reference_date=date(2026, 7, 16),
    )

    assert result.period.start_date == "2026-01-01"
    assert result.period.end_date == "2026-12-31"
    assert result.period.granularity == "year"


@pytest.mark.parametrize(
    ("user_request", "expected_comparison"),
    [
        (
            "Show Actual versus Budget for January 2026",
            "actual_vs_budget",
        ),
        (
            "Show Actual vs Forecast for January 2026",
            "actual_vs_forecast",
        ),
        (
            "Show Actual vs Last Year for January 2026",
            "actual_vs_last_year",
        ),
        (
            "Show Budget vs Forecast for January 2026",
            "budget_vs_forecast",
        ),
        (
            "Show YoY performance for January 2026",
            "actual_vs_last_year",
        ),
    ],
)
def test_parse_supported_comparisons(
    user_request: str,
    expected_comparison: str,
) -> None:
    """Supported comparison phrases should be normalized."""

    result = parse_finance_intent(
        user_request
    )

    assert result.comparison == expected_comparison


def test_variance_word_defaults_to_actual_vs_budget() -> None:
    """Generic variance requests should default to budget comparison."""

    result = parse_finance_intent(
        "Show revenue variance for January 2026"
    )

    assert result.selected_flow == "variance"
    assert result.comparison == "actual_vs_budget"


@pytest.mark.parametrize(
    ("user_request", "expected_category"),
    [
        (
            "Show KPI for 2W for January 2026",
            "2W",
        ),
        (
            "Show KPI for two wheeler for January 2026",
            "2W",
        ),
        (
            "Show forecast for 3W for January 2026",
            "3W",
        ),
        (
            "Show revenue for Tata Ace for January 2026",
            "Tata Ace",
        ),
        (
            "Show revenue for Tata Ace Open for January 2026",
            "Tata Ace Open",
        ),
        (
            "Show KPI for packers and movers for January 2026",
            "Packer & Movers",
        ),
        (
            "Show KPI for Compact Auto for January 2026",
            "Compact Auto",
        ),
    ],
)
def test_parse_supported_categories(
    user_request: str,
    expected_category: str,
) -> None:
    """Supported category aliases should return canonical names."""

    result = parse_finance_intent(
        user_request
    )

    assert result.category == expected_category


def test_longest_category_alias_is_matched_first() -> None:
    """Specific Tata Ace categories should not become generic Tata Ace."""

    result = parse_finance_intent(
        "Show KPI for Tata Ace Closed for January 2026"
    )

    assert result.category == "Tata Ace Closed"


def test_parse_requested_kpis() -> None:
    """Explicit KPI names should be collected without duplicates."""

    result = parse_finance_intent(
        "Show revenue, orders, AOV and fulfillment "
        "for January 2026"
    )

    assert result.selected_flow == "kpi"
    assert set(result.requested_kpis) == {
    "revenue",
    "orders",
    "average_order_value",
    "fulfillment_rate",
}
    


def test_duplicate_kpi_aliases_are_removed() -> None:
    """Multiple aliases for one KPI should not produce duplicates."""

    result = parse_finance_intent(
        "Show order and orders KPI for January 2026"
    )

    assert result.requested_kpis.count(
        "orders"
    ) == 1


@pytest.mark.parametrize(
    ("user_request", "expected_scenario"),
    [
        (
            "Show scenario analysis for January 2026 "
            "using Management Case",
            "Management Case",
        ),
        (
            "Show scenario analysis for January 2026 "
            "using Base Case",
            "Base Case",
        ),
        (
            "Show scenario analysis for January 2026 "
            "using Best Case",
            "Upside Case",
        ),
        (
            "Show scenario analysis for January 2026 "
            "using Worst Case",
            "Downside Case",
        ),
    ],
)
def test_parse_scenario_name(
    user_request: str,
    expected_scenario: str,
) -> None:
    """Scenario aliases should produce canonical scenario names."""

    result = parse_finance_intent(
        user_request
    )

    assert result.scenario_name == expected_scenario


def test_missing_period_uses_all_available_data() -> None:
    """A finance request without a period should use all available data."""

    result = parse_finance_intent(
        "Show Actual vs Budget"
    )

    assert result.selected_flow == "variance"
    assert result.comparison == "actual_vs_budget"

    assert result.period.start_date is None
    assert result.period.end_date is None
    assert result.period.display_value is None
    assert result.period.granularity == "unknown"

    assert result.is_complete is True
    assert result.missing_fields == ()
    assert result.clarification_question is None
 


def test_unknown_request_returns_analysis_clarification() -> None:
    """Unrecognized requests should ask which analysis is needed."""

    result = parse_finance_intent(
        "Please help me understand the information"
    )

    assert result.is_complete is False
    assert "analysis_type" in result.missing_fields
    assert result.clarification_question is not None
    assert "What analysis would you like to run" in (
        result.clarification_question
    )


def test_merge_period_reply_into_pending_intent() -> None:
    """Clarification reply should complete a pending variance request."""

    pending = parse_finance_intent(
        "Show Actual vs Budget"
    )

    completed = merge_finance_intent(
        pending,
        "January 2026",
    )

    assert completed.selected_flow == "variance"
    assert completed.comparison == "actual_vs_budget"
    assert completed.period.start_date == "2026-01-01"
    assert completed.period.end_date == "2026-01-31"
    assert completed.is_complete is True
    assert completed.missing_fields == ()
    assert completed.clarification_question is None


def test_merge_date_range_reply_into_pending_intent() -> None:
    """A clarification reply may provide a complete date range."""

    pending = parse_finance_intent(
        "Show Actual vs Budget"
    )

    completed = merge_finance_intent(
        pending,
        "from 1 January 2026 to 31 March 2026",
    )

    assert completed.period.start_date == "2026-01-01"
    assert completed.period.end_date == "2026-03-31"
    assert completed.period.granularity == "range"
    assert completed.is_complete is True


def test_merge_category_and_period_reply() -> None:
    """Reply may add both category and reporting period."""

    pending = parse_finance_intent(
        "Show KPI"
    )

    completed = merge_finance_intent(
        pending,
        "January 2026 for 3W",
    )

    assert completed.selected_flow == "kpi"
    assert completed.period.display_value == "January 2026"
    assert completed.category == "3W"
    assert completed.is_complete is True


def test_merge_preserves_existing_category() -> None:
    """Existing fields should remain when clarification adds only period."""

    pending = parse_finance_intent(
        "Show KPI for 2W"
    )

    completed = merge_finance_intent(
        pending,
        "January 2026",
    )

    assert completed.category == "2W"
    assert completed.period.display_value == "January 2026"
    assert completed.is_complete is True


def test_merge_preserves_requested_kpis() -> None:
    """Requested KPI slots should survive clarification merging."""

    pending = parse_finance_intent(
        "Show revenue and orders KPI"
    )

    completed = merge_finance_intent(
        pending,
        "January 2026",
    )

    assert "revenue" in completed.requested_kpis
    assert "orders" in completed.requested_kpis


def test_to_filters_returns_graph_compatible_values() -> None:
    """Complete intent should convert into graph-state filters."""

    result = parse_finance_intent(
        "Show Actual vs Budget for January 2026 for 3W"
    )

    filters = result.to_filters()

    assert filters == {
        "start_date": "2026-01-01",
        "end_date": "2026-01-31",
        "period": "January 2026",
        "period_granularity": "month",
        "category": "3W",
        "comparison": "actual_vs_budget",
    }


def test_to_filters_returns_date_range_values() -> None:
    """Date-range intent should preserve both date boundaries."""

    result = parse_finance_intent(
        "Show KPI from 1 January 2026 to 31 March 2026"
    )

    filters = result.to_filters()

    assert filters["start_date"] == "2026-01-01"
    assert filters["end_date"] == "2026-03-31"
    assert filters["period_granularity"] == "range"


def test_to_filters_includes_scenario() -> None:
    """Scenario name should be included in graph filters."""

    result = parse_finance_intent(
        "Show scenario analysis for January 2026 "
        "using Downside Case"
    )

    filters = result.to_filters()

    assert filters["scenario_name"] == "Downside Case"


def test_to_filters_includes_requested_kpis() -> None:
    """Explicitly requested KPIs should be passed to graph filters."""

    result = parse_finance_intent(
        "Show revenue and orders KPI for January 2026"
    )

    filters = result.to_filters()

    assert "requested_kpis" in filters
    assert "revenue" in filters["requested_kpis"]
    assert "orders" in filters["requested_kpis"]


def test_to_filters_omits_empty_values() -> None:
    """Incomplete values should not appear in graph filters."""

    intent = FinanceIntent(
        original_request="Show KPI",
        selected_flow="kpi",
        period=ParsedPeriod(),
    )

    assert intent.to_filters() == {}


def test_parser_normalizes_whitespace() -> None:
    """Extra spaces should not affect parsing."""

    result = parse_finance_intent(
        "  Show   Actual vs Budget   for January 2026  "
    )

    assert result.original_request == (
        "Show Actual vs Budget for January 2026"
    )
    assert result.is_complete is True


def test_invalid_calendar_date_is_not_accepted() -> None:
    """Invalid dates should remain unresolved."""

    result = parse_finance_intent(
        "Show KPI for 31 February 2026"
    )

    assert result.period.start_date is None
    assert result.is_complete is False
    assert "period" in result.missing_fields


def test_parse_request_must_be_string() -> None:
    """Parser should reject non-string requests."""

    with pytest.raises(
        TypeError,
        match="user_request must be a string",
    ):
        parse_finance_intent(
            123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "user_request",
    [
        "",
        "   ",
    ],
)
def test_parse_request_cannot_be_empty(
    user_request: str,
) -> None:
    """Parser should reject empty requests."""

    with pytest.raises(
        ValueError,
        match="user_request cannot be empty",
    ):
        parse_finance_intent(
            user_request
        )


def test_merge_requires_finance_intent() -> None:
    """Pending intent must use the FinanceIntent model."""

    with pytest.raises(
        TypeError,
        match="pending_intent must be a FinanceIntent",
    ):
        merge_finance_intent(
            {},  # type: ignore[arg-type]
            "January 2026",
        )


def test_merge_reply_cannot_be_empty() -> None:
    """Clarification reply must contain information."""

    pending = parse_finance_intent(
        "Show KPI"
    )

    with pytest.raises(
        ValueError,
        match="user_request cannot be empty",
    ):
        merge_finance_intent(
            pending,
            "   ",
        )