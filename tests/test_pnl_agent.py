"""
Unit tests for the Finance P&L Agent.

These tests validate:

- Actual P&L calculation
- Budget P&L calculation using budget_cogs
- Actual vs Budget variance
- Period summary calculation
- Completed-order filtering
- Month filtering
- Missing-column validation
- Invalid-date validation
- Invalid-month validation
- Duplicate-expense-month validation
- Negative-value validation
- Zero-revenue handling
- Non-matching-month handling
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.finance.pnl_agent import PnlAgent, PnlAnalysisResult


@pytest.fixture
def pnl_agent() -> PnlAgent:
    """Return a fresh P&L Agent for each test."""
    return PnlAgent()


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """
    Return sample actual order-level data.

    April completed-order totals:

        Revenue:
            1,000 + 500 = 1,500

        Direct cost:
            First order:
                600 + 50 + 10 + 5 + 20 = 685

            Second order:
                300 + 20 + 5 + 0 + 10 = 335

            Total direct cost:
                685 + 335 = 1,020

    The cancelled order must not be included.
    """
    return pd.DataFrame(
        {
            "order_date": [
                "01-04-2026",
                "15-04-2026",
                "20-04-2026",
                "05-05-2026",
            ],
            "order_status": [
                "completed",
                "Completed",
                "cancelled",
                "completed",
            ],
            "fare": [
                1_000.0,
                500.0,
                2_000.0,
                800.0,
            ],
            "partner_payout": [
                600.0,
                300.0,
                1_200.0,
                480.0,
            ],
            "incentive": [
                50.0,
                20.0,
                100.0,
                30.0,
            ],
            "goodwill": [
                10.0,
                5.0,
                20.0,
                5.0,
            ],
            "dry_run": [
                5.0,
                0.0,
                10.0,
                0.0,
            ],
            "surge": [
                20.0,
                10.0,
                40.0,
                15.0,
            ],
        }
    )


@pytest.fixture
def sample_actual_expenses() -> pd.DataFrame:
    """Return monthly actual corporate expenses."""
    return pd.DataFrame(
        {
            "month": [
                "2026-04",
                "2026-05",
            ],
            "sales_marketing": [
                100.0,
                80.0,
            ],
            "other_opex": [
                50.0,
                40.0,
            ],
            "depreciation": [
                20.0,
                20.0,
            ],
            "interest": [
                10.0,
                10.0,
            ],
        }
    )


@pytest.fixture
def sample_budget() -> pd.DataFrame:
    """
    Return category-level monthly budget data.

    April budget totals:

        Revenue:
            900 + 700 = 1,600

        COGS:
            550 + 450 = 1,000
    """
    return pd.DataFrame(
        {
            "month": [
                "2026-04",
                "2026-04",
                "2026-05",
            ],
            "vehicle_category": [
                "2W",
                "3W",
                "2W",
            ],
            "budget_orders": [
                10,
                8,
                9,
            ],
            "budget_revenue": [
                900.0,
                700.0,
                850.0,
            ],
            "budget_cogs": [
                550.0,
                450.0,
                520.0,
            ],
        }
    )


@pytest.fixture
def sample_budget_expenses() -> pd.DataFrame:
    """Return monthly budget corporate expenses."""
    return pd.DataFrame(
        {
            "month": [
                "2026-04",
                "2026-05",
            ],
            "sales_marketing": [
                90.0,
                75.0,
            ],
            "other_opex": [
                40.0,
                35.0,
            ],
            "depreciation": [
                20.0,
                20.0,
            ],
            "interest": [
                10.0,
                10.0,
            ],
        }
    )


def test_create_actual_pnl_calculates_correct_values(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Actual P&L should calculate all financial lines correctly."""
    result = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["revenue"] == pytest.approx(1_500.0)
    assert april["direct_cost"] == pytest.approx(1_020.0)
    assert april["gross_profit"] == pytest.approx(480.0)

    assert april["gross_margin_percentage"] == pytest.approx(
        32.0
    )

    assert april["sales_marketing"] == pytest.approx(100.0)
    assert april["other_opex"] == pytest.approx(50.0)

    assert april["ebitda"] == pytest.approx(330.0)
    assert april["depreciation"] == pytest.approx(20.0)
    assert april["ebit"] == pytest.approx(310.0)
    assert april["interest"] == pytest.approx(10.0)
    assert april["ebt"] == pytest.approx(300.0)


def test_create_actual_pnl_includes_only_completed_orders(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Cancelled orders must not contribute to actual revenue or cost."""
    result = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["revenue"] == pytest.approx(1_500.0)

    assert april["revenue"] != pytest.approx(
        3_500.0
    )


def test_create_actual_pnl_normalizes_order_status(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Order status matching should ignore case and surrounding spaces."""
    orders = sample_orders.copy()

    orders.loc[0, "order_status"] = " COMPLETED "
    orders.loc[1, "order_status"] = "Completed"

    result = pnl_agent.create_actual_pnl(
        orders_data=orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["revenue"] == pytest.approx(1_500.0)


def test_create_budget_pnl_aggregates_vehicle_categories(
    pnl_agent: PnlAgent,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Category-level budget rows should aggregate into monthly P&L."""
    result = pnl_agent.create_budget_pnl(
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["revenue"] == pytest.approx(1_600.0)
    assert april["direct_cost"] == pytest.approx(1_000.0)
    assert april["gross_profit"] == pytest.approx(600.0)

    assert april["gross_margin_percentage"] == pytest.approx(
        37.5
    )

    assert april["sales_marketing"] == pytest.approx(90.0)
    assert april["other_opex"] == pytest.approx(40.0)

    assert april["ebitda"] == pytest.approx(470.0)
    assert april["depreciation"] == pytest.approx(20.0)
    assert april["ebit"] == pytest.approx(450.0)
    assert april["interest"] == pytest.approx(10.0)
    assert april["ebt"] == pytest.approx(440.0)


def test_create_budget_pnl_uses_budget_cogs_as_direct_cost(
    pnl_agent: PnlAgent,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Budget COGS should become the direct-cost line in Budget P&L."""
    result = pnl_agent.create_budget_pnl(
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["direct_cost"] == pytest.approx(
        sample_budget.loc[
            sample_budget["month"] == "2026-04",
            "budget_cogs",
        ].sum()
    )


def test_calculate_variance_returns_correct_values(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Variance should equal Actual minus Budget."""
    actual_pnl = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    budget_pnl = pnl_agent.create_budget_pnl(
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    result = pnl_agent.calculate_variance(
        actual_pnl=actual_pnl,
        budget_pnl=budget_pnl,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april["revenue_actual"] == pytest.approx(1_500.0)
    assert april["revenue_budget"] == pytest.approx(1_600.0)
    assert april["revenue_variance"] == pytest.approx(-100.0)

    assert april[
        "revenue_variance_percentage"
    ] == pytest.approx(-6.25)

    assert april["direct_cost_variance"] == pytest.approx(20.0)

    assert april["gross_profit_variance"] == pytest.approx(
        -120.0
    )

    assert april["ebitda_variance"] == pytest.approx(-140.0)
    assert april["ebt_variance"] == pytest.approx(-140.0)


def test_gross_margin_variance_is_percentage_point_variance(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Gross margin variance should be reported in percentage points."""
    actual_pnl = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    budget_pnl = pnl_agent.create_budget_pnl(
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    result = pnl_agent.calculate_variance(
        actual_pnl=actual_pnl,
        budget_pnl=budget_pnl,
    )

    april = result.loc[
        result["month"] == "2026-04"
    ].iloc[0]

    assert april[
        "gross_margin_percentage_actual"
    ] == pytest.approx(32.0)

    assert april[
        "gross_margin_percentage_budget"
    ] == pytest.approx(37.5)

    assert april[
        "gross_margin_percentage_point_variance"
    ] == pytest.approx(-5.5)


def test_analyze_returns_complete_pnl_result(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Analyze should return chatbot-ready P&L analysis."""
    result = pnl_agent.analyze(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    assert isinstance(result, PnlAnalysisResult)

    assert len(result.actual_pnl) == 2
    assert len(result.budget_pnl) == 2
    assert len(result.variance_pnl) == 2

    assert result.available_months == [
        "2026-04",
        "2026-05",
    ]

    assert result.excluded_actual_months == []
    assert result.excluded_budget_months == []

    assert "actual" in result.summary
    assert "budget" in result.summary
    assert "variance" in result.summary


def test_analyze_summary_calculates_period_totals(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Summary should consolidate all comparable months."""
    result = pnl_agent.analyze(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    actual_summary = result.summary["actual"]
    budget_summary = result.summary["budget"]
    variance_summary = result.summary["variance"]

    assert actual_summary["revenue"] == pytest.approx(2_300.0)
    assert budget_summary["revenue"] == pytest.approx(2_450.0)

    assert variance_summary[
        "revenue_variance"
    ] == pytest.approx(-150.0)

    assert variance_summary[
        "revenue_variance_percentage"
    ] == pytest.approx(-6.12)


def test_analyze_applies_month_filter(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Analyze should support inclusive YYYY-MM month filters."""
    result = pnl_agent.analyze(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
        start_month="2026-04",
        end_month="2026-04",
    )

    assert result.available_months == ["2026-04"]
    assert len(result.actual_pnl) == 1
    assert len(result.budget_pnl) == 1
    assert len(result.variance_pnl) == 1


def test_actual_pnl_excludes_month_without_expenses(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Actual months without expenses should not produce incomplete P&L."""
    expenses = sample_actual_expenses.loc[
        sample_actual_expenses["month"] == "2026-04"
    ].copy()

    result = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=expenses,
    )

    assert result["month"].tolist() == ["2026-04"]


def test_analyze_reports_excluded_actual_months(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Actual months without budget should appear in result metadata."""
    budget = sample_budget.loc[
        sample_budget["month"] == "2026-04"
    ].copy()

    budget_expenses = sample_budget_expenses.loc[
        sample_budget_expenses["month"] == "2026-04"
    ].copy()

    result = pnl_agent.analyze(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
        budget_data=budget,
        budget_corporate_expenses_data=budget_expenses,
    )

    assert result.available_months == ["2026-04"]
    assert result.excluded_actual_months == ["2026-05"]
    assert result.excluded_budget_months == []


def test_zero_actual_revenue_sets_gross_margin_to_zero(
    pnl_agent: PnlAgent,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Zero revenue should not cause division-by-zero errors."""
    orders = pd.DataFrame(
        {
            "order_date": ["01-04-2026"],
            "order_status": ["completed"],
            "fare": [0.0],
            "partner_payout": [0.0],
            "incentive": [0.0],
            "goodwill": [0.0],
            "dry_run": [0.0],
            "surge": [0.0],
        }
    )

    result = pnl_agent.create_actual_pnl(
        orders_data=orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    april = result.iloc[0]

    assert april["revenue"] == pytest.approx(0.0)
    assert april[
        "gross_margin_percentage"
    ] == pytest.approx(0.0)


def test_zero_budget_returns_zero_variance_percentage(
    pnl_agent: PnlAgent,
) -> None:
    """Zero budget should return a safe zero variance percentage."""
    actual_pnl = pd.DataFrame(
        {
            "month": ["2026-04"],
            "revenue": [100.0],
            "direct_cost": [50.0],
            "gross_profit": [50.0],
            "gross_margin_percentage": [50.0],
            "sales_marketing": [10.0],
            "other_opex": [10.0],
            "ebitda": [30.0],
            "depreciation": [5.0],
            "ebit": [25.0],
            "interest": [5.0],
            "ebt": [20.0],
        }
    )

    budget_pnl = pd.DataFrame(
        {
            "month": ["2026-04"],
            "revenue": [0.0],
            "direct_cost": [0.0],
            "gross_profit": [0.0],
            "gross_margin_percentage": [0.0],
            "sales_marketing": [0.0],
            "other_opex": [0.0],
            "ebitda": [0.0],
            "depreciation": [0.0],
            "ebit": [0.0],
            "interest": [0.0],
            "ebt": [0.0],
        }
    )

    result = pnl_agent.calculate_variance(
        actual_pnl=actual_pnl,
        budget_pnl=budget_pnl,
    )

    assert result.loc[
        0,
        "revenue_variance_percentage",
    ] == pytest.approx(0.0)

    assert result.loc[
        0,
        "ebt_variance_percentage",
    ] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("missing_column", "dataset_name"),
    [
        ("fare", "Orders"),
        ("partner_payout", "Orders"),
        ("order_status", "Orders"),
    ],
)
def test_actual_pnl_rejects_missing_order_columns(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    missing_column: str,
    dataset_name: str,
) -> None:
    """Actual P&L should reject missing required order columns."""
    invalid_orders = sample_orders.drop(
        columns=[missing_column]
    )

    with pytest.raises(
        ValueError,
        match=dataset_name,
    ):
        pnl_agent.create_actual_pnl(
            orders_data=invalid_orders,
            corporate_expenses_data=sample_actual_expenses,
        )


@pytest.mark.parametrize(
    "missing_column",
    [
        "month",
        "budget_revenue",
        "budget_cogs",
    ],
)
def test_budget_pnl_rejects_missing_budget_columns(
    pnl_agent: PnlAgent,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
    missing_column: str,
) -> None:
    """Budget P&L should require budget revenue and budget COGS."""
    invalid_budget = sample_budget.drop(
        columns=[missing_column]
    )

    with pytest.raises(
        ValueError,
        match="Budget is missing required columns",
    ):
        pnl_agent.create_budget_pnl(
            budget_data=invalid_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
        )


def test_actual_pnl_rejects_invalid_order_date(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Invalid order dates should raise a clear validation error."""
    invalid_orders = sample_orders.copy()
    invalid_orders.loc[0, "order_date"] = "2026/99/99"

    with pytest.raises(
        ValueError,
        match="invalid order_date",
    ):
        pnl_agent.create_actual_pnl(
            orders_data=invalid_orders,
            corporate_expenses_data=sample_actual_expenses,
        )


def test_budget_pnl_rejects_invalid_month(
    pnl_agent: PnlAgent,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Invalid budget month values should be rejected."""
    invalid_budget = sample_budget.copy()
    invalid_budget.loc[0, "month"] = "April-2026"

    with pytest.raises(
        ValueError,
        match="invalid month",
    ):
        pnl_agent.create_budget_pnl(
            budget_data=invalid_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
        )


def test_actual_expenses_reject_duplicate_months(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Corporate expenses should contain only one row per month."""
    duplicate_row = sample_actual_expenses.iloc[[0]]

    invalid_expenses = pd.concat(
        [
            sample_actual_expenses,
            duplicate_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate months",
    ):
        pnl_agent.create_actual_pnl(
            orders_data=sample_orders,
            corporate_expenses_data=invalid_expenses,
        )


def test_actual_pnl_rejects_negative_financial_values(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Negative actual revenue and cost input values should be rejected."""
    invalid_orders = sample_orders.copy()
    invalid_orders.loc[0, "fare"] = -1_000.0

    with pytest.raises(
        ValueError,
        match="negative values",
    ):
        pnl_agent.create_actual_pnl(
            orders_data=invalid_orders,
            corporate_expenses_data=sample_actual_expenses,
        )


def test_budget_pnl_rejects_negative_budget_cogs(
    pnl_agent: PnlAgent,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Negative budget COGS should be rejected."""
    invalid_budget = sample_budget.copy()
    invalid_budget.loc[0, "budget_cogs"] = -500.0

    with pytest.raises(
        ValueError,
        match="negative values",
    ):
        pnl_agent.create_budget_pnl(
            budget_data=invalid_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
        )


def test_actual_pnl_rejects_non_numeric_values(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Text inside financial columns should be rejected."""
    invalid_orders = sample_orders.copy()

    invalid_orders["fare"] = invalid_orders["fare"].astype(
        "object"
    )
    invalid_orders.loc[0, "fare"] = "invalid"

    with pytest.raises(
        ValueError,
        match="non-numeric",
    ):
        pnl_agent.create_actual_pnl(
            orders_data=invalid_orders,
            corporate_expenses_data=sample_actual_expenses,
        )


def test_analyze_rejects_invalid_month_range(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Start month cannot be after end month."""
    with pytest.raises(
        ValueError,
        match="start_month cannot be later",
    ):
        pnl_agent.analyze(
            orders_data=sample_orders,
            corporate_expenses_data=sample_actual_expenses,
            budget_data=sample_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
            start_month="2026-06",
            end_month="2026-04",
        )


def test_analyze_rejects_invalid_month_format(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Month filters should use YYYY-MM format."""
    with pytest.raises(
        ValueError,
        match="start_month must use YYYY-MM",
    ):
        pnl_agent.analyze(
            orders_data=sample_orders,
            corporate_expenses_data=sample_actual_expenses,
            budget_data=sample_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
            start_month="April-2026",
        )


def test_create_actual_pnl_rejects_no_completed_orders(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """A dataset containing no completed orders should be rejected."""
    invalid_orders = sample_orders.copy()
    invalid_orders["order_status"] = "cancelled"

    with pytest.raises(
        ValueError,
        match="no completed orders",
    ):
        pnl_agent.create_actual_pnl(
            orders_data=invalid_orders,
            corporate_expenses_data=sample_actual_expenses,
        )


def test_actual_and_budget_without_common_months_raise_error(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Full analysis requires at least one comparable month."""
    shifted_budget = sample_budget.copy()
    shifted_budget["month"] = "2027-01"

    shifted_budget_expenses = sample_budget_expenses.copy()
    shifted_budget_expenses["month"] = [
        "2027-01",
        "2027-02",
    ]

    with pytest.raises(
        ValueError,
        match="no common months",
    ):
        pnl_agent.analyze(
            orders_data=sample_orders,
            corporate_expenses_data=sample_actual_expenses,
            budget_data=shifted_budget,
            budget_corporate_expenses_data=(
                shifted_budget_expenses
            ),
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        [],
        {},
        "orders",
    ],
)
def test_analyze_rejects_non_dataframe_input(
    pnl_agent: PnlAgent,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
    invalid_input: object,
) -> None:
    """Analyze should require pandas DataFrames."""
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        pnl_agent.analyze(
            orders_data=invalid_input,  # type: ignore[arg-type]
            corporate_expenses_data=sample_actual_expenses,
            budget_data=sample_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
        )


def test_analyze_rejects_empty_dataframe(
    pnl_agent: PnlAgent,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """Empty input DataFrames should be rejected."""
    with pytest.raises(
        ValueError,
        match="Orders cannot be empty",
    ):
        pnl_agent.analyze(
            orders_data=pd.DataFrame(),
            corporate_expenses_data=sample_actual_expenses,
            budget_data=sample_budget,
            budget_corporate_expenses_data=(
                sample_budget_expenses
            ),
        )


def test_pnl_output_has_expected_columns(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
) -> None:
    """Actual P&L output should follow the standard statement structure."""
    result = pnl_agent.create_actual_pnl(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
    )

    assert result.columns.tolist() == [
        "month",
        "revenue",
        "direct_cost",
        "gross_profit",
        "gross_margin_percentage",
        "sales_marketing",
        "other_opex",
        "ebitda",
        "depreciation",
        "ebit",
        "interest",
        "ebt",
    ]


def test_input_dataframes_are_not_modified(
    pnl_agent: PnlAgent,
    sample_orders: pd.DataFrame,
    sample_actual_expenses: pd.DataFrame,
    sample_budget: pd.DataFrame,
    sample_budget_expenses: pd.DataFrame,
) -> None:
    """The agent should not mutate caller-owned DataFrames."""
    original_orders = sample_orders.copy(deep=True)
    original_actual_expenses = sample_actual_expenses.copy(
        deep=True
    )
    original_budget = sample_budget.copy(deep=True)
    original_budget_expenses = sample_budget_expenses.copy(
        deep=True
    )

    pnl_agent.analyze(
        orders_data=sample_orders,
        corporate_expenses_data=sample_actual_expenses,
        budget_data=sample_budget,
        budget_corporate_expenses_data=sample_budget_expenses,
    )

    pd.testing.assert_frame_equal(
        sample_orders,
        original_orders,
    )

    pd.testing.assert_frame_equal(
        sample_actual_expenses,
        original_actual_expenses,
    )

    pd.testing.assert_frame_equal(
        sample_budget,
        original_budget,
    )

    pd.testing.assert_frame_equal(
        sample_budget_expenses,
        original_budget_expenses,
    )