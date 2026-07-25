"""Tests for mix, price, and cost GP% decomposition."""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.finance.gp_variance_agent import GrossProfitVarianceAgent


@pytest.fixture
def orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["A1", "A2", "B1", "X1"],
            "order_date": ["01-04-2026"] * 4,
            "vehicle_category": ["A", "A", "B", "A"],
            "order_status": ["completed", "Completed", "completed", "cancelled"],
            "commission_amount": [120.0, 120.0, 80.0, 999.0],
            "incentive": [20.0, 20.0, 10.0, 0.0],
            "goodwill": [0.0] * 4,
            "dry_run": [0.0] * 4,
            "surge": [0.0] * 4,
        }
    )


@pytest.fixture
def budget() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["2026-04", "2026-04"],
            "vehicle_category": ["A", "B"],
            "budget_orders": [1.0, 3.0],
            "budget_revenue": [100.0, 300.0],
            "budget_cogs": [50.0, 210.0],
        }
    )


def test_decomposition_uses_commission_and_reconciles(
    orders: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    result = GrossProfitVarianceAgent().analyze(
        orders,
        budget,
        start_month="2026-04",
        end_month="2026-04",
    )

    assert result.reconciliation_status == "PASS"
    assert result.reconciliation_difference == pytest.approx(0.0)
    assert result.available_months == ["2026-04"]
    assert (
        result.mix_effect_percentage_points
        + result.price_effect_percentage_points
        + result.cost_effect_percentage_points
    ) == pytest.approx(result.total_variance_percentage_points, abs=0.0002)
    category_a = next(
        row for row in result.category_analysis
        if row["vehicle_category"] == "A"
    )
    assert category_a["actual_revenue"] == pytest.approx(240.0)
    assert category_a["actual_price_per_unit"] == pytest.approx(120.0)
    assert category_a["actual_cost_per_unit"] == pytest.approx(20.0)


def test_decomposition_reports_unmatched_categories(
    orders: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    extra = budget.copy()
    extra.loc[len(extra)] = ["2026-04", "C", 1.0, 100.0, 60.0]
    result = GrossProfitVarianceAgent().analyze(orders, extra)
    assert result.excluded_budget_categories == [
        {"month": "2026-04", "vehicle_category": "C"}
    ]


def test_decomposition_rejects_missing_commission(
    orders: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="commission_amount"):
        GrossProfitVarianceAgent().analyze(
            orders.drop(columns=["commission_amount"]),
            budget,
        )


def test_decomposition_respects_month_filter(
    orders: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="no comparable"):
        GrossProfitVarianceAgent().analyze(
            orders,
            budget,
            start_month="2026-05",
            end_month="2026-05",
        )
