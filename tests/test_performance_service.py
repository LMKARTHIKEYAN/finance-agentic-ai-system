"""Tests for FP&A performance calculations."""

from datetime import date
from decimal import Decimal

from src.services.performance_service import PerformanceService


class FakePerformanceRepository:
    """Return deterministic Snowflake-style aggregate rows."""

    def get_monthly_performance(self, *, month, vehicle_category=None):
        assert month == date(2026, 4, 1)
        assert vehicle_category == "10 FT"
        return [
            {
                "month": date(2026, 4, 1),
                "vehicle_category": "10 FT",
                "actual_orders": 110,
                "actual_revenue": Decimal("1200"),
                "actual_cogs": Decimal("700"),
                "budget_orders": 100,
                "budget_revenue": Decimal("1000"),
                "budget_cogs": Decimal("650"),
            }
        ]


def test_service_calculates_variances() -> None:
    service = PerformanceService(FakePerformanceRepository())

    rows = service.get_performance(
        month=date(2026, 4, 1),
        vehicle_category="10 FT",
    )

    assert rows[0]["orders_variance"] == 10
    assert rows[0]["revenue_variance"] == Decimal("200")
    assert rows[0]["revenue_variance_pct"] == Decimal("20.00")
    assert rows[0]["cogs_variance"] == Decimal("50")


def test_service_returns_none_percentage_for_zero_budget() -> None:
    row = {
        "month": date(2026, 4, 1),
        "vehicle_category": "10 FT",
        "actual_orders": 1,
        "actual_revenue": 10,
        "actual_cogs": 1,
        "budget_orders": 0,
        "budget_revenue": 0,
        "budget_cogs": 0,
    }

    result = PerformanceService._calculate_row(row)

    assert result["revenue_variance_pct"] is None
