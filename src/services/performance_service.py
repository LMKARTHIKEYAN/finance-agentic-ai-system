"""FP&A calculations for the performance API."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Mapping, Protocol


class PerformanceDataProvider(Protocol):
    """Repository contract required by the performance service."""

    def get_monthly_performance(
        self,
        *,
        month: date,
        vehicle_category: str | None = None,
    ) -> list[Mapping[str, Any]]: ...


class PerformanceService:
    """Calculate API-ready FP&A variances from Snowflake aggregates."""

    def __init__(self, repository: PerformanceDataProvider) -> None:
        self._repository = repository

    def get_performance(
        self,
        *,
        month: date,
        vehicle_category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return actual, budget, and variance metrics by category."""

        rows = self._repository.get_monthly_performance(
            month=month,
            vehicle_category=vehicle_category,
        )
        return [self._calculate_row(row) for row in rows]

    @classmethod
    def _calculate_row(cls, row: Mapping[str, Any]) -> dict[str, Any]:
        actual_orders = int(row.get("actual_orders") or 0)
        budget_orders = int(row.get("budget_orders") or 0)
        actual_revenue = cls._decimal(row.get("actual_revenue"))
        budget_revenue = cls._decimal(row.get("budget_revenue"))
        actual_cogs = cls._decimal(row.get("actual_cogs"))
        budget_cogs = cls._decimal(row.get("budget_cogs"))

        return {
            "month": row["month"],
            "vehicle_category": str(row["vehicle_category"]),
            "actual_orders": actual_orders,
            "budget_orders": budget_orders,
            "orders_variance": actual_orders - budget_orders,
            "actual_revenue": actual_revenue,
            "budget_revenue": budget_revenue,
            "revenue_variance": actual_revenue - budget_revenue,
            "revenue_variance_pct": cls._percentage(
                actual_revenue - budget_revenue,
                budget_revenue,
            ),
            "actual_cogs": actual_cogs,
            "budget_cogs": budget_cogs,
            "cogs_variance": actual_cogs - budget_cogs,
        }

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        return Decimal(str(value))

    @staticmethod
    def _percentage(value: Decimal, base: Decimal) -> Decimal | None:
        if base == 0:
            return None
        return (value / base * Decimal("100")).quantize(
            Decimal("0.01")
        )
