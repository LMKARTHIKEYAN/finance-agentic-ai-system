"""Read-only Snowflake access for monthly FP&A performance data."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from src.integrations.snowflake_connection import (
    SnowflakeConnectionError,
    SnowflakeConnectionFactory,
    SnowflakeQueryError,
)


class PerformanceRepositoryError(RuntimeError):
    """Raised when performance data cannot be read from Snowflake."""


class SnowflakePerformanceRepository:
    """Read aggregated actual and budget metrics from Snowflake."""

    _PERFORMANCE_QUERY = """
        WITH actual AS (
            SELECT
                DATE_TRUNC('MONTH', order_date)::DATE AS month,
                vehicle_category,
                COUNT(*) AS actual_orders,
                SUM(COALESCE(commission_amount, 0)) AS actual_revenue,
                SUM(
                    COALESCE(incentive, 0)
                    + COALESCE(goodwill, 0)
                    + COALESCE(dry_run, 0)
                    + COALESCE(surge, 0)
                ) AS actual_cogs
            FROM FINANCE_AI.RAW.ORDERS
            WHERE DATE_TRUNC('MONTH', order_date)::DATE = %s
              AND (%s IS NULL OR vehicle_category = %s)
              AND UPPER(COALESCE(order_status, '')) = 'COMPLETED'
            GROUP BY 1, 2
        ),
        budget AS (
            SELECT
                month,
                vehicle_category,
                SUM(COALESCE(budget_orders, 0)) AS budget_orders,
                SUM(COALESCE(budget_revenue, 0)) AS budget_revenue,
                SUM(COALESCE(budget_cogs, 0)) AS budget_cogs
            FROM FINANCE_AI.RAW.BUDGET
            WHERE month = %s
              AND (%s IS NULL OR vehicle_category = %s)
            GROUP BY 1, 2
        )
        SELECT
            COALESCE(actual.month, budget.month) AS month,
            COALESCE(
                actual.vehicle_category,
                budget.vehicle_category
            ) AS vehicle_category,
            COALESCE(actual.actual_orders, 0) AS actual_orders,
            COALESCE(actual.actual_revenue, 0) AS actual_revenue,
            COALESCE(actual.actual_cogs, 0) AS actual_cogs,
            COALESCE(budget.budget_orders, 0) AS budget_orders,
            COALESCE(budget.budget_revenue, 0) AS budget_revenue,
            COALESCE(budget.budget_cogs, 0) AS budget_cogs
        FROM actual
        FULL OUTER JOIN budget
          ON actual.month = budget.month
         AND actual.vehicle_category = budget.vehicle_category
        ORDER BY vehicle_category
    """

    def __init__(self, factory: SnowflakeConnectionFactory) -> None:
        if not isinstance(factory, SnowflakeConnectionFactory):
            raise TypeError("factory must be a SnowflakeConnectionFactory.")
        self._factory = factory

    def get_monthly_performance(
        self,
        *,
        month: date,
        vehicle_category: str | None = None,
    ) -> list[Mapping[str, Any]]:
        """Return monthly actual and budget values using one read query."""

        category = vehicle_category.strip() if vehicle_category else None
        params = (
            month,
            category,
            category,
            month,
            category,
            category,
        )

        try:
            with self._factory.cursor() as cursor:
                cursor.execute(
                    self._PERFORMANCE_QUERY,
                    params,
                    timeout=self._factory.config.query_timeout_seconds,
                )
                rows = cursor.fetchall()
                description = cursor.description or []
        except (SnowflakeConnectionError, SnowflakeQueryError):
            raise
        except Exception as exc:
            raise PerformanceRepositoryError(
                "Snowflake performance query failed."
            ) from exc

        columns = [str(column[0]).lower() for column in description]
        return [
            dict(zip(columns, row, strict=False))
            for row in rows
        ]
