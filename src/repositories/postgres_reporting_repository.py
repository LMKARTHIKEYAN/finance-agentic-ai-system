"""Read-only PostgreSQL datasets for the finance reporting pipelines."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.integrations.postgres_connection import (
    PostgresConnectionError,
    PostgresConnectionFactory,
)


class ReportingRepositoryError(RuntimeError):
    """Raised when a reporting dataset cannot be read."""


class PostgresReportingRepository:
    """Return agent-ready data frames from the local PostgreSQL tables."""

    def __init__(self, factory: PostgresConnectionFactory) -> None:
        if not isinstance(factory, PostgresConnectionFactory):
            raise TypeError("factory must be a PostgresConnectionFactory.")
        self._factory = factory

    def latest_order_date(self) -> date | None:
        frame = self._read(
            "SELECT MAX(order_date) AS latest_order_date FROM public.orders",
            (),
        )
        if frame.empty or pd.isna(frame.iloc[0]["latest_order_date"]):
            return None
        return pd.Timestamp(frame.iloc[0]["latest_order_date"]).date()

    def get_orders(self, start_date: date, end_date: date) -> pd.DataFrame:
        self._validate_range(start_date, end_date)
        return self._read(
            """
            SELECT *
            FROM public.orders
            WHERE order_date BETWEEN %s AND %s
            ORDER BY order_date, order_id
            """,
            (start_date, end_date),
        )

    def get_budget(self, start_month: str, end_month: str) -> pd.DataFrame:
        return self._read_monthly_table(
            "public.budget", start_month, end_month, "month, vehicle_category"
        )

    def get_corporate_expenses(
        self,
        start_month: str,
        end_month: str,
    ) -> pd.DataFrame:
        return self._read_monthly_table(
            "public.corporate_expenses", start_month, end_month, "month"
        )

    def get_budget_corporate_expenses(
        self,
        start_month: str,
        end_month: str,
    ) -> pd.DataFrame:
        return self._read_monthly_table(
            "public.budget_corporate_expenses", start_month, end_month, "month"
        )

    def _read_monthly_table(
        self,
        table: str,
        start_month: str,
        end_month: str,
        order_by: str,
    ) -> pd.DataFrame:
        start, end = self._month_range(start_month, end_month)
        frame = self._read(
            f"SELECT * FROM {table} WHERE month BETWEEN %s AND %s ORDER BY {order_by}",
            (start, end),
        )
        return self._normalize_month_column(frame)

    def _read(self, query: str, params: tuple[Any, ...]) -> pd.DataFrame:
        try:
            with self._factory.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    columns = [
                        str(column.name).lower()
                        for column in (cursor.description or [])
                    ]
        except PostgresConnectionError:
            raise
        except Exception as exc:
            raise ReportingRepositoryError(
                "PostgreSQL reporting query failed."
            ) from exc
        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _normalize_month_column(frame: pd.DataFrame) -> pd.DataFrame:
        if "month" not in frame.columns or frame.empty:
            return frame
        result = frame.copy()
        parsed = pd.to_datetime(result["month"], errors="coerce")
        if parsed.isna().any():
            raise ReportingRepositoryError(
                "PostgreSQL returned invalid month values."
            )
        result["month"] = parsed.dt.strftime("%Y-%m")
        return result

    @staticmethod
    def _validate_range(start_date: date, end_date: date) -> None:
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            raise TypeError("start_date and end_date must be date values.")
        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date.")

    @staticmethod
    def _month_range(start_month: str, end_month: str) -> tuple[date, date]:
        try:
            start = pd.Period(start_month, freq="M").start_time.date()
            end = pd.Period(end_month, freq="M").start_time.date()
        except Exception as exc:
            raise ValueError("Months must use YYYY-MM format.") from exc
        if start > end:
            raise ValueError("start_month cannot be after end_month.")
        return start, end
