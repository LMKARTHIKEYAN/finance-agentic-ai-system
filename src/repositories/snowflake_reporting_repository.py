"""Configurable read-only Snowflake datasets for scheduled finance reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.integrations.snowflake_connection import SnowflakeConnectionFactory


class ReportingRepositoryError(RuntimeError):
    """Raised when a scheduled-report dataset cannot be read."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")


@dataclass(frozen=True)
class ReportingTables:
    """Snowflake objects used by the reporting pipelines."""

    orders: str = "FINANCE_AI.RAW.ORDERS"
    corporate_expenses: str = "FINANCE_AI.RAW.CORPORATE_EXPENSES"
    budget: str = "FINANCE_AI.RAW.BUDGET"
    budget_corporate_expenses: str = "FINANCE_AI.RAW.BUDGET_CORPORATE_EXPENSES"

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value.strip()):
                raise ValueError(f"{name} must be a valid Snowflake identifier.")


class SnowflakeReportingRepository:
    """Return raw, agent-ready pandas frames using parameterized date filters."""

    def __init__(self, factory: SnowflakeConnectionFactory, tables: ReportingTables | None = None) -> None:
        if not isinstance(factory, SnowflakeConnectionFactory):
            raise TypeError("factory must be a SnowflakeConnectionFactory.")
        self._factory = factory
        self.tables = tables or ReportingTables()

    def latest_order_date(self) -> date | None:
        frame = self._read(
            f"SELECT MAX(order_date)::DATE AS latest_order_date FROM {self.tables.orders}",
            (),
        )
        if frame.empty or pd.isna(frame.iloc[0]["latest_order_date"]):
            return None
        return pd.Timestamp(frame.iloc[0]["latest_order_date"]).date()

    def get_orders(self, start_date: date, end_date: date) -> pd.DataFrame:
        self._validate_range(start_date, end_date)
        return self._read(
            f"SELECT * FROM {self.tables.orders} WHERE order_date::DATE BETWEEN %s AND %s ORDER BY order_date",
            (start_date, end_date),
        )

    def get_budget(self, start_month: str, end_month: str) -> pd.DataFrame:
        start, end = self._month_range(start_month, end_month)
        return self._normalize_month_column(self._read(
            f"SELECT * FROM {self.tables.budget} WHERE month::DATE BETWEEN %s AND %s ORDER BY month, vehicle_category",
            (start, end),
        ))

    def get_corporate_expenses(self, start_month: str, end_month: str) -> pd.DataFrame:
        start, end = self._month_range(start_month, end_month)
        return self._normalize_month_column(self._read(
            f"SELECT * FROM {self.tables.corporate_expenses} WHERE month::DATE BETWEEN %s AND %s ORDER BY month",
            (start, end),
        ))

    def get_budget_corporate_expenses(self, start_month: str, end_month: str) -> pd.DataFrame:
        start, end = self._month_range(start_month, end_month)
        return self._normalize_month_column(self._read(
            f"SELECT * FROM {self.tables.budget_corporate_expenses} WHERE month::DATE BETWEEN %s AND %s ORDER BY month",
            (start, end),
        ))

    def _read(self, query: str, params: tuple[Any, ...]) -> pd.DataFrame:
        try:
            with self._factory.cursor() as cursor:
                cursor.execute(query, params, timeout=self._factory.config.query_timeout_seconds)
                frame = cursor.fetch_pandas_all()
        except Exception as exc:
            raise ReportingRepositoryError("Snowflake reporting query failed.") from exc
        if not isinstance(frame, pd.DataFrame):
            raise ReportingRepositoryError("Snowflake did not return a pandas DataFrame.")
        result = frame.copy()
        result.columns = [str(column).lower() for column in result.columns]
        return result

    @staticmethod
    def _normalize_month_column(frame: pd.DataFrame) -> pd.DataFrame:
        """Convert Snowflake DATE month keys to the agents' YYYY-MM contract."""
        if "month" not in frame.columns or frame.empty:
            return frame
        result = frame.copy()
        parsed = pd.to_datetime(result["month"], errors="coerce")
        if parsed.isna().any():
            raise ReportingRepositoryError("Snowflake returned invalid month values.")
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
