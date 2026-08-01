"""Finance dataset repositories for local CSV and Snowflake sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.integrations.snowflake_connection import SnowflakeConnectionFactory


class FinanceDataRepositoryError(RuntimeError):
    """Raised when a finance dataset cannot be loaded."""


class FinanceDataRepository(Protocol):
    """Dataset contract consumed by the active finance service."""

    def get_operations(self) -> pd.DataFrame: ...
    def get_budget(self) -> pd.DataFrame: ...
    def get_assumptions(self) -> pd.DataFrame: ...
    def get_corporate_expenses(self) -> pd.DataFrame: ...
    def get_budget_corporate_expenses(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FinanceDataPaths:
    """Paths used by the local CSV repository."""

    operations: Path
    budget: Path
    assumptions: Path
    corporate_expenses: Path | None = None
    budget_corporate_expenses: Path | None = None


class LocalCsvFinanceRepository:
    """Load finance datasets from the portfolio's local CSV files."""

    def __init__(self, paths: FinanceDataPaths) -> None:
        self.paths = paths

    def get_operations(self) -> pd.DataFrame:
        return self._read(self.paths.operations, "operations")

    def get_budget(self) -> pd.DataFrame:
        return self._read(self.paths.budget, "budget")

    def get_assumptions(self) -> pd.DataFrame:
        return self._read(self.paths.assumptions, "assumptions")

    def get_corporate_expenses(self) -> pd.DataFrame:
        return self._read_optional(
            self.paths.corporate_expenses, "corporate expenses"
        )

    def get_budget_corporate_expenses(self) -> pd.DataFrame:
        return self._read_optional(
            self.paths.budget_corporate_expenses,
            "budget corporate expenses",
        )

    @staticmethod
    def _read(path: Path, dataset: str) -> pd.DataFrame:
        if not path.exists():
            raise FinanceDataRepositoryError(
                f"Required {dataset} data file not found: {path}"
            )
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise FinanceDataRepositoryError(
                f"Unable to load {dataset} data."
            ) from exc

    @classmethod
    def _read_optional(
        cls, path: Path | None, dataset: str
    ) -> pd.DataFrame:
        if path is None:
            raise FinanceDataRepositoryError(
                f"{dataset.title()} data path is not configured."
            )
        return cls._read(path, dataset)


class SnowflakeFinanceRepository:
    """Load finance datasets as Pandas DataFrames using read-only SQL."""

    _TABLES = {
        "operations": "FINANCE_AI.RAW.ORDERS",
        "budget": "FINANCE_AI.RAW.BUDGET",
        "assumptions": "FINANCE_AI.RAW.BUSINESS_ASSUMPTIONS",
        "corporate_expenses": "FINANCE_AI.RAW.CORPORATE_EXPENSES",
        "budget_corporate_expenses": (
            "FINANCE_AI.RAW.BUDGET_CORPORATE_EXPENSES"
        ),
    }

    def __init__(self, factory: SnowflakeConnectionFactory) -> None:
        self._factory = factory

    def get_operations(self) -> pd.DataFrame:
        return self._read_table("operations")

    def get_budget(self) -> pd.DataFrame:
        return self._read_table("budget")

    def get_assumptions(self) -> pd.DataFrame:
        return self._read_table("assumptions")

    def get_corporate_expenses(self) -> pd.DataFrame:
        return self._read_table("corporate_expenses")

    def get_budget_corporate_expenses(self) -> pd.DataFrame:
        return self._read_table("budget_corporate_expenses")

    def _read_table(self, dataset: str) -> pd.DataFrame:
        table = self._TABLES[dataset]
        try:
            with self._factory.cursor() as cursor:
                cursor.execute(
                    f"SELECT * FROM {table}",
                    timeout=self._factory.config.query_timeout_seconds,
                )
                dataframe = cursor.fetch_pandas_all()
        except Exception as exc:
            raise FinanceDataRepositoryError(
                f"Unable to load {dataset} data from Snowflake."
            ) from exc
        dataframe.columns = [str(column).lower() for column in dataframe.columns]
        return self._normalize_for_agents(dataset, dataframe)

    @staticmethod
    def _normalize_for_agents(
        dataset: str,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match the dataframe shapes currently produced by local CSVs."""

        normalized = dataframe.copy()
        if dataset == "operations" and "order_date" in normalized:
            normalized["order_date"] = pd.to_datetime(
                normalized["order_date"], errors="coerce"
            ).dt.strftime("%d-%m-%Y")
        elif "month" in normalized:
            normalized["month"] = pd.to_datetime(
                normalized["month"], errors="coerce"
            ).dt.strftime("%Y-%m")

        text_columns = {
            "order_id", "pickup_cluster", "drop_cluster",
            "vehicle_category", "order_status", "assumption",
            "metric", "impact_type", "direction", "business_reason",
        }
        for column in text_columns.intersection(normalized.columns):
            normalized[column] = normalized[column].fillna("").astype(str)
        return normalized
