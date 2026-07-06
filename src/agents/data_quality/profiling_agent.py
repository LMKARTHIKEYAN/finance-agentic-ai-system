"""
Profiling agent for the Finance Agentic AI System.

This module creates summary profiles for operational and budget data.
It helps users understand the structure, quality, and basic statistics
of the dataset before business analysis starts.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataProfile:
    """
    Stores the profile summary of a dataset.

    Attributes:
        total_rows: Number of rows in the dataset.
        total_columns: Number of columns in the dataset.
        duplicate_rows: Number of fully duplicated rows.
        missing_values: Missing value count by column.
        data_types: Data type of each column.
        summary_statistics: Basic statistics for numeric columns.
        category_distributions: Value distributions for selected columns.
    """

    total_rows: int
    total_columns: int
    duplicate_rows: int
    missing_values: dict[str, int] = field(default_factory=dict)
    data_types: dict[str, str] = field(default_factory=dict)
    summary_statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    category_distributions: dict[str, dict[str, int]] = field(default_factory=dict)


class ProfilingAgent:
    """
    Agent responsible for profiling FP&A datasets.

    This agent creates useful summary information about operational and
    budget data, such as row count, missing values, duplicate rows,
    data types, numeric statistics, and category distributions.
    """

    OPERATIONS_CATEGORY_COLUMNS = [
        "vehicle_category",
        "order_status",
        "pickup_cluster",
        "drop_cluster",
    ]

    BUDGET_CATEGORY_COLUMNS = [
        "vehicle_category",
    ]

    @staticmethod
    def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names into lowercase snake_case format.

        Args:
            dataframe: Input pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with normalized column names.
        """
        profiled_dataframe = dataframe.copy()
        profiled_dataframe.columns = (
            profiled_dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return profiled_dataframe

    @classmethod
    def profile_operations_data(cls, dataframe: pd.DataFrame) -> DataProfile:
        """
        Create a data profile for operational order data.

        Args:
            dataframe: Operational orders DataFrame.

        Returns:
            DataProfile: Summary profile of operational data.
        """
        dataframe = cls.normalize_columns(dataframe)

        return cls._create_profile(
            dataframe=dataframe,
            category_columns=cls.OPERATIONS_CATEGORY_COLUMNS,
        )

    @classmethod
    def profile_budget_data(cls, dataframe: pd.DataFrame) -> DataProfile:
        """
        Create a data profile for budget planning data.

        Args:
            dataframe: Budget DataFrame.

        Returns:
            DataProfile: Summary profile of budget data.
        """
        dataframe = cls.normalize_columns(dataframe)

        return cls._create_profile(
            dataframe=dataframe,
            category_columns=cls.BUDGET_CATEGORY_COLUMNS,
        )

    @staticmethod
    def _create_profile(
        dataframe: pd.DataFrame,
        category_columns: list[str],
    ) -> DataProfile:
        """
        Create a reusable data profile for a DataFrame.

        Args:
            dataframe: Input pandas DataFrame.
            category_columns: Columns for value distribution checks.

        Returns:
            DataProfile: Summary profile of the dataset.
        """
        total_rows = len(dataframe)
        total_columns = len(dataframe.columns)
        duplicate_rows = int(dataframe.duplicated().sum())

        missing_values = {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
            if count > 0
        }

        data_types = {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        }

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns

        summary_statistics = {}

        for column in numeric_columns:
            summary_statistics[column] = {
                "min": float(dataframe[column].min()),
                "max": float(dataframe[column].max()),
                "mean": float(dataframe[column].mean()),
                "sum": float(dataframe[column].sum()),
            }

        category_distributions = {}

        for column in category_columns:
            if column not in dataframe.columns:
                continue

            category_distributions[column] = {
                str(key): int(value)
                for key, value in dataframe[column]
                .value_counts(dropna=False)
                .head(10)
                .items()
            }

        return DataProfile(
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows=duplicate_rows,
            missing_values=missing_values,
            data_types=data_types,
            summary_statistics=summary_statistics,
            category_distributions=category_distributions,
        )