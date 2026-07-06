"""
Validation agent for the Finance Agentic AI System.

This module validates operational and budget data before the data is used
by other analytics, finance, forecasting, and reporting agents.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    """
    Stores the result of a validation check.

    Attributes:
        is_valid: True if validation passed, otherwise False.
        errors: Critical issues that must be fixed before analysis.
        warnings: Non-critical issues that should be reviewed.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ValidationAgent:
    """
    Agent responsible for validating FP&A input data.

    This agent checks operational data and budget data before sending them
    to downstream agents such as operations analysis, variance analysis,
    forecasting, and reporting.
    """

    OPERATIONS_REQUIRED_COLUMNS = {
        "order_id",
        "order_date",
        "pickup_cluster",
        "drop_cluster",
        "vehicle_category",
        "order_status",
        "fare",
        "commission_amount",
        "partner_payout",
        "incentive",
        "goodwill",
        "dry_run",
        "surge",
        "trip_distance_km",
        "delivery_time_minutes",
    }

    BUDGET_REQUIRED_COLUMNS = {
        "month",
        "vehicle_category",
        "budget_orders",
        "budget_revenue",
    }

    VALID_ORDER_STATUSES = {
        "Completed",
        "Cancelled",
        "Rejected",
    }

    @staticmethod
    def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame column names.

        This method converts column names into lowercase snake_case format.

        Example:
            "Order Date" becomes "order_date".

        Args:
            dataframe: Input pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with normalized column names.
        """
        normalized_dataframe = dataframe.copy()
        normalized_dataframe.columns = (
            normalized_dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return normalized_dataframe

    @classmethod
    def validate_operations_data(cls, dataframe: pd.DataFrame) -> ValidationResult:
        """
        Validate operational order data.

        Args:
            dataframe: Operational orders DataFrame.

        Returns:
            ValidationResult: Validation status with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        dataframe = cls.normalize_columns(dataframe)

        missing_columns = cls.OPERATIONS_REQUIRED_COLUMNS - set(dataframe.columns)
        if missing_columns:
            errors.append(
                f"Missing operations columns: {sorted(missing_columns)}"
            )

        if dataframe.empty:
            errors.append("Operations data is empty.")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
            )

        cls._validate_duplicate_order_ids(dataframe, errors)
        cls._validate_required_missing_values(
            dataframe=dataframe,
            required_columns=cls.OPERATIONS_REQUIRED_COLUMNS,
            warnings=warnings,
        )
        cls._validate_order_dates(dataframe, errors)
        cls._validate_order_status(dataframe, warnings)
        cls._validate_numeric_columns(
            dataframe=dataframe,
            numeric_columns=[
                "fare",
                "commission_amount",
                "partner_payout",
                "incentive",
                "goodwill",
                "dry_run",
                "surge",
                "trip_distance_km",
                "delivery_time_minutes",
            ],
            errors=errors,
        )

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def validate_budget_data(cls, dataframe: pd.DataFrame) -> ValidationResult:
        """
        Validate budget planning data.

        Args:
            dataframe: Budget DataFrame.

        Returns:
            ValidationResult: Validation status with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        dataframe = cls.normalize_columns(dataframe)

        missing_columns = cls.BUDGET_REQUIRED_COLUMNS - set(dataframe.columns)
        if missing_columns:
            errors.append(
                f"Missing budget columns: {sorted(missing_columns)}"
            )

        if dataframe.empty:
            errors.append("Budget data is empty.")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
            )

        cls._validate_required_missing_values(
            dataframe=dataframe,
            required_columns=cls.BUDGET_REQUIRED_COLUMNS,
            warnings=warnings,
        )
        cls._validate_budget_month(dataframe, errors)
        cls._validate_numeric_columns(
            dataframe=dataframe,
            numeric_columns=[
                "budget_orders",
                "budget_revenue",
            ],
            errors=errors,
        )

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def _validate_duplicate_order_ids(
        dataframe: pd.DataFrame,
        errors: list[str],
    ) -> None:
        """
        Validate duplicate order IDs.

        Args:
            dataframe: Operational orders DataFrame.
            errors: Error list to update.
        """
        if "order_id" not in dataframe.columns:
            return

        duplicate_count = dataframe["order_id"].duplicated().sum()

        if duplicate_count > 0:
            errors.append(f"Duplicate order_id values found: {duplicate_count}")

    @staticmethod
    def _validate_required_missing_values(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        warnings: list[str],
    ) -> None:
        """
        Validate missing values in required columns.

        Args:
            dataframe: Input pandas DataFrame.
            required_columns: Required column names.
            warnings: Warning list to update.
        """
        for column in required_columns:
            if column not in dataframe.columns:
                continue

            missing_count = dataframe[column].isna().sum()

            if missing_count > 0:
                warnings.append(
                    f"Column '{column}' has {missing_count} missing values."
                )

    @staticmethod
    def _validate_order_dates(
        dataframe: pd.DataFrame,
        errors: list[str],
    ) -> None:
        """
        Validate order date column.

        Args:
            dataframe: Operational orders DataFrame.
            errors: Error list to update.
        """
        if "order_date" not in dataframe.columns:
            return

        parsed_dates = pd.to_datetime(
            dataframe["order_date"],
            errors="coerce",
            dayfirst=True,
        )

        invalid_dates = parsed_dates.isna().sum()

        if invalid_dates > 0:
            errors.append(f"Invalid order_date values found: {invalid_dates}")

    @classmethod
    def _validate_order_status(
        cls,
        dataframe: pd.DataFrame,
        warnings: list[str],
    ) -> None:
        """
        Validate order status values.

        Args:
            dataframe: Operational orders DataFrame.
            warnings: Warning list to update.
        """
        if "order_status" not in dataframe.columns:
            return

        status_values = set(dataframe["order_status"].dropna().unique())
        invalid_statuses = status_values - cls.VALID_ORDER_STATUSES

        if invalid_statuses:
            warnings.append(
                f"Unexpected order_status values found: {sorted(invalid_statuses)}"
            )

    @staticmethod
    def _validate_numeric_columns(
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
        errors: list[str],
    ) -> None:
        """
        Validate numeric columns.

        Args:
            dataframe: Input pandas DataFrame.
            numeric_columns: Columns expected to contain numeric values.
            errors: Error list to update.
        """
        for column in numeric_columns:
            if column not in dataframe.columns:
                continue

            numeric_series = pd.to_numeric(dataframe[column], errors="coerce")
            invalid_count = numeric_series.isna().sum()

            if invalid_count > 0:
                errors.append(
                    f"Column '{column}' has {invalid_count} non-numeric values."
                )

            if (numeric_series.dropna() < 0).any():
                errors.append(
                    f"Column '{column}' contains negative values."
                )

    @staticmethod
    def _validate_budget_month(
        dataframe: pd.DataFrame,
        errors: list[str],
    ) -> None:
        """
        Validate budget month format.

        Expected month format is YYYY-MM.

        Args:
            dataframe: Budget DataFrame.
            errors: Error list to update.
        """
        if "month" not in dataframe.columns:
            return

        parsed_months = pd.to_datetime(
            dataframe["month"],
            format="%Y-%m",
            errors="coerce",
        )

        invalid_months = parsed_months.isna().sum()

        if invalid_months > 0:
            errors.append(f"Invalid month values found: {invalid_months}")