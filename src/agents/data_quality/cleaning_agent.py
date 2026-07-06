"""
Cleaning agent for the Finance Agentic AI System.

This module cleans and standardizes operational and budget data after
validation and before analytics, finance, forecasting, and reporting agents
use the data.
"""

import pandas as pd


class CleaningAgent:
    """
    Agent responsible for cleaning FP&A input data.

    This agent performs safe cleaning operations such as standardizing
    column names, trimming spaces, converting dates, and converting numeric
    fields. It does not silently change important business values.
    """

    OPERATIONS_NUMERIC_COLUMNS = [
        "fare",
        "commission_amount",
        "partner_payout",
        "incentive",
        "goodwill",
        "dry_run",
        "surge",
        "trip_distance_km",
        "delivery_time_minutes",
    ]

    BUDGET_NUMERIC_COLUMNS = [
        "budget_orders",
        "budget_revenue",
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
        cleaned_dataframe = dataframe.copy()
        cleaned_dataframe.columns = (
            cleaned_dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )
        return cleaned_dataframe

    @staticmethod
    def trim_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extra spaces from text columns.

        Args:
            dataframe: Input pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with cleaned text values.
        """
        cleaned_dataframe = dataframe.copy()

        text_columns = cleaned_dataframe.select_dtypes(include=["object"]).columns

        for column in text_columns:
            cleaned_dataframe[column] = cleaned_dataframe[column].str.strip()

        return cleaned_dataframe

    @staticmethod
    def standardize_order_status(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize order status values.

        Args:
            dataframe: Operational orders DataFrame.

        Returns:
            pd.DataFrame: DataFrame with standardized order_status values.
        """
        cleaned_dataframe = dataframe.copy()

        if "order_status" not in cleaned_dataframe.columns:
            return cleaned_dataframe

        status_mapping = {
            "completed": "Completed",
            "complete": "Completed",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
            "rejected": "Rejected",
        }

        cleaned_dataframe["order_status"] = (
            cleaned_dataframe["order_status"]
            .str.strip()
            .str.lower()
            .map(status_mapping)
            .fillna(cleaned_dataframe["order_status"])
        )

        return cleaned_dataframe

    @staticmethod
    def standardize_vehicle_category(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize vehicle category values.

        Args:
            dataframe: Input pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with standardized vehicle_category values.
        """
        cleaned_dataframe = dataframe.copy()

        if "vehicle_category" not in cleaned_dataframe.columns:
            return cleaned_dataframe

        vehicle_mapping = {
            "2w": "2W",
            "3w": "3W",
            "compact auto": "Compact Auto",
            "tata ace open": "Tata Ace Open",
            "tata ace closed": "Tata Ace Closed",
            "8 ft": "8 FT",
            "9 ft": "9 FT",
            "10 ft": "10 FT",
            "14 ft": "14 FT",
            "17 ft": "17 FT",
            "packers & movers": "Packers & Movers",
            "packers and movers": "Packers & Movers",
        }

        cleaned_dataframe["vehicle_category"] = (
            cleaned_dataframe["vehicle_category"]
            .str.strip()
            .str.lower()
            .map(vehicle_mapping)
            .fillna(cleaned_dataframe["vehicle_category"])
        )

        return cleaned_dataframe

    @staticmethod
    def convert_order_date(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert order_date column into pandas datetime format.

        Args:
            dataframe: Operational orders DataFrame.

        Returns:
            pd.DataFrame: DataFrame with converted order_date.
        """
        cleaned_dataframe = dataframe.copy()

        if "order_date" not in cleaned_dataframe.columns:
            return cleaned_dataframe

        cleaned_dataframe["order_date"] = pd.to_datetime(
            cleaned_dataframe["order_date"],
            errors="coerce",
            dayfirst=True,
        )

        return cleaned_dataframe

    @staticmethod
    def convert_budget_month(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Convert budget month into pandas datetime format.

        Args:
            dataframe: Budget DataFrame.

        Returns:
            pd.DataFrame: DataFrame with converted month column.
        """
        cleaned_dataframe = dataframe.copy()

        if "month" not in cleaned_dataframe.columns:
            return cleaned_dataframe

        cleaned_dataframe["month"] = pd.to_datetime(
            cleaned_dataframe["month"],
            format="%Y-%m",
            errors="coerce",
        )

        return cleaned_dataframe

    @staticmethod
    def convert_numeric_columns(
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> pd.DataFrame:
        """
        Convert selected columns into numeric values.

        Args:
            dataframe: Input pandas DataFrame.
            numeric_columns: List of columns to convert.

        Returns:
            pd.DataFrame: DataFrame with numeric columns converted.
        """
        cleaned_dataframe = dataframe.copy()

        for column in numeric_columns:
            if column in cleaned_dataframe.columns:
                cleaned_dataframe[column] = pd.to_numeric(
                    cleaned_dataframe[column],
                    errors="coerce",
                )

        return cleaned_dataframe

    @classmethod
    def clean_operations_data(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean operational order data.

        Args:
            dataframe: Raw operational orders DataFrame.

        Returns:
            pd.DataFrame: Cleaned operational orders DataFrame.
        """
        cleaned_dataframe = cls.normalize_columns(dataframe)
        cleaned_dataframe = cls.trim_text_columns(cleaned_dataframe)
        cleaned_dataframe = cls.standardize_order_status(cleaned_dataframe)
        cleaned_dataframe = cls.standardize_vehicle_category(cleaned_dataframe)
        cleaned_dataframe = cls.convert_order_date(cleaned_dataframe)
        cleaned_dataframe = cls.convert_numeric_columns(
            dataframe=cleaned_dataframe,
            numeric_columns=cls.OPERATIONS_NUMERIC_COLUMNS,
        )

        return cleaned_dataframe

    @classmethod
    def clean_budget_data(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean budget planning data.

        Args:
            dataframe: Raw budget DataFrame.

        Returns:
            pd.DataFrame: Cleaned budget DataFrame.
        """
        cleaned_dataframe = cls.normalize_columns(dataframe)
        cleaned_dataframe = cls.trim_text_columns(cleaned_dataframe)
        cleaned_dataframe = cls.standardize_vehicle_category(cleaned_dataframe)
        cleaned_dataframe = cls.convert_budget_month(cleaned_dataframe)
        cleaned_dataframe = cls.convert_numeric_columns(
            dataframe=cleaned_dataframe,
            numeric_columns=cls.BUDGET_NUMERIC_COLUMNS,
        )

        return cleaned_dataframe