"""
Forecast Agent for the Finance Agentic AI System.

This module creates a base rolling forecast from historical operational
actuals.

The first version uses a simple moving-average rolling forecast.

It forecasts:
- Completed orders
- Revenue
- Average order value

Business assumptions are not applied in this module. They will be applied
later by the Scenario Agent.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


ForecastFrequency = Literal["month", "week"]


@dataclass
class ForecastResult:
    """
    Stores the rolling forecast output.

    Attributes:
        method: Forecast method used.
        frequency: Forecast frequency, such as month or week.
        rolling_window: Number of historical periods used.
        forecast_periods: Number of future periods generated.
        historical_periods_used: Historical periods used for forecasting.
        forecast_summary: Future forecast rows.
    """

    method: str
    frequency: str
    rolling_window: int
    forecast_periods: int
    historical_periods_used: list[str] = field(default_factory=list)
    forecast_summary: list[dict[str, Any]] = field(default_factory=list)


class ForecastAgent:
    """
    Creates a base rolling forecast using historical actual performance.

    Formula:

        Forecast Orders =
        Average completed orders from the latest N periods

        Forecast Revenue =
        Average revenue from the latest N periods

        Forecast AOV =
        Forecast Revenue / Forecast Orders

    For multiple future periods, each newly forecasted period is added to
    the rolling history before calculating the next period.
    """

    REQUIRED_COLUMNS = {
        "period",
        "completed_orders",
        "total_revenue",
    }

    METHOD_NAME = "Simple Moving Average Rolling Forecast"

    def analyze(
        self,
        period_summary: list[dict[str, Any]],
        frequency: ForecastFrequency = "month",
        rolling_window: int = 3,
        forecast_periods: int = 3,
    ) -> ForecastResult:
        """
        Create a rolling forecast from historical period-level actuals.

        Args:
            period_summary:
                Period-level actual data from OperationsAnalysisAgent.

            frequency:
                Forecast frequency. Supported values:
                - month
                - week

            rolling_window:
                Number of latest periods used to calculate each forecast.

                Example:
                rolling_window=3 means the latest three periods are averaged.

            forecast_periods:
                Number of future periods to forecast.

        Returns:
            ForecastResult containing future orders, revenue, and AOV.

        Raises:
            TypeError:
                If period_summary is not a list.

            ValueError:
                If required fields are missing or parameters are invalid.
        """

        self._validate_parameters(
            period_summary=period_summary,
            frequency=frequency,
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
        )

        historical_data = self._prepare_historical_data(
            period_summary=period_summary,
            frequency=frequency,
        )

        if len(historical_data) < rolling_window:
            raise ValueError(
                "Insufficient historical periods for the selected rolling window. "
                f"Available periods: {len(historical_data)}. "
                f"Required periods: {rolling_window}."
            )

        historical_periods_used = (
            historical_data.tail(rolling_window)["period"]
            .astype(str)
            .tolist()
        )

        forecast_summary = self._create_forecast(
            historical_data=historical_data,
            frequency=frequency,
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
        )

        return ForecastResult(
            method=self.METHOD_NAME,
            frequency=frequency,
            rolling_window=rolling_window,
            forecast_periods=forecast_periods,
            historical_periods_used=historical_periods_used,
            forecast_summary=forecast_summary,
        )

    def _validate_parameters(
        self,
        period_summary: list[dict[str, Any]],
        frequency: str,
        rolling_window: int,
        forecast_periods: int,
    ) -> None:
        """
        Validate input data and forecasting parameters.
        """

        if not isinstance(period_summary, list):
            raise TypeError("period_summary must be a list of dictionaries.")

        if not period_summary:
            raise ValueError("period_summary cannot be empty.")

        valid_frequencies = {"month", "week"}

        if frequency not in valid_frequencies:
            raise ValueError(
                "Invalid frequency. Allowed values are: "
                f"{sorted(valid_frequencies)}"
            )

        if not isinstance(rolling_window, int) or rolling_window <= 0:
            raise ValueError("rolling_window must be a positive integer.")

        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            raise ValueError("forecast_periods must be a positive integer.")

        available_columns = set(period_summary[0].keys())
        missing_columns = self.REQUIRED_COLUMNS - available_columns

        if missing_columns:
            raise ValueError(
                "Missing required fields in period_summary: "
                f"{sorted(missing_columns)}"
            )

    def _prepare_historical_data(
        self,
        period_summary: list[dict[str, Any]],
        frequency: ForecastFrequency,
    ) -> pd.DataFrame:
        """
        Convert period summary into a clean forecasting DataFrame.

        This method:
        - Converts orders and revenue to numeric values.
        - Converts period labels to sortable dates.
        - Removes rows with invalid periods.
        - Sorts history from oldest to newest.
        """

        historical_data = pd.DataFrame(period_summary).copy()

        historical_data["completed_orders"] = pd.to_numeric(
            historical_data["completed_orders"],
            errors="coerce",
        )

        historical_data["total_revenue"] = pd.to_numeric(
            historical_data["total_revenue"],
            errors="coerce",
        )

        historical_data = historical_data.dropna(
            subset=[
                "period",
                "completed_orders",
                "total_revenue",
            ]
        )

        if historical_data.empty:
            raise ValueError(
                "No valid historical rows are available after data preparation."
            )

        if (historical_data["completed_orders"] < 0).any():
            raise ValueError("Historical completed orders cannot be negative.")

        if (historical_data["total_revenue"] < 0).any():
            raise ValueError("Historical revenue cannot be negative.")

        historical_data["period_start"] = historical_data["period"].apply(
            lambda period: self._parse_period_start(
                period=str(period),
                frequency=frequency,
            )
        )

        historical_data = historical_data.dropna(subset=["period_start"])

        if historical_data.empty:
            raise ValueError(
                "Historical period values could not be converted into valid dates."
            )

        historical_data = (
            historical_data.sort_values("period_start")
            .drop_duplicates(
                subset=["period_start"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        historical_data["average_order_value"] = historical_data.apply(
            lambda row: self._safe_divide(
                numerator=float(row["total_revenue"]),
                denominator=float(row["completed_orders"]),
            ),
            axis=1,
        )

        return historical_data[
            [
                "period",
                "period_start",
                "completed_orders",
                "total_revenue",
                "average_order_value",
            ]
        ]

    def _create_forecast(
        self,
        historical_data: pd.DataFrame,
        frequency: ForecastFrequency,
        rolling_window: int,
        forecast_periods: int,
    ) -> list[dict[str, Any]]:
        """
        Generate future rolling forecast rows.

        Recursive rolling method:

        1. Use latest N periods to calculate the next forecast.
        2. Add the forecasted value to the working history.
        3. Use the updated latest N periods for the following forecast.
        """

        working_data = historical_data.copy()
        forecast_rows: list[dict[str, Any]] = []

        latest_period_start = working_data["period_start"].max()

        for forecast_number in range(1, forecast_periods + 1):
            rolling_data = working_data.tail(rolling_window)

            forecast_orders = self._calculate_moving_average(
                values=rolling_data["completed_orders"],
            )

            forecast_revenue = self._calculate_moving_average(
                values=rolling_data["total_revenue"],
            )

            forecast_aov = self._safe_divide(
                numerator=forecast_revenue,
                denominator=forecast_orders,
            )

            next_period_start = self._get_next_period_start(
                current_period_start=latest_period_start,
                frequency=frequency,
            )

            forecast_period = self._format_period(
                period_start=next_period_start,
                frequency=frequency,
            )

            source_periods = rolling_data["period"].astype(str).tolist()

            forecast_row = {
                "forecast_number": forecast_number,
                "forecast_period": forecast_period,
                "forecast_orders": int(round(forecast_orders)),
                "forecast_revenue": round(forecast_revenue, 2),
                "forecast_average_order_value": round(forecast_aov, 2),
                "rolling_window": rolling_window,
                "source_periods": source_periods,
                "method": self.METHOD_NAME,
            }

            forecast_rows.append(forecast_row)

            new_history_row = pd.DataFrame(
                [
                    {
                        "period": forecast_period,
                        "period_start": next_period_start,
                        "completed_orders": forecast_orders,
                        "total_revenue": forecast_revenue,
                        "average_order_value": forecast_aov,
                    }
                ]
            )

            working_data = pd.concat(
                [working_data, new_history_row],
                ignore_index=True,
            )

            latest_period_start = next_period_start

        return forecast_rows

    def _calculate_moving_average(self, values: pd.Series) -> float:
        """
        Calculate a simple moving average.
        """

        numeric_values = pd.to_numeric(values, errors="coerce").dropna()

        if numeric_values.empty:
            return 0.0

        return float(numeric_values.mean())

    def _parse_period_start(
        self,
        period: str,
        frequency: ForecastFrequency,
    ) -> pd.Timestamp:
        """
        Convert a period label into its starting date.

        Monthly example:
            2026-04 → 2026-04-01

        Weekly example:
            2026-06-29/2026-07-05 → 2026-06-29
        """

        if frequency == "month":
            parsed_period = pd.Period(period, freq="M")
            return parsed_period.start_time.normalize()

        if frequency == "week":
            period_start_text = period.split("/")[0].strip()

            return pd.to_datetime(
                period_start_text,
                format="%Y-%m-%d",
                errors="raise",
            ).normalize()

        raise ValueError(f"Unsupported forecast frequency: {frequency}")

    def _get_next_period_start(
        self,
        current_period_start: pd.Timestamp,
        frequency: ForecastFrequency,
    ) -> pd.Timestamp:
        """
        Calculate the next forecast period's starting date.
        """

        if frequency == "month":
            return current_period_start + pd.offsets.MonthBegin(1)

        if frequency == "week":
            return current_period_start + pd.Timedelta(weeks=1)

        raise ValueError(f"Unsupported forecast frequency: {frequency}")

    def _format_period(
        self,
        period_start: pd.Timestamp,
        frequency: ForecastFrequency,
    ) -> str:
        """
        Convert forecast period date into a readable label.
        """

        if frequency == "month":
            return period_start.strftime("%Y-%m")

        if frequency == "week":
            period_end = period_start + pd.Timedelta(days=6)

            return (
                f"{period_start.strftime('%Y-%m-%d')}/"
                f"{period_end.strftime('%Y-%m-%d')}"
            )

        raise ValueError(f"Unsupported forecast frequency: {frequency}")

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Divide safely and prevent divide-by-zero errors.
        """

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)