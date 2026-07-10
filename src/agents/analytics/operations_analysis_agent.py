"""
Operations Analysis Agent for the Finance Agentic AI System.

This module calculates FP&A-ready actual operational metrics from cleaned
order-level data.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


GroupByType = Literal["day", "week", "month", "quarter", "year"]


@dataclass
class OperationsAnalysisResult:
    total_orders: int
    completed_orders: int
    cancelled_orders: int
    fulfillment_percentage: float
    cancellation_percentage: float
    total_revenue: float
    average_order_value: float
    vehicle_summary: list[dict[str, Any]] = field(default_factory=list)
    cluster_summary: list[dict[str, Any]] = field(default_factory=list)
    period_summary: list[dict[str, Any]] = field(default_factory=list)


class OperationsAnalysisAgent:
    REQUIRED_COLUMNS = {
        "order_id",
        "order_date",
        "pickup_cluster",
        "order_status",
        "vehicle_category",
        "fare",
    }

    COMPLETED_STATUS = "completed"
    CANCELLED_STATUS = "cancelled"

    def analyze(
        self,
        data: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        vehicle_category: str | None = None,
        pickup_cluster: str | None = None,
        group_by: GroupByType = "day",
    ) -> OperationsAnalysisResult:
        self._validate_input(data=data, group_by=group_by)

        filtered_data = self._apply_filters(
            data=data,
            start_date=start_date,
            end_date=end_date,
            vehicle_category=vehicle_category,
            pickup_cluster=pickup_cluster,
        )

        completed_data = self._get_completed_orders(filtered_data)

        total_orders = len(filtered_data)
        completed_orders = len(completed_data)
        cancelled_orders = self._count_cancelled_orders(filtered_data)

        total_revenue = self._calculate_total_revenue(completed_data)
        average_order_value = self._calculate_average_order_value(
            total_revenue=total_revenue,
            completed_orders=completed_orders,
        )

        return OperationsAnalysisResult(
            total_orders=total_orders,
            completed_orders=completed_orders,
            cancelled_orders=cancelled_orders,
            fulfillment_percentage=self._calculate_percentage(
                completed_orders, total_orders
            ),
            cancellation_percentage=self._calculate_percentage(
                cancelled_orders, total_orders
            ),
            total_revenue=total_revenue,
            average_order_value=average_order_value,
            vehicle_summary=self._calculate_dimension_summary(
                filtered_data, "vehicle_category"
            ),
            cluster_summary=self._calculate_dimension_summary(
                filtered_data, "pickup_cluster"
            ),
            period_summary=self._calculate_period_summary(
                filtered_data, group_by
            ),
        )

    def _validate_input(self, data: pd.DataFrame, group_by: str) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)

        if missing_columns:
            raise ValueError(
                "Missing required columns for operations analysis: "
                f"{sorted(missing_columns)}"
            )

        valid_group_by_values = {"day", "week", "month", "quarter", "year"}

        if group_by not in valid_group_by_values:
            raise ValueError(
                "Invalid group_by value. Allowed values are: "
                f"{sorted(valid_group_by_values)}"
            )

    def _apply_filters(
        self,
        data: pd.DataFrame,
        start_date: str | None,
        end_date: str | None,
        vehicle_category: str | None,
        pickup_cluster: str | None,
    ) -> pd.DataFrame:
        filtered_data = data.copy()

        filtered_data["order_date"] = pd.to_datetime(
            filtered_data["order_date"],
            format="%d-%m-%Y",
            errors="coerce",
        )

        filtered_data["order_status"] = (
            filtered_data["order_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        filtered_data["vehicle_category"] = (
            filtered_data["vehicle_category"]
            .astype(str)
            .str.strip()
        )

        filtered_data["pickup_cluster"] = (
            filtered_data["pickup_cluster"]
            .astype(str)
            .str.strip()
        )

        filtered_data["fare"] = pd.to_numeric(
            filtered_data["fare"],
            errors="coerce",
        ).fillna(0)

        filtered_data = filtered_data.dropna(subset=["order_date"])

        if start_date:
            filtered_data = filtered_data[
                filtered_data["order_date"] >= pd.to_datetime(
                    start_date,
                    format="%Y-%m-%d",
                )
            ]

        if end_date:
            filtered_data = filtered_data[
                filtered_data["order_date"] <= pd.to_datetime(
                    end_date,
                     format="%Y-%m-%d",
                )
            ]

        if vehicle_category:
            filtered_data = filtered_data[
                filtered_data["vehicle_category"].str.lower()
                == vehicle_category.strip().lower()
            ]

        if pickup_cluster:
            filtered_data = filtered_data[
                filtered_data["pickup_cluster"].str.lower()
                == pickup_cluster.strip().lower()
            ]

        return filtered_data

    def _get_completed_orders(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[data["order_status"] == self.COMPLETED_STATUS].copy()

    def _count_cancelled_orders(self, data: pd.DataFrame) -> int:
        return int((data["order_status"] == self.CANCELLED_STATUS).sum())

    def _calculate_total_revenue(self, completed_data: pd.DataFrame) -> float:
        return round(float(completed_data["fare"].sum()), 2)

    def _calculate_average_order_value(
        self,
        total_revenue: float,
        completed_orders: int,
    ) -> float:
        if completed_orders == 0:
            return 0.0

        return round(total_revenue / completed_orders, 2)

    def _calculate_percentage(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0

        return round((numerator / denominator) * 100, 2)

    def _calculate_dimension_summary(
        self,
        data: pd.DataFrame,
        dimension: str,
    ) -> list[dict[str, Any]]:
        if data.empty:
            return []

        summary_rows: list[dict[str, Any]] = []

        for dimension_value, group_data in data.groupby(dimension):
            completed_data = self._get_completed_orders(group_data)

            total_orders = len(group_data)
            completed_orders = len(completed_data)
            cancelled_orders = self._count_cancelled_orders(group_data)
            total_revenue = self._calculate_total_revenue(completed_data)

            summary_rows.append(
                {
                    dimension: dimension_value,
                    "total_orders": total_orders,
                    "completed_orders": completed_orders,
                    "cancelled_orders": cancelled_orders,
                    "fulfillment_percentage": self._calculate_percentage(
                        completed_orders,
                        total_orders,
                    ),
                    "cancellation_percentage": self._calculate_percentage(
                        cancelled_orders,
                        total_orders,
                    ),
                    "total_revenue": total_revenue,
                    "average_order_value": self._calculate_average_order_value(
                        total_revenue,
                        completed_orders,
                    ),
                }
            )

        return summary_rows

    def _calculate_period_summary(
        self,
        data: pd.DataFrame,
        group_by: GroupByType,
    ) -> list[dict[str, Any]]:
        if data.empty:
            return []

        period_data = data.copy()
        period_data["period"] = self._create_period_column(
            data=period_data,
            group_by=group_by,
        )

        summary_rows: list[dict[str, Any]] = []

        for period, group_data in period_data.groupby("period"):
            completed_data = self._get_completed_orders(group_data)

            total_orders = len(group_data)
            completed_orders = len(completed_data)
            cancelled_orders = self._count_cancelled_orders(group_data)
            total_revenue = self._calculate_total_revenue(completed_data)

            summary_rows.append(
                {
                    "period": str(period),
                    "total_orders": total_orders,
                    "completed_orders": completed_orders,
                    "cancelled_orders": cancelled_orders,
                    "fulfillment_percentage": self._calculate_percentage(
                        completed_orders,
                        total_orders,
                    ),
                    "cancellation_percentage": self._calculate_percentage(
                        cancelled_orders,
                        total_orders,
                    ),
                    "total_revenue": total_revenue,
                    "average_order_value": self._calculate_average_order_value(
                        total_revenue,
                        completed_orders,
                    ),
                }
            )

        return summary_rows

    def _create_period_column(
        self,
        data: pd.DataFrame,
        group_by: GroupByType,
    ) -> pd.Series:
        if group_by == "day":
            return data["order_date"].dt.date

        if group_by == "week":
            return data["order_date"].dt.to_period("W").astype(str)

        if group_by == "month":
            return data["order_date"].dt.to_period("M").astype(str)

        if group_by == "quarter":
            return data["order_date"].dt.to_period("Q").astype(str)

        if group_by == "year":
            return data["order_date"].dt.to_period("Y").astype(str)

        raise ValueError(f"Unsupported group_by value: {group_by}")