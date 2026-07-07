"""
Budget Agent for the Finance Agentic AI System.

This module prepares FP&A-ready budget metrics from monthly budget data.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


GroupByType = Literal["month", "quarter", "year"]


@dataclass
class BudgetAnalysisResult:
    total_budget_orders: int
    total_budget_revenue: float
    budget_average_order_value: float
    vehicle_summary: list[dict[str, Any]] = field(default_factory=list)
    period_summary: list[dict[str, Any]] = field(default_factory=list)


class BudgetAgent:
    REQUIRED_COLUMNS = {
        "month",
        "vehicle_category",
        "budget_orders",
        "budget_revenue",
    }

    def analyze(
        self,
        data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
        vehicle_category: str | None = None,
        group_by: GroupByType = "month",
    ) -> BudgetAnalysisResult:
        self._validate_input(data=data, group_by=group_by)

        filtered_data = self._apply_filters(
            data=data,
            start_month=start_month,
            end_month=end_month,
            vehicle_category=vehicle_category,
        )

        total_budget_orders = self._calculate_total_budget_orders(filtered_data)
        total_budget_revenue = self._calculate_total_budget_revenue(filtered_data)

        return BudgetAnalysisResult(
            total_budget_orders=total_budget_orders,
            total_budget_revenue=total_budget_revenue,
            budget_average_order_value=self._calculate_budget_aov(
                total_budget_revenue=total_budget_revenue,
                total_budget_orders=total_budget_orders,
            ),
            vehicle_summary=self._calculate_vehicle_summary(filtered_data),
            period_summary=self._calculate_period_summary(
                data=filtered_data,
                group_by=group_by,
            ),
        )

    def _validate_input(self, data: pd.DataFrame, group_by: str) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS - set(data.columns)

        if missing_columns:
            raise ValueError(
                "Missing required columns for budget analysis: "
                f"{sorted(missing_columns)}"
            )

        valid_group_by_values = {"month", "quarter", "year"}

        if group_by not in valid_group_by_values:
            raise ValueError(
                "Invalid group_by value. Allowed values are: "
                f"{sorted(valid_group_by_values)}"
            )

    def _apply_filters(
        self,
        data: pd.DataFrame,
        start_month: str | None,
        end_month: str | None,
        vehicle_category: str | None,
    ) -> pd.DataFrame:
        filtered_data = data.copy()

        filtered_data["month"] = pd.to_datetime(
            filtered_data["month"],
            format="%Y-%m",
            errors="coerce",
        )

        filtered_data["vehicle_category"] = (
            filtered_data["vehicle_category"]
            .astype(str)
            .str.strip()
        )

        filtered_data["budget_orders"] = pd.to_numeric(
            filtered_data["budget_orders"],
            errors="coerce",
        ).fillna(0)

        filtered_data["budget_revenue"] = pd.to_numeric(
            filtered_data["budget_revenue"],
            errors="coerce",
        ).fillna(0)

        filtered_data = filtered_data.dropna(subset=["month"])

        if start_month:
            filtered_data = filtered_data[
                filtered_data["month"] >= pd.to_datetime(
                    start_month,
                    format="%Y-%m",
                )
            ]

        if end_month:
            filtered_data = filtered_data[
                filtered_data["month"] <= pd.to_datetime(
                    end_month,
                    format="%Y-%m",
                )
            ]

        if vehicle_category:
            filtered_data = filtered_data[
                filtered_data["vehicle_category"].str.lower()
                == vehicle_category.strip().lower()
            ]

        return filtered_data

    def _calculate_total_budget_orders(self, data: pd.DataFrame) -> int:
        return int(data["budget_orders"].sum())

    def _calculate_total_budget_revenue(self, data: pd.DataFrame) -> float:
        return round(float(data["budget_revenue"].sum()), 2)

    def _calculate_budget_aov(
        self,
        total_budget_revenue: float,
        total_budget_orders: int,
    ) -> float:
        if total_budget_orders == 0:
            return 0.0

        return round(total_budget_revenue / total_budget_orders, 2)

    def _calculate_vehicle_summary(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        if data.empty:
            return []

        summary_rows: list[dict[str, Any]] = []

        for vehicle_category, group_data in data.groupby("vehicle_category"):
            budget_orders = self._calculate_total_budget_orders(group_data)
            budget_revenue = self._calculate_total_budget_revenue(group_data)

            summary_rows.append(
                {
                    "vehicle_category": vehicle_category,
                    "budget_orders": budget_orders,
                    "budget_revenue": budget_revenue,
                    "budget_average_order_value": self._calculate_budget_aov(
                        total_budget_revenue=budget_revenue,
                        total_budget_orders=budget_orders,
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
            budget_orders = self._calculate_total_budget_orders(group_data)
            budget_revenue = self._calculate_total_budget_revenue(group_data)

            summary_rows.append(
                {
                    "period": str(period),
                    "budget_orders": budget_orders,
                    "budget_revenue": budget_revenue,
                    "budget_average_order_value": self._calculate_budget_aov(
                        total_budget_revenue=budget_revenue,
                        total_budget_orders=budget_orders,
                    ),
                }
            )

        return summary_rows

    def _create_period_column(
        self,
        data: pd.DataFrame,
        group_by: GroupByType,
    ) -> pd.Series:
        if group_by == "month":
            return data["month"].dt.to_period("M").astype(str)

        if group_by == "quarter":
            return data["month"].dt.to_period("Q").astype(str)

        if group_by == "year":
            return data["month"].dt.to_period("Y").astype(str)

        raise ValueError(f"Unsupported group_by value: {group_by}")