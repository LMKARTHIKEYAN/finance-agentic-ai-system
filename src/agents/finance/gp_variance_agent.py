"""Gross-margin percentage variance decomposition by mix, price, and cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import pandas as pd


COMPLETED_STATUS: Final[str] = "completed"
RECONCILIATION_TOLERANCE: Final[float] = 0.0001


@dataclass(slots=True)
class CategoryUnitEconomics:
    category: str
    product: str
    month: str
    vehicle_category: str
    actual_volume: float
    actual_price_per_unit: float
    actual_cost_per_unit: float
    actual_revenue: float
    actual_direct_cost: float
    actual_gross_profit: float
    actual_gp_percentage: float
    actual_mix_percentage: float
    budget_volume: float
    budget_price_per_unit: float
    budget_cost_per_unit: float
    budget_revenue: float
    budget_direct_cost: float
    budget_gross_profit: float
    budget_gp_percentage: float
    budget_mix_percentage: float
    price_effect_percentage_points: float
    cost_effect_percentage_points: float
    check_percentage_points: float
    mix_indicator: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GrossProfitVarianceResult:
    base_revenue: float
    base_gross_profit: float
    budget_gp_percentage: float
    mix_only_revenue: float
    mix_only_gross_profit: float
    mix_only_gp_percentage: float
    price_only_revenue: float
    price_only_gross_profit: float
    price_only_gp_percentage: float
    actual_revenue: float
    actual_gross_profit: float
    actual_gp_percentage: float
    mix_effect_percentage_points: float
    price_effect_percentage_points: float
    cost_effect_percentage_points: float
    total_variance_percentage_points: float
    mix_effect_basis_points: float
    price_effect_basis_points: float
    cost_effect_basis_points: float
    total_variance_basis_points: float
    reconciliation_difference: float
    reconciliation_status: str
    category_analysis: list[dict[str, Any]] = field(default_factory=list)
    available_months: list[str] = field(default_factory=list)
    excluded_actual_categories: list[dict[str, str]] = field(default_factory=list)
    excluded_budget_categories: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GrossProfitVarianceAgent:
    """Decompose Actual-versus-Budget GP% into mix, price, and cost effects."""

    ACTUAL_COST_COLUMNS: Final[tuple[str, ...]] = (
        "incentive",
        "goodwill",
        "dry_run",
        "surge",
    )

    def analyze(
        self,
        orders_data: pd.DataFrame,
        budget_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> GrossProfitVarianceResult:
        actual = self._prepare_actual(orders_data)
        budget = self._prepare_budget(budget_data)
        actual = self._filter_months(actual, start_month, end_month)
        budget = self._filter_months(budget, start_month, end_month)

        keys = ["month", "vehicle_category"]
        actual_keys = set(map(tuple, actual[keys].to_records(index=False)))
        budget_keys = set(map(tuple, budget[keys].to_records(index=False)))
        common_keys = actual_keys & budget_keys
        if not common_keys:
            raise ValueError("Actual and Budget contain no comparable month/category records.")

        merged = actual.merge(
            budget,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        category_rows = self._category_rows(merged)

        budget_revenue = (merged["budget_volume"] * merged["budget_price_per_unit"]).sum()
        budget_gp = (
            merged["budget_volume"]
            * (merged["budget_price_per_unit"] - merged["budget_cost_per_unit"])
        ).sum()
        mix_revenue = (merged["actual_volume"] * merged["budget_price_per_unit"]).sum()
        mix_gp = (
            merged["actual_volume"]
            * (merged["budget_price_per_unit"] - merged["budget_cost_per_unit"])
        ).sum()
        price_revenue = (merged["actual_volume"] * merged["actual_price_per_unit"]).sum()
        price_gp = (
            merged["actual_volume"]
            * (merged["actual_price_per_unit"] - merged["budget_cost_per_unit"])
        ).sum()
        actual_revenue = price_revenue
        actual_gp = (
            merged["actual_volume"]
            * (merged["actual_price_per_unit"] - merged["actual_cost_per_unit"])
        ).sum()

        budget_pct = self._ratio(budget_gp, budget_revenue)
        mix_pct = self._ratio(mix_gp, mix_revenue)
        price_pct = self._ratio(price_gp, price_revenue)
        actual_pct = self._ratio(actual_gp, actual_revenue)
        mix_effect = mix_pct - budget_pct
        price_effect = price_pct - mix_pct
        cost_effect = actual_pct - price_pct
        total_effect = actual_pct - budget_pct
        difference = mix_effect + price_effect + cost_effect - total_effect

        return GrossProfitVarianceResult(
            base_revenue=round(float(budget_revenue), 2),
            base_gross_profit=round(float(budget_gp), 2),
            budget_gp_percentage=round(budget_pct * 100, 4),
            mix_only_revenue=round(float(mix_revenue), 2),
            mix_only_gross_profit=round(float(mix_gp), 2),
            mix_only_gp_percentage=round(mix_pct * 100, 4),
            price_only_revenue=round(float(price_revenue), 2),
            price_only_gross_profit=round(float(price_gp), 2),
            price_only_gp_percentage=round(price_pct * 100, 4),
            actual_revenue=round(float(actual_revenue), 2),
            actual_gross_profit=round(float(actual_gp), 2),
            actual_gp_percentage=round(actual_pct * 100, 4),
            mix_effect_percentage_points=round(mix_effect * 100, 4),
            price_effect_percentage_points=round(price_effect * 100, 4),
            cost_effect_percentage_points=round(cost_effect * 100, 4),
            total_variance_percentage_points=round(total_effect * 100, 4),
            mix_effect_basis_points=round(mix_effect * 10_000, 2),
            price_effect_basis_points=round(price_effect * 10_000, 2),
            cost_effect_basis_points=round(cost_effect * 10_000, 2),
            total_variance_basis_points=round(total_effect * 10_000, 2),
            reconciliation_difference=round(difference * 100, 8),
            reconciliation_status=(
                "PASS" if abs(difference) <= RECONCILIATION_TOLERANCE else "FAIL"
            ),
            category_analysis=[row.to_dict() for row in category_rows],
            available_months=sorted(merged["month"].unique().tolist()),
            excluded_actual_categories=self._format_exclusions(actual_keys - budget_keys),
            excluded_budget_categories=self._format_exclusions(budget_keys - actual_keys),
        )

    def _prepare_actual(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {
            "order_id", "order_date", "vehicle_category", "order_status",
            "commission_amount", *self.ACTUAL_COST_COLUMNS,
        }
        self._validate(data, required, "Orders")
        actual = data.copy()
        actual["order_date"] = pd.to_datetime(actual["order_date"], dayfirst=True, errors="raise")
        actual = actual[
            actual["order_status"].astype(str).str.strip().str.lower().eq(COMPLETED_STATUS)
        ].copy()
        actual["month"] = actual["order_date"].dt.strftime("%Y-%m")
        numeric = ["commission_amount", *self.ACTUAL_COST_COLUMNS]
        actual[numeric] = actual[numeric].apply(pd.to_numeric, errors="raise")
        actual["actual_direct_cost"] = actual[list(self.ACTUAL_COST_COLUMNS)].sum(axis=1)
        grouped = actual.groupby(["month", "vehicle_category"], as_index=False).agg(
            actual_volume=("order_id", "count"),
            actual_revenue=("commission_amount", "sum"),
            actual_direct_cost=("actual_direct_cost", "sum"),
        )
        grouped["actual_price_per_unit"] = grouped["actual_revenue"] / grouped["actual_volume"]
        grouped["actual_cost_per_unit"] = grouped["actual_direct_cost"] / grouped["actual_volume"]
        return grouped

    def _prepare_budget(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {
            "month", "vehicle_category", "budget_orders",
            "budget_revenue", "budget_cogs",
        }
        self._validate(data, required, "Budget")
        budget = data.copy()
        budget["month"] = pd.to_datetime(budget["month"], errors="raise").dt.strftime("%Y-%m")
        numeric = ["budget_orders", "budget_revenue", "budget_cogs"]
        budget[numeric] = budget[numeric].apply(pd.to_numeric, errors="raise")
        grouped = budget.groupby(["month", "vehicle_category"], as_index=False).agg(
            budget_volume=("budget_orders", "sum"),
            budget_revenue=("budget_revenue", "sum"),
            budget_direct_cost=("budget_cogs", "sum"),
        )
        if (grouped["budget_volume"] <= 0).any():
            raise ValueError("Budget volume must be positive for GP% decomposition.")
        grouped["budget_price_per_unit"] = grouped["budget_revenue"] / grouped["budget_volume"]
        grouped["budget_cost_per_unit"] = grouped["budget_direct_cost"] / grouped["budget_volume"]
        return grouped

    def _category_rows(self, data: pd.DataFrame) -> list[CategoryUnitEconomics]:
        actual_total = float(data["actual_revenue"].sum())
        budget_total = float(data["budget_revenue"].sum())
        portfolio_budget_gp_percentage = self._ratio(
            float(
                (
                    data["budget_revenue"]
                    - data["budget_direct_cost"]
                ).sum()
            ),
            budget_total,
        )
        rows: list[CategoryUnitEconomics] = []
        for row in data.itertuples(index=False):
            actual_gp = row.actual_revenue - row.actual_direct_cost
            budget_gp = row.budget_revenue - row.budget_direct_cost
            actual_gp_percentage = self._ratio(
                actual_gp,
                row.actual_revenue,
            )
            budget_gp_percentage = self._ratio(
                budget_gp,
                row.budget_revenue,
            )
            price_only_gp_percentage = self._ratio(
                row.actual_price_per_unit - row.budget_cost_per_unit,
                row.actual_price_per_unit,
            )
            price_effect = (
                price_only_gp_percentage - budget_gp_percentage
            )
            cost_effect = (
                actual_gp_percentage
                - budget_gp_percentage
                - price_effect
            )
            check = (
                actual_gp_percentage
                - budget_gp_percentage
                - price_effect
                - cost_effect
            )
            actual_mix = self._ratio(row.actual_revenue, actual_total)
            budget_mix = self._ratio(row.budget_revenue, budget_total)
            mix_indicator = (
                1
                if (
                    (
                        budget_gp_percentage
                        > portfolio_budget_gp_percentage
                        and actual_mix > budget_mix
                    )
                    or (
                        budget_gp_percentage
                        < portfolio_budget_gp_percentage
                        and actual_mix < budget_mix
                    )
                )
                else -1
            )
            rows.append(CategoryUnitEconomics(
                category="Vehicle",
                product=str(row.vehicle_category),
                month=row.month,
                vehicle_category=row.vehicle_category,
                actual_volume=float(row.actual_volume),
                actual_price_per_unit=round(float(row.actual_price_per_unit), 4),
                actual_cost_per_unit=round(float(row.actual_cost_per_unit), 4),
                actual_revenue=round(float(row.actual_revenue), 2),
                actual_direct_cost=round(float(row.actual_direct_cost), 2),
                actual_gross_profit=round(float(actual_gp), 2),
                actual_gp_percentage=round(actual_gp_percentage * 100, 4),
                actual_mix_percentage=round(actual_mix * 100, 4),
                budget_volume=float(row.budget_volume),
                budget_price_per_unit=round(float(row.budget_price_per_unit), 4),
                budget_cost_per_unit=round(float(row.budget_cost_per_unit), 4),
                budget_revenue=round(float(row.budget_revenue), 2),
                budget_direct_cost=round(float(row.budget_direct_cost), 2),
                budget_gross_profit=round(float(budget_gp), 2),
                budget_gp_percentage=round(budget_gp_percentage * 100, 4),
                budget_mix_percentage=round(budget_mix * 100, 4),
                price_effect_percentage_points=round(price_effect * 100, 4),
                cost_effect_percentage_points=round(cost_effect * 100, 4),
                check_percentage_points=round(check * 100, 8),
                mix_indicator=mix_indicator,
            ))
        return rows

    @staticmethod
    def _filter_months(data: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        result = data
        if start is not None:
            result = result[result["month"] >= start]
        if end is not None:
            result = result[result["month"] <= end]
        return result.copy()

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _format_exclusions(keys: set[tuple[Any, ...]]) -> list[dict[str, str]]:
        return [
            {"month": str(month), "vehicle_category": str(category)}
            for month, category in sorted(keys)
        ]

    @staticmethod
    def _validate(data: pd.DataFrame, required: set[str], name: str) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{name} data must be a pandas DataFrame.")
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{name} data is missing required columns: {missing}")
        if data.empty:
            raise ValueError(f"{name} data cannot be empty.")
