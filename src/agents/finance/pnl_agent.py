"""
P&L Agent for the Finance Agentic AI System.

This module calculates monthly Actual P&L, Budget P&L, and Actual vs Budget
variance entirely in memory using pandas DataFrames.

It does not create intermediate CSV files.

Actual data sources:
- sample_orders.csv
- sample_corporate_expenses.csv

Budget data sources:
- sample_budget.csv
- sample_budget_corporate_expenses.csv

The resulting analysis can be passed directly to:
- LangGraph state
- Commentary Agent
- Report Agent
- FastAPI
- Streamlit chatbot
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

COMPLETED_ORDER_STATUS: Final[str] = "completed"


@dataclass(slots=True)
class PnlAnalysisResult:
    """
    Store the complete P&L analysis result.

    Attributes:
        actual_pnl:
            Monthly actual P&L records.

        budget_pnl:
            Monthly budget P&L records.

        variance_pnl:
            Monthly Actual vs Budget variance records.

        summary:
            Consolidated Actual, Budget, and variance summary.

        available_months:
            Months available in both Actual and Budget P&L.

        excluded_actual_months:
            Actual months without matching budget data.

        excluded_budget_months:
            Budget months without matching actual data.
    """

    actual_pnl: list[dict[str, Any]] = field(default_factory=list)
    budget_pnl: list[dict[str, Any]] = field(default_factory=list)
    variance_pnl: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    available_months: list[str] = field(default_factory=list)
    excluded_actual_months: list[str] = field(default_factory=list)
    excluded_budget_months: list[str] = field(default_factory=list)


class PnlAgent:
    """
    Calculate Actual P&L, Budget P&L, and Actual vs Budget variance.

    Actual P&L formulas:

        Revenue = Sum of fare from completed orders

        Direct Cost =
            Partner Payout
            + Incentive
            + Goodwill
            + Dry Run
            + Surge

        Gross Profit = Revenue - Direct Cost

        Gross Margin % = Gross Profit / Revenue * 100

        EBITDA =
            Gross Profit
            - Sales & Marketing
            - Other OPEX

        EBIT = EBITDA - Depreciation

        EBT = EBIT - Interest

    Budget P&L formulas:

        Budget Revenue = Sum of budget_revenue

        Budget Direct Cost = Sum of budget_cogs

        Budget Gross Profit =
            Budget Revenue - Budget COGS

        Budget Gross Margin % =
            Budget Gross Profit / Budget Revenue * 100

        Budget EBITDA =
            Budget Gross Profit
            - Budget Sales & Marketing
            - Budget Other OPEX

        Budget EBIT =
            Budget EBITDA - Budget Depreciation

        Budget EBT =
            Budget EBIT - Budget Interest
    """

    ACTUAL_ORDER_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {
            "order_date",
            "order_status",
            "fare",
            "partner_payout",
            "incentive",
            "goodwill",
            "dry_run",
            "surge",
        }
    )

    ACTUAL_EXPENSE_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {
            "month",
            "sales_marketing",
            "other_opex",
            "depreciation",
            "interest",
        }
    )

    BUDGET_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {
            "month",
            "budget_revenue",
            "budget_cogs",
        }
    )

    BUDGET_EXPENSE_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
        {
            "month",
            "sales_marketing",
            "other_opex",
            "depreciation",
            "interest",
        }
    )

    ACTUAL_DIRECT_COST_COLUMNS: Final[tuple[str, ...]] = (
        "partner_payout",
        "incentive",
        "goodwill",
        "dry_run",
        "surge",
    )

    ACTUAL_ORDER_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
        "fare",
        "partner_payout",
        "incentive",
        "goodwill",
        "dry_run",
        "surge",
    )

    EXPENSE_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
        "sales_marketing",
        "other_opex",
        "depreciation",
        "interest",
    )

    BUDGET_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
        "budget_revenue",
        "budget_cogs",
    )

    PNL_COLUMNS: Final[tuple[str, ...]] = (
        "month",
        "revenue",
        "direct_cost",
        "gross_profit",
        "gross_margin_percentage",
        "sales_marketing",
        "other_opex",
        "ebitda",
        "depreciation",
        "ebit",
        "interest",
        "ebt",
    )

    PNL_METRICS: Final[tuple[str, ...]] = (
        "revenue",
        "direct_cost",
        "gross_profit",
        "gross_margin_percentage",
        "sales_marketing",
        "other_opex",
        "ebitda",
        "depreciation",
        "ebit",
        "interest",
        "ebt",
    )

    def analyze(
        self,
        orders_data: pd.DataFrame,
        corporate_expenses_data: pd.DataFrame,
        budget_data: pd.DataFrame,
        budget_corporate_expenses_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> PnlAnalysisResult:
        """
        Perform complete Actual vs Budget P&L analysis.

        Args:
            orders_data:
                Order-level actual operational data.

            corporate_expenses_data:
                Monthly actual corporate-expense data.

            budget_data:
                Monthly or vehicle-category-level budget data containing
                budget revenue and budget COGS.

            budget_corporate_expenses_data:
                Monthly budget corporate-expense data.

            start_month:
                Optional inclusive first month in ``YYYY-MM`` format.

            end_month:
                Optional inclusive final month in ``YYYY-MM`` format.

        Returns:
            PnlAnalysisResult containing chatbot-ready records.

        Raises:
            TypeError:
                If any input is not a pandas DataFrame.

            ValueError:
                If input data is empty, incomplete, or invalid.
        """
        self._validate_dataframe(
            data=orders_data,
            dataset_name="Orders",
        )
        self._validate_dataframe(
            data=corporate_expenses_data,
            dataset_name="Actual corporate expenses",
        )
        self._validate_dataframe(
            data=budget_data,
            dataset_name="Budget",
        )
        self._validate_dataframe(
            data=budget_corporate_expenses_data,
            dataset_name="Budget corporate expenses",
        )

        self._validate_month_filter(
            start_month=start_month,
            end_month=end_month,
        )

        actual_pnl = self.create_actual_pnl(
            orders_data=orders_data,
            corporate_expenses_data=corporate_expenses_data,
            start_month=start_month,
            end_month=end_month,
        )

        budget_pnl = self.create_budget_pnl(
            budget_data=budget_data,
            budget_corporate_expenses_data=(
                budget_corporate_expenses_data
            ),
            start_month=start_month,
            end_month=end_month,
        )

        (
            comparable_actual,
            comparable_budget,
            excluded_actual_months,
            excluded_budget_months,
        ) = self._select_comparable_months(
            actual_pnl=actual_pnl,
            budget_pnl=budget_pnl,
        )

        variance_pnl = self.calculate_variance(
            actual_pnl=comparable_actual,
            budget_pnl=comparable_budget,
        )

        summary = self._create_summary(
            actual_pnl=comparable_actual,
            budget_pnl=comparable_budget,
        )

        available_months = sorted(
            comparable_actual["month"].astype(str).tolist()
        )

        result = PnlAnalysisResult(
            actual_pnl=self._to_records(comparable_actual),
            budget_pnl=self._to_records(comparable_budget),
            variance_pnl=self._to_records(variance_pnl),
            summary=summary,
            available_months=available_months,
            excluded_actual_months=excluded_actual_months,
            excluded_budget_months=excluded_budget_months,
        )

        LOGGER.info(
            "P&L analysis completed for %d comparable months.",
            len(result.available_months),
        )

        return result

    def create_actual_pnl(
        self,
        orders_data: pd.DataFrame,
        corporate_expenses_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> pd.DataFrame:
        """
        Create a monthly actual P&L in memory.

        Only completed orders are included. Months without corresponding
        corporate-expense data are excluded because they do not represent
        a complete P&L statement.

        Args:
            orders_data:
                Actual order-level operational data.

            corporate_expenses_data:
                Monthly actual corporate-expense data.

            start_month:
                Optional inclusive first month.

            end_month:
                Optional inclusive final month.

        Returns:
            Monthly actual P&L DataFrame.
        """
        self._validate_required_columns(
            data=orders_data,
            required_columns=self.ACTUAL_ORDER_REQUIRED_COLUMNS,
            dataset_name="Orders",
        )

        self._validate_required_columns(
            data=corporate_expenses_data,
            required_columns=self.ACTUAL_EXPENSE_REQUIRED_COLUMNS,
            dataset_name="Actual corporate expenses",
        )

        prepared_orders = self._prepare_actual_orders(
            data=orders_data,
        )

        prepared_expenses = self._prepare_expenses(
            data=corporate_expenses_data,
            dataset_name="Actual corporate expenses",
        )

        prepared_orders = self._apply_month_filter(
            data=prepared_orders,
            start_month=start_month,
            end_month=end_month,
        )

        prepared_expenses = self._apply_month_filter(
            data=prepared_expenses,
            start_month=start_month,
            end_month=end_month,
        )

        if prepared_orders.empty:
            raise ValueError(
                "No actual order data remains after applying the month "
                "filter."
            )

        if prepared_expenses.empty:
            raise ValueError(
                "No actual corporate-expense data remains after applying "
                "the month filter."
            )

        monthly_orders = self._aggregate_actual_orders(
            data=prepared_orders,
        )

        merged_data = monthly_orders.merge(
            prepared_expenses,
            on="month",
            how="inner",
            validate="one_to_one",
        )

        if merged_data.empty:
            raise ValueError(
                "Actual orders and actual corporate expenses have no "
                "matching months."
            )

        excluded_order_months = sorted(
            set(monthly_orders["month"])
            - set(prepared_expenses["month"])
        )

        if excluded_order_months:
            LOGGER.warning(
                "Excluded actual order months without corporate expenses: "
                "%s",
                excluded_order_months,
            )

        return self._calculate_pnl(data=merged_data)

    def create_budget_pnl(
        self,
        budget_data: pd.DataFrame,
        budget_corporate_expenses_data: pd.DataFrame,
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> pd.DataFrame:
        """
        Create a monthly budget P&L in memory.

        Budget data may contain multiple vehicle-category rows per month.
        Budget revenue and budget COGS are aggregated to monthly totals.

        Args:
            budget_data:
                Budget dataset containing:

                - month
                - budget_revenue
                - budget_cogs

            budget_corporate_expenses_data:
                Monthly budget corporate-expense data.

            start_month:
                Optional inclusive first month.

            end_month:
                Optional inclusive final month.

        Returns:
            Monthly budget P&L DataFrame.
        """
        self._validate_required_columns(
            data=budget_data,
            required_columns=self.BUDGET_REQUIRED_COLUMNS,
            dataset_name="Budget",
        )

        self._validate_required_columns(
            data=budget_corporate_expenses_data,
            required_columns=self.BUDGET_EXPENSE_REQUIRED_COLUMNS,
            dataset_name="Budget corporate expenses",
        )

        prepared_budget = self._prepare_budget_data(
            data=budget_data,
        )

        prepared_budget_expenses = self._prepare_expenses(
            data=budget_corporate_expenses_data,
            dataset_name="Budget corporate expenses",
        )

        prepared_budget = self._apply_month_filter(
            data=prepared_budget,
            start_month=start_month,
            end_month=end_month,
        )

        prepared_budget_expenses = self._apply_month_filter(
            data=prepared_budget_expenses,
            start_month=start_month,
            end_month=end_month,
        )

        if prepared_budget.empty:
            raise ValueError(
                "No budget data remains after applying the month filter."
            )

        if prepared_budget_expenses.empty:
            raise ValueError(
                "No budget corporate-expense data remains after applying "
                "the month filter."
            )

        monthly_budget = self._aggregate_budget(
            data=prepared_budget,
        )

        merged_data = monthly_budget.merge(
            prepared_budget_expenses,
            on="month",
            how="inner",
            validate="one_to_one",
        )

        if merged_data.empty:
            raise ValueError(
                "Budget data and budget corporate expenses have no "
                "matching months."
            )

        excluded_budget_months = sorted(
            set(monthly_budget["month"])
            - set(prepared_budget_expenses["month"])
        )

        if excluded_budget_months:
            LOGGER.warning(
                "Excluded budget months without budget corporate expenses: "
                "%s",
                excluded_budget_months,
            )

        return self._calculate_pnl(data=merged_data)

    def calculate_variance(
        self,
        actual_pnl: pd.DataFrame,
        budget_pnl: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate monthly Actual vs Budget P&L variance.

        Monetary variance formula:

            Variance = Actual - Budget

        Variance percentage formula:

            Variance % =
                (Actual - Budget) / absolute value of Budget * 100

        Gross margin variance is expressed as a percentage-point movement.

        Args:
            actual_pnl:
                Monthly actual P&L.

            budget_pnl:
                Monthly budget P&L.

        Returns:
            Monthly Actual vs Budget variance DataFrame.
        """
        self._validate_dataframe(
            data=actual_pnl,
            dataset_name="Actual P&L",
        )
        self._validate_dataframe(
            data=budget_pnl,
            dataset_name="Budget P&L",
        )

        self._validate_required_columns(
            data=actual_pnl,
            required_columns=frozenset(self.PNL_COLUMNS),
            dataset_name="Actual P&L",
        )

        self._validate_required_columns(
            data=budget_pnl,
            required_columns=frozenset(self.PNL_COLUMNS),
            dataset_name="Budget P&L",
        )

        merged_data = actual_pnl.merge(
            budget_pnl,
            on="month",
            how="inner",
            suffixes=("_actual", "_budget"),
            validate="one_to_one",
        )

        if merged_data.empty:
            raise ValueError(
                "Actual P&L and Budget P&L have no matching months."
            )

        variance_data = pd.DataFrame(
            {"month": merged_data["month"]}
        )

        for metric in self.PNL_METRICS:
            actual_column = f"{metric}_actual"
            budget_column = f"{metric}_budget"
            variance_column = f"{metric}_variance"

            variance_data[actual_column] = merged_data[actual_column]
            variance_data[budget_column] = merged_data[budget_column]

            variance_data[variance_column] = (
                merged_data[actual_column]
                - merged_data[budget_column]
            )

            if metric == "gross_margin_percentage":
                variance_data[
                    "gross_margin_percentage_point_variance"
                ] = variance_data[variance_column]
                continue

            variance_data[
                f"{metric}_variance_percentage"
            ] = self._calculate_variance_percentage(
                actual=merged_data[actual_column],
                budget=merged_data[budget_column],
            )

        numeric_columns = variance_data.select_dtypes(
            include="number"
        ).columns

        variance_data[numeric_columns] = variance_data[
            numeric_columns
        ].round(2)

        return variance_data.sort_values(
            by="month"
        ).reset_index(drop=True)

    def _prepare_actual_orders(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and prepare completed actual orders."""
        orders = data.copy()

        orders["order_date"] = pd.to_datetime(
            orders["order_date"],
            format="%d-%m-%Y",
            errors="coerce",
        )

        invalid_date_mask = orders["order_date"].isna()

        if invalid_date_mask.any():
            invalid_rows = orders.index[
                invalid_date_mask
            ].tolist()[:10]

            raise ValueError(
                "Orders contain invalid order_date values. Expected "
                f"DD-MM-YYYY. Example rows: {invalid_rows}"
            )

        normalized_status = (
            orders["order_status"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        orders = orders.loc[
            normalized_status == COMPLETED_ORDER_STATUS
        ].copy()

        if orders.empty:
            raise ValueError(
                "Orders dataset contains no completed orders."
            )

        orders = self._convert_numeric_columns(
            data=orders,
            columns=self.ACTUAL_ORDER_NUMERIC_COLUMNS,
            dataset_name="Orders",
        )

        self._validate_non_negative_columns(
            data=orders,
            columns=self.ACTUAL_ORDER_NUMERIC_COLUMNS,
            dataset_name="Orders",
        )

        orders["month"] = (
            orders["order_date"]
            .dt.to_period("M")
            .astype(str)
        )

        return orders

    def _prepare_budget_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and normalize budget revenue and budget COGS."""
        budget = data.copy()

        budget["month"] = self._normalize_month_column(
            data=budget,
            dataset_name="Budget",
        )

        budget = self._convert_numeric_columns(
            data=budget,
            columns=self.BUDGET_NUMERIC_COLUMNS,
            dataset_name="Budget",
        )

        self._validate_non_negative_columns(
            data=budget,
            columns=self.BUDGET_NUMERIC_COLUMNS,
            dataset_name="Budget",
        )

        return budget

    def _prepare_expenses(
        self,
        data: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Validate and normalize monthly corporate expenses."""
        expenses = data.copy()

        expenses["month"] = self._normalize_month_column(
            data=expenses,
            dataset_name=dataset_name,
        )

        duplicate_month_mask = expenses["month"].duplicated(
            keep=False
        )

        if duplicate_month_mask.any():
            duplicate_months = sorted(
                expenses.loc[
                    duplicate_month_mask,
                    "month",
                ].unique()
            )

            raise ValueError(
                f"{dataset_name} must contain one row per month. "
                f"Duplicate months found: {duplicate_months}"
            )

        expenses = self._convert_numeric_columns(
            data=expenses,
            columns=self.EXPENSE_NUMERIC_COLUMNS,
            dataset_name=dataset_name,
        )

        self._validate_non_negative_columns(
            data=expenses,
            columns=self.EXPENSE_NUMERIC_COLUMNS,
            dataset_name=dataset_name,
        )

        return expenses[
            [
                "month",
                "sales_marketing",
                "other_opex",
                "depreciation",
                "interest",
            ]
        ].copy()

    def _aggregate_actual_orders(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate actual revenue and direct cost by month."""
        orders = data.copy()

        orders["direct_cost"] = orders[
            list(self.ACTUAL_DIRECT_COST_COLUMNS)
        ].sum(axis=1)

        return (
            orders.groupby("month", as_index=False)
            .agg(
                revenue=("fare", "sum"),
                direct_cost=("direct_cost", "sum"),
            )
            .sort_values(by="month")
            .reset_index(drop=True)
        )

    def _aggregate_budget(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate category-level budget revenue and budget COGS by month.

        Budget COGS becomes direct cost in the P&L.
        """
        return (
            data.groupby("month", as_index=False)
            .agg(
                revenue=("budget_revenue", "sum"),
                direct_cost=("budget_cogs", "sum"),
            )
            .sort_values(by="month")
            .reset_index(drop=True)
        )

    def _calculate_pnl(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate standard monthly P&L lines."""
        pnl = data.copy()

        pnl["gross_profit"] = (
            pnl["revenue"] - pnl["direct_cost"]
        )

        pnl["gross_margin_percentage"] = self._safe_percentage(
            numerator=pnl["gross_profit"],
            denominator=pnl["revenue"],
        )

        pnl["ebitda"] = (
            pnl["gross_profit"]
            - pnl["sales_marketing"]
            - pnl["other_opex"]
        )

        pnl["ebit"] = (
            pnl["ebitda"] - pnl["depreciation"]
        )

        pnl["ebt"] = (
            pnl["ebit"] - pnl["interest"]
        )

        pnl = pnl.loc[:, self.PNL_COLUMNS].copy()

        numeric_columns = [
            column
            for column in self.PNL_COLUMNS
            if column != "month"
        ]

        pnl[numeric_columns] = pnl[
            numeric_columns
        ].round(2)

        self._validate_calculated_pnl(data=pnl)

        return pnl.sort_values(
            by="month"
        ).reset_index(drop=True)

    def _select_comparable_months(
        self,
        actual_pnl: pd.DataFrame,
        budget_pnl: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        list[str],
        list[str],
    ]:
        """Select months available in both Actual and Budget P&L."""
        actual_months = set(
            actual_pnl["month"].astype(str)
        )

        budget_months = set(
            budget_pnl["month"].astype(str)
        )

        common_months = sorted(
            actual_months.intersection(budget_months)
        )

        if not common_months:
            raise ValueError(
                "Actual and Budget P&L contain no common months."
            )

        comparable_actual = actual_pnl.loc[
            actual_pnl["month"].isin(common_months)
        ].copy()

        comparable_budget = budget_pnl.loc[
            budget_pnl["month"].isin(common_months)
        ].copy()

        excluded_actual_months = sorted(
            actual_months - budget_months
        )

        excluded_budget_months = sorted(
            budget_months - actual_months
        )

        if excluded_actual_months:
            LOGGER.warning(
                "Actual months excluded because Budget is unavailable: %s",
                excluded_actual_months,
            )

        if excluded_budget_months:
            LOGGER.warning(
                "Budget months excluded because Actual is unavailable: %s",
                excluded_budget_months,
            )

        return (
            comparable_actual.reset_index(drop=True),
            comparable_budget.reset_index(drop=True),
            excluded_actual_months,
            excluded_budget_months,
        )

    def _create_summary(
        self,
        actual_pnl: pd.DataFrame,
        budget_pnl: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create consolidated Actual vs Budget P&L summary."""
        actual_totals = self._calculate_period_totals(
            data=actual_pnl,
        )

        budget_totals = self._calculate_period_totals(
            data=budget_pnl,
        )

        variance_summary: dict[str, float] = {}

        for metric in self.PNL_METRICS:
            actual_value = float(actual_totals[metric])
            budget_value = float(budget_totals[metric])
            variance_value = actual_value - budget_value

            if metric == "gross_margin_percentage":
                variance_summary[
                    "gross_margin_percentage_point_variance"
                ] = round(variance_value, 2)
                continue

            variance_summary[
                f"{metric}_variance"
            ] = round(variance_value, 2)

            variance_summary[
                f"{metric}_variance_percentage"
            ] = round(
                self._safe_scalar_variance_percentage(
                    actual=actual_value,
                    budget=budget_value,
                ),
                2,
            )

        return {
            "actual": actual_totals,
            "budget": budget_totals,
            "variance": variance_summary,
        }

    def _calculate_period_totals(
        self,
        data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate consolidated P&L totals for the selected period."""
        revenue = float(data["revenue"].sum())
        direct_cost = float(data["direct_cost"].sum())

        gross_profit = revenue - direct_cost

        sales_marketing = float(
            data["sales_marketing"].sum()
        )

        other_opex = float(
            data["other_opex"].sum()
        )

        depreciation = float(
            data["depreciation"].sum()
        )

        interest = float(
            data["interest"].sum()
        )

        ebitda = (
            gross_profit
            - sales_marketing
            - other_opex
        )

        ebit = ebitda - depreciation
        ebt = ebit - interest

        gross_margin_percentage = (
            gross_profit / revenue * 100
            if revenue != 0
            else 0.0
        )

        return {
            "revenue": round(revenue, 2),
            "direct_cost": round(direct_cost, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_margin_percentage": round(
                gross_margin_percentage,
                2,
            ),
            "sales_marketing": round(
                sales_marketing,
                2,
            ),
            "other_opex": round(other_opex, 2),
            "ebitda": round(ebitda, 2),
            "depreciation": round(depreciation, 2),
            "ebit": round(ebit, 2),
            "interest": round(interest, 2),
            "ebt": round(ebt, 2),
        }

    def _validate_calculated_pnl(
        self,
        data: pd.DataFrame,
    ) -> None:
        """Validate generated P&L formulas and output quality."""
        if data.empty:
            raise ValueError(
                "Calculated P&L contains no rows."
            )

        if data["month"].duplicated().any():
            duplicate_months = sorted(
                data.loc[
                    data["month"].duplicated(keep=False),
                    "month",
                ].unique()
            )

            raise ValueError(
                "Calculated P&L contains duplicate months: "
                f"{duplicate_months}"
            )

        numeric_columns = [
            column
            for column in self.PNL_COLUMNS
            if column != "month"
        ]

        if data[numeric_columns].isna().any().any():
            raise ValueError(
                "Calculated P&L contains missing financial values."
            )

        expected_gross_profit = (
            data["revenue"] - data["direct_cost"]
        ).round(2)

        expected_ebitda = (
            data["gross_profit"]
            - data["sales_marketing"]
            - data["other_opex"]
        ).round(2)

        expected_ebit = (
            data["ebitda"] - data["depreciation"]
        ).round(2)

        expected_ebt = (
            data["ebit"] - data["interest"]
        ).round(2)

        formula_checks = {
            "gross_profit": expected_gross_profit,
            "ebitda": expected_ebitda,
            "ebit": expected_ebit,
            "ebt": expected_ebt,
        }

        for column, expected_values in formula_checks.items():
            if not data[column].round(2).equals(expected_values):
                raise ValueError(
                    f"Calculated P&L failed the {column} formula check."
                )

    def _apply_month_filter(
        self,
        data: pd.DataFrame,
        start_month: str | None,
        end_month: str | None,
    ) -> pd.DataFrame:
        """Apply optional inclusive month filters."""
        filtered_data = data.copy()

        if start_month is not None:
            filtered_data = filtered_data.loc[
                filtered_data["month"] >= start_month
            ]

        if end_month is not None:
            filtered_data = filtered_data.loc[
                filtered_data["month"] <= end_month
            ]

        return filtered_data.reset_index(drop=True)

    def _validate_month_filter(
        self,
        start_month: str | None,
        end_month: str | None,
    ) -> None:
        """Validate optional month filter values."""
        for field_name, value in (
            ("start_month", start_month),
            ("end_month", end_month),
        ):
            if value is None:
                continue

            parsed_value = pd.to_datetime(
                value,
                format="%Y-%m",
                errors="coerce",
            )

            if pd.isna(parsed_value):
                raise ValueError(
                    f"{field_name} must use YYYY-MM format."
                )

        if (
            start_month is not None
            and end_month is not None
            and start_month > end_month
        ):
            raise ValueError(
                "start_month cannot be later than end_month."
            )

    def _validate_dataframe(
        self,
        data: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """Validate that an input is a non-empty pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{dataset_name} must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                f"{dataset_name} cannot be empty."
            )

    def _validate_required_columns(
        self,
        data: pd.DataFrame,
        required_columns: frozenset[str],
        dataset_name: str,
    ) -> None:
        """Validate required dataset columns."""
        missing_columns = sorted(
            required_columns.difference(data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{missing_columns}"
            )

    def _normalize_month_column(
        self,
        data: pd.DataFrame,
        dataset_name: str,
    ) -> pd.Series:
        """Normalize month values to YYYY-MM format."""
        normalized_values = (
            data["month"]
            .astype("string")
            .str.strip()
        )

        parsed_months = pd.to_datetime(
            normalized_values,
            format="%Y-%m",
            errors="coerce",
        )

        invalid_month_mask = parsed_months.isna()

        if invalid_month_mask.any():
            invalid_rows = data.index[
                invalid_month_mask
            ].tolist()[:10]

            raise ValueError(
                f"{dataset_name} contains invalid month values. "
                "Expected YYYY-MM format. "
                f"Example rows: {invalid_rows}"
            )

        return parsed_months.dt.to_period("M").astype(str)

    def _convert_numeric_columns(
        self,
        data: pd.DataFrame,
        columns: tuple[str, ...],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Convert required columns to numeric values."""
        converted_data = data.copy()

        for column in columns:
            converted_data[column] = (
                self._convert_numeric_series(
                    series=converted_data[column],
                    column_name=column,
                    dataset_name=dataset_name,
                )
            )

        return converted_data

    def _convert_numeric_series(
        self,
        series: pd.Series,
        column_name: str,
        dataset_name: str,
    ) -> pd.Series:
        """Convert a pandas Series to numeric and reject invalid values."""
        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        invalid_value_mask = numeric_series.isna()

        if invalid_value_mask.any():
            invalid_rows = series.index[
                invalid_value_mask
            ].tolist()[:10]

            raise ValueError(
                f"{dataset_name} column '{column_name}' contains "
                "missing or non-numeric values. "
                f"Example rows: {invalid_rows}"
            )

        return numeric_series.astype(float)

    def _validate_non_negative_columns(
        self,
        data: pd.DataFrame,
        columns: tuple[str, ...],
        dataset_name: str,
    ) -> None:
        """Reject negative revenue and cost input values."""
        for column in columns:
            negative_value_mask = data[column] < 0

            if negative_value_mask.any():
                invalid_rows = data.index[
                    negative_value_mask
                ].tolist()[:10]

                raise ValueError(
                    f"{dataset_name} column '{column}' contains "
                    f"negative values. Example rows: {invalid_rows}"
                )

    def _safe_percentage(
        self,
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """Calculate percentages while handling zero denominators."""
        result = pd.Series(
            0.0,
            index=numerator.index,
            dtype=float,
        )

        valid_mask = denominator != 0

        result.loc[valid_mask] = (
            numerator.loc[valid_mask]
            / denominator.loc[valid_mask]
            * 100
        )

        return result

    def _calculate_variance_percentage(
        self,
        actual: pd.Series,
        budget: pd.Series,
    ) -> pd.Series:
        """Calculate variance percentage using absolute budget values."""
        result = pd.Series(
            0.0,
            index=actual.index,
            dtype=float,
        )

        valid_mask = budget != 0

        result.loc[valid_mask] = (
            (
                actual.loc[valid_mask]
                - budget.loc[valid_mask]
            )
            / budget.loc[valid_mask].abs()
            * 100
        )

        return result

    def _safe_scalar_variance_percentage(
        self,
        actual: float,
        budget: float,
    ) -> float:
        """Calculate a scalar variance percentage safely."""
        if budget == 0:
            return 0.0

        return (
            (actual - budget)
            / abs(budget)
            * 100
        )

    def _to_records(
        self,
        data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Convert a DataFrame to API and chatbot-ready records."""
        return data.to_dict(orient="records")