"""
KPI Agent for the Finance Agentic AI System.

This module selects requested KPIs from outputs produced by finance and
analytics agents.

The KPI Agent does not recalculate core financial metrics. It retrieves and
standardizes already-calculated values so downstream reporting and commentary
agents receive only the metrics requested by the user or manager.

Supported KPI levels:
- Overall
- Vehicle category
- Pickup cluster
- Period
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KPIValue:
    """
    Stores one selected KPI.

    Attributes:
        kpi:
            Standard internal KPI name.

        display_name:
            Human-readable KPI name.

        value:
            KPI value.

        unit:
            Unit such as count, currency, percentage, or status.

        source:
            Agent that produced the KPI.

        period:
            Optional forecast, scenario, or operational period.

        dimension:
            Optional dimension such as vehicle_category or pickup_cluster.

        dimension_value:
            Selected value inside the dimension, such as 2W or Chennai.
    """

    kpi: str
    display_name: str
    value: Any
    unit: str
    source: str
    period: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None


@dataclass
class KPIResult:
    """
    Stores the complete KPI selection result.

    Attributes:
        requested_kpis:
            Original KPI names requested by the user.

        selected_kpis:
            Successfully retrieved KPI values.

        unavailable_kpis:
            Valid KPI names whose source result or dimension value was
            unavailable.

        unknown_kpis:
            KPI names not recognized by the KPI Agent.

        dimension:
            Optional dimension requested by the user.

        dimension_value:
            Optional selected dimension value.
    """

    requested_kpis: list[str] = field(default_factory=list)
    selected_kpis: list[KPIValue] = field(default_factory=list)
    unavailable_kpis: list[str] = field(default_factory=list)
    unknown_kpis: list[str] = field(default_factory=list)
    dimension: str | None = None
    dimension_value: str | None = None


class KPIAgent:
    """
    Selects requested KPIs from existing agent outputs.

    Example overall request:

        requested_kpis = [
            "total revenue",
            "fulfillment",
            "revenue variance",
        ]

    Example vehicle-category request:

        requested_kpis = [
            "total revenue",
            "completed orders",
            "fulfillment",
        ]

        dimension = "vehicle_category"
        dimension_value = "2W"

    Example cluster request:

        dimension = "pickup_cluster"
        dimension_value = "Chennai"
    """

    SUPPORTED_DIMENSIONS = {
        "vehicle_category",
        "pickup_cluster",
        "period",
    }

    DIMENSION_ALIASES = {
        "vehicle": "vehicle_category",
        "vehicle category": "vehicle_category",
        "vehicle_category": "vehicle_category",
        "category": "vehicle_category",
        "cluster": "pickup_cluster",
        "pickup cluster": "pickup_cluster",
        "pickup_cluster": "pickup_cluster",
        "location": "pickup_cluster",
        "period": "period",
        "month": "period",
        "week": "period",
        "quarter": "period",
        "year": "period",
    }

    KPI_ALIASES = {
        # Operations KPIs
        "total orders": "total_orders",
        "orders": "total_orders",
        "actual orders": "total_orders",
        "completed orders": "completed_orders",
        "fulfilled orders": "completed_orders",
        "cancelled orders": "cancelled_orders",
        "canceled orders": "cancelled_orders",
        "fulfillment": "fulfillment_percentage",
        "fulfillment percentage": "fulfillment_percentage",
        "fulfillment percent": "fulfillment_percentage",
        "fulfillment rate": "fulfillment_percentage",
        "cancellation": "cancellation_percentage",
        "cancellation percentage": "cancellation_percentage",
        "cancellation percent": "cancellation_percentage",
        "cancellation rate": "cancellation_percentage",
        "total revenue": "actual_revenue",
        "actual revenue": "actual_revenue",
        "revenue": "actual_revenue",
        "actual aov": "actual_aov",
        "average order value": "actual_aov",
        "aov": "actual_aov",

        # Budget KPIs
        "budget orders": "budget_orders",
        "budget revenue": "budget_revenue",
        "budget aov": "budget_aov",
        "budget average order value": "budget_aov",

        # Revenue variance KPIs
        "order variance": "order_variance",
        "revenue variance": "revenue_variance",
        "aov variance": "aov_variance",
        "price effect": "price_effect",
        "volume effect": "volume_effect",
        "new discontinued effect": "new_discontinued_effect",
        "new launch effect": "new_discontinued_effect",
        "variance check": "variance_check",

        # Portfolio GP KPIs
        "base gp": "base_gp_percentage",
        "base gp percentage": "base_gp_percentage",
        "base gp percent": "base_gp_percentage",
        "actual gp": "actual_gp_percentage",
        "actual gp percentage": "actual_gp_percentage",
        "actual gp percent": "actual_gp_percentage",
        "mix effect": "mix_effect",
        "gp price effect": "gp_price_effect",
        "cost effect": "cost_effect",
        "total gp change": "total_gp_change",
        "gp check": "gp_check",

        # Forecast KPIs
        "forecast orders": "forecast_orders",
        "forecast revenue": "forecast_revenue",
        "forecast aov": "forecast_aov",
        "forecast average order value": "forecast_aov",

        # Scenario KPIs
        "adjusted orders": "adjusted_orders",
        "scenario orders": "adjusted_orders",
        "adjusted revenue": "adjusted_revenue",
        "scenario revenue": "adjusted_revenue",
        "adjusted aov": "adjusted_aov",
        "scenario aov": "adjusted_aov",

        # Finance controls
        "finance rules status": "finance_rules_status",
        "rules status": "finance_rules_status",
        "warning count": "warning_count",
        "error count": "error_count",
    }

    KPI_METADATA = {
        "total_orders": (
            "Total Orders",
            "count",
            "OperationsAnalysisAgent",
        ),
        "completed_orders": (
            "Completed Orders",
            "count",
            "OperationsAnalysisAgent",
        ),
        "cancelled_orders": (
            "Cancelled Orders",
            "count",
            "OperationsAnalysisAgent",
        ),
        "fulfillment_percentage": (
            "Fulfillment %",
            "percentage",
            "OperationsAnalysisAgent",
        ),
        "cancellation_percentage": (
            "Cancellation %",
            "percentage",
            "OperationsAnalysisAgent",
        ),
        "actual_revenue": (
            "Actual Revenue",
            "currency",
            "OperationsAnalysisAgent",
        ),
        "actual_aov": (
            "Actual AOV",
            "currency_per_order",
            "OperationsAnalysisAgent",
        ),
        "budget_orders": (
            "Budget Orders",
            "count",
            "BudgetAgent",
        ),
        "budget_revenue": (
            "Budget Revenue",
            "currency",
            "BudgetAgent",
        ),
        "budget_aov": (
            "Budget AOV",
            "currency_per_order",
            "BudgetAgent",
        ),
        "order_variance": (
            "Order Variance",
            "count",
            "RevenueVarianceAgent",
        ),
        "revenue_variance": (
            "Revenue Variance",
            "currency",
            "RevenueVarianceAgent",
        ),
        "aov_variance": (
            "AOV Variance",
            "currency_per_order",
            "RevenueVarianceAgent",
        ),
        "price_effect": (
            "Price Effect",
            "currency",
            "RevenueVarianceAgent",
        ),
        "volume_effect": (
            "Volume Effect",
            "currency",
            "RevenueVarianceAgent",
        ),
        "new_discontinued_effect": (
            "New/Discontinued Effect",
            "currency",
            "RevenueVarianceAgent",
        ),
        "variance_check": (
            "Variance Check",
            "currency",
            "RevenueVarianceAgent",
        ),
        "base_gp_percentage": (
            "Base GP%",
            "decimal_percentage",
            "GPPortfolioVarianceAgent",
        ),
        "actual_gp_percentage": (
            "Actual GP%",
            "decimal_percentage",
            "GPPortfolioVarianceAgent",
        ),
        "mix_effect": (
            "Mix Effect",
            "percentage_points",
            "GPPortfolioVarianceAgent",
        ),
        "gp_price_effect": (
            "GP Price Effect",
            "percentage_points",
            "GPPortfolioVarianceAgent",
        ),
        "cost_effect": (
            "Cost Effect",
            "percentage_points",
            "GPPortfolioVarianceAgent",
        ),
        "total_gp_change": (
            "Total GP% Change",
            "percentage_points",
            "GPPortfolioVarianceAgent",
        ),
        "gp_check": (
            "GP Bridge Check",
            "percentage_points",
            "GPPortfolioVarianceAgent",
        ),
        "forecast_orders": (
            "Forecast Orders",
            "count",
            "ForecastAgent",
        ),
        "forecast_revenue": (
            "Forecast Revenue",
            "currency",
            "ForecastAgent",
        ),
        "forecast_aov": (
            "Forecast AOV",
            "currency_per_order",
            "ForecastAgent",
        ),
        "adjusted_orders": (
            "Adjusted Orders",
            "count",
            "ScenarioAgent",
        ),
        "adjusted_revenue": (
            "Adjusted Revenue",
            "currency",
            "ScenarioAgent",
        ),
        "adjusted_aov": (
            "Adjusted AOV",
            "currency_per_order",
            "ScenarioAgent",
        ),
        "finance_rules_status": (
            "Finance Rules Status",
            "status",
            "FinanceRulesAgent",
        ),
        "warning_count": (
            "Finance Rule Warnings",
            "count",
            "FinanceRulesAgent",
        ),
        "error_count": (
            "Finance Rule Errors",
            "count",
            "FinanceRulesAgent",
        ),
    }

    OPERATIONS_KPIS = {
        "total_orders",
        "completed_orders",
        "cancelled_orders",
        "fulfillment_percentage",
        "cancellation_percentage",
        "actual_revenue",
        "actual_aov",
    }

    DIMENSION_KPI_COLUMN_MAPPING = {
        "total_orders": "total_orders",
        "completed_orders": "completed_orders",
        "cancelled_orders": "cancelled_orders",
        "fulfillment_percentage": "fulfillment_percentage",
        "cancellation_percentage": "cancellation_percentage",
        "actual_revenue": "total_revenue",
        "actual_aov": "average_order_value",
    }

    def analyze(
        self,
        requested_kpis: list[str],
        operations_result: Any | None = None,
        budget_result: Any | None = None,
        revenue_variance_result: Any | None = None,
        gp_portfolio_result: Any | None = None,
        forecast_result: Any | None = None,
        scenario_result: Any | None = None,
        finance_rules_result: Any | None = None,
        forecast_period: str | None = None,
        scenario_period: str | None = None,
        dimension: str | None = None,
        dimension_value: str | None = None,
    ) -> KPIResult:
        """
        Retrieve only the KPIs requested by the user.

        Args:
            requested_kpis:
                KPI names requested by the manager.

            operations_result:
                Output from OperationsAnalysisAgent.

            budget_result:
                Output from BudgetAgent.

            revenue_variance_result:
                Output from RevenueVarianceAgent.

            gp_portfolio_result:
                Output from GPPortfolioVarianceAgent.

            forecast_result:
                Output from ForecastAgent.

            scenario_result:
                Output from ScenarioAgent.

            finance_rules_result:
                Output from FinanceRulesAgent.

            forecast_period:
                Optional forecast period such as 2026-07.

            scenario_period:
                Optional adjusted scenario period such as 2026-07.

            dimension:
                Optional operational dimension:
                - vehicle_category
                - pickup_cluster
                - period

            dimension_value:
                Requested value in the dimension.

                Examples:
                - 2W
                - Chennai
                - 2026-06

        Returns:
            KPIResult containing selected, unavailable, and unknown KPIs.
        """

        self._validate_requested_kpis(requested_kpis)

        normalized_dimension = self._normalize_dimension(dimension)

        self._validate_dimension_request(
            dimension=normalized_dimension,
            dimension_value=dimension_value,
        )

        selected_kpis: list[KPIValue] = []
        unavailable_kpis: list[str] = []
        unknown_kpis: list[str] = []
        already_selected: set[str] = set()

        for requested_name in requested_kpis:
            normalized_name = self._normalize_name(requested_name)
            canonical_kpi = self.KPI_ALIASES.get(normalized_name)

            if canonical_kpi is None:
                unknown_kpis.append(requested_name)
                continue

            if canonical_kpi in already_selected:
                continue

            kpi_value = self._get_kpi_value(
                canonical_kpi=canonical_kpi,
                operations_result=operations_result,
                budget_result=budget_result,
                revenue_variance_result=revenue_variance_result,
                gp_portfolio_result=gp_portfolio_result,
                forecast_result=forecast_result,
                scenario_result=scenario_result,
                finance_rules_result=finance_rules_result,
                forecast_period=forecast_period,
                scenario_period=scenario_period,
                dimension=normalized_dimension,
                dimension_value=dimension_value,
            )

            if kpi_value is None:
                unavailable_kpis.append(requested_name)
                continue

            selected_kpis.append(kpi_value)
            already_selected.add(canonical_kpi)

        return KPIResult(
            requested_kpis=requested_kpis.copy(),
            selected_kpis=selected_kpis,
            unavailable_kpis=unavailable_kpis,
            unknown_kpis=unknown_kpis,
            dimension=normalized_dimension,
            dimension_value=dimension_value,
        )

    def available_kpis(self) -> list[dict[str, str]]:
        """
        Return all KPIs currently supported by the agent.
        """

        available: list[dict[str, str]] = []

        for canonical_kpi, metadata in self.KPI_METADATA.items():
            display_name, unit, source = metadata

            available.append(
                {
                    "kpi": canonical_kpi,
                    "display_name": display_name,
                    "unit": unit,
                    "source": source,
                }
            )

        return available

    def available_dimensions(self) -> list[str]:
        """
        Return all supported operational dimensions.
        """

        return sorted(self.SUPPORTED_DIMENSIONS)

    def _validate_requested_kpis(
        self,
        requested_kpis: list[str],
    ) -> None:
        """
        Validate the requested KPI list.
        """

        if not isinstance(requested_kpis, list):
            raise TypeError(
                "requested_kpis must be a list of strings."
            )

        if not requested_kpis:
            raise ValueError(
                "At least one KPI must be requested."
            )

        if not all(
            isinstance(kpi, str) and kpi.strip()
            for kpi in requested_kpis
        ):
            raise ValueError(
                "Every requested KPI must be a non-empty string."
            )

    def _validate_dimension_request(
        self,
        dimension: str | None,
        dimension_value: str | None,
    ) -> None:
        """
        Validate optional dimension inputs.
        """

        if dimension is None and dimension_value is None:
            return

        if dimension is None and dimension_value is not None:
            raise ValueError(
                "dimension must be provided when dimension_value is used."
            )

        if dimension is not None and not dimension_value:
            raise ValueError(
                "dimension_value must be provided when dimension is used."
            )

        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                "Invalid dimension. Allowed values are: "
                f"{sorted(self.SUPPORTED_DIMENSIONS)}"
            )

    def _get_kpi_value(
        self,
        canonical_kpi: str,
        operations_result: Any | None,
        budget_result: Any | None,
        revenue_variance_result: Any | None,
        gp_portfolio_result: Any | None,
        forecast_result: Any | None,
        scenario_result: Any | None,
        finance_rules_result: Any | None,
        forecast_period: str | None,
        scenario_period: str | None,
        dimension: str | None,
        dimension_value: str | None,
    ) -> KPIValue | None:
        """
        Route a KPI request to the correct agent result.
        """

        operations_mapping = {
            "total_orders": "total_orders",
            "completed_orders": "completed_orders",
            "cancelled_orders": "cancelled_orders",
            "fulfillment_percentage": "fulfillment_percentage",
            "cancellation_percentage": "cancellation_percentage",
            "actual_revenue": "total_revenue",
            "actual_aov": "average_order_value",
        }

        budget_mapping = {
            "budget_orders": "total_budget_orders",
            "budget_revenue": "total_budget_revenue",
            "budget_aov": "budget_average_order_value",
        }

        variance_mapping = {
            "order_variance": "order_variance",
            "revenue_variance": "revenue_variance",
            "aov_variance": "aov_variance",
            "price_effect": "price_effect",
            "volume_effect": "volume_effect",
            "new_discontinued_effect": (
                "new_discontinued_effect"
            ),
            "variance_check": "variance_check",
        }

        gp_mapping = {
            "base_gp_percentage": "base_gp_percentage",
            "actual_gp_percentage": "actual_gp_percentage",
            "mix_effect": "mix_effect",
            "gp_price_effect": "price_effect",
            "cost_effect": "cost_effect",
            "total_gp_change": "total_gp_change",
            "gp_check": "check",
        }

        rules_mapping = {
            "finance_rules_status": "overall_status",
            "warning_count": "warning_count",
            "error_count": "error_count",
        }

        if canonical_kpi in operations_mapping:
            if dimension is not None:
                return self._get_dimension_kpi(
                    canonical_kpi=canonical_kpi,
                    operations_result=operations_result,
                    dimension=dimension,
                    dimension_value=dimension_value,
                )

            return self._get_attribute_kpi(
                canonical_kpi=canonical_kpi,
                result=operations_result,
                attribute=operations_mapping[canonical_kpi],
            )

        if canonical_kpi in budget_mapping:
            return self._get_attribute_kpi(
                canonical_kpi=canonical_kpi,
                result=budget_result,
                attribute=budget_mapping[canonical_kpi],
            )

        if canonical_kpi in variance_mapping:
            return self._get_attribute_kpi(
                canonical_kpi=canonical_kpi,
                result=revenue_variance_result,
                attribute=variance_mapping[canonical_kpi],
            )

        if canonical_kpi in gp_mapping:
            return self._get_attribute_kpi(
                canonical_kpi=canonical_kpi,
                result=gp_portfolio_result,
                attribute=gp_mapping[canonical_kpi],
            )

        if canonical_kpi in {
            "forecast_orders",
            "forecast_revenue",
            "forecast_aov",
        }:
            return self._get_forecast_kpi(
                canonical_kpi=canonical_kpi,
                forecast_result=forecast_result,
                requested_period=forecast_period,
            )

        if canonical_kpi in {
            "adjusted_orders",
            "adjusted_revenue",
            "adjusted_aov",
        }:
            return self._get_scenario_kpi(
                canonical_kpi=canonical_kpi,
                scenario_result=scenario_result,
                requested_period=scenario_period,
            )

        if canonical_kpi in rules_mapping:
            return self._get_attribute_kpi(
                canonical_kpi=canonical_kpi,
                result=finance_rules_result,
                attribute=rules_mapping[canonical_kpi],
            )

        return None

    def _get_dimension_kpi(
        self,
        canonical_kpi: str,
        operations_result: Any | None,
        dimension: str,
        dimension_value: str | None,
    ) -> KPIValue | None:
        """
        Retrieve an operational KPI for a selected dimension value.

        Supported dimensions:
        - vehicle_category
        - pickup_cluster
        - period
        """

        if operations_result is None or dimension_value is None:
            return None

        summary_mapping = {
            "vehicle_category": "vehicle_summary",
            "pickup_cluster": "cluster_summary",
            "period": "period_summary",
        }

        summary_attribute = summary_mapping[dimension]

        if not hasattr(operations_result, summary_attribute):
            return None

        summary_rows = getattr(
            operations_result,
            summary_attribute,
        )

        if not isinstance(summary_rows, list) or not summary_rows:
            return None

        selected_row = self._find_dimension_row(
            rows=summary_rows,
            dimension=dimension,
            dimension_value=dimension_value,
        )

        if selected_row is None:
            return None

        value_column = self.DIMENSION_KPI_COLUMN_MAPPING.get(
            canonical_kpi
        )

        if value_column is None or value_column not in selected_row:
            return None

        period = (
            str(selected_row.get("period"))
            if dimension == "period"
            else None
        )

        return self._create_kpi_value(
            canonical_kpi=canonical_kpi,
            value=selected_row[value_column],
            period=period,
            dimension=dimension,
            dimension_value=str(
                selected_row.get(dimension, dimension_value)
            ),
        )

    def _find_dimension_row(
        self,
        rows: list[dict[str, Any]],
        dimension: str,
        dimension_value: str,
    ) -> dict[str, Any] | None:
        """
        Find a dimension row using case-insensitive matching.

        Examples:
            2w matches 2W
            chennai matches Chennai
            2026-06 matches 2026-06
        """

        normalized_requested_value = self._normalize_dimension_value(
            dimension_value
        )

        for row in rows:
            row_value = row.get(dimension)

            if row_value is None:
                continue

            normalized_row_value = self._normalize_dimension_value(
                str(row_value)
            )

            if normalized_row_value == normalized_requested_value:
                return row

        return None

    def _get_attribute_kpi(
        self,
        canonical_kpi: str,
        result: Any | None,
        attribute: str,
    ) -> KPIValue | None:
        """
        Retrieve a KPI stored directly as a result-object attribute.
        """

        if result is None or not hasattr(result, attribute):
            return None

        return self._create_kpi_value(
            canonical_kpi=canonical_kpi,
            value=getattr(result, attribute),
        )

    def _get_forecast_kpi(
        self,
        canonical_kpi: str,
        forecast_result: Any | None,
        requested_period: str | None,
    ) -> KPIValue | None:
        """
        Retrieve a KPI from a forecast period.
        """

        if (
            forecast_result is None
            or not hasattr(
                forecast_result,
                "forecast_summary",
            )
            or not forecast_result.forecast_summary
        ):
            return None

        row = self._select_period_row(
            rows=forecast_result.forecast_summary,
            period_column="forecast_period",
            requested_period=requested_period,
        )

        if row is None:
            return None

        column_mapping = {
            "forecast_orders": "forecast_orders",
            "forecast_revenue": "forecast_revenue",
            "forecast_aov": (
                "forecast_average_order_value"
            ),
        }

        value_column = column_mapping[canonical_kpi]

        if value_column not in row:
            return None

        return self._create_kpi_value(
            canonical_kpi=canonical_kpi,
            value=row[value_column],
            period=str(row.get("forecast_period")),
        )

    def _get_scenario_kpi(
        self,
        canonical_kpi: str,
        scenario_result: Any | None,
        requested_period: str | None,
    ) -> KPIValue | None:
        """
        Retrieve a KPI from a scenario-adjusted period.
        """

        if (
            scenario_result is None
            or not hasattr(
                scenario_result,
                "adjusted_forecast",
            )
            or not scenario_result.adjusted_forecast
        ):
            return None

        row = self._select_period_row(
            rows=scenario_result.adjusted_forecast,
            period_column="forecast_period",
            requested_period=requested_period,
        )

        if row is None:
            return None

        column_mapping = {
            "adjusted_orders": "adjusted_orders",
            "adjusted_revenue": "adjusted_revenue",
            "adjusted_aov": (
                "adjusted_average_order_value"
            ),
        }

        value_column = column_mapping[canonical_kpi]

        if value_column not in row:
            return None

        return self._create_kpi_value(
            canonical_kpi=canonical_kpi,
            value=row[value_column],
            period=str(row.get("forecast_period")),
        )

    def _select_period_row(
        self,
        rows: list[dict[str, Any]],
        period_column: str,
        requested_period: str | None,
    ) -> dict[str, Any] | None:
        """
        Select a requested period or default to the first available period.
        """

        if not rows:
            return None

        if requested_period is None:
            return rows[0]

        normalized_period = requested_period.strip()

        for row in rows:
            row_period = str(
                row.get(period_column, "")
            ).strip()

            if row_period == normalized_period:
                return row

        return None

    def _create_kpi_value(
        self,
        canonical_kpi: str,
        value: Any,
        period: str | None = None,
        dimension: str | None = None,
        dimension_value: str | None = None,
    ) -> KPIValue:
        """
        Create a standardized KPI output record.
        """

        display_name, unit, source = self.KPI_METADATA[
            canonical_kpi
        ]

        return KPIValue(
            kpi=canonical_kpi,
            display_name=display_name,
            value=value,
            unit=unit,
            source=source,
            period=period,
            dimension=dimension,
            dimension_value=dimension_value,
        )

    def _normalize_dimension(
        self,
        dimension: str | None,
    ) -> str | None:
        """
        Normalize dimension aliases.

        Examples:
            vehicle → vehicle_category
            cluster → pickup_cluster
            month → period
        """

        if dimension is None:
            return None

        normalized_dimension = self._normalize_name(dimension)

        return self.DIMENSION_ALIASES.get(
            normalized_dimension,
            normalized_dimension,
        )

    def _normalize_dimension_value(
        self,
        value: str,
    ) -> str:
        """
        Normalize dimension values for matching.
        """

        return " ".join(
            value.strip()
            .lower()
            .replace("_", " ")
            .split()
        )

    def _normalize_name(self, value: str) -> str:
        """
        Normalize a user-provided KPI or dimension name.

        Example:
            " Revenue_Variance " becomes "revenue variance".
        """

        return " ".join(
            value.strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )