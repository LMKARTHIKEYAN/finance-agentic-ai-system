"""
Scenario Agent for the Finance Agentic AI System.

This module applies structured business assumptions to the base rolling
forecast produced by ForecastAgent.

Flow:
    Historical Actuals
        ↓
    ForecastAgent
        ↓
    Base Forecast
        ↓
    ScenarioAgent
        ↓
    Assumption-Adjusted Forecast

Supported assumption metrics:
- orders
- revenue
- cost, when forecast cost is available

Supported impact types:
- percentage
- absolute
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


ImpactType = Literal["percentage", "absolute"]


@dataclass
class ScenarioResult:
    """
    Stores the scenario-adjusted forecast output.

    Attributes:
        scenario_name:
            Name of the scenario, such as Management Case.

        base_method:
            Forecasting method used to create the base forecast.

        adjusted_forecast:
            Forecast rows after applying business assumptions.

        applied_assumptions:
            Assumptions successfully applied to forecast values.

        unapplied_assumptions:
            Assumptions that could not be applied.

        total_assumptions:
            Total number of assumptions received.

        applied_assumption_count:
            Number of successfully applied assumptions.
    """

    scenario_name: str
    base_method: str
    total_assumptions: int
    applied_assumption_count: int
    adjusted_forecast: list[dict[str, Any]] = field(default_factory=list)
    applied_assumptions: list[dict[str, Any]] = field(default_factory=list)
    unapplied_assumptions: list[dict[str, Any]] = field(default_factory=list)


class ScenarioAgent:
    """
    Applies business assumptions to a base forecast.

    Percentage formula:

        Adjusted Value =
        Base Value × (1 + Impact Value / 100)

    Example:

        Base Orders = 10,000
        Marketing Impact = +15%

        Adjusted Orders =
        10,000 × (1 + 15 / 100)
        = 11,500

    Absolute formula:

        Adjusted Value =
        Base Value + Impact Value

    Example:

        Base Orders = 10,000
        New Cluster Increment = 2,000

        Adjusted Orders =
        10,000 + 2,000
        = 12,000
    """

    REQUIRED_ASSUMPTION_COLUMNS = {
        "month",
        "assumption",
        "metric",
        "impact_type",
        "impact_value",
    }

    SUPPORTED_METRICS = {
        "orders",
        "revenue",
        "cost",
    }

    SUPPORTED_IMPACT_TYPES = {
        "percentage",
        "absolute",
    }

    METRIC_COLUMN_MAPPING = {
        "orders": "adjusted_orders",
        "revenue": "adjusted_revenue",
        "cost": "adjusted_cost",
    }

    def analyze(
        self,
        forecast_result: Any,
        assumptions: pd.DataFrame | list[dict[str, Any]],
        scenario_name: str = "Management Case",
    ) -> ScenarioResult:
        """
        Apply business assumptions to a base forecast.

        Args:
            forecast_result:
                ForecastResult returned by ForecastAgent.

            assumptions:
                Business assumptions as a Pandas DataFrame or list of
                dictionaries.

            scenario_name:
                Name given to the scenario.

        Returns:
            ScenarioResult containing the adjusted forecast.

        Raises:
            TypeError:
                If assumptions are not a DataFrame or list.

            ValueError:
                If required forecast or assumption data is missing.
        """

        base_forecast = self._prepare_forecast_data(forecast_result)

        assumption_data = self._prepare_assumption_data(assumptions)

        adjusted_forecast, applied, unapplied = self._apply_assumptions(
            base_forecast=base_forecast,
            assumption_data=assumption_data,
        )

        base_method = getattr(
            forecast_result,
            "method",
            "Unknown Forecast Method",
        )

        return ScenarioResult(
            scenario_name=scenario_name,
            base_method=base_method,
            total_assumptions=len(assumption_data),
            applied_assumption_count=len(applied),
            adjusted_forecast=adjusted_forecast,
            applied_assumptions=applied,
            unapplied_assumptions=unapplied,
        )

    def _prepare_forecast_data(self, forecast_result: Any) -> pd.DataFrame:
        """
        Convert ForecastResult into a scenario-ready DataFrame.

        Expected forecast fields:
        - forecast_period
        - forecast_orders
        - forecast_revenue
        - forecast_average_order_value
        """

        if not hasattr(forecast_result, "forecast_summary"):
            raise ValueError(
                "forecast_result must contain a forecast_summary attribute."
            )

        forecast_summary = forecast_result.forecast_summary

        if not isinstance(forecast_summary, list):
            raise TypeError("forecast_summary must be a list of dictionaries.")

        if not forecast_summary:
            raise ValueError("forecast_summary cannot be empty.")

        forecast_data = pd.DataFrame(forecast_summary).copy()

        required_columns = {
            "forecast_period",
            "forecast_orders",
            "forecast_revenue",
        }

        missing_columns = required_columns - set(forecast_data.columns)

        if missing_columns:
            raise ValueError(
                "Missing required forecast fields: "
                f"{sorted(missing_columns)}"
            )

        forecast_data["forecast_period"] = (
            forecast_data["forecast_period"]
            .astype(str)
            .str.strip()
        )

        forecast_data["forecast_orders"] = pd.to_numeric(
            forecast_data["forecast_orders"],
            errors="coerce",
        )

        forecast_data["forecast_revenue"] = pd.to_numeric(
            forecast_data["forecast_revenue"],
            errors="coerce",
        )

        forecast_data = forecast_data.dropna(
            subset=[
                "forecast_period",
                "forecast_orders",
                "forecast_revenue",
            ]
        )

        if forecast_data.empty:
            raise ValueError("No valid forecast rows are available.")

        if (forecast_data["forecast_orders"] < 0).any():
            raise ValueError("Forecast orders cannot be negative.")

        if (forecast_data["forecast_revenue"] < 0).any():
            raise ValueError("Forecast revenue cannot be negative.")

        forecast_data["base_orders"] = forecast_data["forecast_orders"]
        forecast_data["base_revenue"] = forecast_data["forecast_revenue"]

        forecast_data["adjusted_orders"] = (forecast_data["forecast_orders"].astype(float) )
                                            
        forecast_data["adjusted_revenue"] = (forecast_data["forecast_revenue"].astype(float) )

        if "forecast_cost" in forecast_data.columns:
            forecast_data["forecast_cost"] = pd.to_numeric(
                forecast_data["forecast_cost"],
                errors="coerce",
            ).fillna(0)

            forecast_data["base_cost"] = forecast_data["forecast_cost"]
            forecast_data["adjusted_cost"] = forecast_data["forecast_cost"]

        forecast_data["base_average_order_value"] = forecast_data.apply(
            lambda row: self._safe_divide(
                numerator=float(row["base_revenue"]),
                denominator=float(row["base_orders"]),
            ),
            axis=1,
        )

        return forecast_data

    def _prepare_assumption_data(
        self,
        assumptions: pd.DataFrame | list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Clean and standardize business assumptions.
        """

        if isinstance(assumptions, pd.DataFrame):
            assumption_data = assumptions.copy()

        elif isinstance(assumptions, list):
            assumption_data = pd.DataFrame(assumptions)

        else:
            raise TypeError(
                "assumptions must be a pandas DataFrame or list of dictionaries."
            )

        if assumption_data.empty:
            raise ValueError("Business assumptions cannot be empty.")

        assumption_data.columns = [
            str(column).strip().lower()
            for column in assumption_data.columns
        ]

        missing_columns = (
            self.REQUIRED_ASSUMPTION_COLUMNS
            - set(assumption_data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required assumption columns: "
                f"{sorted(missing_columns)}"
            )

        assumption_data["month"] = (
            assumption_data["month"]
            .astype(str)
            .str.strip()
        )

        assumption_data["assumption"] = (
            assumption_data["assumption"]
            .astype(str)
            .str.strip()
        )

        assumption_data["metric"] = (
            assumption_data["metric"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        assumption_data["impact_type"] = (
            assumption_data["impact_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        assumption_data["impact_value"] = pd.to_numeric(
            assumption_data["impact_value"],
            errors="coerce",
        )

        assumption_data = assumption_data.dropna(
            subset=[
                "month",
                "assumption",
                "metric",
                "impact_type",
                "impact_value",
            ]
        )

        if assumption_data.empty:
            raise ValueError(
                "No valid assumptions remain after data preparation."
            )

        if "direction" not in assumption_data.columns:
            assumption_data["direction"] = ""

        if "business_reason" not in assumption_data.columns:
            assumption_data["business_reason"] = ""

        return assumption_data.reset_index(drop=True)

    def _apply_assumptions(
        self,
        base_forecast: pd.DataFrame,
        assumption_data: pd.DataFrame,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """
        Apply assumptions to matching forecast periods.

        Multiple assumptions for the same period and metric are applied
        sequentially.

        Example:

            Base Orders = 10,000

            Marketing Campaign = +15%
            Driver Constraint = -3%

            Step 1:
            10,000 × 1.15 = 11,500

            Step 2:
            11,500 × 0.97 = 11,155
        """

        adjusted_data = base_forecast.copy()

        adjusted_data["applied_assumption_names"] = [
            [] for _ in range(len(adjusted_data))
        ]

        applied_assumptions: list[dict[str, Any]] = []
        unapplied_assumptions: list[dict[str, Any]] = []

        for _, assumption_row in assumption_data.iterrows():
            period = str(assumption_row["month"])
            metric = str(assumption_row["metric"])
            impact_type = str(assumption_row["impact_type"])
            impact_value = float(assumption_row["impact_value"])
            assumption_name = str(assumption_row["assumption"])

            forecast_match = adjusted_data[
                adjusted_data["forecast_period"] == period
            ]

            if forecast_match.empty:
                unapplied_assumptions.append(
                    self._create_unapplied_record(
                        assumption_row=assumption_row,
                        reason="Forecast period was not found.",
                    )
                )
                continue

            if metric not in self.SUPPORTED_METRICS:
                unapplied_assumptions.append(
                    self._create_unapplied_record(
                        assumption_row=assumption_row,
                        reason=f"Unsupported metric: {metric}.",
                    )
                )
                continue

            if impact_type not in self.SUPPORTED_IMPACT_TYPES:
                unapplied_assumptions.append(
                    self._create_unapplied_record(
                        assumption_row=assumption_row,
                        reason=f"Unsupported impact type: {impact_type}.",
                    )
                )
                continue

            target_column = self.METRIC_COLUMN_MAPPING[metric]

            if target_column not in adjusted_data.columns:
                unapplied_assumptions.append(
                    self._create_unapplied_record(
                        assumption_row=assumption_row,
                        reason=(
                            f"The base forecast does not contain "
                            f"a value for metric: {metric}."
                        ),
                    )
                )
                continue

            matching_indexes = forecast_match.index.tolist()

            for row_index in matching_indexes:
                value_before = float(
                    adjusted_data.at[row_index, target_column]
                )

                value_after = self._apply_single_impact(
                    current_value=value_before,
                    impact_type=impact_type,
                    impact_value=impact_value,
                )

                value_after = max(value_after, 0.0)

                adjusted_data.at[row_index, target_column] = value_after

                current_names = list(
                    adjusted_data.at[
                        row_index,
                        "applied_assumption_names",
                    ]
                )

                current_names.append(assumption_name)

                adjusted_data.at[
                    row_index,
                    "applied_assumption_names",
                ] = current_names

                applied_assumptions.append(
                    {
                        "forecast_period": period,
                        "assumption": assumption_name,
                        "metric": metric,
                        "impact_type": impact_type,
                        "impact_value": impact_value,
                        "value_before": round(value_before, 2),
                        "value_after": round(value_after, 2),
                        "absolute_change": round(
                            value_after - value_before,
                            2,
                        ),
                        "business_reason": str(
                            assumption_row.get("business_reason", "")
                        ),
                    }
                )

        adjusted_data["adjusted_average_order_value"] = adjusted_data.apply(
            lambda row: self._safe_divide(
                numerator=float(row["adjusted_revenue"]),
                denominator=float(row["adjusted_orders"]),
            ),
            axis=1,
        )

        adjusted_data["orders_adjustment"] = (
            adjusted_data["adjusted_orders"]
            - adjusted_data["base_orders"]
        )

        adjusted_data["revenue_adjustment"] = (
            adjusted_data["adjusted_revenue"]
            - adjusted_data["base_revenue"]
        )

        adjusted_data["scenario_commentary"] = adjusted_data.apply(
            self._generate_period_commentary,
            axis=1,
        )

        adjusted_rows = self._format_adjusted_output(adjusted_data)

        return (
            adjusted_rows,
            applied_assumptions,
            unapplied_assumptions,
        )

    def _apply_single_impact(
        self,
        current_value: float,
        impact_type: str,
        impact_value: float,
    ) -> float:
        """
        Apply one assumption to one forecast value.
        """

        if impact_type == "percentage":
            return current_value * (1 + impact_value / 100)

        if impact_type == "absolute":
            return current_value + impact_value

        raise ValueError(f"Unsupported impact type: {impact_type}")

    def _format_adjusted_output(
        self,
        adjusted_data: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Convert adjusted scenario results into clean dictionaries.
        """

        result_rows: list[dict[str, Any]] = []

        for _, row in adjusted_data.iterrows():
            output_row: dict[str, Any] = {
                "forecast_period": row["forecast_period"],
                "base_orders": int(round(row["base_orders"])),
                "adjusted_orders": int(round(row["adjusted_orders"])),
                "orders_adjustment": int(
                    round(row["orders_adjustment"])
                ),
                "base_revenue": round(float(row["base_revenue"]), 2),
                "adjusted_revenue": round(
                    float(row["adjusted_revenue"]),
                    2,
                ),
                "revenue_adjustment": round(
                    float(row["revenue_adjustment"]),
                    2,
                ),
                "base_average_order_value": round(
                    float(row["base_average_order_value"]),
                    2,
                ),
                "adjusted_average_order_value": round(
                    float(row["adjusted_average_order_value"]),
                    2,
                ),
                "applied_assumptions": list(
                    row["applied_assumption_names"]
                ),
                "scenario_commentary": row[
                    "scenario_commentary"
                ],
            }

            if "base_cost" in adjusted_data.columns:
                output_row["base_cost"] = round(
                    float(row["base_cost"]),
                    2,
                )

                output_row["adjusted_cost"] = round(
                    float(row["adjusted_cost"]),
                    2,
                )

                output_row["cost_adjustment"] = round(
                    float(row["adjusted_cost"] - row["base_cost"]),
                    2,
                )

            result_rows.append(output_row)

        return result_rows

    def _generate_period_commentary(self, row: pd.Series) -> str:
        """
        Generate simple scenario commentary for one forecast period.
        """

        comments: list[str] = []

        assumption_names = row["applied_assumption_names"]

        if assumption_names:
            comments.append(
                "Applied assumptions: "
                + ", ".join(assumption_names)
                + "."
            )
        else:
            comments.append(
                "No business assumptions were applied."
            )

        orders_change = float(row["orders_adjustment"])
        revenue_change = float(row["revenue_adjustment"])

        if orders_change > 0:
            comments.append(
                "Business assumptions increased forecast orders."
            )
        elif orders_change < 0:
            comments.append(
                "Business assumptions reduced forecast orders."
            )
        else:
            comments.append(
                "Forecast orders were unchanged."
            )

        if revenue_change > 0:
            comments.append(
                "Business assumptions increased forecast revenue."
            )
        elif revenue_change < 0:
            comments.append(
                "Business assumptions reduced forecast revenue."
            )
        else:
            comments.append(
                "Forecast revenue was unchanged."
            )

        return " ".join(comments)

    def _create_unapplied_record(
        self,
        assumption_row: pd.Series,
        reason: str,
    ) -> dict[str, Any]:
        """
        Create a record for an assumption that could not be applied.
        """

        return {
            "month": str(assumption_row.get("month", "")),
            "assumption": str(
                assumption_row.get("assumption", "")
            ),
            "metric": str(assumption_row.get("metric", "")),
            "impact_type": str(
                assumption_row.get("impact_type", "")
            ),
            "impact_value": float(
                assumption_row.get("impact_value", 0)
            ),
            "reason": reason,
        }

    def _safe_divide(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        """
        Divide safely and prevent divide-by-zero errors.
        """

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)