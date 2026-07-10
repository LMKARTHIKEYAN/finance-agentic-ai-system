"""
Commentary Agent for the Finance Agentic AI System.

This module converts validated finance and operational results into
management-friendly business commentary.

The first version is deterministic and rule-based.

It does not:
- Recalculate financial metrics
- Modify finance results
- Call an LLM
- Invent unsupported explanations

A future LLM tool can use this structured commentary to improve wording.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommentaryResult:
    """
    Stores the complete management commentary.

    Attributes:
        executive_summary:
            High-level summary for management.

        kpi_commentary:
            Commentary for selected KPIs.

        variance_commentary:
            Explanation of revenue variance drivers.

        forecast_commentary:
            Explanation of the base forecast.

        scenario_commentary:
            Explanation of business-assumption adjustments.

        control_commentary:
            Finance-rule validation commentary.

        positive_drivers:
            Favourable business drivers.

        risks:
            Unfavourable results or control concerns.

        management_attention:
            Items that management should review.

        source_kpis:
            Standard KPI values used to generate commentary.
    """

    executive_summary: str
    kpi_commentary: list[str] = field(default_factory=list)
    variance_commentary: list[str] = field(default_factory=list)
    forecast_commentary: list[str] = field(default_factory=list)
    scenario_commentary: list[str] = field(default_factory=list)
    control_commentary: list[str] = field(default_factory=list)
    positive_drivers: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    management_attention: list[str] = field(default_factory=list)
    source_kpis: list[dict[str, Any]] = field(default_factory=list)


class CommentaryAgent:
    """
    Converts finance outputs into structured business commentary.

    The agent uses deterministic finance rules.

    Example:

        Revenue Variance > 0
            → Revenue exceeded budget.

        Price Effect > 0
            → Improved price/AOV supported revenue performance.

        Finance Rules Status = FAIL
            → Results should not be released until errors are resolved.
    """

    STRONG_FULFILLMENT_THRESHOLD = 95.0
    ACCEPTABLE_FULFILLMENT_THRESHOLD = 90.0

    HIGH_CANCELLATION_THRESHOLD = 10.0
    MODERATE_CANCELLATION_THRESHOLD = 5.0

    LARGE_VARIANCE_PERCENTAGE = 20.0
    MATERIAL_SCENARIO_CHANGE_PERCENTAGE = 10.0

    def analyze(
        self,
        kpi_result: Any,
        revenue_variance_result: Any | None = None,
        forecast_result: Any | None = None,
        scenario_result: Any | None = None,
        finance_rules_result: Any | None = None,
    ) -> CommentaryResult:
        """
        Generate structured management commentary.

        Args:
            kpi_result:
                KPIResult returned by KPIAgent.

            revenue_variance_result:
                Optional RevenueVarianceResult.

            forecast_result:
                Optional ForecastResult.

            scenario_result:
                Optional ScenarioResult.

            finance_rules_result:
                Optional FinanceRulesResult.

        Returns:
            CommentaryResult containing management-ready commentary.

        Raises:
            ValueError:
                If kpi_result is missing or invalid.
        """

        self._validate_kpi_result(kpi_result)

        source_kpis = self._extract_source_kpis(kpi_result)

        kpi_commentary = self._generate_kpi_commentary(
            kpi_result=kpi_result,
        )

        variance_commentary = self._generate_variance_commentary(
            revenue_variance_result=revenue_variance_result,
        )

        forecast_commentary = self._generate_forecast_commentary(
            forecast_result=forecast_result,
        )

        scenario_commentary = self._generate_scenario_commentary(
            scenario_result=scenario_result,
        )

        control_commentary = self._generate_control_commentary(
            finance_rules_result=finance_rules_result,
        )

        positive_drivers = self._identify_positive_drivers(
            kpi_result=kpi_result,
            revenue_variance_result=revenue_variance_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )

        risks = self._identify_risks(
            kpi_result=kpi_result,
            revenue_variance_result=revenue_variance_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )

        management_attention = self._identify_management_attention(
            kpi_result=kpi_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )

        executive_summary = self._generate_executive_summary(
            kpi_result=kpi_result,
            revenue_variance_result=revenue_variance_result,
            forecast_result=forecast_result,
            scenario_result=scenario_result,
            finance_rules_result=finance_rules_result,
        )

        return CommentaryResult(
            executive_summary=executive_summary,
            kpi_commentary=kpi_commentary,
            variance_commentary=variance_commentary,
            forecast_commentary=forecast_commentary,
            scenario_commentary=scenario_commentary,
            control_commentary=control_commentary,
            positive_drivers=positive_drivers,
            risks=risks,
            management_attention=management_attention,
            source_kpis=source_kpis,
        )

    def _validate_kpi_result(self, kpi_result: Any) -> None:
        """
        Validate the KPI Agent result.
        """

        if kpi_result is None:
            raise ValueError("kpi_result is required.")

        if not hasattr(kpi_result, "selected_kpis"):
            raise ValueError(
                "kpi_result must contain a selected_kpis attribute."
            )

        if not isinstance(kpi_result.selected_kpis, list):
            raise TypeError("selected_kpis must be a list.")

        if not kpi_result.selected_kpis:
            raise ValueError(
                "At least one selected KPI is required for commentary."
            )

    def _extract_source_kpis(
        self,
        kpi_result: Any,
    ) -> list[dict[str, Any]]:
        """
        Convert selected KPI objects into simple dictionaries.
        """

        source_rows: list[dict[str, Any]] = []

        for kpi in kpi_result.selected_kpis:
            source_rows.append(
                {
                    "kpi": getattr(kpi, "kpi", ""),
                    "display_name": getattr(kpi, "display_name", ""),
                    "value": getattr(kpi, "value", None),
                    "unit": getattr(kpi, "unit", ""),
                    "source": getattr(kpi, "source", ""),
                    "period": getattr(kpi, "period", None),
                    "dimension": getattr(kpi, "dimension", None),
                    "dimension_value": getattr(
                        kpi,
                        "dimension_value",
                        None,
                    ),
                }
            )

        return source_rows

    def _generate_executive_summary(
        self,
        kpi_result: Any,
        revenue_variance_result: Any | None,
        forecast_result: Any | None,
        scenario_result: Any | None,
        finance_rules_result: Any | None,
    ) -> str:
        """
        Generate a concise management-level executive summary.
        """

        sentences: list[str] = []

        actual_revenue = self._find_kpi_value(
            kpi_result,
            "actual_revenue",
        )

        fulfillment = self._find_kpi_value(
            kpi_result,
            "fulfillment_percentage",
        )

        revenue_variance = self._find_kpi_value(
            kpi_result,
            "revenue_variance",
        )

        forecast_revenue = self._find_kpi_value(
            kpi_result,
            "forecast_revenue",
        )

        finance_status = self._find_kpi_value(
            kpi_result,
            "finance_rules_status",
        )

        if actual_revenue is not None:
            sentences.append(
                "Actual revenue reached "
                f"{self._format_currency(actual_revenue)}."
            )

        if fulfillment is not None:
            sentences.append(
                "Fulfillment was "
                f"{self._format_percentage(fulfillment)}."
            )

        if revenue_variance is not None:
            if float(revenue_variance) > 0:
                sentences.append(
                    "Revenue exceeded budget by "
                    f"{self._format_currency(revenue_variance)}."
                )
            elif float(revenue_variance) < 0:
                sentences.append(
                    "Revenue was below budget by "
                    f"{self._format_currency(abs(float(revenue_variance)))}."
                )
            else:
                sentences.append(
                    "Revenue was in line with budget."
                )

        if forecast_revenue is not None:
            forecast_period = self._find_kpi_period(
                kpi_result,
                "forecast_revenue",
            )

            period_text = (
                f" for {forecast_period}"
                if forecast_period
                else ""
            )

            sentences.append(
                f"Base forecast revenue{period_text} is "
                f"{self._format_currency(forecast_revenue)}."
            )

        if scenario_result is not None:
            applied_count = int(
                getattr(
                    scenario_result,
                    "applied_assumption_count",
                    0,
                )
            )

            if applied_count > 0:
                sentences.append(
                    f"The management scenario applied "
                    f"{applied_count} business assumptions."
                )

        if finance_status is None and finance_rules_result is not None:
            finance_status = getattr(
                finance_rules_result,
                "overall_status",
                None,
            )

        if finance_status:
            status = str(finance_status).upper()

            if status == "PASS":
                sentences.append(
                    "All supplied finance-control checks passed."
                )
            elif status == "WARNING":
                sentences.append(
                    "Finance controls identified warnings requiring review."
                )
            elif status == "FAIL":
                sentences.append(
                    "Finance controls identified errors that must be resolved."
                )

        if not sentences:
            return (
                "Financial results were received, but insufficient "
                "information was available for an executive summary."
            )

        return " ".join(sentences)

    def _generate_kpi_commentary(
        self,
        kpi_result: Any,
    ) -> list[str]:
        """
        Generate commentary for each selected KPI.
        """

        comments: list[str] = []

        for kpi in kpi_result.selected_kpis:
            name = str(getattr(kpi, "display_name", "KPI"))
            value = getattr(kpi, "value", None)
            unit = str(getattr(kpi, "unit", ""))
            period = getattr(kpi, "period", None)
            dimension = getattr(kpi, "dimension", None)
            dimension_value = getattr(
                kpi,
                "dimension_value",
                None,
            )

            context_text = self._build_context_text(
                period=period,
                dimension=dimension,
                dimension_value=dimension_value,
            )

            formatted_value = self._format_value(
                value=value,
                unit=unit,
            )

            comments.append(
                f"{name}{context_text} was {formatted_value}."
            )

        return comments

    def _generate_variance_commentary(
        self,
        revenue_variance_result: Any | None,
    ) -> list[str]:
        """
        Explain revenue variance and its major drivers.
        """

        if revenue_variance_result is None:
            return []

        comments: list[str] = []

        revenue_variance = float(
            getattr(
                revenue_variance_result,
                "revenue_variance",
                0.0,
            )
        )

        price_effect = float(
            getattr(
                revenue_variance_result,
                "price_effect",
                0.0,
            )
        )

        volume_effect = float(
            getattr(
                revenue_variance_result,
                "volume_effect",
                0.0,
            )
        )

        new_discontinued_effect = float(
            getattr(
                revenue_variance_result,
                "new_discontinued_effect",
                0.0,
            )
        )

        variance_check = float(
            getattr(
                revenue_variance_result,
                "variance_check",
                0.0,
            )
        )

        if revenue_variance > 0:
            comments.append(
                "Revenue performance was favourable compared with budget."
            )
        elif revenue_variance < 0:
            comments.append(
                "Revenue performance was unfavourable compared with budget."
            )
        else:
            comments.append(
                "Revenue performance was in line with budget."
            )

        comments.append(
            self._describe_effect(
                effect_name="Price/AOV effect",
                value=price_effect,
                positive_text=(
                    "Higher price or AOV improved revenue."
                ),
                negative_text=(
                    "Lower price or AOV reduced revenue."
                ),
                neutral_text=(
                    "Price or AOV had no material revenue impact."
                ),
            )
        )

        comments.append(
            self._describe_effect(
                effect_name="Volume effect",
                value=volume_effect,
                positive_text=(
                    "Higher completed-order volume improved revenue."
                ),
                negative_text=(
                    "Lower completed-order volume reduced revenue."
                ),
                neutral_text=(
                    "Completed-order volume had no material impact."
                ),
            )
        )

        if new_discontinued_effect > 0:
            comments.append(
                "New categories generated a favourable revenue contribution "
                f"of {self._format_currency(new_discontinued_effect)}."
            )
        elif new_discontinued_effect < 0:
            comments.append(
                "Discontinued categories reduced revenue by "
                f"{self._format_currency(abs(new_discontinued_effect))}."
            )

        if abs(variance_check) <= 0.05:
            comments.append(
                "The revenue variance decomposition reconciled successfully."
            )
        else:
            comments.append(
                "The revenue variance decomposition did not reconcile and "
                "requires review."
            )

        return comments

    def _generate_forecast_commentary(
        self,
        forecast_result: Any | None,
    ) -> list[str]:
        """
        Explain base forecast results.
        """

        if (
            forecast_result is None
            or not hasattr(forecast_result, "forecast_summary")
            or not forecast_result.forecast_summary
        ):
            return []

        comments: list[str] = []

        method = str(
            getattr(
                forecast_result,
                "method",
                "Unknown forecast method",
            )
        )

        comments.append(
            f"The base forecast was prepared using {method}."
        )

        forecast_rows = forecast_result.forecast_summary

        first_row = forecast_rows[0]

        comments.append(
            "For "
            f"{first_row.get('forecast_period')}, forecast orders are "
            f"{self._format_count(first_row.get('forecast_orders', 0))} "
            "and forecast revenue is "
            f"{self._format_currency(first_row.get('forecast_revenue', 0))}."
        )

        if len(forecast_rows) > 1:
            first_revenue = float(
                first_row.get("forecast_revenue", 0)
            )

            last_row = forecast_rows[-1]

            last_revenue = float(
                last_row.get("forecast_revenue", 0)
            )

            change = last_revenue - first_revenue

            if change > 0:
                comments.append(
                    "Forecast revenue increases from the first to the "
                    "last forecast period."
                )
            elif change < 0:
                comments.append(
                    "Forecast revenue decreases from the first to the "
                    "last forecast period."
                )
            else:
                comments.append(
                    "Forecast revenue remains stable across the horizon."
                )

        return comments

    def _generate_scenario_commentary(
        self,
        scenario_result: Any | None,
    ) -> list[str]:
        """
        Explain assumption-adjusted scenario results.
        """

        if (
            scenario_result is None
            or not hasattr(scenario_result, "adjusted_forecast")
            or not scenario_result.adjusted_forecast
        ):
            return []

        comments: list[str] = []

        scenario_name = str(
            getattr(
                scenario_result,
                "scenario_name",
                "Scenario",
            )
        )

        applied_count = int(
            getattr(
                scenario_result,
                "applied_assumption_count",
                0,
            )
        )

        total_assumptions = int(
            getattr(
                scenario_result,
                "total_assumptions",
                0,
            )
        )

        comments.append(
            f"{scenario_name} applied {applied_count} of "
            f"{total_assumptions} supplied assumptions."
        )

        for row in scenario_result.adjusted_forecast:
            period = str(row.get("forecast_period", ""))

            orders_adjustment = float(
                row.get("orders_adjustment", 0)
            )

            revenue_adjustment = float(
                row.get("revenue_adjustment", 0)
            )

            if orders_adjustment != 0:
                direction = (
                    "increased"
                    if orders_adjustment > 0
                    else "reduced"
                )

                comments.append(
                    f"For {period}, business assumptions {direction} "
                    "forecast orders by "
                    f"{self._format_count(abs(orders_adjustment))}."
                )

            if revenue_adjustment != 0:
                direction = (
                    "increased"
                    if revenue_adjustment > 0
                    else "reduced"
                )

                comments.append(
                    f"For {period}, business assumptions {direction} "
                    "forecast revenue by "
                    f"{self._format_currency(abs(revenue_adjustment))}."
                )

        unapplied = getattr(
            scenario_result,
            "unapplied_assumptions",
            [],
        )

        if unapplied:
            comments.append(
                f"{len(unapplied)} assumption(s) could not be applied and "
                "require review."
            )

        return comments

    def _generate_control_commentary(
        self,
        finance_rules_result: Any | None,
    ) -> list[str]:
        """
        Explain finance-control validation status.
        """

        if finance_rules_result is None:
            return []

        status = str(
            getattr(
                finance_rules_result,
                "overall_status",
                "UNKNOWN",
            )
        ).upper()

        rules_checked = int(
            getattr(
                finance_rules_result,
                "rules_checked",
                0,
            )
        )

        passed_rules = int(
            getattr(
                finance_rules_result,
                "passed_rules",
                0,
            )
        )

        warning_count = int(
            getattr(
                finance_rules_result,
                "warning_count",
                0,
            )
        )

        error_count = int(
            getattr(
                finance_rules_result,
                "error_count",
                0,
            )
        )

        comments = [
            f"Finance controls checked {rules_checked} rules, with "
            f"{passed_rules} passing."
        ]

        if status == "PASS":
            comments.append(
                "No finance-control warnings or errors were identified."
            )
        elif status == "WARNING":
            comments.append(
                f"{warning_count} warning(s) require management review."
            )
        elif status == "FAIL":
            comments.append(
                f"{error_count} finance-control error(s) must be resolved "
                "before reporting."
            )
        else:
            comments.append(
                "Finance-control status is unavailable."
            )

        return comments

    def _identify_positive_drivers(
        self,
        kpi_result: Any,
        revenue_variance_result: Any | None,
        scenario_result: Any | None,
        finance_rules_result: Any | None,
    ) -> list[str]:
        """
        Identify favourable business drivers.
        """

        drivers: list[str] = []

        fulfillment = self._find_kpi_value(
            kpi_result,
            "fulfillment_percentage",
        )

        if fulfillment is not None:
            if float(fulfillment) >= self.STRONG_FULFILLMENT_THRESHOLD:
                drivers.append(
                    "Fulfillment performance was strong."
                )
            elif float(fulfillment) >= self.ACCEPTABLE_FULFILLMENT_THRESHOLD:
                drivers.append(
                    "Fulfillment performance remained within an "
                    "acceptable range."
                )

        if revenue_variance_result is not None:
            price_effect = float(
                getattr(
                    revenue_variance_result,
                    "price_effect",
                    0,
                )
            )

            volume_effect = float(
                getattr(
                    revenue_variance_result,
                    "volume_effect",
                    0,
                )
            )

            if price_effect > 0:
                drivers.append(
                    "Favourable price/AOV movement supported revenue."
                )

            if volume_effect > 0:
                drivers.append(
                    "Higher completed-order volume supported revenue."
                )

        if scenario_result is not None:
            for row in scenario_result.adjusted_forecast:
                if float(row.get("orders_adjustment", 0)) > 0:
                    drivers.append(
                        "Business assumptions created additional "
                        "forecast-order upside."
                    )
                    break

        if finance_rules_result is not None:
            if str(
                getattr(
                    finance_rules_result,
                    "overall_status",
                    "",
                )
            ).upper() == "PASS":
                drivers.append(
                    "All finance-control checks passed."
                )

        return self._remove_duplicates(drivers)

    def _identify_risks(
        self,
        kpi_result: Any,
        revenue_variance_result: Any | None,
        scenario_result: Any | None,
        finance_rules_result: Any | None,
    ) -> list[str]:
        """
        Identify unfavourable results and risks.
        """

        risks: list[str] = []

        cancellation = self._find_kpi_value(
            kpi_result,
            "cancellation_percentage",
        )

        fulfillment = self._find_kpi_value(
            kpi_result,
            "fulfillment_percentage",
        )

        if cancellation is not None:
            if float(cancellation) >= self.HIGH_CANCELLATION_THRESHOLD:
                risks.append(
                    "Cancellation is high and may require operational action."
                )
            elif (
                float(cancellation)
                >= self.MODERATE_CANCELLATION_THRESHOLD
            ):
                risks.append(
                    "Cancellation should be monitored."
                )

        if fulfillment is not None:
            if float(fulfillment) < self.ACCEPTABLE_FULFILLMENT_THRESHOLD:
                risks.append(
                    "Fulfillment is below the acceptable threshold."
                )

        if revenue_variance_result is not None:
            revenue_variance = float(
                getattr(
                    revenue_variance_result,
                    "revenue_variance",
                    0,
                )
            )

            price_effect = float(
                getattr(
                    revenue_variance_result,
                    "price_effect",
                    0,
                )
            )

            volume_effect = float(
                getattr(
                    revenue_variance_result,
                    "volume_effect",
                    0,
                )
            )

            if revenue_variance < 0:
                risks.append(
                    "Revenue is below budget."
                )

            if price_effect < 0:
                risks.append(
                    "Price/AOV movement reduced revenue."
                )

            if volume_effect < 0:
                risks.append(
                    "Lower completed-order volume reduced revenue."
                )

        if scenario_result is not None:
            unapplied = getattr(
                scenario_result,
                "unapplied_assumptions",
                [],
            )

            if unapplied:
                risks.append(
                    "Some business assumptions could not be applied."
                )

        if finance_rules_result is not None:
            status = str(
                getattr(
                    finance_rules_result,
                    "overall_status",
                    "",
                )
            ).upper()

            if status == "WARNING":
                risks.append(
                    "Finance controls identified warnings."
                )
            elif status == "FAIL":
                risks.append(
                    "Finance controls identified errors."
                )

        return self._remove_duplicates(risks)

    def _identify_management_attention(
        self,
        kpi_result: Any,
        scenario_result: Any | None,
        finance_rules_result: Any | None,
    ) -> list[str]:
        """
        Identify items requiring management attention.
        """

        items: list[str] = []

        if getattr(kpi_result, "unavailable_kpis", []):
            items.append(
                "Some requested KPIs were unavailable."
            )

        if getattr(kpi_result, "unknown_kpis", []):
            items.append(
                "Some requested KPI names were not recognized."
            )

        if scenario_result is not None:
            unapplied = getattr(
                scenario_result,
                "unapplied_assumptions",
                [],
            )

            if unapplied:
                items.append(
                    "Review unapplied business assumptions."
                )

        if finance_rules_result is not None:
            status = str(
                getattr(
                    finance_rules_result,
                    "overall_status",
                    "",
                )
            ).upper()

            if status == "WARNING":
                items.append(
                    "Review finance-control warnings before final reporting."
                )
            elif status == "FAIL":
                items.append(
                    "Resolve finance-control errors before releasing results."
                )

        return self._remove_duplicates(items)

    def _find_kpi_value(
        self,
        kpi_result: Any,
        canonical_kpi: str,
    ) -> Any | None:
        """
        Find one selected KPI value using its canonical name.
        """

        for kpi in kpi_result.selected_kpis:
            if getattr(kpi, "kpi", None) == canonical_kpi:
                return getattr(kpi, "value", None)

        return None

    def _find_kpi_period(
        self,
        kpi_result: Any,
        canonical_kpi: str,
    ) -> str | None:
        """
        Find the period attached to one selected KPI.
        """

        for kpi in kpi_result.selected_kpis:
            if getattr(kpi, "kpi", None) == canonical_kpi:
                period = getattr(kpi, "period", None)

                return str(period) if period else None

        return None

    def _describe_effect(
        self,
        effect_name: str,
        value: float,
        positive_text: str,
        negative_text: str,
        neutral_text: str,
    ) -> str:
        """
        Generate commentary for a positive, negative, or neutral effect.
        """

        if value > 0:
            return (
                f"{effect_name} was favourable at "
                f"{self._format_currency(value)}. "
                f"{positive_text}"
            )

        if value < 0:
            return (
                f"{effect_name} was unfavourable at "
                f"{self._format_currency(abs(value))}. "
                f"{negative_text}"
            )

        return neutral_text

    def _build_context_text(
        self,
        period: str | None,
        dimension: str | None,
        dimension_value: str | None,
    ) -> str:
        """
        Build readable KPI context.
        """

        context_parts: list[str] = []

        if dimension and dimension_value:
            readable_dimension = dimension.replace("_", " ")

            context_parts.append(
                f"for {readable_dimension} {dimension_value}"
            )

        if period:
            context_parts.append(f"for {period}")

        if not context_parts:
            return ""

        return " " + " ".join(context_parts)

    def _format_value(
        self,
        value: Any,
        unit: str,
    ) -> str:
        """
        Format a KPI value based on its unit.
        """

        if unit == "currency":
            return self._format_currency(value)

        if unit == "currency_per_order":
            return self._format_currency(value)

        if unit == "percentage":
            return self._format_percentage(value)

        if unit == "decimal_percentage":
            return self._format_decimal_percentage(value)

        if unit == "percentage_points":
            return self._format_percentage_points(value)

        if unit == "count":
            return self._format_count(value)

        return str(value)

    def _format_currency(self, value: Any) -> str:
        """
        Format currency using Indian-style business units.
        """

        numeric_value = float(value)
        absolute_value = abs(numeric_value)
        sign = "-" if numeric_value < 0 else ""

        if absolute_value >= 10_000_000:
            return (
                f"{sign}₹{absolute_value / 10_000_000:,.2f} crore"
            )

        if absolute_value >= 100_000:
            return (
                f"{sign}₹{absolute_value / 100_000:,.2f} lakh"
            )

        return f"{sign}₹{absolute_value:,.2f}"

    def _format_percentage(self, value: Any) -> str:
        """
        Format a percentage stored from 0 to 100.
        """

        return f"{float(value):,.2f}%"

    def _format_decimal_percentage(self, value: Any) -> str:
        """
        Format a percentage stored as a decimal.

        Example:
            0.40 becomes 40.00%.
        """

        return f"{float(value) * 100:,.2f}%"

    def _format_percentage_points(self, value: Any) -> str:
        """
        Format percentage-point movement stored as a decimal.
        """

        return f"{float(value) * 100:,.2f} percentage points"

    def _format_count(self, value: Any) -> str:
        """
        Format order or record counts.
        """

        return f"{int(round(float(value))):,}"

    def _remove_duplicates(
        self,
        values: list[str],
    ) -> list[str]:
        """
        Remove duplicate commentary while preserving order.
        """

        return list(dict.fromkeys(values))