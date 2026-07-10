"""
Finance Rules Agent for the Finance Agentic AI System.

This module applies financial control rules to outputs produced by finance
and analytics agents.

The agent checks:
- Operational actuals
- Budget metrics
- Revenue variance
- Product-level GP variance
- Portfolio-level GP bridge
- Base forecasts
- Scenario-adjusted forecasts

This agent does not modify financial results.
It only identifies passed rules, warnings, and failed rules.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class RuleSeverity(str, Enum):
    """
    Defines the seriousness of a finance-rule result.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class FinanceRuleIssue:
    """
    Stores one finance-rule finding.

    Attributes:
        rule_code:
            Unique code used to identify the rule.

        rule_name:
            Human-readable name of the rule.

        severity:
            Information, warning, or error.

        message:
            Explanation of the finding.

        metric:
            Name of the affected financial metric.

        actual_value:
            Value found by the rule.

        expected_value:
            Expected value, range, or condition.

        context:
            Extra information such as period, vehicle, or product.
    """

    rule_code: str
    rule_name: str
    severity: RuleSeverity
    message: str
    metric: str | None = None
    actual_value: Any = None
    expected_value: Any = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinanceRulesResult:
    """
    Stores the complete finance-rules assessment.

    Attributes:
        overall_status:
            PASS, WARNING, or FAIL.

        rules_checked:
            Total number of control checks performed.

        passed_rules:
            Number of checks that passed.

        warning_count:
            Number of warning findings.

        error_count:
            Number of failed rules.

        issues:
            Detailed warning and error records.
    """

    overall_status: str
    rules_checked: int
    passed_rules: int
    warning_count: int
    error_count: int
    issues: list[FinanceRuleIssue] = field(default_factory=list)


class FinanceRulesAgent:
    """
    Applies finance and business control rules to calculated results.

    This agent performs control checks only.

    It does not:
    - Clean source data
    - Recalculate business metrics
    - Change financial results
    - Generate LLM commentary
    """

    DEFAULT_VARIANCE_TOLERANCE = 0.05
    DEFAULT_PERCENTAGE_TOLERANCE = 0.000001

    def __init__(
        self,
        variance_tolerance: float = DEFAULT_VARIANCE_TOLERANCE,
        percentage_tolerance: float = DEFAULT_PERCENTAGE_TOLERANCE,
        maximum_aov: float | None = None,
        maximum_forecast_growth_percentage: float | None = 50.0,
    ) -> None:
        """
        Configure finance-rule thresholds.

        Args:
            variance_tolerance:
                Maximum acceptable currency difference in a variance check.

                Example:
                A variance check between -0.05 and +0.05 is considered balanced.

            percentage_tolerance:
                Maximum acceptable difference for GP% bridge checks.

            maximum_aov:
                Optional business-defined upper limit for average order value.

                If None, only negative AOV is treated as invalid.

            maximum_forecast_growth_percentage:
                Optional warning threshold for scenario changes.

                Example:
                50 means an adjustment above +50% or below -50%
                produces a warning.
        """

        if variance_tolerance < 0:
            raise ValueError("variance_tolerance cannot be negative.")

        if percentage_tolerance < 0:
            raise ValueError("percentage_tolerance cannot be negative.")

        if maximum_aov is not None and maximum_aov <= 0:
            raise ValueError("maximum_aov must be positive when provided.")

        if (
            maximum_forecast_growth_percentage is not None
            and maximum_forecast_growth_percentage <= 0
        ):
            raise ValueError(
                "maximum_forecast_growth_percentage must be positive."
            )

        self.variance_tolerance = variance_tolerance
        self.percentage_tolerance = percentage_tolerance
        self.maximum_aov = maximum_aov
        self.maximum_forecast_growth_percentage = (
            maximum_forecast_growth_percentage
        )

        self._rules_checked = 0
        self._passed_rules = 0
        self._issues: list[FinanceRuleIssue] = []

    def analyze(
        self,
        operations_result: Any | None = None,
        budget_result: Any | None = None,
        revenue_variance_result: Any | None = None,
        gp_product_result: Any | None = None,
        gp_portfolio_result: Any | None = None,
        forecast_result: Any | None = None,
        scenario_result: Any | None = None,
    ) -> FinanceRulesResult:
        """
        Run finance controls on available agent outputs.

        All inputs are optional, but at least one result must be provided.

        Args:
            operations_result:
                Output from OperationsAnalysisAgent.

            budget_result:
                Output from BudgetAgent.

            revenue_variance_result:
                Output from RevenueVarianceAgent.

            gp_product_result:
                Output from GPProductVarianceAgent.

            gp_portfolio_result:
                Output from GPPortfolioVarianceAgent.

            forecast_result:
                Output from ForecastAgent.

            scenario_result:
                Output from ScenarioAgent.

        Returns:
            FinanceRulesResult containing the control assessment.
        """

        supplied_results = [
            operations_result,
            budget_result,
            revenue_variance_result,
            gp_product_result,
            gp_portfolio_result,
            forecast_result,
            scenario_result,
        ]

        if all(result is None for result in supplied_results):
            raise ValueError(
                "At least one finance or analytics result must be provided."
            )

        self._reset_assessment()

        if operations_result is not None:
            self._check_operations_result(operations_result)

        if budget_result is not None:
            self._check_budget_result(budget_result)

        if revenue_variance_result is not None:
            self._check_revenue_variance_result(
                revenue_variance_result
            )

        if gp_product_result is not None:
            self._check_gp_product_result(gp_product_result)

        if gp_portfolio_result is not None:
            self._check_gp_portfolio_result(gp_portfolio_result)

        if forecast_result is not None:
            self._check_forecast_result(forecast_result)

        if scenario_result is not None:
            self._check_scenario_result(scenario_result)

        return self._create_result()

    def _check_operations_result(self, result: Any) -> None:
        """
        Check operational actual metrics.
        """

        self._check_non_negative(
            value=result.total_orders,
            metric="total_orders",
            rule_code="OPS_001",
            rule_name="Total orders must not be negative",
        )

        self._check_non_negative(
            value=result.completed_orders,
            metric="completed_orders",
            rule_code="OPS_002",
            rule_name="Completed orders must not be negative",
        )

        self._check_non_negative(
            value=result.cancelled_orders,
            metric="cancelled_orders",
            rule_code="OPS_003",
            rule_name="Cancelled orders must not be negative",
        )

        self._check_non_negative(
            value=result.total_revenue,
            metric="total_revenue",
            rule_code="OPS_004",
            rule_name="Actual revenue must not be negative",
        )

        self._check_non_negative(
            value=result.average_order_value,
            metric="average_order_value",
            rule_code="OPS_005",
            rule_name="Actual AOV must not be negative",
        )

        self._check_percentage_range(
            value=result.fulfillment_percentage,
            metric="fulfillment_percentage",
            rule_code="OPS_006",
            rule_name="Fulfillment must be between 0% and 100%",
        )

        self._check_percentage_range(
            value=result.cancellation_percentage,
            metric="cancellation_percentage",
            rule_code="OPS_007",
            rule_name="Cancellation must be between 0% and 100%",
        )

        self._check_less_than_or_equal(
            left_value=result.completed_orders,
            right_value=result.total_orders,
            left_metric="completed_orders",
            right_metric="total_orders",
            rule_code="OPS_008",
            rule_name="Completed orders cannot exceed total orders",
        )

        self._check_less_than_or_equal(
            left_value=result.cancelled_orders,
            right_value=result.total_orders,
            left_metric="cancelled_orders",
            right_metric="total_orders",
            rule_code="OPS_009",
            rule_name="Cancelled orders cannot exceed total orders",
        )

        self._check_status_coverage(result)

        if self.maximum_aov is not None:
            self._check_maximum_value(
                value=result.average_order_value,
                maximum_value=self.maximum_aov,
                metric="average_order_value",
                rule_code="OPS_011",
                rule_name="Actual AOV is within configured limit",
            )

    def _check_budget_result(self, result: Any) -> None:
        """
        Check budget metrics.
        """

        self._check_non_negative(
            value=result.total_budget_orders,
            metric="total_budget_orders",
            rule_code="BUD_001",
            rule_name="Budget orders must not be negative",
        )

        self._check_non_negative(
            value=result.total_budget_revenue,
            metric="total_budget_revenue",
            rule_code="BUD_002",
            rule_name="Budget revenue must not be negative",
        )

        self._check_non_negative(
            value=result.budget_average_order_value,
            metric="budget_average_order_value",
            rule_code="BUD_003",
            rule_name="Budget AOV must not be negative",
        )

        if self.maximum_aov is not None:
            self._check_maximum_value(
                value=result.budget_average_order_value,
                maximum_value=self.maximum_aov,
                metric="budget_average_order_value",
                rule_code="BUD_004",
                rule_name="Budget AOV is within configured limit",
            )

    def _check_revenue_variance_result(self, result: Any) -> None:
        """
        Check revenue variance decomposition.
        """

        self._check_finite_number(
            value=result.revenue_variance,
            metric="revenue_variance",
            rule_code="VAR_001",
            rule_name="Revenue variance must be a valid number",
        )

        self._check_variance_balance(
            check_value=result.variance_check,
            context={"level": "total"},
            rule_code="VAR_002",
            rule_name="Total revenue variance decomposition must balance",
        )

        for row in result.vehicle_variance_summary:
            self._check_variance_balance(
                check_value=row.get("variance_check", 0),
                context={
                    "level": "vehicle_category",
                    "vehicle_category": row.get("vehicle_category"),
                },
                rule_code="VAR_003",
                rule_name=(
                    "Vehicle revenue variance decomposition must balance"
                ),
            )

    def _check_gp_product_result(self, result: Any) -> None:
        """
        Check product-level GP% analysis.
        """

        for row in result.product_analysis:
            context = {
                "category": row.get("category"),
                "product": row.get("product"),
            }

            self._check_decimal_percentage_range(
                value=row.get("base_gp_percentage", 0),
                metric="base_gp_percentage",
                context=context,
                rule_code="GPP_001",
                rule_name="Base product GP% must be between -100% and 100%",
            )

            self._check_decimal_percentage_range(
                value=row.get("actual_gp_percentage", 0),
                metric="actual_gp_percentage",
                context=context,
                rule_code="GPP_002",
                rule_name="Actual product GP% must be between -100% and 100%",
            )

            self._check_gp_balance(
                check_value=row.get("check", 0),
                context=context,
                rule_code="GPP_003",
                rule_name="Product GP% decomposition must balance",
            )

    def _check_gp_portfolio_result(self, result: Any) -> None:
        """
        Check portfolio-level GP% bridge.
        """

        self._check_decimal_percentage_range(
            value=result.base_gp_percentage,
            metric="base_gp_percentage",
            context={"level": "portfolio"},
            rule_code="GPF_001",
            rule_name="Base portfolio GP% must be between -100% and 100%",
        )

        self._check_decimal_percentage_range(
            value=result.actual_gp_percentage,
            metric="actual_gp_percentage",
            context={"level": "portfolio"},
            rule_code="GPF_002",
            rule_name="Actual portfolio GP% must be between -100% and 100%",
        )

        self._check_gp_balance(
            check_value=result.check,
            context={"level": "portfolio"},
            rule_code="GPF_003",
            rule_name="Portfolio GP% bridge must balance",
        )

    def _check_forecast_result(self, result: Any) -> None:
        """
        Check base forecast results.
        """

        if not result.forecast_summary:
            self._record_issue(
                rule_code="FCT_001",
                rule_name="Forecast must contain future periods",
                severity=RuleSeverity.ERROR,
                message="Forecast summary is empty.",
                metric="forecast_summary",
                actual_value=0,
                expected_value="At least one forecast period",
            )
            return

        self._record_pass()

        seen_periods: set[str] = set()

        for row in result.forecast_summary:
            period = str(row.get("forecast_period", ""))
            context = {"forecast_period": period}

            self._check_non_negative(
                value=row.get("forecast_orders", 0),
                metric="forecast_orders",
                context=context,
                rule_code="FCT_002",
                rule_name="Forecast orders must not be negative",
            )

            self._check_non_negative(
                value=row.get("forecast_revenue", 0),
                metric="forecast_revenue",
                context=context,
                rule_code="FCT_003",
                rule_name="Forecast revenue must not be negative",
            )

            self._check_non_negative(
                value=row.get("forecast_average_order_value", 0),
                metric="forecast_average_order_value",
                context=context,
                rule_code="FCT_004",
                rule_name="Forecast AOV must not be negative",
            )

            self._check_unique_period(
                period=period,
                seen_periods=seen_periods,
                context=context,
            )

    def _check_scenario_result(self, result: Any) -> None:
        """
        Check assumption-adjusted forecast results.
        """

        if not result.adjusted_forecast:
            self._record_issue(
                rule_code="SCN_001",
                rule_name="Scenario must contain adjusted forecast periods",
                severity=RuleSeverity.ERROR,
                message="Adjusted forecast is empty.",
                metric="adjusted_forecast",
                actual_value=0,
                expected_value="At least one adjusted forecast period",
            )
            return

        self._record_pass()

        if result.applied_assumption_count > result.total_assumptions:
            self._record_issue(
                rule_code="SCN_002",
                rule_name="Applied assumptions cannot exceed total assumptions",
                severity=RuleSeverity.ERROR,
                message=(
                    "Applied assumption count is greater than the total "
                    "assumption count."
                ),
                metric="applied_assumption_count",
                actual_value=result.applied_assumption_count,
                expected_value=f"<= {result.total_assumptions}",
            )
        else:
            self._record_pass()

        for row in result.adjusted_forecast:
            period = str(row.get("forecast_period", ""))
            context = {"forecast_period": period}

            self._check_non_negative(
                value=row.get("adjusted_orders", 0),
                metric="adjusted_orders",
                context=context,
                rule_code="SCN_003",
                rule_name="Adjusted orders must not be negative",
            )

            self._check_non_negative(
                value=row.get("adjusted_revenue", 0),
                metric="adjusted_revenue",
                context=context,
                rule_code="SCN_004",
                rule_name="Adjusted revenue must not be negative",
            )

            self._check_non_negative(
                value=row.get("adjusted_average_order_value", 0),
                metric="adjusted_average_order_value",
                context=context,
                rule_code="SCN_005",
                rule_name="Adjusted AOV must not be negative",
            )

            self._check_scenario_adjustment_size(
                row=row,
                context=context,
            )

    def _check_status_coverage(self, result: Any) -> None:
        """
        Check whether completed and cancelled percentages exceed 100%.

        The sum may be below 100% because other statuses can exist,
        such as pending, failed, or in-progress orders.
        """

        combined_percentage = (
            float(result.fulfillment_percentage)
            + float(result.cancellation_percentage)
        )

        self._rules_checked += 1

        if combined_percentage > 100 + self.percentage_tolerance:
            self._record_issue_without_increment(
                rule_code="OPS_010",
                rule_name="Order-status percentages must not exceed 100%",
                severity=RuleSeverity.ERROR,
                message=(
                    "Fulfillment and cancellation percentages together "
                    "exceed 100%."
                ),
                metric="status_percentage_total",
                actual_value=round(combined_percentage, 6),
                expected_value="<= 100%",
            )
        else:
            self._passed_rules += 1

    def _check_scenario_adjustment_size(
        self,
        row: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """
        Warn when scenario adjustment is unusually large.
        """

        if self.maximum_forecast_growth_percentage is None:
            return

        base_orders = float(row.get("base_orders", 0))
        adjusted_orders = float(row.get("adjusted_orders", 0))

        base_revenue = float(row.get("base_revenue", 0))
        adjusted_revenue = float(row.get("adjusted_revenue", 0))

        orders_change_percentage = self._calculate_change_percentage(
            base_value=base_orders,
            adjusted_value=adjusted_orders,
        )

        revenue_change_percentage = self._calculate_change_percentage(
            base_value=base_revenue,
            adjusted_value=adjusted_revenue,
        )

        self._check_adjustment_threshold(
            change_percentage=orders_change_percentage,
            metric="orders_change_percentage",
            context=context,
            rule_code="SCN_006",
            rule_name="Scenario order adjustment is within threshold",
        )

        self._check_adjustment_threshold(
            change_percentage=revenue_change_percentage,
            metric="revenue_change_percentage",
            context=context,
            rule_code="SCN_007",
            rule_name="Scenario revenue adjustment is within threshold",
        )

    def _check_non_negative(
        self,
        value: Any,
        metric: str,
        rule_code: str,
        rule_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Check that a metric is numeric and not negative.
        """

        self._rules_checked += 1

        if not self._is_valid_number(value):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=f"{metric} is not a valid finite number.",
                metric=metric,
                actual_value=value,
                expected_value="A valid number greater than or equal to zero",
                context=context,
            )
            return

        if float(value) < 0:
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=f"{metric} cannot be negative.",
                metric=metric,
                actual_value=value,
                expected_value=">= 0",
                context=context,
            )
            return

        self._passed_rules += 1

    def _check_percentage_range(
        self,
        value: Any,
        metric: str,
        rule_code: str,
        rule_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Check percentages expressed from 0 to 100.
        """

        self._rules_checked += 1

        if not self._is_valid_number(value) or not 0 <= float(value) <= 100:
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=f"{metric} must be between 0% and 100%.",
                metric=metric,
                actual_value=value,
                expected_value="0% to 100%",
                context=context,
            )
            return

        self._passed_rules += 1

    def _check_decimal_percentage_range(
        self,
        value: Any,
        metric: str,
        rule_code: str,
        rule_name: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Check percentages expressed as decimals from -1 to 1.

        Example:
            40% is stored as 0.40.
        """

        self._rules_checked += 1

        if not self._is_valid_number(value) or not -1 <= float(value) <= 1:
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.WARNING,
                message=(
                    f"{metric} is outside the expected range "
                    "of -100% to 100%."
                ),
                metric=metric,
                actual_value=value,
                expected_value="-1.0 to 1.0",
                context=context,
            )
            return

        self._passed_rules += 1

    def _check_less_than_or_equal(
        self,
        left_value: Any,
        right_value: Any,
        left_metric: str,
        right_metric: str,
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Check that one financial metric does not exceed another.
        """

        self._rules_checked += 1

        if (
            not self._is_valid_number(left_value)
            or not self._is_valid_number(right_value)
            or float(left_value) > float(right_value)
        ):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=f"{left_metric} cannot exceed {right_metric}.",
                metric=left_metric,
                actual_value=left_value,
                expected_value=f"<= {right_value}",
            )
            return

        self._passed_rules += 1

    def _check_variance_balance(
        self,
        check_value: Any,
        context: dict[str, Any],
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Check revenue variance decomposition balance.
        """

        self._rules_checked += 1

        if (
            not self._is_valid_number(check_value)
            or abs(float(check_value)) > self.variance_tolerance
        ):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=(
                    "Revenue variance decomposition does not balance "
                    "within the configured tolerance."
                ),
                metric="variance_check",
                actual_value=check_value,
                expected_value=(
                    f"Between {-self.variance_tolerance} "
                    f"and {self.variance_tolerance}"
                ),
                context=context,
            )
            return

        self._passed_rules += 1

    def _check_gp_balance(
        self,
        check_value: Any,
        context: dict[str, Any],
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Check GP% bridge balance.
        """

        self._rules_checked += 1

        if (
            not self._is_valid_number(check_value)
            or abs(float(check_value)) > self.percentage_tolerance
        ):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=(
                    "GP% decomposition does not balance within "
                    "the configured tolerance."
                ),
                metric="gp_percentage_check",
                actual_value=check_value,
                expected_value=(
                    f"Between {-self.percentage_tolerance} "
                    f"and {self.percentage_tolerance}"
                ),
                context=context,
            )
            return

        self._passed_rules += 1

    def _check_finite_number(
        self,
        value: Any,
        metric: str,
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Check that a metric is a finite numeric value.
        """

        self._rules_checked += 1

        if not self._is_valid_number(value):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.ERROR,
                message=f"{metric} is not a valid finite number.",
                metric=metric,
                actual_value=value,
                expected_value="Valid finite number",
            )
            return

        self._passed_rules += 1

    def _check_maximum_value(
        self,
        value: Any,
        maximum_value: float,
        metric: str,
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Warn when a value exceeds a configured business threshold.
        """

        self._rules_checked += 1

        if (
            not self._is_valid_number(value)
            or float(value) > maximum_value
        ):
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.WARNING,
                message=f"{metric} exceeds the configured maximum.",
                metric=metric,
                actual_value=value,
                expected_value=f"<= {maximum_value}",
            )
            return

        self._passed_rules += 1

    def _check_unique_period(
        self,
        period: str,
        seen_periods: set[str],
        context: dict[str, Any],
    ) -> None:
        """
        Check that each forecast period appears only once.
        """

        self._rules_checked += 1

        if period in seen_periods:
            self._record_issue_without_increment(
                rule_code="FCT_005",
                rule_name="Forecast periods must be unique",
                severity=RuleSeverity.ERROR,
                message="Duplicate forecast period found.",
                metric="forecast_period",
                actual_value=period,
                expected_value="Unique period",
                context=context,
            )
            return

        seen_periods.add(period)
        self._passed_rules += 1

    def _check_adjustment_threshold(
        self,
        change_percentage: float | None,
        metric: str,
        context: dict[str, Any],
        rule_code: str,
        rule_name: str,
    ) -> None:
        """
        Warn when scenario impact exceeds the configured percentage.
        """

        self._rules_checked += 1

        if change_percentage is None:
            self._passed_rules += 1
            return

        threshold = float(
            self.maximum_forecast_growth_percentage or 0
        )

        if abs(change_percentage) > threshold:
            self._record_issue_without_increment(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=RuleSeverity.WARNING,
                message=(
                    f"{metric} exceeds the configured scenario "
                    "adjustment threshold."
                ),
                metric=metric,
                actual_value=round(change_percentage, 2),
                expected_value=f"Between {-threshold}% and {threshold}%",
                context=context,
            )
            return

        self._passed_rules += 1

    def _calculate_change_percentage(
        self,
        base_value: float,
        adjusted_value: float,
    ) -> float | None:
        """
        Calculate percentage movement from base to adjusted value.
        """

        if base_value == 0:
            return None

        return ((adjusted_value - base_value) / base_value) * 100

    def _is_valid_number(self, value: Any) -> bool:
        """
        Return True when value is a finite integer or float.
        """

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _record_pass(self) -> None:
        """
        Record one successfully passed rule.
        """

        self._rules_checked += 1
        self._passed_rules += 1

    def _record_issue(
        self,
        rule_code: str,
        rule_name: str,
        severity: RuleSeverity,
        message: str,
        metric: str | None = None,
        actual_value: Any = None,
        expected_value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a failed rule and increment the rule count.
        """

        self._rules_checked += 1

        self._record_issue_without_increment(
            rule_code=rule_code,
            rule_name=rule_name,
            severity=severity,
            message=message,
            metric=metric,
            actual_value=actual_value,
            expected_value=expected_value,
            context=context,
        )

    def _record_issue_without_increment(
        self,
        rule_code: str,
        rule_name: str,
        severity: RuleSeverity,
        message: str,
        metric: str | None = None,
        actual_value: Any = None,
        expected_value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a failed rule without changing the rule count.
        """

        self._issues.append(
            FinanceRuleIssue(
                rule_code=rule_code,
                rule_name=rule_name,
                severity=severity,
                message=message,
                metric=metric,
                actual_value=actual_value,
                expected_value=expected_value,
                context=context or {},
            )
        )

    def _reset_assessment(self) -> None:
        """
        Reset internal state before each analysis.
        """

        self._rules_checked = 0
        self._passed_rules = 0
        self._issues = []

    def _create_result(self) -> FinanceRulesResult:
        """
        Create final finance-rules result.
        """

        warning_count = sum(
            issue.severity == RuleSeverity.WARNING
            for issue in self._issues
        )

        error_count = sum(
            issue.severity == RuleSeverity.ERROR
            for issue in self._issues
        )

        if error_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        return FinanceRulesResult(
            overall_status=overall_status,
            rules_checked=self._rules_checked,
            passed_rules=self._passed_rules,
            warning_count=warning_count,
            error_count=error_count,
            issues=self._issues.copy(),
        )