"""
Anomaly Agent for the Finance Agentic AI System.

This module detects unusual operational and financial performance from
summaries produced by OperationsAnalysisAgent.

Supported analysis levels:
- Period
- Vehicle category
- Pickup cluster

Detection methods:
- Period-over-period percentage change
- Peer-group z-score comparison

The agent does not modify financial results.
It only identifies unusual values that may require investigation.
"""

from dataclasses import dataclass, field
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Literal


AnomalyLevel = Literal[
    "period",
    "vehicle_category",
    "pickup_cluster",
]

AnomalySeverity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


@dataclass
class AnomalyFinding:
    """
    Stores one detected anomaly.

    Attributes:
        anomaly_code:
            Unique identifier for the anomaly.

        level:
            Analysis level where the anomaly was detected.

        dimension_value:
            Period, vehicle category, or pickup cluster.

        metric:
            Metric containing the unusual value.

        actual_value:
            Current value found in the data.

        comparison_value:
            Previous-period value or peer-group average.

        percentage_change:
            Percentage movement from the comparison value.

        z_score:
            Distance from the peer average in standard deviations.

        direction:
            Increase, decrease, high, or low.

        severity:
            Low, medium, high, or critical.

        message:
            Business-readable explanation.

        context:
            Additional supporting information.
    """

    anomaly_code: str
    level: str
    dimension_value: str
    metric: str
    actual_value: float
    comparison_value: float | None
    percentage_change: float | None
    z_score: float | None
    direction: str
    severity: AnomalySeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """
    Stores the complete anomaly detection result.

    Attributes:
        overall_status:
            NORMAL, REVIEW, or CRITICAL.

        analysis_levels:
            Levels checked by the agent.

        rows_checked:
            Number of summary rows reviewed.

        metrics_checked:
            Number of metric comparisons performed.

        anomaly_count:
            Total number of anomalies detected.

        high_priority_count:
            Number of high or critical anomalies.

        findings:
            Detailed anomaly findings.
    """

    overall_status: str
    analysis_levels: list[str]
    rows_checked: int
    metrics_checked: int
    anomaly_count: int
    high_priority_count: int
    findings: list[AnomalyFinding] = field(default_factory=list)


class AnomalyAgent:
    """
    Detects unusual operational and financial performance.

    Period analysis:
        Compares each period with the immediately previous period.

    Vehicle and cluster analysis:
        Compares each category or cluster with the peer-group average.

    Percentage-change formula:

        Percentage Change =
        (Current Value - Previous Value)
        / Previous Value
        × 100

    Z-score formula:

        Z-score =
        (Current Value - Group Average)
        / Group Standard Deviation
    """

    SUPPORTED_LEVELS = {
        "period",
        "vehicle_category",
        "pickup_cluster",
    }

    DEFAULT_METRICS = [
        "total_orders",
        "completed_orders",
        "total_revenue",
        "average_order_value",
        "fulfillment_percentage",
        "cancellation_percentage",
    ]

    SUPPORTED_METRICS = {
        "total_orders",
        "completed_orders",
        "cancelled_orders",
        "total_revenue",
        "average_order_value",
        "fulfillment_percentage",
        "cancellation_percentage",
    }

    METRIC_DISPLAY_NAMES = {
        "total_orders": "Total Orders",
        "completed_orders": "Completed Orders",
        "cancelled_orders": "Cancelled Orders",
        "total_revenue": "Revenue",
        "average_order_value": "Average Order Value",
        "fulfillment_percentage": "Fulfillment Percentage",
        "cancellation_percentage": "Cancellation Percentage",
    }

    LEVEL_SUMMARY_MAPPING = {
        "period": "period_summary",
        "vehicle_category": "vehicle_summary",
        "pickup_cluster": "cluster_summary",
    }

    def __init__(
        self,
        percentage_change_threshold: float = 20.0,
        z_score_threshold: float = 2.0,
        high_percentage_threshold: float = 40.0,
        critical_percentage_threshold: float = 60.0,
        high_z_score_threshold: float = 3.0,
        critical_z_score_threshold: float = 3.5,
        minimum_comparison_value: float = 1.0,
    ) -> None:
        """
        Configure anomaly-detection thresholds.

        Args:
            percentage_change_threshold:
                Minimum absolute period-over-period change required to flag
                an anomaly.

            z_score_threshold:
                Minimum absolute z-score required to flag a peer anomaly.

            high_percentage_threshold:
                Percentage movement classified as high severity.

            critical_percentage_threshold:
                Percentage movement classified as critical severity.

            high_z_score_threshold:
                Z-score classified as high severity.

            critical_z_score_threshold:
                Z-score classified as critical severity.

            minimum_comparison_value:
                Prevents unreliable percentage calculations when the
                comparison value is very close to zero.
        """

        self._validate_thresholds(
            percentage_change_threshold=percentage_change_threshold,
            z_score_threshold=z_score_threshold,
            high_percentage_threshold=high_percentage_threshold,
            critical_percentage_threshold=critical_percentage_threshold,
            high_z_score_threshold=high_z_score_threshold,
            critical_z_score_threshold=critical_z_score_threshold,
            minimum_comparison_value=minimum_comparison_value,
        )

        self.percentage_change_threshold = percentage_change_threshold
        self.z_score_threshold = z_score_threshold
        self.high_percentage_threshold = high_percentage_threshold
        self.critical_percentage_threshold = critical_percentage_threshold
        self.high_z_score_threshold = high_z_score_threshold
        self.critical_z_score_threshold = critical_z_score_threshold
        self.minimum_comparison_value = minimum_comparison_value

    def analyze(
        self,
        operations_result: Any,
        levels: list[AnomalyLevel] | None = None,
        metrics: list[str] | None = None,
    ) -> AnomalyResult:
        """
        Detect anomalies from OperationsAnalysisResult.

        Args:
            operations_result:
                Output from OperationsAnalysisAgent.

            levels:
                Optional levels to analyze.

                Supported:
                - period
                - vehicle_category
                - pickup_cluster

                If omitted, all supported levels are checked.

            metrics:
                Optional metrics to analyze.

                If omitted, DEFAULT_METRICS are checked.

        Returns:
            AnomalyResult containing detected anomalies.
        """

        self._validate_operations_result(operations_result)

        selected_levels = self._prepare_levels(levels)
        selected_metrics = self._prepare_metrics(metrics)

        findings: list[AnomalyFinding] = []
        rows_checked = 0
        metrics_checked = 0

        for level in selected_levels:
            summary_rows = self._get_summary_rows(
                operations_result=operations_result,
                level=level,
            )

            rows_checked += len(summary_rows)

            if level == "period":
                level_findings, level_checks = (
                    self._analyze_period_anomalies(
                        rows=summary_rows,
                        metrics=selected_metrics,
                    )
                )
            else:
                level_findings, level_checks = (
                    self._analyze_peer_anomalies(
                        rows=summary_rows,
                        level=level,
                        metrics=selected_metrics,
                    )
                )

            findings.extend(level_findings)
            metrics_checked += level_checks

        sorted_findings = self._sort_findings(findings)

        high_priority_count = sum(
            finding.severity in {"high", "critical"}
            for finding in sorted_findings
        )

        return AnomalyResult(
            overall_status=self._determine_status(sorted_findings),
            analysis_levels=selected_levels,
            rows_checked=rows_checked,
            metrics_checked=metrics_checked,
            anomaly_count=len(sorted_findings),
            high_priority_count=high_priority_count,
            findings=sorted_findings,
        )

    def _validate_thresholds(
        self,
        percentage_change_threshold: float,
        z_score_threshold: float,
        high_percentage_threshold: float,
        critical_percentage_threshold: float,
        high_z_score_threshold: float,
        critical_z_score_threshold: float,
        minimum_comparison_value: float,
    ) -> None:
        """
        Validate configured anomaly thresholds.
        """

        if percentage_change_threshold <= 0:
            raise ValueError(
                "percentage_change_threshold must be positive."
            )

        if z_score_threshold <= 0:
            raise ValueError("z_score_threshold must be positive.")

        if high_percentage_threshold <= percentage_change_threshold:
            raise ValueError(
                "high_percentage_threshold must be greater than "
                "percentage_change_threshold."
            )

        if critical_percentage_threshold <= high_percentage_threshold:
            raise ValueError(
                "critical_percentage_threshold must be greater than "
                "high_percentage_threshold."
            )

        if high_z_score_threshold <= z_score_threshold:
            raise ValueError(
                "high_z_score_threshold must be greater than "
                "z_score_threshold."
            )

        if critical_z_score_threshold <= high_z_score_threshold:
            raise ValueError(
                "critical_z_score_threshold must be greater than "
                "high_z_score_threshold."
            )

        if minimum_comparison_value < 0:
            raise ValueError(
                "minimum_comparison_value cannot be negative."
            )

    def _validate_operations_result(
        self,
        operations_result: Any,
    ) -> None:
        """
        Validate OperationsAnalysisResult.
        """

        if operations_result is None:
            raise ValueError("operations_result is required.")

        required_attributes = {
            "vehicle_summary",
            "cluster_summary",
            "period_summary",
        }

        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(operations_result, attribute)
        ]

        if missing_attributes:
            raise ValueError(
                "operations_result is missing required summaries: "
                f"{sorted(missing_attributes)}"
            )

    def _prepare_levels(
        self,
        levels: list[AnomalyLevel] | None,
    ) -> list[str]:
        """
        Validate and standardize requested analysis levels.
        """

        if levels is None:
            return [
                "period",
                "vehicle_category",
                "pickup_cluster",
            ]

        if not isinstance(levels, list) or not levels:
            raise ValueError(
                "levels must be a non-empty list when provided."
            )

        normalized_levels = [
            str(level).strip().lower()
            for level in levels
        ]

        invalid_levels = set(normalized_levels) - self.SUPPORTED_LEVELS

        if invalid_levels:
            raise ValueError(
                "Unsupported anomaly levels: "
                f"{sorted(invalid_levels)}"
            )

        return list(dict.fromkeys(normalized_levels))

    def _prepare_metrics(
        self,
        metrics: list[str] | None,
    ) -> list[str]:
        """
        Validate and standardize requested metrics.
        """

        if metrics is None:
            return self.DEFAULT_METRICS.copy()

        if not isinstance(metrics, list) or not metrics:
            raise ValueError(
                "metrics must be a non-empty list when provided."
            )

        normalized_metrics = [
            str(metric).strip().lower()
            for metric in metrics
        ]

        invalid_metrics = (
            set(normalized_metrics)
            - self.SUPPORTED_METRICS
        )

        if invalid_metrics:
            raise ValueError(
                "Unsupported anomaly metrics: "
                f"{sorted(invalid_metrics)}"
            )

        return list(dict.fromkeys(normalized_metrics))

    def _get_summary_rows(
        self,
        operations_result: Any,
        level: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the correct operational summary.
        """

        summary_attribute = self.LEVEL_SUMMARY_MAPPING[level]

        summary_rows = getattr(
            operations_result,
            summary_attribute,
            [],
        )

        if not isinstance(summary_rows, list):
            raise TypeError(
                f"{summary_attribute} must be a list."
            )

        return summary_rows

    def _analyze_period_anomalies(
        self,
        rows: list[dict[str, Any]],
        metrics: list[str],
    ) -> tuple[list[AnomalyFinding], int]:
        """
        Detect period anomalies by comparing each period with the previous one.
        """

        if len(rows) < 2:
            return [], 0

        sorted_rows = sorted(
            rows,
            key=lambda row: str(row.get("period", "")),
        )

        findings: list[AnomalyFinding] = []
        checks = 0

        for row_index in range(1, len(sorted_rows)):
            previous_row = sorted_rows[row_index - 1]
            current_row = sorted_rows[row_index]

            previous_period = str(
                previous_row.get("period", "")
            )

            current_period = str(
                current_row.get("period", "")
            )

            for metric in metrics:
                checks += 1

                previous_value = self._to_number(
                    previous_row.get(metric)
                )

                current_value = self._to_number(
                    current_row.get(metric)
                )

                if previous_value is None or current_value is None:
                    continue

                percentage_change = self._calculate_percentage_change(
                    current_value=current_value,
                    comparison_value=previous_value,
                )

                if percentage_change is None:
                    continue

                if (
                    abs(percentage_change)
                    < self.percentage_change_threshold
                ):
                    continue

                direction = (
                    "increase"
                    if percentage_change > 0
                    else "decrease"
                )

                findings.append(
                    AnomalyFinding(
                        anomaly_code=self._build_anomaly_code(
                            level="period",
                            metric=metric,
                            position=len(findings) + 1,
                        ),
                        level="period",
                        dimension_value=current_period,
                        metric=metric,
                        actual_value=round(current_value, 2),
                        comparison_value=round(
                            previous_value,
                            2,
                        ),
                        percentage_change=round(
                            percentage_change,
                            2,
                        ),
                        z_score=None,
                        direction=direction,
                        severity=(
                            self._classify_percentage_severity(
                                percentage_change
                            )
                        ),
                        message=self._create_period_message(
                            metric=metric,
                            current_period=current_period,
                            previous_period=previous_period,
                            percentage_change=percentage_change,
                        ),
                        context={
                            "current_period": current_period,
                            "previous_period": previous_period,
                            "comparison_method": (
                                "previous_period_percentage_change"
                            ),
                        },
                    )
                )

        return findings, checks

    def _analyze_peer_anomalies(
        self,
        rows: list[dict[str, Any]],
        level: str,
        metrics: list[str],
    ) -> tuple[list[AnomalyFinding], int]:
        """
        Detect vehicle or cluster anomalies using peer-group z-scores.
        """

        if len(rows) < 3:
            return [], 0

        findings: list[AnomalyFinding] = []
        checks = 0

        for metric in metrics:
            numeric_rows: list[tuple[dict[str, Any], float]] = []

            for row in rows:
                value = self._to_number(row.get(metric))

                if value is not None:
                    numeric_rows.append((row, value))

            if len(numeric_rows) < 3:
                continue

            metric_values = [
                value
                for _, value in numeric_rows
            ]

            group_mean = mean(metric_values)
            group_standard_deviation = pstdev(metric_values)

            if group_standard_deviation == 0:
                continue

            for row, actual_value in numeric_rows:
                checks += 1

                z_score = (
                    actual_value - group_mean
                ) / group_standard_deviation

                if abs(z_score) < self.z_score_threshold:
                    continue

                dimension_value = str(
                    row.get(level, "")
                )

                direction = (
                    "high"
                    if z_score > 0
                    else "low"
                )

                percentage_change = self._calculate_percentage_change(
                    current_value=actual_value,
                    comparison_value=group_mean,
                )

                findings.append(
                    AnomalyFinding(
                        anomaly_code=self._build_anomaly_code(
                            level=level,
                            metric=metric,
                            position=len(findings) + 1,
                        ),
                        level=level,
                        dimension_value=dimension_value,
                        metric=metric,
                        actual_value=round(actual_value, 2),
                        comparison_value=round(group_mean, 2),
                        percentage_change=(
                            round(percentage_change, 2)
                            if percentage_change is not None
                            else None
                        ),
                        z_score=round(z_score, 3),
                        direction=direction,
                        severity=self._classify_z_score_severity(
                            z_score
                        ),
                        message=self._create_peer_message(
                            level=level,
                            dimension_value=dimension_value,
                            metric=metric,
                            z_score=z_score,
                            group_mean=group_mean,
                        ),
                        context={
                            "group_average": round(
                                group_mean,
                                2,
                            ),
                            "group_standard_deviation": round(
                                group_standard_deviation,
                                2,
                            ),
                            "comparison_method": (
                                "peer_group_z_score"
                            ),
                        },
                    )
                )

        return findings, checks

    def _calculate_percentage_change(
        self,
        current_value: float,
        comparison_value: float,
    ) -> float | None:
        """
        Calculate percentage movement from comparison to current value.
        """

        if abs(comparison_value) < self.minimum_comparison_value:
            return None

        return (
            (current_value - comparison_value)
            / abs(comparison_value)
        ) * 100

    def _classify_percentage_severity(
        self,
        percentage_change: float,
    ) -> AnomalySeverity:
        """
        Classify period anomaly severity.
        """

        absolute_change = abs(percentage_change)

        if absolute_change >= self.critical_percentage_threshold:
            return "critical"

        if absolute_change >= self.high_percentage_threshold:
            return "high"

        if absolute_change >= self.percentage_change_threshold:
            return "medium"

        return "low"

    def _classify_z_score_severity(
        self,
        z_score: float,
    ) -> AnomalySeverity:
        """
        Classify peer anomaly severity.
        """

        absolute_z_score = abs(z_score)

        if absolute_z_score >= self.critical_z_score_threshold:
            return "critical"

        if absolute_z_score >= self.high_z_score_threshold:
            return "high"

        if absolute_z_score >= self.z_score_threshold:
            return "medium"

        return "low"

    def _create_period_message(
        self,
        metric: str,
        current_period: str,
        previous_period: str,
        percentage_change: float,
    ) -> str:
        """
        Generate a period anomaly message.
        """

        display_name = self.METRIC_DISPLAY_NAMES.get(
            metric,
            metric,
        )

        direction = (
            "increased"
            if percentage_change > 0
            else "decreased"
        )

        return (
            f"{display_name} {direction} by "
            f"{abs(percentage_change):,.2f}% in {current_period} "
            f"compared with {previous_period}."
        )

    def _create_peer_message(
        self,
        level: str,
        dimension_value: str,
        metric: str,
        z_score: float,
        group_mean: float,
    ) -> str:
        """
        Generate a vehicle or cluster anomaly message.
        """

        display_name = self.METRIC_DISPLAY_NAMES.get(
            metric,
            metric,
        )

        readable_level = level.replace("_", " ")

        position = (
            "above"
            if z_score > 0
            else "below"
        )

        return (
            f"{display_name} for {readable_level} "
            f"{dimension_value} is unusually {position} the peer average "
            f"of {group_mean:,.2f}."
        )

    def _build_anomaly_code(
        self,
        level: str,
        metric: str,
        position: int,
    ) -> str:
        """
        Build a readable anomaly identifier.
        """

        level_codes = {
            "period": "PER",
            "vehicle_category": "VEH",
            "pickup_cluster": "CLU",
        }

        metric_code = "".join(
            word[0].upper()
            for word in metric.split("_")
            if word
        )

        return (
            f"ANM_{level_codes[level]}_"
            f"{metric_code}_{position:03d}"
        )

    def _to_number(
        self,
        value: Any,
    ) -> float | None:
        """
        Convert a value into a finite float.
        """

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not isfinite(numeric_value):
            return None

        return numeric_value

    def _sort_findings(
        self,
        findings: list[AnomalyFinding],
    ) -> list[AnomalyFinding]:
        """
        Sort findings by severity and magnitude.
        """

        severity_order = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        return sorted(
            findings,
            key=lambda finding: (
                severity_order[finding.severity],
                self._get_anomaly_magnitude(finding),
            ),
            reverse=True,
        )

    def _get_anomaly_magnitude(
        self,
        finding: AnomalyFinding,
    ) -> float:
        """
        Return anomaly magnitude for sorting.
        """

        if finding.z_score is not None:
            return abs(finding.z_score)

        if finding.percentage_change is not None:
            return abs(finding.percentage_change)

        return 0.0

    def _determine_status(
        self,
        findings: list[AnomalyFinding],
    ) -> str:
        """
        Determine overall anomaly status.
        """

        if any(
            finding.severity == "critical"
            for finding in findings
        ):
            return "CRITICAL"

        if findings:
            return "REVIEW"

        return "NORMAL"