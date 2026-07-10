"""
Root Cause Agent for the Finance Agentic AI System.

This module investigates anomalies detected by AnomalyAgent and identifies
likely operational or financial drivers.

Supported inputs:
- AnomalyResult from AnomalyAgent
- OperationsAnalysisResult from OperationsAnalysisAgent
- Optional RevenueVarianceResult from RevenueVarianceAgent

The agent is deterministic and rule-based. It does not claim that a cause is
proven; it ranks likely causes using available supporting evidence.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal


CauseConfidence = Literal["low", "medium", "high"]
CauseImpact = Literal["favorable", "unfavorable", "neutral"]


@dataclass
class RootCauseFinding:
    """Stores one root-cause hypothesis for an anomaly."""

    root_cause_code: str
    anomaly_code: str
    level: str
    dimension_value: str
    target_metric: str
    anomaly_direction: str
    anomaly_severity: str
    cause_metric: str
    cause_description: str
    confidence: CauseConfidence
    impact: CauseImpact
    evidence_score: float
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_next_check: str = ""


@dataclass
class RootCauseResult:
    """Stores the complete root-cause analysis result."""

    overall_status: str
    anomalies_received: int
    anomalies_analyzed: int
    root_cause_count: int
    high_confidence_count: int
    unresolved_count: int
    findings: list[RootCauseFinding] = field(default_factory=list)
    unresolved_anomalies: list[dict[str, Any]] = field(default_factory=list)


class RootCauseAgent:
    """
    Identifies likely causes of operational and revenue anomalies.

    Main business relationships:

        Revenue = Completed Orders × Average Order Value

        Completed Orders = Total Orders × Fulfillment Percentage

        Cancellation Percentage can reduce fulfillment and completed orders.

    Revenue variance decomposition, when supplied:

        Revenue Variance = Price Effect + Volume Effect
                         + New / Discontinued Effect

    The agent uses related metrics from the same analysis row and ranks causes
    by evidence strength. Results are hypotheses for investigation, not proof
    of causality.
    """

    SUPPORTED_LEVELS = {
        "period",
        "vehicle_category",
        "pickup_cluster",
    }

    LEVEL_SUMMARY_MAPPING = {
        "period": "period_summary",
        "vehicle_category": "vehicle_summary",
        "pickup_cluster": "cluster_summary",
    }

    LEVEL_KEY_MAPPING = {
        "period": "period",
        "vehicle_category": "vehicle_category",
        "pickup_cluster": "pickup_cluster",
    }

    NEGATIVE_DIRECTION_METRICS = {
        "total_orders",
        "completed_orders",
        "total_revenue",
        "average_order_value",
        "fulfillment_percentage",
    }

    POSITIVE_DIRECTION_METRICS = {
        "cancelled_orders",
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
        "price_effect": "Price Effect",
        "volume_effect": "Volume Effect",
        "new_discontinued_effect": "New / Discontinued Effect",
    }

    def __init__(
        self,
        minimum_evidence_score: float = 20.0,
        high_confidence_score: float = 70.0,
        medium_confidence_score: float = 40.0,
        max_causes_per_anomaly: int = 3,
    ) -> None:
        """Configure root-cause ranking thresholds."""

        if minimum_evidence_score < 0:
            raise ValueError("minimum_evidence_score cannot be negative.")

        if not 0 <= medium_confidence_score <= 100:
            raise ValueError(
                "medium_confidence_score must be between 0 and 100."
            )

        if not 0 <= high_confidence_score <= 100:
            raise ValueError(
                "high_confidence_score must be between 0 and 100."
            )

        if high_confidence_score <= medium_confidence_score:
            raise ValueError(
                "high_confidence_score must be greater than "
                "medium_confidence_score."
            )

        if max_causes_per_anomaly <= 0:
            raise ValueError("max_causes_per_anomaly must be positive.")

        self.minimum_evidence_score = float(minimum_evidence_score)
        self.high_confidence_score = float(high_confidence_score)
        self.medium_confidence_score = float(medium_confidence_score)
        self.max_causes_per_anomaly = int(max_causes_per_anomaly)

    def analyze(
        self,
        anomaly_result: Any,
        operations_result: Any,
        variance_result: Any | None = None,
    ) -> RootCauseResult:
        """
        Analyze anomalies and return ranked root-cause hypotheses.

        Args:
            anomaly_result:
                Output from AnomalyAgent.

            operations_result:
                Output from OperationsAnalysisAgent.

            variance_result:
                Optional output from RevenueVarianceAgent. It is used mainly
                for revenue anomalies at overall or vehicle-category level.

        Returns:
            RootCauseResult containing ranked causes and unresolved anomalies.
        """

        self._validate_inputs(
            anomaly_result=anomaly_result,
            operations_result=operations_result,
        )

        all_findings: list[RootCauseFinding] = []
        unresolved_anomalies: list[dict[str, Any]] = []
        anomalies_analyzed = 0

        for anomaly in anomaly_result.findings:
            anomalies_analyzed += 1

            summary_row = self._find_matching_summary_row(
                anomaly=anomaly,
                operations_result=operations_result,
            )

            causes = self._identify_causes(
                anomaly=anomaly,
                summary_row=summary_row,
                operations_result=operations_result,
                variance_result=variance_result,
            )

            causes = [
                cause
                for cause in causes
                if cause.evidence_score >= self.minimum_evidence_score
            ]

            causes = self._sort_causes(causes)[
                : self.max_causes_per_anomaly
            ]

            if causes:
                all_findings.extend(causes)
            else:
                unresolved_anomalies.append(
                    {
                        "anomaly_code": str(anomaly.anomaly_code),
                        "level": str(anomaly.level),
                        "dimension_value": str(anomaly.dimension_value),
                        "metric": str(anomaly.metric),
                        "message": (
                            "No sufficiently strong cause was found from the "
                            "available operational metrics."
                        ),
                    }
                )

        sorted_findings = self._sort_causes(all_findings)
        high_confidence_count = sum(
            finding.confidence == "high"
            for finding in sorted_findings
        )

        return RootCauseResult(
            overall_status=self._determine_overall_status(
                findings=sorted_findings,
                unresolved_count=len(unresolved_anomalies),
            ),
            anomalies_received=len(anomaly_result.findings),
            anomalies_analyzed=anomalies_analyzed,
            root_cause_count=len(sorted_findings),
            high_confidence_count=high_confidence_count,
            unresolved_count=len(unresolved_anomalies),
            findings=sorted_findings,
            unresolved_anomalies=unresolved_anomalies,
        )

    def _validate_inputs(
        self,
        anomaly_result: Any,
        operations_result: Any,
    ) -> None:
        if anomaly_result is None:
            raise ValueError("anomaly_result is required.")

        if not hasattr(anomaly_result, "findings"):
            raise ValueError(
                "anomaly_result must contain a findings attribute."
            )

        if not isinstance(anomaly_result.findings, list):
            raise TypeError("anomaly_result.findings must be a list.")

        if operations_result is None:
            raise ValueError("operations_result is required.")

        missing_summaries = [
            attribute
            for attribute in self.LEVEL_SUMMARY_MAPPING.values()
            if not hasattr(operations_result, attribute)
        ]

        if missing_summaries:
            raise ValueError(
                "operations_result is missing required summaries: "
                f"{sorted(missing_summaries)}"
            )

        for anomaly in anomaly_result.findings:
            required_attributes = {
                "anomaly_code",
                "level",
                "dimension_value",
                "metric",
                "actual_value",
                "comparison_value",
                "percentage_change",
                "z_score",
                "direction",
                "severity",
            }

            missing_attributes = [
                attribute
                for attribute in required_attributes
                if not hasattr(anomaly, attribute)
            ]

            if missing_attributes:
                raise ValueError(
                    "An anomaly finding is missing required attributes: "
                    f"{sorted(missing_attributes)}"
                )

            if str(anomaly.level) not in self.SUPPORTED_LEVELS:
                raise ValueError(
                    f"Unsupported anomaly level: {anomaly.level}"
                )

    def _find_matching_summary_row(
        self,
        anomaly: Any,
        operations_result: Any,
    ) -> dict[str, Any] | None:
        level = str(anomaly.level)
        summary_attribute = self.LEVEL_SUMMARY_MAPPING[level]
        dimension_key = self.LEVEL_KEY_MAPPING[level]
        summary_rows = getattr(operations_result, summary_attribute, [])

        if not isinstance(summary_rows, list):
            return None

        target = self._normalize_dimension_value(
            anomaly.dimension_value
        )

        for row in summary_rows:
            if not isinstance(row, dict):
                continue

            candidate = self._normalize_dimension_value(
                row.get(dimension_key)
            )

            if candidate == target:
                return row

        return None

    def _identify_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
        operations_result: Any,
        variance_result: Any | None,
    ) -> list[RootCauseFinding]:
        metric = str(anomaly.metric)
        causes: list[RootCauseFinding] = []

        if metric == "total_revenue":
            causes.extend(
                self._revenue_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                    variance_result=variance_result,
                )
            )

        elif metric == "completed_orders":
            causes.extend(
                self._completed_order_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                )
            )

        elif metric == "total_orders":
            causes.extend(
                self._total_order_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                )
            )

        elif metric == "average_order_value":
            causes.extend(
                self._aov_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                    variance_result=variance_result,
                )
            )

        elif metric == "fulfillment_percentage":
            causes.extend(
                self._fulfillment_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                )
            )

        elif metric in {
            "cancelled_orders",
            "cancellation_percentage",
        }:
            causes.extend(
                self._cancellation_causes(
                    anomaly=anomaly,
                    summary_row=summary_row,
                )
            )

        causes.extend(
            self._same_dimension_anomaly_evidence(
                target_anomaly=anomaly,
                anomaly_result_findings=getattr(
                    operations_result,
                    "_anomaly_findings",
                    [],
                ),
            )
        )

        return self._deduplicate_causes(causes)

    def _revenue_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
        variance_result: Any | None,
    ) -> list[RootCauseFinding]:
        causes: list[RootCauseFinding] = []
        unfavorable = self._is_unfavorable(anomaly)

        if summary_row:
            completed_orders = self._number(
                summary_row.get("completed_orders")
            )
            total_orders = self._number(summary_row.get("total_orders"))
            fulfillment = self._number(
                summary_row.get("fulfillment_percentage")
            )
            cancellation = self._number(
                summary_row.get("cancellation_percentage")
            )
            aov = self._number(summary_row.get("average_order_value"))

            if completed_orders is not None:
                score = self._base_score(anomaly) + 12.0
                causes.append(
                    self._build_finding(
                        anomaly=anomaly,
                        cause_metric="completed_orders",
                        description=(
                            "Completed-order volume is a direct revenue "
                            "driver because revenue equals completed orders "
                            "multiplied by average order value."
                        ),
                        score=score,
                        impact=(
                            "unfavorable" if unfavorable else "favorable"
                        ),
                        evidence={
                            "completed_orders": completed_orders,
                            "total_orders": total_orders,
                            "formula": (
                                "Revenue = Completed Orders × "
                                "Average Order Value"
                            ),
                        },
                        next_check=(
                            "Compare completed orders with the previous period "
                            "or budget at vehicle and cluster level."
                        ),
                    )
                )

            if aov is not None:
                score = self._base_score(anomaly) + 10.0
                causes.append(
                    self._build_finding(
                        anomaly=anomaly,
                        cause_metric="average_order_value",
                        description=(
                            "Average order value is a direct revenue driver "
                            "and may explain the revenue movement."
                        ),
                        score=score,
                        impact=(
                            "unfavorable" if unfavorable else "favorable"
                        ),
                        evidence={
                            "average_order_value": aov,
                            "formula": (
                                "Revenue = Completed Orders × "
                                "Average Order Value"
                            ),
                        },
                        next_check=(
                            "Review pricing, discounting, trip mix, distance, "
                            "and vehicle-category mix."
                        ),
                    )
                )

            if fulfillment is not None and unfavorable:
                score = self._rate_risk_score(
                    value=fulfillment,
                    warning_below=85.0,
                    critical_below=70.0,
                )
                if score > 0:
                    causes.append(
                        self._build_finding(
                            anomaly=anomaly,
                            cause_metric="fulfillment_percentage",
                            description=(
                                "Low fulfillment may have reduced completed "
                                "orders and therefore revenue."
                            ),
                            score=score,
                            impact="unfavorable",
                            evidence={
                                "fulfillment_percentage": fulfillment,
                            },
                            next_check=(
                                "Check partner availability, allocation delay, "
                                "acceptance rate, and peak-hour capacity."
                            ),
                        )
                    )

            if cancellation is not None and unfavorable:
                score = self._rate_risk_score(
                    value=cancellation,
                    warning_above=10.0,
                    critical_above=20.0,
                )
                if score > 0:
                    causes.append(
                        self._build_finding(
                            anomaly=anomaly,
                            cause_metric="cancellation_percentage",
                            description=(
                                "High cancellation may have reduced completed "
                                "orders and revenue."
                            ),
                            score=score,
                            impact="unfavorable",
                            evidence={
                                "cancellation_percentage": cancellation,
                            },
                            next_check=(
                                "Separate customer, partner, operational, and "
                                "pricing-related cancellation reasons."
                            ),
                        )
                    )

        causes.extend(
            self._variance_causes(
                anomaly=anomaly,
                variance_result=variance_result,
            )
        )

        return causes

    def _completed_order_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
    ) -> list[RootCauseFinding]:
        if not summary_row:
            return []

        causes: list[RootCauseFinding] = []
        unfavorable = self._is_unfavorable(anomaly)
        total_orders = self._number(summary_row.get("total_orders"))
        fulfillment = self._number(
            summary_row.get("fulfillment_percentage")
        )
        cancellation = self._number(
            summary_row.get("cancellation_percentage")
        )

        if total_orders is not None:
            causes.append(
                self._build_finding(
                    anomaly=anomaly,
                    cause_metric="total_orders",
                    description=(
                        "Demand volume is a direct driver of completed orders."
                    ),
                    score=self._base_score(anomaly) + 10.0,
                    impact=(
                        "unfavorable" if unfavorable else "favorable"
                    ),
                    evidence={
                        "total_orders": total_orders,
                        "formula": (
                            "Completed Orders ≈ Total Orders × "
                            "Fulfillment Percentage"
                        ),
                    },
                    next_check=(
                        "Review order creation by period, cluster, customer, "
                        "and vehicle category."
                    ),
                )
            )

        if fulfillment is not None:
            score = self._base_score(anomaly) + 8.0
            if unfavorable:
                score += self._rate_risk_score(
                    value=fulfillment,
                    warning_below=85.0,
                    critical_below=70.0,
                ) / 3

            causes.append(
                self._build_finding(
                    anomaly=anomaly,
                    cause_metric="fulfillment_percentage",
                    description=(
                        "Fulfillment performance controls how many requested "
                        "orders become completed orders."
                    ),
                    score=score,
                    impact=(
                        "unfavorable" if unfavorable else "favorable"
                    ),
                    evidence={
                        "fulfillment_percentage": fulfillment,
                    },
                    next_check=(
                        "Investigate partner supply, acceptance, allocation, "
                        "and serviceability."
                    ),
                )
            )

        if cancellation is not None and unfavorable:
            causes.append(
                self._build_finding(
                    anomaly=anomaly,
                    cause_metric="cancellation_percentage",
                    description=(
                        "Cancellation is a leakage between total orders and "
                        "completed orders."
                    ),
                    score=self._rate_risk_score(
                        value=cancellation,
                        warning_above=10.0,
                        critical_above=20.0,
                    ),
                    impact="unfavorable",
                    evidence={
                        "cancellation_percentage": cancellation,
                    },
                    next_check=(
                        "Analyze cancellation reasons and responsible party."
                    ),
                )
            )

        return causes

    def _total_order_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
    ) -> list[RootCauseFinding]:
        evidence = dict(summary_row or {})
        unfavorable = self._is_unfavorable(anomaly)

        return [
            self._build_finding(
                anomaly=anomaly,
                cause_metric="demand_or_order_creation",
                description=(
                    "The total-order anomaly is most likely connected to "
                    "changes in customer demand, order creation, seasonality, "
                    "market coverage, or service availability."
                ),
                score=self._base_score(anomaly),
                impact="unfavorable" if unfavorable else "favorable",
                evidence=evidence,
                next_check=(
                    "Compare customer count, repeat rate, acquisition, "
                    "seasonality, pricing, and serviceable-area coverage."
                ),
            )
        ]

    def _aov_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
        variance_result: Any | None,
    ) -> list[RootCauseFinding]:
        causes: list[RootCauseFinding] = []
        unfavorable = self._is_unfavorable(anomaly)

        causes.append(
            self._build_finding(
                anomaly=anomaly,
                cause_metric="price_or_trip_mix",
                description=(
                    "Average order value is mainly driven by realized price, "
                    "discounting, trip distance, vehicle mix, and customer or "
                    "cluster mix."
                ),
                score=self._base_score(anomaly) + 10.0,
                impact="unfavorable" if unfavorable else "favorable",
                evidence=dict(summary_row or {}),
                next_check=(
                    "Decompose AOV by vehicle category, distance band, customer, "
                    "cluster, base fare, surge, and discount."
                ),
            )
        )

        causes.extend(
            self._variance_causes(
                anomaly=anomaly,
                variance_result=variance_result,
                include_only={"price_effect"},
            )
        )

        return causes

    def _fulfillment_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
    ) -> list[RootCauseFinding]:
        causes: list[RootCauseFinding] = []
        unfavorable = self._is_unfavorable(anomaly)
        cancellation = self._number(
            (summary_row or {}).get("cancellation_percentage")
        )

        causes.append(
            self._build_finding(
                anomaly=anomaly,
                cause_metric="partner_capacity_or_allocation",
                description=(
                    "Fulfillment changes are commonly driven by partner "
                    "availability, acceptance, allocation speed, serviceability, "
                    "or demand-supply imbalance."
                ),
                score=self._base_score(anomaly) + 8.0,
                impact="unfavorable" if unfavorable else "favorable",
                evidence=dict(summary_row or {}),
                next_check=(
                    "Check active partners, login rate, acceptance rate, "
                    "allocation time, and orders per active partner."
                ),
            )
        )

        if cancellation is not None and unfavorable:
            causes.append(
                self._build_finding(
                    anomaly=anomaly,
                    cause_metric="cancellation_percentage",
                    description=(
                        "Elevated cancellation may directly reduce fulfillment."
                    ),
                    score=self._rate_risk_score(
                        value=cancellation,
                        warning_above=10.0,
                        critical_above=20.0,
                    ),
                    impact="unfavorable",
                    evidence={
                        "cancellation_percentage": cancellation,
                    },
                    next_check=(
                        "Review cancellation reasons by customer, partner, "
                        "cluster, and vehicle category."
                    ),
                )
            )

        return causes

    def _cancellation_causes(
        self,
        anomaly: Any,
        summary_row: dict[str, Any] | None,
    ) -> list[RootCauseFinding]:
        unfavorable = self._is_unfavorable(anomaly)

        return [
            self._build_finding(
                anomaly=anomaly,
                cause_metric="cancellation_process_drivers",
                description=(
                    "Cancellation changes may arise from partner rejection, "
                    "long allocation or arrival time, customer behavior, pricing, "
                    "vehicle unavailability, or serviceability issues."
                ),
                score=self._base_score(anomaly) + 8.0,
                impact="unfavorable" if unfavorable else "favorable",
                evidence=dict(summary_row or {}),
                next_check=(
                    "Split cancellations by reason, responsible party, time of "
                    "day, cluster, and vehicle category."
                ),
            )
        ]

    def _variance_causes(
        self,
        anomaly: Any,
        variance_result: Any | None,
        include_only: set[str] | None = None,
    ) -> list[RootCauseFinding]:
        if variance_result is None:
            return []

        effect_values = self._get_relevant_variance_effects(
            anomaly=anomaly,
            variance_result=variance_result,
        )

        if include_only is not None:
            effect_values = {
                key: value
                for key, value in effect_values.items()
                if key in include_only
            }

        total_absolute_effect = sum(abs(value) for value in effect_values.values())
        if total_absolute_effect == 0:
            return []

        causes: list[RootCauseFinding] = []

        descriptions = {
            "price_effect": (
                "Price or average-order-value movement contributed to the "
                "revenue variance."
            ),
            "volume_effect": (
                "Completed-order volume movement contributed to the revenue "
                "variance."
            ),
            "new_discontinued_effect": (
                "A new or discontinued vehicle category contributed to the "
                "revenue variance."
            ),
        }

        checks = {
            "price_effect": (
                "Review realized fare, discount, distance, surge, and mix."
            ),
            "volume_effect": (
                "Review demand, fulfillment, cancellation, and partner capacity."
            ),
            "new_discontinued_effect": (
                "Confirm category launch, discontinuation, and budget mapping."
            ),
        }

        for effect_name, effect_value in effect_values.items():
            contribution = abs(effect_value) / total_absolute_effect * 100
            score = min(100.0, self._base_score(anomaly) + contribution / 2)

            causes.append(
                self._build_finding(
                    anomaly=anomaly,
                    cause_metric=effect_name,
                    description=descriptions[effect_name],
                    score=score,
                    impact=(
                        "unfavorable" if effect_value < 0 else "favorable"
                    ),
                    evidence={
                        effect_name: round(effect_value, 2),
                        "absolute_contribution_percentage": round(
                            contribution,
                            2,
                        ),
                    },
                    next_check=checks[effect_name],
                )
            )

        return causes

    def _get_relevant_variance_effects(
        self,
        anomaly: Any,
        variance_result: Any,
    ) -> dict[str, float]:
        level = str(anomaly.level)

        if level == "vehicle_category" and hasattr(
            variance_result,
            "vehicle_variance_summary",
        ):
            target = self._normalize_dimension_value(
                anomaly.dimension_value
            )

            for row in variance_result.vehicle_variance_summary:
                if not isinstance(row, dict):
                    continue

                candidate = self._normalize_dimension_value(
                    row.get("vehicle_category")
                )

                if candidate == target:
                    return self._extract_effects(row)

        return self._extract_effects(variance_result)

    def _extract_effects(self, source: Any) -> dict[str, float]:
        effects: dict[str, float] = {}

        for effect_name in {
            "price_effect",
            "volume_effect",
            "new_discontinued_effect",
        }:
            if isinstance(source, dict):
                raw_value = source.get(effect_name)
            else:
                raw_value = getattr(source, effect_name, None)

            value = self._number(raw_value)
            if value is not None and value != 0:
                effects[effect_name] = value

        return effects

    def _same_dimension_anomaly_evidence(
        self,
        target_anomaly: Any,
        anomaly_result_findings: list[Any],
    ) -> list[RootCauseFinding]:
        # Reserved for future orchestration where related anomaly findings may
        # be supplied through a richer context object.
        del target_anomaly, anomaly_result_findings
        return []

    def _build_finding(
        self,
        anomaly: Any,
        cause_metric: str,
        description: str,
        score: float,
        impact: CauseImpact,
        evidence: dict[str, Any],
        next_check: str,
    ) -> RootCauseFinding:
        bounded_score = round(max(0.0, min(float(score), 100.0)), 2)
        confidence = self._score_to_confidence(bounded_score)

        return RootCauseFinding(
            root_cause_code=(
                f"RC-{str(anomaly.anomaly_code)}-"
                f"{cause_metric.upper().replace(' ', '_')}"
            ),
            anomaly_code=str(anomaly.anomaly_code),
            level=str(anomaly.level),
            dimension_value=str(anomaly.dimension_value),
            target_metric=str(anomaly.metric),
            anomaly_direction=str(anomaly.direction),
            anomaly_severity=str(anomaly.severity),
            cause_metric=cause_metric,
            cause_description=description,
            confidence=confidence,
            impact=impact,
            evidence_score=bounded_score,
            evidence=evidence,
            recommended_next_check=next_check,
        )

    def _base_score(self, anomaly: Any) -> float:
        severity_scores = {
            "low": 30.0,
            "medium": 45.0,
            "high": 60.0,
            "critical": 75.0,
        }

        score = severity_scores.get(str(anomaly.severity).lower(), 30.0)

        percentage_change = self._number(anomaly.percentage_change)
        if percentage_change is not None:
            score += min(abs(percentage_change), 100.0) * 0.15

        z_score = self._number(anomaly.z_score)
        if z_score is not None:
            score += min(abs(z_score), 5.0) * 4.0

        return min(score, 100.0)

    def _rate_risk_score(
        self,
        value: float,
        warning_below: float | None = None,
        critical_below: float | None = None,
        warning_above: float | None = None,
        critical_above: float | None = None,
    ) -> float:
        if warning_below is not None and value < warning_below:
            if critical_below is not None and value < critical_below:
                return 80.0
            return 55.0

        if warning_above is not None and value > warning_above:
            if critical_above is not None and value > critical_above:
                return 80.0
            return 55.0

        return 0.0

    def _score_to_confidence(self, score: float) -> CauseConfidence:
        if score >= self.high_confidence_score:
            return "high"

        if score >= self.medium_confidence_score:
            return "medium"

        return "low"

    def _is_unfavorable(self, anomaly: Any) -> bool:
        metric = str(anomaly.metric)
        direction = str(anomaly.direction).strip().lower()

        decrease_directions = {"decrease", "low", "negative", "down"}
        increase_directions = {"increase", "high", "positive", "up"}

        if metric in self.NEGATIVE_DIRECTION_METRICS:
            return direction in decrease_directions

        if metric in self.POSITIVE_DIRECTION_METRICS:
            return direction in increase_directions

        return False

    def _determine_overall_status(
        self,
        findings: list[RootCauseFinding],
        unresolved_count: int,
    ) -> str:
        if any(
            finding.confidence == "high"
            and finding.impact == "unfavorable"
            for finding in findings
        ):
            return "ACTION_REQUIRED"

        if findings or unresolved_count:
            return "REVIEW"

        return "NO_ROOT_CAUSE_REQUIRED"

    def _sort_causes(
        self,
        causes: list[RootCauseFinding],
    ) -> list[RootCauseFinding]:
        confidence_rank = {
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        return sorted(
            causes,
            key=lambda cause: (
                confidence_rank.get(cause.confidence, 0),
                cause.evidence_score,
                cause.anomaly_severity,
            ),
            reverse=True,
        )

    def _deduplicate_causes(
        self,
        causes: list[RootCauseFinding],
    ) -> list[RootCauseFinding]:
        best_by_key: dict[tuple[str, str], RootCauseFinding] = {}

        for cause in causes:
            key = (cause.anomaly_code, cause.cause_metric)
            existing = best_by_key.get(key)

            if existing is None or cause.evidence_score > existing.evidence_score:
                best_by_key[key] = cause

        return list(best_by_key.values())

    def _normalize_dimension_value(self, value: Any) -> str:
        return str(value).strip().lower()

    def _number(self, value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not isfinite(number):
            return None

        return number