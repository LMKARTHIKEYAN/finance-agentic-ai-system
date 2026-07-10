"""
Recommendation Agent for the Finance Agentic AI System.

This module converts root-cause findings into prioritized and measurable
business actions.

Supported input:
- RootCauseResult from RootCauseAgent

The agent is deterministic and rule-based. It does not execute actions.
It recommends the next best actions based on the identified cause, evidence,
confidence, impact, and anomaly severity.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal


RecommendationPriority = Literal["low", "medium", "high", "critical"]
RecommendationStatus = Literal["open", "monitor", "not_required"]


@dataclass
class RecommendationFinding:
    """Stores one recommended action for one root-cause finding."""

    recommendation_code: str
    root_cause_code: str
    anomaly_code: str
    level: str
    dimension_value: str
    target_metric: str
    cause_metric: str
    priority: RecommendationPriority
    status: RecommendationStatus
    recommended_action: str
    business_rationale: str
    expected_impact: str
    owner: str
    time_horizon: str
    success_metric: str
    target_direction: str
    confidence: str
    evidence_score: float
    supporting_evidence: dict[str, Any] = field(default_factory=dict)
    implementation_steps: list[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Stores the complete recommendation result."""

    overall_status: str
    root_causes_received: int
    root_causes_processed: int
    recommendation_count: int
    critical_priority_count: int
    high_priority_count: int
    monitor_count: int
    recommendations: list[RecommendationFinding] = field(default_factory=list)
    unresolved_root_causes: list[dict[str, Any]] = field(default_factory=list)


class RecommendationAgent:
    """
    Converts root-cause hypotheses into practical business recommendations.

    Main flow:

        Root-cause finding
                ↓
        Action template selection
                ↓
        Priority calculation
                ↓
        Owner and time-horizon assignment
                ↓
        Success metric definition
                ↓
        Ranked recommendation output

    The agent recommends actions but does not claim that the root cause is
    proven and does not perform the recommended action.
    """

    PRIORITY_RANK = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    SEVERITY_RANK = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    CONFIDENCE_RANK = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    ACTION_LIBRARY: dict[str, dict[str, Any]] = {
        "completed_orders": {
            "action": (
                "Identify the vehicle categories, clusters, customers, and time "
                "bands with the largest completed-order decline and implement a "
                "targeted order-recovery plan."
            ),
            "rationale": (
                "Completed orders are a direct revenue driver, so restoring lost "
                "completed-order volume can improve revenue performance."
            ),
            "expected_impact": (
                "Higher completed-order volume and improved revenue realization."
            ),
            "owner": "Operations Manager",
            "time_horizon": "1-2 weeks",
            "success_metric": "completed_orders",
            "target_direction": "increase",
            "steps": [
                "Rank dimensions by completed-order decline.",
                "Separate demand loss from fulfillment loss.",
                "Assign recovery targets to the responsible operations owners.",
                "Track completed orders daily against the recovery baseline.",
            ],
        },
        "total_orders": {
            "action": (
                "Investigate the largest sources of order-demand decline and run "
                "targeted customer acquisition, retention, and reactivation actions."
            ),
            "rationale": (
                "Lower demand limits the maximum number of orders that can be "
                "completed and can reduce revenue."
            ),
            "expected_impact": (
                "Recovery in order creation, active customers, and repeat orders."
            ),
            "owner": "Business Growth Manager",
            "time_horizon": "2-4 weeks",
            "success_metric": "total_orders",
            "target_direction": "increase",
            "steps": [
                "Compare new and repeat customer order trends.",
                "Identify weak clusters and vehicle categories.",
                "Review pricing, seasonality, and service-area availability.",
                "Launch targeted acquisition or reactivation actions.",
            ],
        },
        "demand_or_order_creation": {
            "action": (
                "Perform a demand-driver analysis and address the largest decline "
                "in customer acquisition, repeat usage, market coverage, pricing, "
                "or service availability."
            ),
            "rationale": (
                "The total-order anomaly indicates a possible demand or order-"
                "creation issue that requires deeper commercial analysis."
            ),
            "expected_impact": (
                "Improved demand generation and a larger order pipeline."
            ),
            "owner": "Business Growth Manager",
            "time_horizon": "2-4 weeks",
            "success_metric": "total_orders",
            "target_direction": "increase",
            "steps": [
                "Decompose orders into new and repeat customers.",
                "Compare current performance with seasonality and prior periods.",
                "Review price, promotion, and serviceable-area changes.",
                "Prioritize the largest controllable demand gap.",
            ],
        },
        "average_order_value": {
            "action": (
                "Review realized fare, discounts, trip distance, customer mix, "
                "cluster mix, and vehicle mix, then correct avoidable price or mix "
                "dilution."
            ),
            "rationale": (
                "Average order value is a direct revenue driver and can reduce "
                "revenue even when completed-order volume is stable."
            ),
            "expected_impact": (
                "Improved revenue per completed order and reduced pricing leakage."
            ),
            "owner": "Pricing and Revenue Manager",
            "time_horizon": "1-3 weeks",
            "success_metric": "average_order_value",
            "target_direction": "increase",
            "steps": [
                "Decompose AOV by vehicle, distance, cluster, and customer.",
                "Measure discount and pricing leakage.",
                "Separate price effects from business-mix effects.",
                "Implement approved pricing or mix corrections.",
            ],
        },
        "price_or_trip_mix": {
            "action": (
                "Decompose average order value into price, discount, distance, and "
                "business-mix drivers and correct the largest unfavorable driver."
            ),
            "rationale": (
                "AOV movements may arise from realized pricing or from changes in "
                "trip and customer mix."
            ),
            "expected_impact": (
                "Better price realization and improved revenue per order."
            ),
            "owner": "Pricing and Revenue Manager",
            "time_horizon": "1-3 weeks",
            "success_metric": "average_order_value",
            "target_direction": "increase",
            "steps": [
                "Build an AOV bridge by price and mix.",
                "Identify excessive discounting or weak-fare segments.",
                "Validate commercial and operational constraints.",
                "Track post-action AOV without damaging order volume.",
            ],
        },
        "price_effect": {
            "action": (
                "Review realized fare, discounts, surge, distance pricing, and rate-"
                "card compliance in the affected dimension and remove avoidable "
                "pricing leakage."
            ),
            "rationale": (
                "The variance decomposition shows that price movement contributed "
                "to the revenue variance."
            ),
            "expected_impact": (
                "Improved price realization and a smaller unfavorable price variance."
            ),
            "owner": "Pricing and Revenue Manager",
            "time_horizon": "1-2 weeks",
            "success_metric": "price_effect",
            "target_direction": "increase",
            "steps": [
                "Compare actual and budget price by dimension.",
                "Quantify discount, surge, and rate-card leakage.",
                "Review exceptions with commercial owners.",
                "Track price variance after corrective action.",
            ],
        },
        "volume_effect": {
            "action": (
                "Recover completed-order volume by addressing the largest demand, "
                "fulfillment, cancellation, or partner-capacity gap in the affected "
                "dimension."
            ),
            "rationale": (
                "The variance decomposition shows that volume movement contributed "
                "to the revenue variance."
            ),
            "expected_impact": (
                "Higher completed-order volume and a smaller unfavorable volume variance."
            ),
            "owner": "Operations Manager",
            "time_horizon": "1-3 weeks",
            "success_metric": "volume_effect",
            "target_direction": "increase",
            "steps": [
                "Identify the largest order-volume gaps.",
                "Separate demand, fulfillment, and cancellation drivers.",
                "Assign actions to cluster and vehicle owners.",
                "Track actual volume against budget and prior period.",
            ],
        },
        "new_discontinued_effect": {
            "action": (
                "Validate new and discontinued category mapping, confirm launch or "
                "closure assumptions, and update the budget baseline where required."
            ),
            "rationale": (
                "New or discontinued categories can create structural variance that "
                "is not explained by normal price and volume movement."
            ),
            "expected_impact": (
                "More accurate variance attribution and corrected planning assumptions."
            ),
            "owner": "FP&A Manager",
            "time_horizon": "Within 1 week",
            "success_metric": "unexplained_revenue_variance",
            "target_direction": "decrease",
            "steps": [
                "Validate category master-data mapping.",
                "Confirm actual launch and discontinuation dates.",
                "Reconcile actual and budget category coverage.",
                "Update forecast assumptions after approval.",
            ],
        },
        "fulfillment_percentage": {
            "action": (
                "Improve fulfillment in the affected dimension by correcting partner "
                "availability, acceptance, allocation delay, serviceability, and peak-"
                "hour capacity gaps."
            ),
            "rationale": (
                "Low fulfillment prevents requested orders from becoming completed "
                "orders and can reduce revenue."
            ),
            "expected_impact": (
                "Higher fulfillment, more completed orders, and lower lost demand."
            ),
            "owner": "Supply and Operations Manager",
            "time_horizon": "1-2 weeks",
            "success_metric": "fulfillment_percentage",
            "target_direction": "increase",
            "steps": [
                "Measure fulfillment by hour, cluster, and vehicle category.",
                "Compare demand with active partner supply.",
                "Correct acceptance and allocation bottlenecks.",
                "Monitor fulfillment daily against the target.",
            ],
        },
        "partner_capacity_or_allocation": {
            "action": (
                "Rebalance partner supply and improve allocation performance in the "
                "affected cluster, vehicle category, and time band."
            ),
            "rationale": (
                "Partner shortages or slow allocation can directly reduce fulfillment."
            ),
            "expected_impact": (
                "Faster allocation, higher acceptance, and improved fulfillment."
            ),
            "owner": "Supply and Operations Manager",
            "time_horizon": "Within 1 week",
            "success_metric": "fulfillment_percentage",
            "target_direction": "increase",
            "steps": [
                "Calculate demand-to-active-partner gaps.",
                "Identify low-login and low-acceptance partner groups.",
                "Move or activate supply in constrained time bands.",
                "Track allocation time and fulfillment after intervention.",
            ],
        },
        "cancellation_percentage": {
            "action": (
                "Reduce cancellation by identifying the largest cancellation reasons "
                "and responsible parties, then implement targeted corrective actions."
            ),
            "rationale": (
                "Cancellation creates leakage between requested and completed orders."
            ),
            "expected_impact": (
                "Lower cancellation, higher completed orders, and improved customer experience."
            ),
            "owner": "Operations Quality Manager",
            "time_horizon": "1-2 weeks",
            "success_metric": "cancellation_percentage",
            "target_direction": "decrease",
            "steps": [
                "Split cancellations by reason and responsible party.",
                "Rank clusters, vehicles, and time bands by cancellation loss.",
                "Assign corrective actions to the responsible teams.",
                "Track cancellation percentage and saved orders.",
            ],
        },
        "cancellation_process_drivers": {
            "action": (
                "Perform a detailed cancellation root-cause review covering partner "
                "rejection, allocation delay, arrival time, customer behavior, pricing, "
                "vehicle availability, and serviceability."
            ),
            "rationale": (
                "The available evidence points to the cancellation process, but the "
                "specific operational driver requires deeper classification."
            ),
            "expected_impact": (
                "Clear ownership of cancellation drivers and reduced avoidable cancellations."
            ),
            "owner": "Operations Quality Manager",
            "time_horizon": "Within 1 week",
            "success_metric": "cancellation_percentage",
            "target_direction": "decrease",
            "steps": [
                "Standardize cancellation-reason categories.",
                "Measure cancellation by responsible party.",
                "Identify the top three avoidable causes.",
                "Implement and monitor cause-specific controls.",
            ],
        },
    }

    DEFAULT_ACTION = {
        "action": (
            "Investigate the supporting evidence for this root cause, validate the "
            "driver with the responsible business owner, and implement a targeted "
            "corrective action."
        ),
        "rationale": (
            "The root-cause finding requires business validation before a specific "
            "operational or financial action is finalized."
        ),
        "expected_impact": (
            "Better understanding of the performance driver and reduced recurrence."
        ),
        "owner": "Business Performance Manager",
        "time_horizon": "1-2 weeks",
        "success_metric": "target_metric_performance",
        "target_direction": "improve",
        "steps": [
            "Validate the evidence with the data owner.",
            "Confirm whether the cause is controllable.",
            "Define one measurable corrective action.",
            "Review the target metric after implementation.",
        ],
    }

    def __init__(
        self,
        minimum_evidence_score: float = 20.0,
        critical_evidence_score: float = 85.0,
        high_evidence_score: float = 70.0,
        max_recommendations_per_root_cause: int = 1,
        include_favorable_monitoring: bool = True,
    ) -> None:
        """Configure recommendation thresholds and output limits."""

        if minimum_evidence_score < 0 or minimum_evidence_score > 100:
            raise ValueError(
                "minimum_evidence_score must be between 0 and 100."
            )

        if not 0 <= high_evidence_score <= 100:
            raise ValueError(
                "high_evidence_score must be between 0 and 100."
            )

        if not 0 <= critical_evidence_score <= 100:
            raise ValueError(
                "critical_evidence_score must be between 0 and 100."
            )

        if critical_evidence_score <= high_evidence_score:
            raise ValueError(
                "critical_evidence_score must be greater than "
                "high_evidence_score."
            )

        if high_evidence_score < minimum_evidence_score:
            raise ValueError(
                "high_evidence_score cannot be below minimum_evidence_score."
            )

        if max_recommendations_per_root_cause <= 0:
            raise ValueError(
                "max_recommendations_per_root_cause must be positive."
            )

        if not isinstance(include_favorable_monitoring, bool):
            raise TypeError("include_favorable_monitoring must be a bool.")

        self.minimum_evidence_score = float(minimum_evidence_score)
        self.critical_evidence_score = float(critical_evidence_score)
        self.high_evidence_score = float(high_evidence_score)
        self.max_recommendations_per_root_cause = int(
            max_recommendations_per_root_cause
        )
        self.include_favorable_monitoring = include_favorable_monitoring

    def analyze(self, root_cause_result: Any) -> RecommendationResult:
        """
        Convert RootCauseResult into ranked business recommendations.

        Args:
            root_cause_result:
                Output from RootCauseAgent.

        Returns:
            RecommendationResult containing prioritized actions.
        """

        self._validate_input(root_cause_result)

        recommendations: list[RecommendationFinding] = []
        unresolved_root_causes: list[dict[str, Any]] = []
        root_causes_processed = 0

        for root_cause in root_cause_result.findings:
            root_causes_processed += 1

            evidence_score = self._number(root_cause.evidence_score)
            if evidence_score is None:
                unresolved_root_causes.append(
                    self._unresolved_record(
                        root_cause,
                        "Root cause has an invalid evidence score.",
                    )
                )
                continue

            if evidence_score < self.minimum_evidence_score:
                unresolved_root_causes.append(
                    self._unresolved_record(
                        root_cause,
                        "Root-cause evidence is below the minimum threshold.",
                    )
                )
                continue

            impact = str(root_cause.impact).strip().lower()

            if impact == "favorable" and not self.include_favorable_monitoring:
                continue

            finding = self._create_recommendation(root_cause)
            recommendations.append(finding)

        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._sort_recommendations(recommendations)

        critical_priority_count = sum(
            item.priority == "critical" for item in recommendations
        )
        high_priority_count = sum(
            item.priority == "high" for item in recommendations
        )
        monitor_count = sum(
            item.status == "monitor" for item in recommendations
        )

        return RecommendationResult(
            overall_status=self._determine_overall_status(
                recommendations=recommendations,
                unresolved_count=len(unresolved_root_causes),
            ),
            root_causes_received=len(root_cause_result.findings),
            root_causes_processed=root_causes_processed,
            recommendation_count=len(recommendations),
            critical_priority_count=critical_priority_count,
            high_priority_count=high_priority_count,
            monitor_count=monitor_count,
            recommendations=recommendations,
            unresolved_root_causes=unresolved_root_causes,
        )

    def _validate_input(self, root_cause_result: Any) -> None:
        if root_cause_result is None:
            raise ValueError("root_cause_result is required.")

        if not hasattr(root_cause_result, "findings"):
            raise ValueError(
                "root_cause_result must contain a findings attribute."
            )

        if not isinstance(root_cause_result.findings, list):
            raise TypeError("root_cause_result.findings must be a list.")

        required_attributes = {
            "root_cause_code",
            "anomaly_code",
            "level",
            "dimension_value",
            "target_metric",
            "anomaly_direction",
            "anomaly_severity",
            "cause_metric",
            "cause_description",
            "confidence",
            "impact",
            "evidence_score",
            "evidence",
            "recommended_next_check",
        }

        for root_cause in root_cause_result.findings:
            missing_attributes = [
                attribute
                for attribute in required_attributes
                if not hasattr(root_cause, attribute)
            ]

            if missing_attributes:
                raise ValueError(
                    "A root-cause finding is missing required attributes: "
                    f"{sorted(missing_attributes)}"
                )

            confidence = str(root_cause.confidence).strip().lower()
            if confidence not in self.CONFIDENCE_RANK:
                raise ValueError(
                    f"Unsupported root-cause confidence: {root_cause.confidence}"
                )

            impact = str(root_cause.impact).strip().lower()
            if impact not in {"favorable", "unfavorable", "neutral"}:
                raise ValueError(
                    f"Unsupported root-cause impact: {root_cause.impact}"
                )

            if not isinstance(root_cause.evidence, dict):
                raise TypeError("root_cause.evidence must be a dictionary.")

    def _create_recommendation(
        self,
        root_cause: Any,
    ) -> RecommendationFinding:
        cause_metric = str(root_cause.cause_metric).strip().lower()
        template = self.ACTION_LIBRARY.get(cause_metric, self.DEFAULT_ACTION)
        impact = str(root_cause.impact).strip().lower()
        priority = self._calculate_priority(root_cause)

        if impact == "favorable":
            status: RecommendationStatus = "monitor"
            recommended_action = (
                "Preserve the favorable driver and monitor it for sustainability. "
                + str(template["action"])
            )
            expected_impact = (
                "Sustained favorable performance without creating an adverse "
                "trade-off in other metrics."
            )
        elif impact == "neutral":
            status = "monitor"
            recommended_action = (
                "Validate the neutral driver before taking corrective action. "
                + str(template["action"])
            )
            expected_impact = (
                "Improved understanding of the driver and prevention of unnecessary action."
            )
        else:
            status = "open"
            recommended_action = str(template["action"])
            expected_impact = str(template["expected_impact"])

        success_metric = str(template["success_metric"])
        if success_metric == "target_metric_performance":
            success_metric = str(root_cause.target_metric)

        supporting_evidence = dict(root_cause.evidence)
        supporting_evidence.update(
            {
                "root_cause_description": str(root_cause.cause_description),
                "root_cause_next_check": str(
                    root_cause.recommended_next_check
                ),
                "root_cause_confidence": str(root_cause.confidence),
                "root_cause_impact": impact,
            }
        )

        return RecommendationFinding(
            recommendation_code=(
                f"REC-{str(root_cause.root_cause_code)}-"
                f"{cause_metric.upper().replace(' ', '_')}"
            ),
            root_cause_code=str(root_cause.root_cause_code),
            anomaly_code=str(root_cause.anomaly_code),
            level=str(root_cause.level),
            dimension_value=str(root_cause.dimension_value),
            target_metric=str(root_cause.target_metric),
            cause_metric=cause_metric,
            priority=priority,
            status=status,
            recommended_action=recommended_action,
            business_rationale=str(template["rationale"]),
            expected_impact=expected_impact,
            owner=str(template["owner"]),
            time_horizon=str(template["time_horizon"]),
            success_metric=success_metric,
            target_direction=str(template["target_direction"]),
            confidence=str(root_cause.confidence).strip().lower(),
            evidence_score=round(float(root_cause.evidence_score), 2),
            supporting_evidence=supporting_evidence,
            implementation_steps=list(template["steps"]),
        )

    def _calculate_priority(
        self,
        root_cause: Any,
    ) -> RecommendationPriority:
        impact = str(root_cause.impact).strip().lower()
        severity = str(root_cause.anomaly_severity).strip().lower()
        confidence = str(root_cause.confidence).strip().lower()
        evidence_score = float(root_cause.evidence_score)

        if impact != "unfavorable":
            return "low"

        if (
            severity == "critical"
            and confidence == "high"
            and evidence_score >= self.critical_evidence_score
        ):
            return "critical"

        if (
            confidence == "high"
            and evidence_score >= self.high_evidence_score
        ) or severity == "critical":
            return "high"

        if confidence == "medium" or severity == "high":
            return "medium"

        return "low"

    def _determine_overall_status(
        self,
        recommendations: list[RecommendationFinding],
        unresolved_count: int,
    ) -> str:
        if any(item.priority == "critical" for item in recommendations):
            return "CRITICAL_ACTION_REQUIRED"

        if any(item.priority == "high" for item in recommendations):
            return "ACTION_REQUIRED"

        if recommendations or unresolved_count:
            return "REVIEW"

        return "NO_RECOMMENDATION_REQUIRED"

    def _sort_recommendations(
        self,
        recommendations: list[RecommendationFinding],
    ) -> list[RecommendationFinding]:
        return sorted(
            recommendations,
            key=lambda item: (
                self.PRIORITY_RANK.get(item.priority, 0),
                self.CONFIDENCE_RANK.get(item.confidence, 0),
                item.evidence_score,
                self.SEVERITY_RANK.get(
                    str(
                        item.supporting_evidence.get(
                            "anomaly_severity",
                            "low",
                        )
                    ).lower(),
                    0,
                ),
            ),
            reverse=True,
        )

    def _deduplicate_recommendations(
        self,
        recommendations: list[RecommendationFinding],
    ) -> list[RecommendationFinding]:
        best_by_key: dict[
            tuple[str, str],
            RecommendationFinding,
        ] = {}

        for recommendation in recommendations:
            key = (
                recommendation.root_cause_code,
                recommendation.cause_metric,
            )
            existing = best_by_key.get(key)

            if (
                existing is None
                or recommendation.evidence_score > existing.evidence_score
            ):
                best_by_key[key] = recommendation

        return list(best_by_key.values())

    def _unresolved_record(
        self,
        root_cause: Any,
        message: str,
    ) -> dict[str, Any]:
        return {
            "root_cause_code": str(root_cause.root_cause_code),
            "anomaly_code": str(root_cause.anomaly_code),
            "level": str(root_cause.level),
            "dimension_value": str(root_cause.dimension_value),
            "target_metric": str(root_cause.target_metric),
            "cause_metric": str(root_cause.cause_metric),
            "message": message,
        }

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