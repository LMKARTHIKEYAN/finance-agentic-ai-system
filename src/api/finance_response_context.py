"""Build structured finance evidence for AI response generation.

This module organizes existing LangGraph finance-agent outputs into a
flow-aware context. It does not calculate, modify, infer, or enrich any
financial values.

The resulting structure helps the RAG prompt distinguish between:

- the primary analysis required to answer the user's question;
- supporting analysis from other finance agents;
- unavailable outputs that must not be invented; and
- the internal agent sources used to build the answer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentContextDefinition:
    """Describe how one agent result should be used in an AI response."""

    result_key: str
    source_name: str
    purpose: str


class FinanceResponseContextBuilder:
    """Create a flow-specific evidence package from finance-agent outputs.

    The builder accepts the JSON-compatible dictionary produced by
    ``FinanceAskService._extract_finance_analysis``. It groups the available
    outputs according to the selected LangGraph flow without changing their
    contents.
    """

    _AGENT_DEFINITIONS: Mapping[str, AgentContextDefinition] = MappingProxyType(
        {
            "operations_result": AgentContextDefinition(
                result_key="operations_result",
                source_name="Operations Analysis Agent",
                purpose=(
                    "Provides actual operating volumes, revenue, fulfilment, "
                    "cancellations, average order value, and dimensional summaries."
                ),
            ),
            "budget_result": AgentContextDefinition(
                result_key="budget_result",
                source_name="Budget Agent",
                purpose=(
                    "Provides budget orders, budget revenue, budget average order "
                    "value, and budget summaries."
                ),
            ),
            "forecast_result": AgentContextDefinition(
                result_key="forecast_result",
                source_name="Forecast Agent",
                purpose=(
                    "Provides forecast methodology, forecast periods, historical "
                    "coverage, and forecast values."
                ),
            ),
            "scenario_result": AgentContextDefinition(
                result_key="scenario_result",
                source_name="Scenario Agent",
                purpose=(
                    "Provides scenario assumptions, adjusted forecasts, and applied "
                    "or unapplied assumption details."
                ),
            ),
            "variance_result": AgentContextDefinition(
                result_key="variance_result",
                source_name="Revenue Variance Agent",
                purpose=(
                    "Provides actual-versus-budget performance, price and volume "
                    "effects, variance checks, and dimensional variance detail."
                ),
            ),
            "pnl_result": AgentContextDefinition(
                result_key="pnl_result",
                source_name="P&L Agent",
                purpose=(
                    "Provides the calculated Actual P&L, Budget P&L, variance P&L, "
                    "variance amounts, variance percentages, actual and budget gross "
                    "margin percentages, gross-margin percentage-point variance, "
                    "monthly P&L results, consolidated summary, available months, "
                    "and excluded-month information."
                ),
            ),
            "gp_variance_result": AgentContextDefinition(
                result_key="gp_variance_result",
                source_name="GP% Variance Agent",
                purpose=(
                    "Provides the calculated Budget, Mix-only, Price-only, "
                    "and Actual Gross Margin percentages; mix, price, and cost "
                    "effects in percentage points and basis points; category "
                    "unit economics; exclusions; and reconciliation status."
                ),
            ),
            "pnl_commentary_result": AgentContextDefinition(
                result_key="pnl_commentary_result",
                source_name="P&L Commentary Agent",
                purpose=(
                    "Provides management commentary for the calculated P&L, including "
                    "material favourable and unfavourable movements, profitability "
                    "observations, risks, exceptions, and management attention items."
                ),
            ),
            "finance_rules_result": AgentContextDefinition(
                result_key="finance_rules_result",
                source_name="Finance Rules Agent",
                purpose=(
                    "Provides control checks, warnings, errors, and finance-rule "
                    "validation results."
                ),
            ),
            "anomaly_result": AgentContextDefinition(
                result_key="anomaly_result",
                source_name="Anomaly Agent",
                purpose=(
                    "Provides unusual movements, priority levels, and anomaly findings."
                ),
            ),
            "root_cause_result": AgentContextDefinition(
                result_key="root_cause_result",
                source_name="Root Cause Agent",
                purpose=(
                    "Provides identified causes, confidence levels, unresolved items, "
                    "and supporting findings."
                ),
            ),
            "recommendation_result": AgentContextDefinition(
                result_key="recommendation_result",
                source_name="Recommendation Agent",
                purpose=(
                    "Provides prioritized management actions, monitoring actions, and "
                    "unresolved root causes."
                ),
            ),
            "kpi_result": AgentContextDefinition(
                result_key="kpi_result",
                source_name="KPI Agent",
                purpose=(
                    "Provides selected KPI values, dimensions, unavailable KPIs, and "
                    "unknown KPI requests."
                ),
            ),
            "commentary_result": AgentContextDefinition(
                result_key="commentary_result",
                source_name="Commentary Agent",
                purpose=(
                    "Provides management commentary, positive drivers, risks, and "
                    "items requiring management attention."
                ),
            ),
            "report_result": AgentContextDefinition(
                result_key="report_result",
                source_name="Report Agent",
                purpose=(
                    "Provides the assembled management report, executive summary, "
                    "key risks, and management actions."
                ),
            ),
        }
    )

    _PRIMARY_RESULTS_BY_FLOW: Mapping[str, tuple[str, ...]] = MappingProxyType(
        {
            "kpi": (
                "kpi_result",
                "operations_result",
            ),
            "budget": (
                "budget_result",
            ),
            "forecast": (
                "forecast_result",
                "operations_result",
            ),
            "variance": (
                "variance_result",
                "operations_result",
                "budget_result",
            ),
            "scenario": (
                "scenario_result",
                "forecast_result",
            ),
            "pnl": (
                "pnl_result",
                "pnl_commentary_result",
            ),
            "gp_variance": (
                "gp_variance_result",
            ),
            "full": (
                "report_result",
                "commentary_result",
                "kpi_result",
                "variance_result",
                "forecast_result",
                "scenario_result",
                "budget_result",
                "operations_result",
            ),
        }
    )

    _SUPPORTING_RESULTS_BY_FLOW: Mapping[str, tuple[str, ...]] = MappingProxyType(
        {
            "kpi": (
                "anomaly_result",
                "commentary_result",
                "finance_rules_result",
            ),
            "budget": (
                "operations_result",
                "variance_result",
                "finance_rules_result",
                "commentary_result",
            ),
            "forecast": (
                "scenario_result",
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "kpi_result",
                "commentary_result",
            ),
            "variance": (
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "kpi_result",
                "finance_rules_result",
                "commentary_result",
                "report_result",
            ),
            "scenario": (
                "operations_result",
                "budget_result",
                "variance_result",
                "kpi_result",
                "recommendation_result",
                "commentary_result",
            ),
            "pnl": (
                "finance_rules_result",
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "commentary_result",
                "report_result",
            ),
            "full": (
                "anomaly_result",
                "root_cause_result",
                "recommendation_result",
                "finance_rules_result",
            ),
        }
    )

    _DEFAULT_PRIMARY_RESULTS: tuple[str, ...] = (
        "report_result",
        "commentary_result",
        "kpi_result",
        "variance_result",
        "forecast_result",
        "scenario_result",
        "budget_result",
        "operations_result",
    )

    _DEFAULT_SUPPORTING_RESULTS: tuple[str, ...] = (
        "gp_variance_result",
        "pnl_result",
        "pnl_commentary_result",
        "anomaly_result",
        "root_cause_result",
        "recommendation_result",
        "finance_rules_result",
    )

    def build(
        self,
        selected_flow: str,
        finance_analysis: Mapping[str, Any],
        hybrid_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a structured, flow-aware finance response context.

        Args:
            selected_flow:
                LangGraph finance flow such as ``variance``, ``forecast``,
                or ``pnl``.
            finance_analysis:
                JSON-compatible outputs collected from the finance agents.

        Returns:
            A dictionary containing primary analysis, supporting analysis,
            source metadata, and explicit availability information.

        Raises:
            TypeError:
                If ``selected_flow`` is not a string or ``finance_analysis`` is
                not a mapping.
            ValueError:
                If ``selected_flow`` is empty after whitespace is removed.
        """

        normalized_flow = self._normalize_flow(selected_flow)
        validated_analysis = self._validate_analysis(finance_analysis)
        if hybrid_metadata is not None and not isinstance(
            hybrid_metadata,
            Mapping,
        ):
            raise TypeError("hybrid_metadata must be a mapping or None.")

        primary_keys = self._PRIMARY_RESULTS_BY_FLOW.get(
            normalized_flow,
            self._DEFAULT_PRIMARY_RESULTS,
        )
        supporting_keys = self._SUPPORTING_RESULTS_BY_FLOW.get(
            normalized_flow,
            self._DEFAULT_SUPPORTING_RESULTS,
        )

        primary_analysis = self._select_results(
            validated_analysis,
            primary_keys,
        )
        supporting_analysis = self._select_results(
            validated_analysis,
            supporting_keys,
            excluded_keys=set(primary_analysis),
        )

        included_keys = tuple(primary_analysis) + tuple(supporting_analysis)
        remaining_analysis = {
            key: deepcopy(value)
            for key, value in validated_analysis.items()
            if key not in included_keys and self._has_meaningful_value(value)
        }

        if remaining_analysis:
            supporting_analysis.update(remaining_analysis)
            included_keys = tuple(primary_analysis) + tuple(supporting_analysis)

        available_keys = [
            key
            for key in self._AGENT_DEFINITIONS
            if key in validated_analysis
            and self._has_meaningful_value(validated_analysis[key])
        ]
        unavailable_keys = [
            key
            for key in self._AGENT_DEFINITIONS
            if key not in available_keys
        ]

        result = {
            "context_version": "1.0",
            "selected_flow": normalized_flow,
            "response_instruction": (
                "Use only the supplied values. Do not invent missing financial "
                "figures, causes, recommendations, or source references. Do not "
                "recalculate values already supplied by an agent. Clearly label "
                "information as unavailable only when the required source data is "
                "genuinely absent."
            ),
            "primary_analysis": primary_analysis,
            "supporting_analysis": supporting_analysis,
            "internal_sources": self._build_internal_sources(included_keys),
            "data_availability": {
                "available_results": available_keys,
                "unavailable_results": unavailable_keys,
                "has_primary_analysis": bool(primary_analysis),
                "has_supporting_analysis": bool(supporting_analysis),
            },
        }
        if hybrid_metadata is not None:
            result["hybrid_execution"] = {
                key: deepcopy(hybrid_metadata[key])
                for key in (
                    "execution_mode",
                    "autonomous_status",
                    "fallback_used",
                    "fallback_reason",
                    "review_decision",
                    "evidence_ids",
                    "usage",
                )
                if key in hybrid_metadata
            }
        return result

    @staticmethod
    def _normalize_flow(selected_flow: str) -> str:
        """Validate and normalize a LangGraph flow name."""

        if not isinstance(selected_flow, str):
            raise TypeError("selected_flow must be a string.")

        normalized_flow = selected_flow.strip().lower()
        if not normalized_flow:
            raise ValueError("selected_flow cannot be empty.")

        return normalized_flow

    @staticmethod
    def _validate_analysis(
        finance_analysis: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Validate the finance-analysis container."""

        if not isinstance(finance_analysis, Mapping):
            raise TypeError("finance_analysis must be a mapping.")

        return finance_analysis

    @classmethod
    def _select_results(
        cls,
        finance_analysis: Mapping[str, Any],
        result_keys: tuple[str, ...],
        *,
        excluded_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Select available results while preserving configured order."""

        excluded = excluded_keys or set()

        return {
            key: deepcopy(finance_analysis[key])
            for key in result_keys
            if key not in excluded
            and key in finance_analysis
            and cls._has_meaningful_value(finance_analysis[key])
        }

    @classmethod
    def _build_internal_sources(
        cls,
        included_keys: tuple[str, ...],
    ) -> list[dict[str, str]]:
        """Build human-readable metadata for included internal agent outputs."""

        sources: list[dict[str, str]] = []
        seen: set[str] = set()

        for result_key in included_keys:
            if result_key in seen:
                continue

            definition = cls._AGENT_DEFINITIONS.get(result_key)
            if definition is None:
                sources.append(
                    {
                        "result_key": result_key,
                        "source_name": result_key,
                        "purpose": "Provides additional supplied finance analysis.",
                    }
                )
            else:
                sources.append(
                    {
                        "result_key": definition.result_key,
                        "source_name": definition.source_name,
                        "purpose": definition.purpose,
                    }
                )

            seen.add(result_key)

        return sources

    @staticmethod
    def _has_meaningful_value(value: Any) -> bool:
        """Return whether a result contains usable supplied information."""

        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return bool(value)

        return True


def build_finance_response_context(
    selected_flow: str,
    finance_analysis: Mapping[str, Any],
    hybrid_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function for building structured finance response context."""

    return FinanceResponseContextBuilder().build(
        selected_flow=selected_flow,
        finance_analysis=finance_analysis,
        hybrid_metadata=hybrid_metadata,
    )
