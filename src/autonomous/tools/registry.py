"""Allow-list registry for deterministic autonomous tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from src.autonomous.tools.diagnostic_tools import (
    generate_supported_recommendations,
    identify_supported_root_causes,
)
from src.autonomous.tools.gp_decomposition_tools import (
    calculate_validated_gp_decomposition,
)
from src.autonomous.tools.kpi_tools import calculate_validated_kpis
from src.autonomous.tools.pnl_tools import (
    generate_validated_pnl_analysis,
)
from src.autonomous.tools.retrieval_tools import retrieve_company_context
from src.autonomous.tools.revenue_variance_tools import (
    calculate_validated_revenue_variance,
)
from src.autonomous.tools.trend_analysis_tool import analyze_trends
from src.autonomous.tools.period_comparison_tool import compare_periods
from src.autonomous.tools.drilldown_tool import analyze_drilldown
from src.autonomous.tools.root_cause_tool import identify_operational_drivers
from src.autonomous.tools.anomaly_detection_tool import detect_anomalies
from src.autonomous.tools.category_profitability_tool import analyze_category_profitability
from src.autonomous.tools.customer_route_tool import analyze_customers_and_routes
from src.autonomous.tools.forecast_accuracy_tool import analyze_forecast_accuracy
from src.autonomous.tools.scenario_analysis_tool import analyze_scenario
from src.autonomous.tools.driver_forecast_tool import forecast_from_drivers
from src.autonomous.tools.profitability_alert_tool import detect_profitability_alerts
from src.autonomous.tools.action_tracker_tool import prepare_management_action


ToolCallable = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata for one allow-listed deterministic tool."""

    name: str
    description: str
    function: ToolCallable
    required_inputs: tuple[str, ...]
    result_type: str
    performs_finance_calculation: bool
    external_write: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name cannot be empty.")
        if not self.description.strip():
            raise ValueError("tool description cannot be empty.")
        if not callable(self.function):
            raise TypeError("tool function must be callable.")


class ToolRegistry:
    """Immutable lookup for explicitly approved tools."""

    def __init__(
        self,
        definitions: tuple[ToolDefinition, ...],
    ) -> None:
        names = [item.name for item in definitions]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique.")
        if any(item.external_write for item in definitions):
            raise ValueError(
                "External-write tools require a separate approval registry."
            )
        self._definitions: Mapping[str, ToolDefinition] = MappingProxyType(
            {item.name: item for item in definitions}
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def get(self, name: str) -> ToolDefinition:
        if name not in self._definitions:
            raise KeyError(f"Unknown autonomous tool: {name!r}.")
        return self._definitions[name]

    def as_mapping(self) -> Mapping[str, ToolDefinition]:
        return self._definitions


DEFAULT_TOOL_REGISTRY = ToolRegistry(
    (
        ToolDefinition(
            "calculate_validated_kpis",
            "Select approved KPIs from existing deterministic results.",
            calculate_validated_kpis,
            ("requested_kpis",),
            "kpi",
            True,
        ),
        ToolDefinition(
            "generate_validated_pnl_analysis",
            "Generate Actual, Budget, and variance P&L.",
            generate_validated_pnl_analysis,
            (
                "operations_data",
                "budget_data",
                "corporate_expenses_data",
                "budget_corporate_expenses_data",
            ),
            "pnl",
            True,
        ),
        ToolDefinition(
            "calculate_validated_revenue_variance",
            "Calculate Actual-versus-Budget revenue variance.",
            calculate_validated_revenue_variance,
            ("operations_result", "budget_result"),
            "revenue_variance",
            True,
        ),
        ToolDefinition(
            "calculate_validated_gp_decomposition",
            "Calculate Product- and Portfolio-Level GP% decomposition.",
            calculate_validated_gp_decomposition,
            ("operations_data", "budget_data"),
            "gp_decomposition",
            True,
        ),
        ToolDefinition(
            "identify_supported_root_causes",
            "Identify deterministic causes from reconciled finance evidence.",
            identify_supported_root_causes,
            (),
            "root_cause",
            False,
        ),
        ToolDefinition("analyze_trends", "Calculate daily, weekly, or monthly KPI trends.", analyze_trends, ("finance_context",), "trend", True),
        ToolDefinition("compare_periods", "Compare two periods across operational and financial KPIs.", compare_periods, ("finance_context", "current_start", "current_end", "comparison_start", "comparison_end"), "period_comparison", True),
        ToolDefinition("analyze_drilldown", "Drill into vehicle, cluster, status, or weekday performance.", analyze_drilldown, ("finance_context", "dimension"), "drilldown", True),
        ToolDefinition("identify_operational_drivers", "Identify evidence-supported operating associations without claiming causation.", identify_operational_drivers, ("finance_context",), "operational_drivers", True),
        ToolDefinition("detect_anomalies", "Detect robust daily KPI anomalies.", detect_anomalies, ("finance_context",), "anomaly", True),
        ToolDefinition("analyze_category_profitability", "Calculate category P&L and unit economics.", analyze_category_profitability, ("finance_context",), "category_profitability", True),
        ToolDefinition("analyze_customers_and_routes", "Analyze customer concentration and route profitability when fields are available.", analyze_customers_and_routes, ("finance_context",), "customer_route", True),
        ToolDefinition("analyze_forecast_accuracy", "Calculate forecast error, bias, MAPE, and accuracy.", analyze_forecast_accuracy, ("actuals", "forecasts"), "forecast_accuracy", True),
        ToolDefinition("analyze_scenario", "Calculate deterministic what-if scenarios.", analyze_scenario, ("finance_context",), "scenario", True),
        ToolDefinition("forecast_from_drivers", "Forecast revenue and GP from orders, AOV, and direct-cost drivers.", forecast_from_drivers, ("finance_context",), "driver_forecast", True),
        ToolDefinition("detect_profitability_alerts", "Flag GP and cancellation threshold breaches.", detect_profitability_alerts, ("finance_context",), "profitability_alert", True),
        ToolDefinition("prepare_management_action", "Prepare an approval-aware management action draft.", prepare_management_action, ("issue", "recommended_action", "owner", "due_date"), "management_action", False),
        ToolDefinition(
            "generate_supported_recommendations",
            "Generate deterministic recommendations from root causes.",
            generate_supported_recommendations,
            ("root_cause_result",),
            "recommendation",
            False,
        ),
        ToolDefinition(
            "retrieve_company_context",
            "Retrieve compact read-only company-context excerpts.",
            retrieve_company_context,
            ("retriever", "query"),
            "retrieval",
            False,
        ),
    )
)
