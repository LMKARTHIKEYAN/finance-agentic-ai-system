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
