"""
LangGraph workflow for the Finance Agentic AI System.

This module connects:

- Shared graph state
- Deterministic request router
- Existing agent node wrappers
- Conditional error handling
- Successful graph completion

Business logic must remain inside the existing agents. This module only
defines workflow sequencing and conditional routing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.orchestrator.nodes import (
    anomaly_node,
    budget_node,
    clean_budget_node,
    clean_operations_node,
    commentary_node,
    complete_node,
    error_node,
    finance_rules_node,
    forecast_node,
    kpi_node,
    operations_analysis_node,
    profile_budget_node,
    profile_operations_node,
    recommendation_node,
    report_node,
    root_cause_node,
    scenario_node,
    validate_budget_node,
    validate_operations_node,
    variance_node,
)
from src.orchestrator.router import route_request
from src.orchestrator.state import FinanceGraphState


NodeFunction = Callable[[FinanceGraphState], FinanceGraphState]

SUPPORTED_FLOWS = (
    "kpi",
    "budget",
    "forecast",
    "variance",
    "scenario",
    "full",
)


# ----------------------------------------------------------------------
# Router conditions
# ----------------------------------------------------------------------


def select_initial_route(
    state: FinanceGraphState,
) -> Literal[
    "kpi",
    "budget",
    "forecast",
    "variance",
    "scenario",
    "full",
    "error",
]:
    """
    Select the first graph route after the router node.

    Args:
        state:
            Shared graph state after ``route_request`` has executed.

    Returns:
        A supported route name or ``error``.
    """

    if state.get("execution_status") == "failed":
        return "error"

    selected_flow = state.get("selected_flow", "unknown")

    if selected_flow in SUPPORTED_FLOWS:
        return selected_flow

    return "error"


def select_next_step(
    state: FinanceGraphState,
) -> Literal["continue", "error"]:
    """
    Decide whether graph execution should continue.

    Every operational node stores failures in the shared state instead of
    raising them out of the graph. This condition checks that status after
    each node.

    Args:
        state:
            Current shared graph state.

    Returns:
        ``continue`` when the previous node succeeded.
        ``error`` when the previous node failed.
    """

    if state.get("execution_status") == "failed":
        return "error"

    return "continue"


# ----------------------------------------------------------------------
# Route registration helper
# ----------------------------------------------------------------------


def _add_linear_route(
    builder: StateGraph,
    route_name: str,
    nodes: Sequence[tuple[str, NodeFunction]],
    *,
    final_node_ends_graph: bool = False,
) -> str:
    """
    Add one route as a safe sequential workflow.

    Route-specific node names are generated even when different routes use
    the same Python node function. This prevents multiple unconditional
    outgoing edges from the same graph node.

    Example:

        Route name:
            ``kpi``

        Logical node:
            ``validate_operations``

        Graph node name:
            ``kpi__validate_operations``

    Args:
        builder:
            LangGraph StateGraph builder.

        route_name:
            Supported workflow name.

        nodes:
            Ordered collection of logical node names and node functions.

        final_node_ends_graph:
            Set to True when the final node itself completes execution,
            such as ``report_node``.

    Returns:
        Name of the first node in the registered route.

    Raises:
        ValueError:
            If no route nodes are supplied.
    """

    if not nodes:
        raise ValueError(
            f"Route '{route_name}' must contain at least one node."
        )

    graph_node_names: list[str] = []

    for logical_name, node_function in nodes:
        graph_node_name = f"{route_name}__{logical_name}"

        builder.add_node(
            graph_node_name,
            node_function,
        )

        graph_node_names.append(graph_node_name)

    for index, current_node_name in enumerate(graph_node_names):
        is_final_node = index == len(graph_node_names) - 1

        if is_final_node:
            if final_node_ends_graph:
                builder.add_edge(current_node_name, END)
                continue

            builder.add_conditional_edges(
                current_node_name,
                select_next_step,
                {
                    "continue": END,
                    "error": "error",
                },
            )
            continue

        next_node_name = graph_node_names[index + 1]

        builder.add_conditional_edges(
            current_node_name,
            select_next_step,
            {
                "continue": next_node_name,
                "error": "error",
            },
        )

    return graph_node_names[0]


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


def build_finance_graph() -> Any:
    """
    Build and compile the Finance Agentic AI LangGraph workflow.

    Supported workflows:

    KPI:
        Router
        -> Validate Operations
        -> Clean Operations
        -> Operations Analysis
        -> KPI
        -> Commentary
        -> Complete

    Budget:
        Router
        -> Validate Budget
        -> Clean Budget
        -> Profile Budget
        -> Budget
        -> KPI
        -> Commentary
        -> Complete

    Forecast:
        Router
        -> Validate Operations
        -> Clean Operations
        -> Operations Analysis
        -> Forecast
        -> KPI
        -> Commentary
        -> Complete

    Variance:
        Router
        -> Validate Operations
        -> Validate Budget
        -> Clean Operations
        -> Clean Budget
        -> Operations Analysis
        -> Budget
        -> Variance
        -> Anomaly
        -> Root Cause
        -> Recommendation
        -> KPI
        -> Commentary
        -> Complete

    Scenario:
        Router
        -> Validate Operations
        -> Clean Operations
        -> Operations Analysis
        -> Forecast
        -> Scenario
        -> KPI
        -> Commentary
        -> Complete

    Full:
        Router
        -> Validate Operations
        -> Validate Budget
        -> Clean Operations
        -> Clean Budget
        -> Profile Operations
        -> Profile Budget
        -> Operations Analysis
        -> Budget
        -> Forecast
        -> Scenario
        -> Variance
        -> Finance Rules
        -> Anomaly
        -> Root Cause
        -> Recommendation
        -> KPI
        -> Commentary
        -> Report

    Returns:
        Compiled LangGraph application.
    """

    builder = StateGraph(FinanceGraphState)

    # ------------------------------------------------------------------
    # Common control nodes
    # ------------------------------------------------------------------

    builder.add_node(
        "router",
        route_request,
    )

    builder.add_node(
        "error",
        error_node,
    )

    # ------------------------------------------------------------------
    # KPI route
    # ------------------------------------------------------------------

    kpi_entry = _add_linear_route(
        builder,
        "kpi",
        [
            (
                "validate_operations",
                validate_operations_node,
            ),
            (
                "clean_operations",
                clean_operations_node,
            ),
            (
                "operations_analysis",
                operations_analysis_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "complete",
                complete_node,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Budget route
    # ------------------------------------------------------------------

    budget_entry = _add_linear_route(
        builder,
        "budget",
        [
            (
                "validate_budget",
                validate_budget_node,
            ),
            (
                "clean_budget",
                clean_budget_node,
            ),
            (
                "profile_budget",
                profile_budget_node,
            ),
            (
                "budget",
                budget_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "complete",
                complete_node,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Forecast route
    # ------------------------------------------------------------------

    forecast_entry = _add_linear_route(
        builder,
        "forecast",
        [
            (
                "validate_operations",
                validate_operations_node,
            ),
            (
                "clean_operations",
                clean_operations_node,
            ),
            (
                "operations_analysis",
                operations_analysis_node,
            ),
            (
                "forecast",
                forecast_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "complete",
                complete_node,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Variance and root-cause route
    # ------------------------------------------------------------------

    variance_entry = _add_linear_route(
        builder,
        "variance",
        [
            (
                "validate_operations",
                validate_operations_node,
            ),
            (
                "validate_budget",
                validate_budget_node,
            ),
            (
                "clean_operations",
                clean_operations_node,
            ),
            (
                "clean_budget",
                clean_budget_node,
            ),
            (
                "operations_analysis",
                operations_analysis_node,
            ),
            (
                "budget",
                budget_node,
            ),
            (
                "variance",
                variance_node,
            ),
            (
                "anomaly",
                anomaly_node,
            ),
            (
                "root_cause",
                root_cause_node,
            ),
            (
                "recommendation",
                recommendation_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "complete",
                complete_node,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Scenario route
    # ------------------------------------------------------------------

    scenario_entry = _add_linear_route(
        builder,
        "scenario",
        [
            (
                "validate_operations",
                validate_operations_node,
            ),
            (
                "clean_operations",
                clean_operations_node,
            ),
            (
                "operations_analysis",
                operations_analysis_node,
            ),
            (
                "forecast",
                forecast_node,
            ),
            (
                "scenario",
                scenario_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "complete",
                complete_node,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Full management-analysis route
    # ------------------------------------------------------------------

    full_entry = _add_linear_route(
        builder,
        "full",
        [
            (
                "validate_operations",
                validate_operations_node,
            ),
            (
                "validate_budget",
                validate_budget_node,
            ),
            (
                "clean_operations",
                clean_operations_node,
            ),
            (
                "clean_budget",
                clean_budget_node,
            ),
            (
                "profile_operations",
                profile_operations_node,
            ),
            (
                "profile_budget",
                profile_budget_node,
            ),
            (
                "operations_analysis",
                operations_analysis_node,
            ),
            (
                "budget",
                budget_node,
            ),
            (
                "forecast",
                forecast_node,
            ),
            (
                "scenario",
                scenario_node,
            ),
            (
                "variance",
                variance_node,
            ),
            (
                "finance_rules",
                finance_rules_node,
            ),
            (
                "anomaly",
                anomaly_node,
            ),
            (
                "root_cause",
                root_cause_node,
            ),
            (
                "recommendation",
                recommendation_node,
            ),
            (
                "kpi",
                kpi_node,
            ),
            (
                "commentary",
                commentary_node,
            ),
            (
                "report",
                report_node,
            ),
        ],
        final_node_ends_graph=True,
    )

    # ------------------------------------------------------------------
    # Entry and routing
    # ------------------------------------------------------------------

    builder.add_edge(
        START,
        "router",
    )

    builder.add_conditional_edges(
        "router",
        select_initial_route,
        {
            "kpi": kpi_entry,
            "budget": budget_entry,
            "forecast": forecast_entry,
            "variance": variance_entry,
            "scenario": scenario_entry,
            "full": full_entry,
            "error": "error",
        },
    )

    builder.add_edge(
        "error",
        END,
    )

    return builder.compile()


# Compiled graph available for direct import.
finance_graph = build_finance_graph()


def run_finance_graph(
    initial_state: FinanceGraphState,
) -> FinanceGraphState:
    """
    Execute the compiled Finance Agentic AI graph.

    Args:
        initial_state:
            Initial graph state containing at least ``user_request`` and the
            data required by the selected flow.

    Returns:
        Final graph state after successful completion or safe failure.
    """

    result = finance_graph.invoke(initial_state)

    return FinanceGraphState(**result)