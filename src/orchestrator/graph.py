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

from src.memory.memory_manager import MemoryManager
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
    pnl_node,
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
    "pnl",
    "full",
)


AGENT_RESULT_FIELDS: tuple[str, ...] = (
    "operations_validation",
    "budget_validation",
    "corporate_expenses_validation",
    "budget_corporate_expenses_validation",
    "validation_result",
    "operations_profile",
    "budget_profile",
    "corporate_expenses_profile",
    "budget_corporate_expenses_profile",
    "profile_result",
    "operations_result",
    "budget_result",
    "forecast_result",
    "scenario_result",
    "variance_result",
    "pnl_result",
    "finance_rules_result",
    "anomaly_result",
    "root_cause_result",
    "recommendation_result",
    "kpi_result",
    "commentary_result",
    "report_result",
)


MEMORY_CONTEXT_FIELDS: tuple[str, ...] = (
    "selected_flow",
    "start_date",
    "end_date",
    "start_month",
    "end_month",
    "group_by",
    "filters",
    "frequency",
    "rolling_window",
    "forecast_periods",
    "scenario_name",
    "requested_kpis",
)


def _build_memory_workflow_context(
    state: FinanceGraphState,
) -> dict[str, Any]:
    """Build lightweight workflow context for memory persistence."""

    context: dict[str, Any] = {}

    for field_name in MEMORY_CONTEXT_FIELDS:
        if field_name in state:
            context[field_name] = state[field_name]

    return context


def _prepare_memory_state(
    initial_state: FinanceGraphState,
    memory_manager: MemoryManager,
) -> FinanceGraphState:
    """Create or restore a memory session before graph execution."""

    prepared_state = FinanceGraphState(**dict(initial_state))

    try:
        supplied_session_id = prepared_state.get("session_id")

        if (
            isinstance(supplied_session_id, str)
            and supplied_session_id.strip()
            and memory_manager.session_exists(supplied_session_id)
        ):
            session_id = supplied_session_id.strip()
            memory_manager.set_question(
                session_id,
                prepared_state.get("user_request", ""),
            )
            memory_manager.set_uploaded_files(
                session_id,
                prepared_state.get("uploaded_files", []),
            )
            memory_manager.update_workflow_context(
                session_id,
                _build_memory_workflow_context(prepared_state),
            )
        else:
            session_id = memory_manager.create_session(
                user_id=prepared_state.get("user_id"),
                question=prepared_state.get("user_request"),
                uploaded_files=prepared_state.get(
                    "uploaded_files",
                    [],
                ),
                workflow_context=(
                    _build_memory_workflow_context(prepared_state)
                ),
                metadata={
                    "source": "langgraph",
                    "component": "finance_graph",
                },
            )

        workflow_id = memory_manager.get_workflow_id(session_id)
        memory_manager.set_workflow_status(session_id, "RUNNING")
        memory_context = memory_manager.build_context(session_id)

        prepared_state.update(
            {
                "session_id": session_id,
                "workflow_id": workflow_id,
                "memory_context": memory_context,
                "memory_status": "context_loaded",
                "memory_error": None,
                "agent_memory_ids": {},
                "report_memory_id": None,
                "workflow_memory_id": None,
            }
        )
    except Exception as exc:  # pragma: no cover
        prepared_state.update(
            {
                "memory_status": "failed",
                "memory_error": f"{type(exc).__name__}: {exc}",
            }
        )

    return prepared_state


def _store_graph_outputs_in_session(
    result: FinanceGraphState,
    memory_manager: MemoryManager,
) -> None:
    """Store available graph outputs in temporary session memory."""

    session_id = result.get("session_id")

    if not isinstance(session_id, str) or not session_id:
        return

    for field_name in AGENT_RESULT_FIELDS:
        output = result.get(field_name)

        if output is not None:
            memory_manager.store_agent_output(
                session_id,
                field_name,
                output,
            )


def _finalize_memory_state(
    result: FinanceGraphState,
    memory_manager: MemoryManager,
) -> FinanceGraphState:
    """Persist graph outputs and return generated memory identifiers."""

    finalized_state = FinanceGraphState(**dict(result))
    session_id = finalized_state.get("session_id")

    if not isinstance(session_id, str) or not session_id:
        return finalized_state

    try:
        finalized_state["memory_status"] = "saving"
        _store_graph_outputs_in_session(
            finalized_state,
            memory_manager,
        )

        if finalized_state.get("execution_status") == "failed":
            memory_manager.set_workflow_status(
                session_id,
                "FAILED",
            )
            finalized_state["memory_status"] = "saved"
            return finalized_state

        memory_result = memory_manager.complete_workflow(
            session_id=session_id,
            report=finalized_state.get("report_result"),
            save_agents=True,
        )

        finalized_state.update(
            {
                "workflow_id": memory_result.workflow_id,
                "report_memory_id": memory_result.report_memory_id,
                "agent_memory_ids": memory_result.agent_memory_ids,
                "memory_status": "saved",
                "memory_error": None,
            }
        )
    except Exception as exc:  # pragma: no cover
        finalized_state.update(
            {
                "memory_status": "failed",
                "memory_error": f"{type(exc).__name__}: {exc}",
            }
        )

    return finalized_state


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
    "pnl",
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

    P&L:
        Router
        -> Validate Operations
        -> Validate Budget
        -> Clean Operations
        -> Clean Budget
        -> P&L
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
    # P&L route
    # ------------------------------------------------------------------

    pnl_entry = _add_linear_route(
        builder,
        "pnl",
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
                "pnl",
                pnl_node,
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
            "pnl": pnl_entry,
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
    *,
    memory_manager: MemoryManager | None = None,
) -> FinanceGraphState:
    """
    Execute the compiled Finance Agentic AI graph.

    Memory integration is enabled when ``memory_manager`` is supplied. This
    keeps existing callers backward-compatible while allowing the service
    layer to provide one shared MemoryManager in production.

    Args:
        initial_state:
            Initial graph state containing ``user_request`` and route data.

        memory_manager:
            Optional shared MemoryManager used to create or restore a session,
            load context, store agent outputs, and persist workflow results.

    Returns:
        Final graph state after successful completion or safe failure.
    """

    prepared_state = (
        _prepare_memory_state(initial_state, memory_manager)
        if memory_manager is not None
        else FinanceGraphState(**dict(initial_state))
    )

    result = FinanceGraphState(
        **finance_graph.invoke(prepared_state)
    )

    if memory_manager is None:
        return result

    return _finalize_memory_state(
        result,
        memory_manager,
    )