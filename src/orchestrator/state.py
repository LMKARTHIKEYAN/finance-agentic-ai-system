"""
Shared state for the Finance Agentic AI LangGraph workflow.

Every LangGraph node reads information from this state and returns updates
that are merged back into the same state.

This module contains only state definitions. It must not contain finance
calculations, routing logic, agent execution logic, or graph construction.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import pandas as pd


FlowType = Literal[
    "kpi",
    "budget",
    "forecast",
    "variance",
    "scenario",
    "full",
    "unknown",
]


ExecutionStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]


class FinanceGraphState(TypedDict, total=False):
    """
    Shared state used by the Finance Agentic AI LangGraph workflow.

    The state is defined with ``total=False`` because different graph routes
    require different fields. For example, the KPI route does not necessarily
    require scenario or forecast results.

    Every orchestration node should:

    1. Read the inputs it requires from this state.
    2. Call one existing agent.
    3. Return the result under the appropriate state field.
    4. Record failures in ``errors`` and ``error_message``.
    5. Set ``execution_status`` to ``failed`` when execution cannot continue.
    """

    # ------------------------------------------------------------------
    # User request and routing
    # ------------------------------------------------------------------

    user_request: str
    selected_flow: FlowType

    # ------------------------------------------------------------------
    # Raw input data
    # ------------------------------------------------------------------

    operations_data: pd.DataFrame
    budget_data: pd.DataFrame

    # Optional data inputs retained for future workflows.
    forecast_data: pd.DataFrame
    scenario_data: pd.DataFrame

    # ------------------------------------------------------------------
    # Date and analysis configuration
    # ------------------------------------------------------------------

    start_date: str | None
    end_date: str | None

    start_month: str | None
    end_month: str | None

    group_by: str | list[str] | None
    filters: dict[str, Any]

    # ------------------------------------------------------------------
    # Forecast configuration
    # ------------------------------------------------------------------

    frequency: str
    rolling_window: int
    forecast_periods: int

    # ------------------------------------------------------------------
    # Scenario configuration
    # ------------------------------------------------------------------

    scenario_name: str
    business_assumptions: pd.DataFrame | list[dict[str, Any]]

    # ------------------------------------------------------------------
    # KPI configuration
    # ------------------------------------------------------------------

    requested_kpis: list[str]

    # ------------------------------------------------------------------
    # Validation results
    # ------------------------------------------------------------------

    operations_validation: Any
    budget_validation: Any

    # Backward-compatible general field.
    # This can be used by a smaller route that has only one validation result.
    validation_result: Any

    # ------------------------------------------------------------------
    # Cleaned data
    # ------------------------------------------------------------------

    cleaned_operations_data: pd.DataFrame
    cleaned_budget_data: pd.DataFrame

    # Backward-compatible general field.
    cleaned_data: pd.DataFrame

    # ------------------------------------------------------------------
    # Profiling results
    # ------------------------------------------------------------------

    operations_profile: Any
    budget_profile: Any

    # Backward-compatible general field.
    profile_result: Any

    # ------------------------------------------------------------------
    # Agent results
    # ------------------------------------------------------------------

    operations_result: Any
    budget_result: Any
    forecast_result: Any
    scenario_result: Any
    variance_result: Any
    finance_rules_result: Any
    anomaly_result: Any
    root_cause_result: Any
    recommendation_result: Any
    kpi_result: Any
    commentary_result: Any
    report_result: Any

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    errors: list[str]
    error_message: str

    # Optional information identifying where a failure occurred.
    failed_node: str

    # ------------------------------------------------------------------
    # Execution tracking
    # ------------------------------------------------------------------

    execution_status: ExecutionStatus

    # Optional list of successfully executed graph nodes.
    executed_nodes: list[str]