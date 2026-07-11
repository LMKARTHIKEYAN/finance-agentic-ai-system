"""
Deterministic request router for the Finance Agentic AI LangGraph workflow.

The router reads the user's request from the shared graph state and selects
the appropriate workflow.

The first version intentionally uses keyword-based routing instead of an LLM.
"""

from __future__ import annotations

from src.orchestrator.state import FinanceGraphState, FlowType


def identify_flow(user_request: str) -> FlowType:
    """
    Identify the workflow requested by the user.

    Args:
        user_request: Natural-language request received from the user.

    Returns:
        One of the supported workflow names:
        ``kpi``, ``budget``, ``forecast``, ``variance``, ``scenario``,
        ``full``, or ``unknown``.
    """

    normalized_request = user_request.strip().lower()

    if not normalized_request:
        return "unknown"

    full_keywords = (
        "complete management analysis",
        "complete management report",
        "full analysis",
        "full management analysis",
        "full management report",
        "run complete analysis",
        "run full analysis",
        "management report",
        "complete report",
        "end to end analysis",
        "end-to-end analysis",
    )

    variance_keywords = (
        "variance",
        "below budget",
        "above budget",
        "actual vs budget",
        "actual versus budget",
        "budget vs actual",
        "budget versus actual",
        "revenue below budget",
        "revenue above budget",
        "explain revenue",
        "why is revenue",
        "root cause",
        "root-cause",
    )

    scenario_keywords = (
        "scenario",
        "what if",
        "what-if",
        "sensitivity",
        "best case",
        "worst case",
        "base case",
    )

    forecast_keywords = (
        "forecast",
        "projection",
        "predict",
        "next month",
        "next 3 months",
        "next three months",
        "future revenue",
        "future performance",
    )

    budget_keywords = (
        "budget",
        "budget analysis",
        "budget performance",
        "budget plan",
        "prepare budget",
    )

    kpi_keywords = (
        "kpi",
        "key performance indicator",
        "performance metric",
        "performance metrics",
        "business performance",
        "show performance",
    )

    # More specific and broader workflows must be checked first.
    if any(keyword in normalized_request for keyword in full_keywords):
        return "full"

    if any(keyword in normalized_request for keyword in variance_keywords):
        return "variance"

    if any(keyword in normalized_request for keyword in scenario_keywords):
        return "scenario"

    if any(keyword in normalized_request for keyword in forecast_keywords):
        return "forecast"

    if any(keyword in normalized_request for keyword in budget_keywords):
        return "budget"

    if any(keyword in normalized_request for keyword in kpi_keywords):
        return "kpi"

    return "unknown"


def route_request(state: FinanceGraphState) -> FinanceGraphState:
    """
    Update the graph state with the selected workflow.

    This function is intended to be used as the LangGraph router node.

    Args:
        state: Current FinanceGraphState.

    Returns:
        A state update containing the selected flow and execution status.
    """

    user_request = state.get("user_request", "")
    selected_flow = identify_flow(user_request)

    if selected_flow == "unknown":
        return {
            **state,
            "selected_flow": "unknown",
            "execution_status": "failed",
            "error_message": "Unable to identify a supported workflow.",
            "errors": [
                *state.get("errors", []),
                "Unable to identify a supported workflow.",
            ],
        }

    return {
        **state,
        "selected_flow": selected_flow,
        "execution_status": "running",
        "error_message": "",
        "errors": state.get("errors", []),
    }