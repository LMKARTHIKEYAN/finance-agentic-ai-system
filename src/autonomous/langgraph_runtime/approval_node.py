"""Pause controlled actions pending explicit human approval."""

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def approval_node(state: FinanceGraphState) -> dict[str, object]:
    decision = state.get("decision", {})
    return {"approval_request": decision.get("question") or "Approval is required for this action.", "completed": True}
