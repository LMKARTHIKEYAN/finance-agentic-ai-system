"""Pause the graph when essential goal details are missing."""

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def clarification_node(state: FinanceGraphState) -> dict[str, object]:
    decision = state.get("decision", {})
    return {"pending_question": decision.get("question") or "Please clarify the required period and scope.", "completed": True}
