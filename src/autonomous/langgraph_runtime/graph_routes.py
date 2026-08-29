"""Conditional edges for the Decide-Act-Observe loop."""

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def route_after_supervisor(state: FinanceGraphState) -> str:
    if state.get("step_count", 0) >= state.get("max_steps", 8):
        return "recover"
    return {"call_tool": "execute_tool", "validate": "validate", "ask_user": "clarify",
            "request_approval": "approve", "finish": "validate"}.get(state.get("next_action", ""), "recover")


def route_after_observation(state: FinanceGraphState) -> str:
    return "supervisor" if state.get("next_action") == "supervise" else "recover"


def route_after_validation(state: FinanceGraphState) -> str:
    return "reviewer" if state.get("next_action") == "review" else "supervisor"


def route_after_reviewer(state: FinanceGraphState) -> str:
    return "final_answer" if state.get("next_action") == "finalize" else "supervisor"


def route_after_recovery(state: FinanceGraphState) -> str:
    return "final_answer" if state.get("completed") else "supervisor"
