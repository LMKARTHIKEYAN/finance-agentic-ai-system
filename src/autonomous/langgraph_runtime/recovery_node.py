"""Bounded recovery for failed or invalid graph actions."""

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def recovery_node(state: FinanceGraphState) -> dict[str, object]:
    steps = state.get("step_count", 0) + 1
    if steps >= state.get("max_steps", 8):
        return {"step_count": steps, "final_answer": "The analysis stopped safely after reaching its execution limit.", "completed": True, "next_action": "finish"}
    return {"step_count": steps, "next_action": "supervise"}
