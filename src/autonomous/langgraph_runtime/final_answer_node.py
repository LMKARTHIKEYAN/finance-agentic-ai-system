"""Produce a cited final answer from verified evidence only."""

from __future__ import annotations

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def final_answer_node(state: FinanceGraphState) -> dict[str, object]:
    existing = state.get("final_answer")
    if existing:
        return {"completed": True, "execution_trace": [*state.get("execution_trace", []), {
            "step": state.get("step_count", 0), "node": "final_answer", "action": "completed",
        }]}
    evidence = state.get("evidence", [])
    refs = [f"[{item.get('evidence_id')}]" for item in evidence if item.get("evidence_id")]
    return {"final_answer": "Validated autonomous analysis completed " + " ".join(refs) + ".", "completed": True,
            "execution_trace": [*state.get("execution_trace", []), {
                "step": state.get("step_count", 0), "node": "final_answer", "action": "completed",
            }]}
