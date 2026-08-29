"""Normalize tool results into observations and evidence."""

from __future__ import annotations

from typing import Any

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


def observation_node(state: FinanceGraphState) -> dict[str, Any]:
    result = state.get("context", {}).get("last_tool_result", {})
    observation = {
        "tool_name": result.get("tool_name"), "status": result.get("status"),
        "payload": result.get("payload", {}), "error": result.get("error"),
        "evidence_id": result.get("evidence_id"),
    }
    observations = [*state.get("observations", []), observation]
    evidence = list(state.get("evidence", []))
    if result.get("evidence_id"):
        evidence.append(observation)
    return {"observations": observations, "evidence": evidence,
            "step_count": state.get("step_count", 0) + 1,
            "next_action": "supervise" if result.get("status") == "completed" else "recover",
            "execution_trace": [*state.get("execution_trace", []), {
                "step": state.get("step_count", 0) + 1, "node": "observation",
                "action": result.get("status"), "tool_name": result.get("tool_name"),
                "evidence_id": result.get("evidence_id"),
            }]}
