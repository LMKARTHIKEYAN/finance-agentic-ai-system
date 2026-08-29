"""Gate completion on deterministic validation supplied by existing tools."""

from __future__ import annotations

from typing import Any, Callable

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState


class ValidationNode:
    def __init__(self, validator: Callable[[FinanceGraphState], dict[str, Any]] | None = None) -> None:
        self.validator = validator or self._default

    @staticmethod
    def _default(state: FinanceGraphState) -> dict[str, Any]:
        ok = bool(state.get("evidence")) and not state.get("error")
        return {"passed": ok, "details": "Evidence present." if ok else "Verified evidence is required."}

    def __call__(self, state: FinanceGraphState) -> dict[str, Any]:
        validation = self.validator(state)
        return {"validation": validation, "next_action": "review" if validation.get("passed") else "supervise",
                "execution_trace": [*state.get("execution_trace", []), {
                    "step": state.get("step_count", 0), "node": "validation",
                    "action": "passed" if validation.get("passed") else "failed",
                }]}
