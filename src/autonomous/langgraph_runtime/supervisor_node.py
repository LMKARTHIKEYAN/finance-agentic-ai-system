"""LLM supervisor that chooses the next approved action."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState
from src.autonomous.langgraph_runtime.prompts import SUPERVISOR_PROMPT
from src.autonomous.langgraph_runtime.tool_catalog import LangGraphToolCatalog
from src.llm.client import StructuredLLMClient


class SupervisorToolArguments(BaseModel):
    """Strict LLM-visible business inputs; trusted objects are excluded."""

    model_config = ConfigDict(extra="forbid")
    start_date: str | None = None
    end_date: str | None = None
    current_start: str | None = None
    current_end: str | None = None
    comparison_start: str | None = None
    comparison_end: str | None = None
    comparison_label: str | None = None
    category: str | None = None
    dimension: str | None = None
    frequency: str | None = None
    metric: str | None = None
    analysis_scope: str | None = None
    requested_kpis: list[str] | None = None
    threshold: float | None = None
    periods: int | None = None
    top_k: int | None = None
    score_threshold: float | None = None
    max_excerpt_characters: int | None = None
    order_change_percentage: float | None = None
    aov_change_amount: float | None = None
    direct_cost_change_percentage: float | None = None
    target_gp_percentage: float | None = None
    minimum_gp_percentage: float | None = None
    maximum_cancellation_percentage: float | None = None
    issue: str | None = None
    recommended_action: str | None = None
    owner: str | None = None
    due_date: str | None = None
    financial_impact: float | None = None


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["call_tool", "validate", "ask_user", "request_approval", "finish"]
    rationale: str = Field(min_length=1)
    tool_name: str | None = None
    arguments: SupervisorToolArguments = Field(
        default_factory=SupervisorToolArguments
    )
    question: str | None = None


class SupervisorNode:
    def __init__(self, llm: StructuredLLMClient, catalog: LangGraphToolCatalog) -> None:
        self.llm, self.catalog = llm, catalog

    def __call__(self, state: FinanceGraphState) -> dict[str, Any]:
        response = self.llm.generate_structured(
            messages=[
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": json.dumps({
                    "goal": state.get("request", ""),
                    "available_tools": self.catalog.describe(),
                    "observations": state.get("observations", []),
                    "validation": state.get("validation", {}),
                    "step_count": state.get("step_count", 0),
                }, default=str)},
            ],
            response_model=SupervisorDecision,
        )
        decision = response.output.model_dump(mode="python")
        decision["arguments"] = {
            key: value for key, value in decision["arguments"].items()
            if value is not None
        }
        if decision["action"] == "call_tool" and not self.catalog.contains(decision.get("tool_name") or ""):
            return {"error": "Supervisor selected an unapproved tool.", "next_action": "recover",
                    "execution_trace": [*state.get("execution_trace", []), {
                        "step": state.get("step_count", 0), "node": "supervisor",
                        "action": "rejected_tool", "tool_name": decision.get("tool_name"),
                    }]}
        return {"decision": decision, "next_action": decision["action"],
                "execution_trace": [*state.get("execution_trace", []), {
                    "step": state.get("step_count", 0), "node": "supervisor",
                    "action": decision["action"], "tool_name": decision.get("tool_name"),
                    "rationale": decision.get("rationale"),
                }]}
