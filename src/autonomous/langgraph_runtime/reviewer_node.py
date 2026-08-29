"""LLM evidence reviewer for the complex path."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState
from src.autonomous.langgraph_runtime.prompts import REVIEWER_PROMPT
from src.llm.client import StructuredLLMClient


class GraphReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approved", "more_evidence", "rejected"]
    rationale: str = Field(min_length=1)
    caveats: list[str] = Field(default_factory=list)


class ReviewerNode:
    def __init__(self, llm: StructuredLLMClient) -> None:
        self.llm = llm

    def __call__(self, state: FinanceGraphState) -> dict[str, object]:
        response = self.llm.generate_structured(
            messages=[{"role": "system", "content": REVIEWER_PROMPT},
                      {"role": "user", "content": json.dumps({"goal": state.get("request"), "evidence": state.get("evidence", []), "validation": state.get("validation", {})}, default=str)}],
            response_model=GraphReview,
        )
        review = response.output.model_dump(mode="python")
        return {"review": review, "next_action": "finalize" if review["decision"] == "approved" else "supervise",
                "execution_trace": [*state.get("execution_trace", []), {
                    "step": state.get("step_count", 0), "node": "reviewer",
                    "action": review["decision"], "rationale": review["rationale"],
                }]}
