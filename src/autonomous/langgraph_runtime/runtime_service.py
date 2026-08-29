"""Application-facing service for the complex LangGraph path."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.autonomous.langgraph_runtime.llm_client import build_langgraph_llm
from src.autonomous.langgraph_runtime.reviewer_node import ReviewerNode
from src.autonomous.langgraph_runtime.supervisor_node import SupervisorNode
from src.autonomous.langgraph_runtime.tool_catalog import LangGraphToolCatalog
from src.autonomous.langgraph_runtime.tool_executor_node import ToolExecutorNode
from src.autonomous.langgraph_runtime.validation_node import ValidationNode
from src.autonomous.langgraph_runtime.workflow import build_workflow


class LangGraphRuntimeService:
    def __init__(self, *, llm=None, validator=None, checkpointer=None, max_steps: int = 8) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        self.max_steps = max_steps
        self._trusted_contexts: dict[str, dict[str, Any]] = {}
        client = llm or build_langgraph_llm()
        catalog = LangGraphToolCatalog()
        self.graph = build_workflow(supervisor=SupervisorNode(client, catalog),
                                    tool_executor=ToolExecutorNode(context_provider=self._trusted_context),
                                    validator=ValidationNode(validator),
                                    reviewer=ReviewerNode(client), checkpointer=checkpointer)

    def _trusted_context(self, state: dict[str, Any]) -> dict[str, Any]:
        context_id = str(state.get("context", {}).get("context_id", ""))
        if context_id not in self._trusted_contexts:
            raise ValueError("Trusted execution context is unavailable.")
        return self._trusted_contexts[context_id]

    def run(self, request: str, *, context: dict[str, Any] | None = None, thread_id: str | None = None) -> dict[str, Any]:
        if not request.strip():
            raise ValueError("request cannot be empty.")
        resolved_thread_id = thread_id or uuid4().hex
        context_id = f"context-{uuid4().hex}"
        self._trusted_contexts[context_id] = dict(context or {})
        config = {"configurable": {"thread_id": resolved_thread_id}}
        return dict(self.graph.invoke({"request": request, "context": {"context_id": context_id},
                                       "observations": [], "evidence": [], "step_count": 0,
                                       "max_steps": self.max_steps, "completed": False,
                                       "executed_calls": [], "execution_trace": []}, config=config))
