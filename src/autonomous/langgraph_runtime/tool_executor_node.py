"""Resolve and execute one supervisor-selected allow-listed tool."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from uuid import uuid4

from src.autonomous.langgraph_runtime.graph_state import FinanceGraphState
from src.autonomous.langgraph_runtime.tool_argument_resolver import (
    ToolArgumentResolutionError,
    ToolArgumentResolver,
)
from src.autonomous.schemas import ToolResult
from src.autonomous.tools.data_tools import serialize_tool_payload
from src.autonomous.tools.registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


class ToolExecutorNode:
    def __init__(
        self,
        *,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
        resolver: ToolArgumentResolver | None = None,
        context_provider: Callable[[FinanceGraphState], Mapping[str, Any]] | None = None,
    ) -> None:
        self.registry = registry
        self.resolver = resolver or ToolArgumentResolver(registry)
        self.context_provider = context_provider

    def __call__(self, state: FinanceGraphState) -> dict[str, Any]:
        raw = state.get("decision", {})
        tool_name = str(raw.get("tool_name") or "")
        proposed = dict(raw.get("arguments") or {})
        signature = json.dumps(
            {"tool_name": tool_name, "arguments": proposed},
            sort_keys=True,
            default=str,
        )
        executed = list(state.get("executed_calls", []))
        if signature in executed:
            result = ToolResult(
                call_id=f"graph-{uuid4().hex}",
                tool_name=tool_name or "unknown",
                status="failed",
                error="Duplicate tool call blocked.",
            )
            return _result_update(state, result, executed)

        try:
            definition = self.registry.get(tool_name)
            trusted_context = (
                self.context_provider(state)
                if self.context_provider is not None
                else state.get("context", {})
            )
            arguments = self.resolver.resolve(
                tool_name,
                proposed,
                trusted_context,
                request=state.get("request", ""),
            )
            value = definition.function(**arguments)
            result = value if isinstance(value, ToolResult) else ToolResult(
                call_id=f"graph-{uuid4().hex}",
                tool_name=tool_name,
                status="completed",
                evidence_id=f"graph-evidence-{uuid4().hex}",
                payload=serialize_tool_payload(value),
            )
            if result.status == "completed" and not result.evidence_id:
                result = result.model_copy(update={
                    "call_id": f"graph-{uuid4().hex}",
                    "tool_name": tool_name,
                    "evidence_id": f"graph-evidence-{uuid4().hex}",
                })
        except (KeyError, ToolArgumentResolutionError) as exc:
            result = ToolResult(
                call_id=f"graph-{uuid4().hex}",
                tool_name=tool_name or "unknown",
                status="failed",
                error=str(exc),
            )
        except Exception as exc:  # boundary intentionally hides data/details
            result = ToolResult(
                call_id=f"graph-{uuid4().hex}",
                tool_name=tool_name or "unknown",
                status="failed",
                error=f"{type(exc).__name__}: approved tool execution failed.",
            )
        executed.append(signature)
        return _result_update(state, result, executed)


def _result_update(
    state: FinanceGraphState,
    result: ToolResult,
    executed: list[str],
) -> dict[str, Any]:
    # ToolResult payloads are already compact/serialized; trusted context stays
    # in graph state and is never copied into the LLM-facing observation.
    return {
        "context": {
            **state.get("context", {}),
            "last_tool_result": result.model_dump(mode="python"),
        },
        "executed_calls": executed,
        "execution_trace": [*state.get("execution_trace", []), {
            "step": state.get("step_count", 0), "node": "tool_executor",
            "action": result.status, "tool_name": result.tool_name,
            "evidence_id": result.evidence_id,
        }],
    }
