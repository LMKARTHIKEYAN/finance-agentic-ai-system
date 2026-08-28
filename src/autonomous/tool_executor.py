"""Safe execution boundary for allow-listed autonomous tools."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from src.autonomous.schemas import AutonomousDecision, ToolResult
from src.autonomous.tools.registry import DEFAULT_TOOL_REGISTRY, ToolRegistry


class ToolExecutionError(RuntimeError):
    """Raised only for invalid autonomous tool-execution requests."""


class ToolExecutor:
    """Validate and execute one approved tool call at a time."""

    def __init__(
        self,
        *,
        registry: ToolRegistry = DEFAULT_TOOL_REGISTRY,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry.")
        self.registry = registry

    def execute(self, decision: AutonomousDecision) -> ToolResult:
        if not isinstance(decision, AutonomousDecision):
            raise TypeError("decision must be an AutonomousDecision.")
        if decision.action != "call_tool" or not decision.tool_name:
            raise ToolExecutionError(
                "ToolExecutor accepts only call_tool decisions."
            )
        try:
            definition = self.registry.get(decision.tool_name)
        except KeyError as exc:
            raise ToolExecutionError(
                f"Tool is not approved: {decision.tool_name!r}."
            ) from exc

        missing = set(definition.required_inputs) - set(decision.arguments)
        if missing:
            raise ToolExecutionError(
                "Required tool arguments are missing: "
                + ", ".join(sorted(missing))
                + "."
            )
        unknown = set(decision.arguments) - set(definition.required_inputs)
        if unknown:
            raise ToolExecutionError(
                "Unsupported tool arguments were supplied: "
                + ", ".join(sorted(unknown))
                + "."
            )

        call_id = f"call-{uuid4().hex}"
        try:
            value = definition.function(**decision.arguments)
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                tool_name=definition.name,
                status="failed",
                error=f"{type(exc).__name__}: tool execution failed",
            )
        if isinstance(value, ToolResult):
            return value.model_copy(
                update={"call_id": call_id, "tool_name": definition.name}
            )
        return ToolResult(
            call_id=call_id,
            tool_name=definition.name,
            status="completed",
            evidence_id=f"evidence-{uuid4().hex}",
            payload=_payload(value),
        )


def _payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        return dict(result) if isinstance(result, dict) else {"value": result}
    return {"value": value}
