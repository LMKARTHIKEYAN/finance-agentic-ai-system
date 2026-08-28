"""Normalize tool and validation results into autonomous observations."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.autonomous.schemas import AutonomousDecision, Observation, ToolResult


class ObservationManager:
    """Create compact observations that can drive the next decision."""

    def from_tool_result(
        self,
        decision: AutonomousDecision,
        result: ToolResult,
    ) -> Observation:
        if not isinstance(decision, AutonomousDecision):
            raise TypeError("decision must be an AutonomousDecision.")
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult.")
        completed = result.status == "completed"
        summary = (
            _summarize_payload(result.tool_name, result.payload)
            if completed
            else result.error or f"Tool {result.tool_name} failed."
        )
        return Observation(
            observation_id=f"observation-{uuid4().hex}",
            decision_id=decision.decision_id,
            source=result.tool_name,
            status="completed" if completed else "failed",
            summary=summary,
            payload=dict(result.payload) if completed else {},
            satisfied_criteria=(
                decision.target_criteria if completed else ()
            ),
            evidence_ids=(
                (result.evidence_id,)
                if completed and result.evidence_id
                else ()
            ),
            warnings=result.warnings,
            error_code="tool_failed" if not completed else None,
        )

    def validation_observation(
        self,
        decision: AutonomousDecision,
        *,
        passed: bool,
        details: str,
        payload: dict[str, Any] | None = None,
    ) -> Observation:
        if decision.action != "validate":
            raise ValueError("validation observation requires validate decision.")
        cleaned = _required_text(details, "details")
        return Observation(
            observation_id=f"observation-{uuid4().hex}",
            decision_id=decision.decision_id,
            source="final_validation",
            status="completed" if passed else "failed",
            summary=cleaned,
            payload=dict(payload or {}),
            satisfied_criteria=decision.target_criteria if passed else (),
            error_code=None if passed else "validation_failed",
        )


def _summarize_payload(tool_name: str, payload: dict[str, Any]) -> str:
    explicit = payload.get("summary")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    keys = sorted(str(key) for key in payload)[:6]
    suffix = f" Result fields: {', '.join(keys)}." if keys else ""
    return f"Tool {tool_name} completed successfully.{suffix}"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned
