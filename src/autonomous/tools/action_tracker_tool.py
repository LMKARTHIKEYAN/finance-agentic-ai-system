"""Approval-aware management action drafting and status tracking."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from src.autonomous.schemas import ToolResult


ALLOWED_STATUSES = {"draft", "approved", "in_progress", "completed", "cancelled"}


def prepare_management_action(
    *, issue: str, recommended_action: str, owner: str, due_date: str,
    financial_impact: float | None = None, approved: bool = False,
) -> ToolResult:
    parsed_due_date = date.fromisoformat(due_date)
    status = "approved" if approved else "draft"
    action = {"action_id": f"action-{uuid4().hex}", "issue": issue.strip(), "recommended_action": recommended_action.strip(), "owner": owner.strip(), "due_date": parsed_due_date.isoformat(), "financial_impact": financial_impact, "status": status, "requires_human_approval": not approved}
    if not all((action["issue"], action["recommended_action"], action["owner"])):
        raise ValueError("issue, recommended_action, and owner are required.")
    return ToolResult(call_id="action-tracker", tool_name="prepare_management_action", status="completed", payload={"action": action, "persistence_status": "draft only; connect an approved action repository to persist updates"})
