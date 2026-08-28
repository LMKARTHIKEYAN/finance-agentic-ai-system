"""Human-in-the-loop approval records for high-impact actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ApprovalRecord:
    action_id: str
    approved: bool
    approved_by: str
    approved_at: datetime


class ApprovalGateway:
    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}

    def approve(self, action_id: str, approved_by: str) -> ApprovalRecord:
        if not action_id.strip() or not approved_by.strip():
            raise ValueError("action_id and approved_by are required.")
        record = ApprovalRecord(action_id, True, approved_by, datetime.now(timezone.utc))
        self._records[action_id] = record
        return record

    def is_approved(self, action_id: str) -> bool:
        return bool(self._records.get(action_id) and self._records[action_id].approved)
