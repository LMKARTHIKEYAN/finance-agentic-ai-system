"""Structured, secret-safe autonomous execution events and metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExecutionEvent:
    event_type: str
    goal_id: str
    details: dict[str, Any]
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ObservabilityRecorder:
    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    def record(self, event_type: str, goal_id: str, **details: Any) -> None:
        forbidden = {"password", "token", "secret", "smtp_password"}
        safe = {key: value for key, value in details.items() if key.lower() not in forbidden}
        self.events.append(ExecutionEvent(event_type, goal_id, safe))

    def metrics(self) -> dict[str, Any]:
        counts = Counter(item.event_type for item in self.events)
        return {"event_count": len(self.events), "by_type": dict(counts)}
