"""Scheduler trigger adapter for autonomous report goals."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.autonomous.scheduled_goal_builder import ScheduledGoalBuilder


class AutonomousScheduler:
    def __init__(self, *, service: Any, goal_builder: ScheduledGoalBuilder | None = None) -> None:
        self.service = service
        self.goal_builder = goal_builder or ScheduledGoalBuilder()

    def run_due(self, run_date: date | None = None) -> dict[str, Any]:
        today = run_date or date.today()
        report_types = ["daily"]
        if today.weekday() == 0:
            report_types.append("weekly")
        if today.day == 1:
            report_types.append("monthly")
        return {
            kind: self.service.execute_goal(self.goal_builder.build(kind, run_date=today))
            for kind in report_types
        }
