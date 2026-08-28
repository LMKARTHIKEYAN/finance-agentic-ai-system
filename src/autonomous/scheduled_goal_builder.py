"""Convert scheduler triggers into the same goals used by CFO questions."""

from __future__ import annotations

from datetime import date, timedelta

from src.autonomous.goal_builder import GoalBuilder
from src.autonomous.schemas import FinanceGoal


class ScheduledGoalBuilder:
    def __init__(self, goal_builder: GoalBuilder | None = None) -> None:
        self.goal_builder = goal_builder or GoalBuilder()

    def build(self, report_type: str, *, run_date: date) -> FinanceGoal:
        if report_type == "daily":
            request = f"Generate the daily KPI report for {run_date.isoformat()}"
        elif report_type == "weekly":
            start = run_date - timedelta(days=6)
            request = f"Generate the weekly KPI report from {start.isoformat()} to {run_date.isoformat()}"
        elif report_type == "monthly":
            request = (
                f"Generate monthly P&L, revenue variance, GP decomposition, "
                f"and rolling forecast for {run_date:%B %Y}"
            )
        else:
            raise ValueError("report_type must be daily, weekly, or monthly.")
        return self.goal_builder.build(request, reference_date=run_date)
